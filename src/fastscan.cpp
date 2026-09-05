#include "recastlib/fastscan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/quantize_lut.h>

#include "interval_selection_internal.h"
#include "recastlib/detail/exact_pq4.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define RECAST_FASTSCAN_X86 1
#else
#define RECAST_FASTSCAN_X86 0
#endif

#if RECAST_FASTSCAN_X86 && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(RECASTLIB_LEGACY_SCALAR_PIPELINE)
#include <immintrin.h>
#define RECAST_COMPILE_POSTPROCESS_AVX512 1
#else
#define RECAST_COMPILE_POSTPROCESS_AVX512 0
#endif

namespace recastlib {
namespace {

constexpr size_t kFastScanBlockSize = 32;
constexpr size_t kMetadataPlaneCount = 3;
constexpr size_t kMetadataBlockBytes =
    kFastScanBlockSize * kMetadataPlaneCount;

}  // namespace

size_t packed_metadata_size_bbs32(size_t padded_size) {
  if (padded_size % kFastScanBlockSize != 0) {
    throw std::invalid_argument(
        "packed metadata row count must be a multiple of 32");
  }
  if (padded_size >
      std::numeric_limits<size_t>::max() / kMetadataPlaneCount) {
    throw std::overflow_error("packed metadata size overflow");
  }
  return padded_size * kMetadataPlaneCount;
}

void pack_metadata_bbs32(const Metadata* metadata,
                         size_t size,
                         size_t padded_size,
                         uint8_t* output) {
  if (size > padded_size || padded_size % kFastScanBlockSize != 0 ||
      (size > 0 && metadata == nullptr) ||
      (padded_size > 0 && output == nullptr)) {
    throw std::invalid_argument("invalid bbs=32 metadata packing input");
  }
  if (padded_size == 0) return;

  // A conventional Metadata array is AoS:
  //
  //   [n0 s0 e0] [n1 s1 e1] ... [n31 s31 e31]
  //
  // FastScan consumes 32 vectors together, so each tile is transposed to SoA:
  //
  //   byte  0 .. 31 : n0  n1  ... n31
  //   byte 32 .. 63 : s0  s1  ... s31
  //   byte 64 .. 95 : e0  e1  ... e31
  //
  // Thus one 16-byte load feeds sixteen norm/scale/error lanes directly. The
  // physical size remains 96 bytes for 32 records, exactly 3 bytes per full
  // vector. Tail padding uses 255, matching the conservative overflow sentinel.
  std::memset(output, 255, packed_metadata_size_bbs32(padded_size));
  for (size_t i = 0; i < size; ++i) {
    const size_t block_base = (i / kFastScanBlockSize) * kMetadataBlockBytes;
    const size_t lane = i % kFastScanBlockSize;
    output[block_base + lane] = metadata[i].norm;
    output[block_base + kFastScanBlockSize + lane] = metadata[i].scale;
    output[block_base + 2 * kFastScanBlockSize + lane] = metadata[i].error;
  }
}

namespace {

struct QuantizedLut {
  faiss::AlignedTable<uint8_t> values;
  float scale = 1.0f;
  float bias = 0.0f;
};

QuantizedLut quantize_lut(const std::vector<float>& lut, size_t M) {
  QuantizedLut result;
  result.values.resize(M * 16);

  // Faiss uses one global scale but subtracts a separate minimum from each LUT
  // row. The sum of those row minima becomes the query-level reconstruction
  // bias, while the widest row determines the shared quantization resolution.
  float max_span = 0.0f;
  float constant_sum = 0.0f;
  for (size_t m = 0; m < M; ++m) {
    const float* row = lut.data() + m * 16;
    const auto minmax = std::minmax_element(row, row + 16);
    max_span = std::max(max_span, *minmax.second - *minmax.first);
    constant_sum += *minmax.first;
  }

  if (max_span <= 1e-20f) {
    // Every row is constant, so code values cannot affect the accumulated dot
    // product; represent the complete result with bias alone.
    std::memset(result.values.get(), 0, result.values.nbytes());
    result.bias = constant_sum;
    return result;
  }

  faiss::quantize_lut::quantize_LUT_and_bias(
      1, M, 16, false, lut.data(), nullptr, result.values.get(), M,
      nullptr, &result.scale, &result.bias);
  if (!std::isfinite(result.scale) || result.scale <= 0.0f) {
    throw std::runtime_error("Faiss produced an invalid PQ4 LUT scale");
  }
  return result;
}

LutMode choose_mode(LutMode requested) {
  if (requested == LutMode::Auto) {
    return detail::exact_pq4_avx512_supported()
        ? LutMode::ExactFloat
        : LutMode::QuantizedUint8;
  }
  return requested;
}

/**
 * Four independent float planes produced for one 32-vector FastScan tile.
 *
 * This materialized representation is retained for scan_all(), portable
 * fallback, and reference testing. The production AVX-512 interval-selection
 * path bypasses it and passes ZMM values directly to FusedIntervalAccumulator.
 */
struct alignas(64) PostprocessedBlock32 {
  float estimated_distances[kFastScanBlockSize];
  float uncertainties[kFastScanBlockSize];
  float lower_bounds[kFastScanBlockSize];
  float upper_bounds[kFastScanBlockSize];
};

/**
 * Production interval collector specialized for the FastScan tile stream.
 *
 * A simple record-oriented selector would materialize one DistanceEstimate per
 * scanned row. This production collector instead stores the bounded top-R
 * reservoir as three structure-of-arrays planes. AVX-512 scan output can then
 * compare and compact sixteen lanes directly, without constructing a record or
 * loading an explicit ID for a lane that fails reservoir admission.
 *
 * Two rankings are maintained independently:
 *
 *   1. smallest_uppers_ keeps the global K smallest upper endpoints. Its root
 *      is therefore the final K-th-upper pruning threshold theta_U.
 *   2. The SoA reservoir keeps the global R smallest point estimates, ordered
 *      by (estimated_distance, id). This is the configured candidate cap.
 *
 * Every scanned vector participates in (1), even if it is rejected by (2).
 * Only after all IVF lists have been scanned do we apply lower <= theta_U to
 * the final top-R reservoir. This ordering is part of the selector semantics,
 * not merely a performance optimization.
 *
 * Bounded layout for R > 0:
 *
 *   estimates_:     [e0, e1, ...]
 *   uncertainties_: [u0, u1, ...]
 *   ids_:           [id0,id1,...]
 *
 * The three arrays always share the same logical index and reservoir_size_.
 */
class FusedIntervalAccumulator {
 public:
  FusedIntervalAccumulator(size_t topk, float z_score, size_t max_candidates)
      : topk_(topk),
        z_score_(z_score),
        max_candidates_(max_candidates) {
    if (topk == 0 || !std::isfinite(z_score) || z_score < 0.0f ||
        (max_candidates > 0 && max_candidates < topk) ||
        max_candidates > std::numeric_limits<size_t>::max() / 2) {
      throw std::invalid_argument("invalid fused interval selection parameters");
    }
    if (max_candidates_ > 0) {
      // Growing to 2R amortizes partitioning. Each fuzzy shrink retains an
      // exact rank prefix in [R,1.5R]; finish() performs one exact shrink to R.
      reservoir_capacity_ = 2 * max_candidates_;
      // One 16-lane append may cross the shrink boundary. The extra tile avoids
      // splitting that mask and is discarded by the immediate fuzzy shrink.
      const size_t allocated = reservoir_capacity_ + 16;
      estimates_.resize(allocated);
      uncertainties_.resize(allocated);
      ids_.resize(allocated);
    }
  }

  void add_scalar_block(const VectorId* ids,
                        VectorId implicit_id_base,
                        const float* estimates,
                        const float* uncertainties,
                        const float* uppers,
                        size_t count) {
    // Portable fallback mirrors the AVX-512 ordering exactly: update the
    // global upper threshold first, then admit point estimates to top-R.
    seen_ += count;
    for (size_t lane = 0; lane < count; ++lane) {
      update_upper(uppers[lane]);
    }
    for (size_t lane = 0; lane < count; ++lane) {
      const VectorId id = ids == nullptr
          ? static_cast<VectorId>(implicit_id_base + lane)
          : ids[lane];
      if (!admit(estimates[lane], id)) continue;
      append_scalar(estimates[lane], uncertainties[lane], id);
    }
  }

#if RECAST_COMPILE_POSTPROCESS_AVX512
  /**
   * Consumes sixteen postprocessed lanes while values are still in ZMMs.
   *
   * This is the hot-path handoff between metric reconstruction and selection.
   * No lower endpoints or DistanceEstimate objects exist at this point.
   */
  __attribute__((target("avx512f"), always_inline))
  inline void add_16_avx512(const VectorId* explicit_ids,
                            VectorId implicit_id_base,
                            __m512 estimates,
                            __m512 uncertainties,
                            __m512 uppers,
                            size_t count) {
    if (count == 0) return;
    const __mmask16 logical = count == 16
        ? static_cast<__mmask16>(0xffffu)
        : static_cast<__mmask16>((uint32_t{1} << count) - 1);
    seen_ += count;

    // Stage 1: maintain theta_U, the global K-th smallest upper endpoint.
    //
    // theta_U can only decrease as scanning proceeds. Therefore a lane with
    // upper >= theta_U_before cannot enter the final K smallest uppers and can
    // be rejected by one SIMD compare. Passing uppers are compressed to a tiny
    // stack array and inserted into the exact scalar K-element heap. Rechecking
    // them one by one preserves correctness while theta_U changes within this
    // 16-lane batch.
    __mmask16 upper_mask = logical;
    if (smallest_uppers_.size() >= topk_) {
      upper_mask &= _mm512_cmp_ps_mask(
          uppers, _mm512_set1_ps(smallest_uppers_.top()), _CMP_LT_OQ);
    }
    alignas(64) float passing_uppers[16];
    _mm512_mask_compressstoreu_ps(passing_uppers, upper_mask, uppers);
    const size_t upper_count = static_cast<size_t>(
        __builtin_popcount(static_cast<unsigned>(upper_mask)));
    for (size_t i = 0; i < upper_count; ++i) {
      update_upper(passing_uppers[i]);
    }

    // Stage 2: admit lanes to the point-estimate top-R reservoir.
    //
    // admission_threshold_ is the worst (estimate,id) retained by the most
    // recent partition. It may be stale and too loose, but never too strict;
    // using it can admit extra work but cannot discard a true global top-R row.
    __mmask16 admission_mask = logical;
    __m512i lane_ids;
    bool ids_loaded = false;
    if (has_admission_threshold_) {
      admission_mask &= _mm512_cmp_ps_mask(
          estimates,
          _mm512_set1_ps(admission_threshold_estimate_),
          _CMP_LT_OQ);
      const __mmask16 equal = logical & _mm512_cmp_ps_mask(
          estimates,
          _mm512_set1_ps(admission_threshold_estimate_),
          _CMP_EQ_OQ);
      if (equal != 0) {
        // IDs are only needed to resolve exact estimate ties. Most blocks avoid
        // this load entirely once the estimate threshold becomes selective.
        lane_ids = load_ids_avx512(explicit_ids, implicit_id_base);
        ids_loaded = true;
        admission_mask |= equal & _mm512_cmplt_epu32_mask(
            lane_ids, _mm512_set1_epi32(
                          static_cast<int>(admission_threshold_id_)));
      }
    }
    if (admission_mask == 0) return;
    if (!ids_loaded) {
      // At least one lane survived by estimate. A single contiguous 64-byte ID
      // load is cheaper than materializing one record per original lane.
      lane_ids = load_ids_avx512(explicit_ids, implicit_id_base);
    }

    // vcompressps/vpcompressd write only selected lanes, preserving lane order
    // and keeping all three SoA planes aligned at the same logical positions.
    const size_t accepted = static_cast<size_t>(
        __builtin_popcount(static_cast<unsigned>(admission_mask)));
    ensure_uncapped_space(accepted);
    _mm512_mask_compressstoreu_ps(
        estimates_.data() + reservoir_size_, admission_mask, estimates);
    _mm512_mask_compressstoreu_ps(
        uncertainties_.data() + reservoir_size_, admission_mask, uncertainties);
    _mm512_mask_compressstoreu_epi32(
        ids_.data() + reservoir_size_, admission_mask, lane_ids);
    reservoir_size_ += accepted;
    shrink_if_full();
  }
#endif

  SelectionResult finish() {
    if (seen_ < topk_) {
      throw std::invalid_argument("fewer scanned estimates than topk");
    }

    SelectionResult result;
    // The heap root is valid only after every scanned lane has contributed its
    // upper endpoint. This is the global threshold, not a per-list threshold.
    result.kth_upper_bound = smallest_uppers_.top();
    if (max_candidates_ > 0) {
      shrink_shortlist(max_candidates_);
      result.truncated = seen_ > max_candidates_;
    }
    // Lower endpoints are required only for the final top-R rows. Deferring
    // this arithmetic removes lower construction for almost every scanned row.
    filter_by_lower(result.kth_upper_bound);

    result.candidates.reserve(reservoir_size_);
    for (size_t i = 0; i < reservoir_size_; ++i) {
      result.candidates.push_back(
          DistanceEstimate{ids_[i], estimates_[i], uncertainties_[i]});
    }
    std::sort(
        result.candidates.begin(), result.candidates.end(),
        [](const DistanceEstimate& lhs, const DistanceEstimate& rhs) {
          if (lhs.estimated_distance != rhs.estimated_distance) {
            return lhs.estimated_distance < rhs.estimated_distance;
          }
          return lhs.id < rhs.id;
        });
    return result;
  }

 private:
  // Convert IEEE-754 floats to unsigned integers whose ordinary integer order
  // matches float order. Appending the 32-bit ID then gives one sortable key
  // for the exact lexicographic order (estimated_distance, id), so nth_element
  // moves only eight-byte keys instead of three-array candidate records.
  static uint32_t ordered_float_key(float value) {
    // The selector treats -0 and +0 as equal and then breaks ties by ID.
    if (value == 0.0f) value = 0.0f;
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
  }

  static float float_from_ordered_key(uint32_t ordered) {
    const uint32_t bits = (ordered & 0x80000000u) != 0
        ? ordered ^ 0x80000000u
        : ~ordered;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  static uint64_t rank_key(float estimate, VectorId id) {
    return (static_cast<uint64_t>(ordered_float_key(estimate)) << 32) |
        static_cast<uint64_t>(id);
  }

  bool admit(float estimate, VectorId id) const {
    return !has_admission_threshold_ ||
        estimate < admission_threshold_estimate_ ||
        (estimate == admission_threshold_estimate_ &&
         id < admission_threshold_id_);
  }

  void update_upper(float upper) {
    // A max heap of size K stores the K smallest uppers seen so far. The root
    // is the largest member of that set, i.e. the current K-th smallest upper.
    if (smallest_uppers_.size() < topk_) {
      smallest_uppers_.push(upper);
    } else if (upper < smallest_uppers_.top()) {
      smallest_uppers_.pop();
      smallest_uppers_.push(upper);
    }
  }

  void ensure_uncapped_space(size_t count) {
    // max_candidates=0 deliberately preserves the diagnostic uncapped mode;
    // only that mode grows with the total number of scanned vectors.
    if (max_candidates_ > 0) return;
    estimates_.resize(reservoir_size_ + count);
    uncertainties_.resize(reservoir_size_ + count);
    ids_.resize(reservoir_size_ + count);
  }

  void append_scalar(float estimate, float uncertainty, VectorId id) {
    ensure_uncapped_space(1);
    estimates_[reservoir_size_] = estimate;
    uncertainties_[reservoir_size_] = uncertainty;
    ids_[reservoir_size_] = id;
    ++reservoir_size_;
    shrink_if_full();
  }

  void shrink_if_full() {
    if (max_candidates_ == 0 || reservoir_size_ < reservoir_capacity_) return;
    const size_t fuzzy_max = max_candidates_ +
        (reservoir_capacity_ - max_candidates_) / 2;
    fuzzy_shrink_shortlist(max_candidates_, fuzzy_max);
  }

#if RECAST_COMPILE_POSTPROCESS_AVX512
  __attribute__((target("avx512f"), always_inline))
  static inline __m512i load_ids_avx512(const VectorId* explicit_ids,
                                        VectorId implicit_id_base) {
    if (explicit_ids != nullptr) {
      return _mm512_loadu_si512(
          reinterpret_cast<const void*>(explicit_ids));
    }
    // Contiguous implicit IDs need no memory access: synthesize all sixteen by
    // adding a compile-time lane-number vector to the block base.
    return _mm512_add_epi32(
        _mm512_set1_epi32(static_cast<int>(implicit_id_base)),
        _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15));
  }

  __attribute__((target("avx512f"), noinline))
  static size_t compact_threshold_avx512(float* estimates,
                                         float* uncertainties,
                                         VectorId* ids,
                                         size_t count,
                                         float threshold_estimate,
                                         VectorId threshold_id,
                                         bool include_equal) {
    size_t read = 0;
    size_t write = 0;
    const __m512 threshold = _mm512_set1_ps(threshold_estimate);
    const __m512i threshold_ids =
        _mm512_set1_epi32(static_cast<int>(threshold_id));
    for (; read + 16 <= count; read += 16) {
      const __m512 current_estimates = _mm512_loadu_ps(estimates + read);
      const __m512 current_uncertainties =
          _mm512_loadu_ps(uncertainties + read);
      const __m512i current_ids = _mm512_loadu_si512(
          reinterpret_cast<const void*>(ids + read));
      __mmask16 keep = _mm512_cmp_ps_mask(
          current_estimates, threshold, _CMP_LT_OQ);
      if (include_equal) {
        const __mmask16 equal = _mm512_cmp_ps_mask(
            current_estimates, threshold, _CMP_EQ_OQ);
        keep |= equal & _mm512_cmple_epu32_mask(current_ids, threshold_ids);
      }
      _mm512_mask_compressstoreu_ps(
          estimates + write, keep, current_estimates);
      _mm512_mask_compressstoreu_ps(
          uncertainties + write, keep, current_uncertainties);
      _mm512_mask_compressstoreu_epi32(ids + write, keep, current_ids);
      write += static_cast<size_t>(
          __builtin_popcount(static_cast<unsigned>(keep)));
    }
    for (; read < count; ++read) {
      if (estimates[read] < threshold_estimate ||
          (include_equal && estimates[read] == threshold_estimate &&
           ids[read] <= threshold_id)) {
        estimates[write] = estimates[read];
        uncertainties[write] = uncertainties[read];
        ids[write] = ids[read];
        ++write;
      }
    }
    return write;
  }

  __attribute__((target("avx512f"), noinline))
  static size_t filter_lower_avx512(float* estimates,
                                    float* uncertainties,
                                    VectorId* ids,
                                    size_t count,
                                    float z_score,
                                    float kth_upper) {
    // This is intentionally run after the exact top-R shrink. At this point the
    // lower endpoint is simply estimate - z * uncertainty, and all survivors
    // can be compacted in-place because write never advances beyond read.
    size_t read = 0;
    size_t write = 0;
    const __m512 z = _mm512_set1_ps(z_score);
    const __m512 threshold = _mm512_set1_ps(kth_upper);
    for (; read + 16 <= count; read += 16) {
      const __m512 current_estimates = _mm512_loadu_ps(estimates + read);
      const __m512 current_uncertainties =
          _mm512_loadu_ps(uncertainties + read);
      const __m512 lowers = z_score == 0.0f
          ? current_estimates
          : _mm512_sub_ps(
                current_estimates,
                _mm512_mul_ps(current_uncertainties, z));
      const __mmask16 keep = _mm512_cmp_ps_mask(
          lowers, threshold, _CMP_LE_OQ);
      const __m512i current_ids = _mm512_loadu_si512(
          reinterpret_cast<const void*>(ids + read));
      _mm512_mask_compressstoreu_ps(
          estimates + write, keep, current_estimates);
      _mm512_mask_compressstoreu_ps(
          uncertainties + write, keep, current_uncertainties);
      _mm512_mask_compressstoreu_epi32(ids + write, keep, current_ids);
      write += static_cast<size_t>(
          __builtin_popcount(static_cast<unsigned>(keep)));
    }
    for (; read < count; ++read) {
      const float lower = z_score == 0.0f
          ? estimates[read]
          : estimates[read] - z_score * uncertainties[read];
      if (lower <= kth_upper) {
        estimates[write] = estimates[read];
        uncertainties[write] = uncertainties[read];
        ids[write] = ids[read];
        ++write;
      }
    }
    return write;
  }
#endif

  void fuzzy_shrink_shortlist(size_t q_min, size_t q_max) {
    if (reservoir_size_ <= q_max) return;
    const detail::FuzzyRankThreshold threshold =
        detail::find_fuzzy_rank_threshold(
            estimates_.data(), ids_.data(), reservoir_size_, q_min, q_max,
            &fuzzy_value_scratch_, &fuzzy_id_scratch_);
    admission_threshold_estimate_ = threshold.estimate;
    admission_threshold_id_ = threshold.id;
    has_admission_threshold_ = true;

#if RECAST_COMPILE_POSTPROCESS_AVX512
    if (detail::exact_pq4_avx512_supported()) {
      reservoir_size_ = compact_threshold_avx512(
          estimates_.data(), uncertainties_.data(), ids_.data(),
          reservoir_size_, admission_threshold_estimate_,
          admission_threshold_id_, threshold.equal_to_keep > 0);
    } else
#endif
    {
      size_t write = 0;
      for (size_t read = 0; read < reservoir_size_; ++read) {
        if (estimates_[read] < admission_threshold_estimate_ ||
            (threshold.equal_to_keep > 0 &&
             estimates_[read] == admission_threshold_estimate_ &&
             ids_[read] <= admission_threshold_id_)) {
          estimates_[write] = estimates_[read];
          uncertainties_[write] = uncertainties_[read];
          ids_[write] = ids_[read];
          ++write;
        }
      }
      reservoir_size_ = write;
    }
    if (reservoir_size_ != threshold.output_size) {
      throw std::logic_error("fuzzy reservoir compaction size mismatch");
    }
  }

  void shrink_shortlist(size_t keep) {
    if (reservoir_size_ <= keep) return;
    // Partition only compact rank keys. The resulting K-th key defines an exact
    // threshold; a second linear pass compacts the corresponding SoA entries.
    // This avoids random three-array swaps inside std::nth_element.
    rank_keys_.resize(reservoir_size_);
    for (size_t i = 0; i < reservoir_size_; ++i) {
      rank_keys_[i] = rank_key(estimates_[i], ids_[i]);
    }
    auto middle = rank_keys_.begin() + static_cast<std::ptrdiff_t>(keep - 1);
    std::nth_element(rank_keys_.begin(), middle, rank_keys_.end());
    const uint64_t threshold_key = *middle;
    admission_threshold_estimate_ = float_from_ordered_key(
        static_cast<uint32_t>(threshold_key >> 32));
    admission_threshold_id_ = static_cast<VectorId>(threshold_key);
    has_admission_threshold_ = true;

#if RECAST_COMPILE_POSTPROCESS_AVX512
    if (detail::exact_pq4_avx512_supported()) {
      reservoir_size_ = compact_threshold_avx512(
          estimates_.data(), uncertainties_.data(), ids_.data(),
          reservoir_size_, admission_threshold_estimate_,
          admission_threshold_id_, true);
    } else
#endif
    {
      size_t write = 0;
      for (size_t read = 0; read < reservoir_size_; ++read) {
        if (estimates_[read] < admission_threshold_estimate_ ||
            (estimates_[read] == admission_threshold_estimate_ &&
             ids_[read] <= admission_threshold_id_)) {
          estimates_[write] = estimates_[read];
          uncertainties_[write] = uncertainties_[read];
          ids_[write] = ids_[read];
          ++write;
        }
      }
      reservoir_size_ = write;
    }
    // Vector IDs are unique for an index. Keep this defensive truncation so a
    // malformed duplicate-ID input cannot grow beyond the requested reservoir.
    reservoir_size_ = std::min(reservoir_size_, keep);
  }

  void filter_by_lower(float kth_upper) {
#if RECAST_COMPILE_POSTPROCESS_AVX512
    if (detail::exact_pq4_avx512_supported()) {
      reservoir_size_ = filter_lower_avx512(
          estimates_.data(), uncertainties_.data(), ids_.data(),
          reservoir_size_, z_score_, kth_upper);
      return;
    }
#endif
    size_t write = 0;
    for (size_t read = 0; read < reservoir_size_; ++read) {
      const DistanceEstimate estimate{
          ids_[read], estimates_[read], uncertainties_[read]};
      if (estimate.lower(z_score_) <= kth_upper) {
        estimates_[write] = estimates_[read];
        uncertainties_[write] = uncertainties_[read];
        ids_[write] = ids_[read];
        ++write;
      }
    }
    reservoir_size_ = write;
  }

  size_t topk_;              // K used by the global upper-endpoint threshold.
  float z_score_;            // Query-level confidence-radius multiplier.
  size_t max_candidates_;    // R; zero selects the uncapped diagnostic mode.
  size_t reservoir_capacity_ = 0;  // Fuzzy partition trigger, normally 2R.
  size_t reservoir_size_ = 0;      // Shared logical length of all SoA planes.
  size_t seen_ = 0;                // All logical rows across all probed lists.

  // Independent state for the two orderings described in the class contract.
  std::priority_queue<float> smallest_uppers_;  // Current global K uppers.
  std::vector<float> estimates_;                // SoA plane 0.
  std::vector<float> uncertainties_;            // SoA plane 1.
  std::vector<VectorId> ids_;                    // SoA plane 2.
  std::vector<uint64_t> rank_keys_;              // Scratch for nth_element.
  std::vector<float> fuzzy_value_scratch_;
  std::vector<VectorId> fuzzy_id_scratch_;

  // Worst retained (estimate,id) from the latest reservoir partition. This is
  // an online admission accelerator; finish() still performs an exact top-R.
  float admission_threshold_estimate_ =
      std::numeric_limits<float>::infinity();
  VectorId admission_threshold_id_ = std::numeric_limits<VectorId>::max();
  bool has_admission_threshold_ = false;
};

Metadata metadata_at_lane(const uint8_t* block_metadata, size_t lane) {
  return Metadata{
      block_metadata[lane],
      block_metadata[kFastScanBlockSize + lane],
      block_metadata[2 * kFastScanBlockSize + lane]};
}

template <DistanceMetric Metric>
void scalar_metric_values(const QueryContext& query,
                          float inverse_sqrt_dimension,
                          float norm,
                          float scale,
                          float direction_error,
                          bool overflow,
                          float dot,
                          float* estimate,
                          float* uncertainty) {
  static_assert(
      Metric == DistanceMetric::SquaredL2 ||
          Metric == DistanceMetric::InnerProduct ||
          Metric == DistanceMetric::Cosine,
      "unsupported Recast metric");

  if constexpr (Metric == DistanceMetric::SquaredL2) {
    *estimate = query.query_norm_squared + norm * norm - 2.0f * scale * dot;
    if (!overflow) {
      *uncertainty = 2.0f * norm * query.query_norm * direction_error *
          inverse_sqrt_dimension;
    }
  } else if constexpr (Metric == DistanceMetric::InnerProduct) {
    // Recast's selectors are smaller-is-better. Maximum inner product is thus
    // represented as negative similarity, with q.mean added exactly once.
    *estimate = -(query.mean_dot_query + scale * dot);
    if (!overflow) {
      *uncertainty = norm * query.query_norm * direction_error *
          inverse_sqrt_dimension;
    }
  } else {
    // prepare_query() unit-normalizes q. The stored scale normalizes the raw PQ
    // reconstruction, so scale * dot estimates cos(q, x).
    *estimate = 1.0f - scale * dot;
    if (!overflow) {
      *uncertainty = direction_error * inverse_sqrt_dimension;
    }
  }
}

template <DistanceMetric Metric, bool QuantizedDot, bool ComputeIntervals>
void postprocess_block_scalar(
    const RecastQuantizer& quantizer,
    const QueryContext& query,
    const uint8_t* block_metadata,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    float z_score,
    size_t logical_count,
    PostprocessedBlock32* output) {
  if constexpr (!ComputeIntervals) {
    // The same reference decoder also serves estimate-only scans. Keep that
    // instantiation warning-clean without introducing a runtime branch.
    (void)z_score;
  }
  const float inverse_sqrt_dimension =
      1.0f / std::sqrt(static_cast<float>(quantizer.dimension()));
  for (size_t lane = 0; lane < logical_count; ++lane) {
    float norm = 0.0f;
    float scale = 0.0f;
    float direction_error = 0.0f;
    bool overflow = false;
    quantizer.decode_metadata(
        metadata_at_lane(block_metadata, lane),
        &norm, &scale, &direction_error, &overflow);

    const float dot = QuantizedDot
        ? quantized_bias +
              static_cast<float>(quantized_dots[lane]) *
                  quantized_inverse_scale
        : exact_dots[lane];
    float estimate = std::numeric_limits<float>::infinity();
    float uncertainty = std::numeric_limits<float>::infinity();
    scalar_metric_values<Metric>(
        query, inverse_sqrt_dimension, norm, scale, direction_error,
        overflow, dot, &estimate, &uncertainty);
    if (!std::isfinite(estimate)) {
      estimate = std::numeric_limits<float>::infinity();
    }
    output->estimated_distances[lane] = estimate;
    output->uncertainties[lane] = uncertainty;

    if constexpr (ComputeIntervals) {
      // Reuse the public endpoint semantics for infinity and z=0. In particular,
      // an overflow lane has [-inf,+inf] unless z is exactly zero.
      const DistanceEstimate record{0, estimate, uncertainty};
      output->lower_bounds[lane] = record.lower(z_score);
      output->upper_bounds[lane] = record.upper(z_score);
    }
  }
}

#if RECAST_COMPILE_POSTPROCESS_AVX512

/**
 * Decodes metadata and reconstructs metric intervals for sixteen vector lanes.
 *
 * One Faiss bbs=32 tile calls this function twice: lane_base=0 and 16. For each
 * call, three 16-byte loads read contiguous norm/scale/error planes. The bytes
 * widen to sixteen int32 lanes and then float lanes:
 *
 *   uint8 code[16] -> vpmovzxbd -> int32[16] -> vcvtdq2ps -> float[16]
 *
 * PQ dot values are already lane-aligned by the preceding kernel. Exact mode
 * loads float[16] directly. Uint8-LUT mode widens sixteen uint16 accumulators
 * and reverses Faiss's query-level scale/bias. Metric is a template argument,
 * so only one formula exists in each machine-code instantiation: there is no
 * per-vector or per-block metric switch in this function.
 */
struct MetricValues16 {
  __m512 estimates;           // Point estimates for sixteen vector lanes.
  __m512 uncertainties;       // Tau values; +inf marks an unbounded lane.
  __mmask16 finite_uncertainty;  // Lanes where z*tau is safe to evaluate.
};

template <DistanceMetric Metric, bool QuantizedDot>
__attribute__((target("avx512f"), always_inline))
inline MetricValues16 compute_metric_values_16_avx512(
    const QueryContext& query,
    const MetadataDecodeParameters& parameters,
    float inverse_sqrt_dimension,
    const uint8_t* block_metadata,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    size_t lane_base) {
  static_assert(
      Metric == DistanceMetric::SquaredL2 ||
          Metric == DistanceMetric::InnerProduct ||
          Metric == DistanceMetric::Cosine,
      "unsupported Recast metric");

  // Keep all template instantiations warning-clean. if constexpr removes the
  // pointer/scalars belonging to the unselected dot representation entirely.
  if constexpr (QuantizedDot) {
    (void)exact_dots;
  } else {
    (void)quantized_dots;
    (void)quantized_inverse_scale;
    (void)quantized_bias;
  }

  // Metadata is block-SoA. For lane_base=0 or 16, these are three independent
  // 16-byte contiguous loads rather than 16 gathers from AoS records:
  //
  //   metadata[ 0..31] = norm codes
  //   metadata[32..63] = scale codes
  //   metadata[64..95] = direction-error codes
  const __m128i norm_bytes = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(block_metadata + lane_base));
  const __m128i scale_bytes = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(
          block_metadata + kFastScanBlockSize + lane_base));
  const __m128i error_bytes = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(
          block_metadata + 2 * kFastScanBlockSize + lane_base));

  const __m512i norm_codes = _mm512_cvtepu8_epi32(norm_bytes);
  const __m512i scale_codes = _mm512_cvtepu8_epi32(scale_bytes);
  const __m512i error_codes = _mm512_cvtepu8_epi32(error_bytes);
  const __m512i sentinel = _mm512_set1_epi32(255);
  const __mmask16 norm_overflow =
      _mm512_cmpeq_epi32_mask(norm_codes, sentinel);
  const __mmask16 scale_overflow =
      _mm512_cmpeq_epi32_mask(scale_codes, sentinel);

  __m512 norms = _mm512_mul_ps(
      _mm512_cvtepi32_ps(norm_codes), _mm512_set1_ps(parameters.norm_step));
  norms = _mm512_mask_mov_ps(
      norms, norm_overflow, _mm512_set1_ps(parameters.norm_max));
  __m512 scales = _mm512_mul_ps(
      _mm512_cvtepi32_ps(scale_codes),
      _mm512_set1_ps(parameters.scale_step));
  scales = _mm512_mask_mov_ps(
      scales, scale_overflow, _mm512_set1_ps(parameters.scale_max));
  const __m512 errors = _mm512_mul_ps(
      _mm512_cvtepi32_ps(error_codes),
      _mm512_set1_ps(2.0f / 255.0f));

  __m512 dots;
  if constexpr (QuantizedDot) {
    // Faiss's uint8 LUT kernel emits uint16 accumulators. Widen to int32/float
    // and reverse its query-level affine quantization in sixteen lanes.
    const __m256i accumulated = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(quantized_dots + lane_base));
    dots = _mm512_add_ps(
        _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(accumulated)),
            _mm512_set1_ps(quantized_inverse_scale)),
        _mm512_set1_ps(quantized_bias));
  } else {
    // ExactPQ4AVX512 already produced one float dot product per vector lane.
    dots = _mm512_loadu_ps(exact_dots + lane_base);
  }

  __m512 estimates;
  __m512 uncertainties;
  __mmask16 overflow;
  if constexpr (Metric == DistanceMetric::SquaredL2) {
    estimates = _mm512_sub_ps(
        _mm512_add_ps(
            _mm512_set1_ps(query.query_norm_squared),
            _mm512_mul_ps(norms, norms)),
        _mm512_mul_ps(
            _mm512_set1_ps(2.0f), _mm512_mul_ps(scales, dots)));
    uncertainties = _mm512_mul_ps(
        _mm512_mul_ps(norms, errors),
        _mm512_set1_ps(
            2.0f * query.query_norm * inverse_sqrt_dimension));
    overflow = norm_overflow | scale_overflow;
  } else if constexpr (Metric == DistanceMetric::InnerProduct) {
    estimates = _mm512_sub_ps(
        _mm512_setzero_ps(),
        _mm512_add_ps(
            _mm512_set1_ps(query.mean_dot_query),
            _mm512_mul_ps(scales, dots)));
    uncertainties = _mm512_mul_ps(
        _mm512_mul_ps(norms, errors),
        _mm512_set1_ps(query.query_norm * inverse_sqrt_dimension));
    overflow = norm_overflow | scale_overflow;
  } else {
    estimates = _mm512_sub_ps(
        _mm512_set1_ps(1.0f), _mm512_mul_ps(scales, dots));
    uncertainties = _mm512_mul_ps(
        errors, _mm512_set1_ps(inverse_sqrt_dimension));
    // Source norm is irrelevant to cosine scanning. Its byte remains available
    // for decode(), but only scale overflow makes the interval unbounded.
    overflow = scale_overflow;
  }

  // Replace invalid point estimates and interval metadata with conservative
  // infinities in SIMD. The production path therefore needs no scalar
  // std::isfinite/std::isnan validation loop per scanned vector.
  const __m512 positive_infinity =
      _mm512_set1_ps(std::numeric_limits<float>::infinity());
  const __m512 negative_infinity =
      _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  const __mmask16 finite_estimate =
      _mm512_cmp_ps_mask(estimates, positive_infinity, _CMP_LT_OQ) &
      _mm512_cmp_ps_mask(estimates, negative_infinity, _CMP_GT_OQ);
  estimates = _mm512_mask_mov_ps(
      positive_infinity, finite_estimate, estimates);

  // Overflow metadata and any non-finite arithmetic both map to +infinity.
  // This reproduces DistanceEstimate's conservative endpoint contract.
  const __mmask16 finite_uncertainty =
      _mm512_cmp_ps_mask(uncertainties, positive_infinity, _CMP_LT_OQ) &
      _mm512_cmp_ps_mask(
          uncertainties, _mm512_setzero_ps(), _CMP_GE_OQ) &
      static_cast<__mmask16>(~overflow);
  uncertainties = _mm512_mask_mov_ps(
      positive_infinity, finite_uncertainty, uncertainties);

  return MetricValues16{estimates, uncertainties, finite_uncertainty};
}

template <DistanceMetric Metric, bool QuantizedDot, bool ComputeIntervals>
__attribute__((target("avx512f"), always_inline))
inline void postprocess_16_avx512(
    const QueryContext& query,
    const MetadataDecodeParameters& parameters,
    float inverse_sqrt_dimension,
    const uint8_t* block_metadata,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    float z_score,
    size_t lane_base,
    PostprocessedBlock32* output) {
  const MetricValues16 values =
      compute_metric_values_16_avx512<Metric, QuantizedDot>(
          query, parameters, inverse_sqrt_dimension, block_metadata,
          exact_dots, quantized_dots, quantized_inverse_scale,
          quantized_bias, lane_base);
  const __m512 estimates = values.estimates;
  const __m512 uncertainties = values.uncertainties;

  _mm512_store_ps(
      output->estimated_distances + lane_base, estimates);
  _mm512_store_ps(output->uncertainties + lane_base, uncertainties);

  if constexpr (ComputeIntervals) {
    const __m512 positive_infinity =
        _mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __m512 negative_infinity =
        _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    __m512 lowers;
    __m512 uppers;
    if (z_score == 0.0f) {
      // 0 * infinity is NaN, while the public API defines z=0 as the point
      // estimate. Handle this uniform query-level case before multiplication.
      lowers = estimates;
      uppers = estimates;
    } else {
      const __m512 radii =
          _mm512_mul_ps(uncertainties, _mm512_set1_ps(z_score));
      lowers = _mm512_sub_ps(estimates, radii);
      uppers = _mm512_add_ps(estimates, radii);
      const __mmask16 unbounded =
          static_cast<__mmask16>(~values.finite_uncertainty);
      lowers = _mm512_mask_mov_ps(lowers, unbounded, negative_infinity);
      uppers = _mm512_mask_mov_ps(uppers, unbounded, positive_infinity);
    }
    _mm512_store_ps(output->lower_bounds + lane_base, lowers);
    _mm512_store_ps(output->upper_bounds + lane_base, uppers);
  }
}

/**
 * Runs the complete metadata/interval pipeline for one IVF list in AVX-512.
 *
 * This wrapper is intentionally the ISA-dispatch boundary. The portable caller
 * checks CPU support once per list and enters this function once; all bbs=32
 * tiles are then processed without crossing the generic/AVX-512 function
 * boundary again. Keeping postprocess_16_avx512() always-inline here matters:
 * otherwise a list with N rows would pay two out-of-line calls for every 32
 * rows, which can be comparable to the small amount of arithmetic itself.
 *
 * Within each iteration, lanes 0..15 and 16..31 are independent zmm batches,
 * but both read the same physical metadata tile:
 *
 *   metadata_block +  0: norm[0..31]
 *   metadata_block + 32: scale[0..31]
 *   metadata_block + 64: error[0..31]
 *
 * The two inlined calls fill four contiguous float planes. The consumer then
 * sees one complete logical tile and may perform candidate selection in bulk.
 */
template <DistanceMetric Metric,
          bool QuantizedDot,
          bool ComputeIntervals,
          typename Consumer>
__attribute__((target("avx512f"), noinline))
void emit_postprocessed_blocks_avx512(
    const QueryContext& query,
    const MetadataDecodeParameters& parameters,
    float inverse_sqrt_dimension,
    const CodeBlockView& block,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    float z_score,
    Consumer& consume) {
  for (size_t offset = 0; offset < block.size;
       offset += kFastScanBlockSize) {
    const size_t logical_count =
        std::min(kFastScanBlockSize, block.size - offset);
    const uint8_t* metadata_block =
        block.packed_metadata + (offset / kFastScanBlockSize) *
            kMetadataBlockBytes;
    PostprocessedBlock32 values;

    // One bbs=32 PQ-code tile maps to two 16-float zmm batches. Because both
    // helpers inline into this target function, these are ordinary instruction
    // sequences rather than two calls inside the hot block loop.
    postprocess_16_avx512<Metric, QuantizedDot, ComputeIntervals>(
        query, parameters, inverse_sqrt_dimension, metadata_block,
        exact_dots == nullptr ? nullptr : exact_dots + offset,
        quantized_dots == nullptr ? nullptr : quantized_dots + offset,
        quantized_inverse_scale, quantized_bias, z_score, 0, &values);
    postprocess_16_avx512<Metric, QuantizedDot, ComputeIntervals>(
        query, parameters, inverse_sqrt_dimension, metadata_block,
        exact_dots == nullptr ? nullptr : exact_dots + offset,
        quantized_dots == nullptr ? nullptr : quantized_dots + offset,
        quantized_inverse_scale, quantized_bias, z_score, 16, &values);
    consume(offset, logical_count, values);
  }
}

/**
 * Fused production path for interval selection.
 *
 * Unlike emit_postprocessed_blocks_avx512(), this function never materializes
 * the four 32-float PostprocessedBlock32 planes. Estimates and uncertainties
 * remain in ZMM registers; only the upper endpoint needed by the top-K
 * threshold is formed, and admitted lanes are compressed directly into the
 * selector's SoA reservoir. Lower endpoints are deferred until the final top-R
 * shortlist, where at most R lanes remain.
 */
template <DistanceMetric Metric, bool QuantizedDot>
__attribute__((target("avx512f"), noinline))
void emit_selected_blocks_avx512(
    const QueryContext& query,
    const MetadataDecodeParameters& parameters,
    float inverse_sqrt_dimension,
    const CodeBlockView& block,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    float z_score,
    FusedIntervalAccumulator& accumulator) {
  const __m512 z = _mm512_set1_ps(z_score);
  const __m512 positive_infinity =
      _mm512_set1_ps(std::numeric_limits<float>::infinity());
  for (size_t offset = 0; offset < block.size;
       offset += kFastScanBlockSize) {
    const size_t logical_count =
        std::min(kFastScanBlockSize, block.size - offset);
    const uint8_t* metadata_block =
        block.packed_metadata + (offset / kFastScanBlockSize) *
            kMetadataBlockBytes;

    // One physical bbs=32 FastScan tile becomes two 16-lane ZMM batches. Each
    // batch follows the same register-only path:
    //
    //   PQ dot + metadata -> estimate/tau -> upper -> fused selector
    //
    // estimate and tau are not stored to a temporary 32-row result block.
    const MetricValues16 first =
        compute_metric_values_16_avx512<Metric, QuantizedDot>(
            query, parameters, inverse_sqrt_dimension, metadata_block,
            exact_dots == nullptr ? nullptr : exact_dots + offset,
            quantized_dots == nullptr ? nullptr : quantized_dots + offset,
            quantized_inverse_scale, quantized_bias, 0);
    __m512 first_upper;
    if (z_score == 0.0f) {
      // Avoid 0 * +inf = NaN. The public endpoint contract defines z=0 as the
      // point estimate even for metadata-overflow lanes.
      first_upper = first.estimates;
    } else {
      first_upper = _mm512_add_ps(
          first.estimates, _mm512_mul_ps(first.uncertainties, z));
      first_upper = _mm512_mask_mov_ps(
          first_upper,
          static_cast<__mmask16>(~first.finite_uncertainty),
          positive_infinity);
    }
    accumulator.add_16_avx512(
        block.ids == nullptr ? nullptr : block.ids + offset,
        static_cast<VectorId>(block.implicit_id_base + offset),
        first.estimates, first.uncertainties, first_upper,
        std::min<size_t>(16, logical_count));

    if (logical_count > 16) {
      const MetricValues16 second =
          compute_metric_values_16_avx512<Metric, QuantizedDot>(
              query, parameters, inverse_sqrt_dimension, metadata_block,
              exact_dots == nullptr ? nullptr : exact_dots + offset,
              quantized_dots == nullptr ? nullptr : quantized_dots + offset,
              quantized_inverse_scale, quantized_bias, 16);
      __m512 second_upper;
      if (z_score == 0.0f) {
        second_upper = second.estimates;
      } else {
        second_upper = _mm512_add_ps(
            second.estimates, _mm512_mul_ps(second.uncertainties, z));
        second_upper = _mm512_mask_mov_ps(
            second_upper,
            static_cast<__mmask16>(~second.finite_uncertainty),
            positive_infinity);
      }
      accumulator.add_16_avx512(
          block.ids == nullptr ? nullptr : block.ids + offset + 16,
          static_cast<VectorId>(block.implicit_id_base + offset + 16),
          second.estimates, second.uncertainties, second_upper,
          logical_count - 16);
    }
  }
}

#endif  // RECAST_COMPILE_POSTPROCESS_AVX512

/**
 * Postprocesses all logical 32-row tiles in one IVF list.
 *
 * SIMD capability is tested once before the tile loop. Metric, dot storage, and
 * interval generation are compile-time template dimensions. The hot loop thus
 * contains no metric switch and no per-tile backend branch.
 */
template <DistanceMetric Metric,
          bool QuantizedDot,
          bool ComputeIntervals,
          typename Consumer>
void emit_postprocessed_blocks(
    const RecastQuantizer& quantizer,
    const QueryContext& query,
    const CodeBlockView& block,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    float z_score,
    Consumer&& consume) {
#if RECAST_COMPILE_POSTPROCESS_AVX512
  const MetadataDecodeParameters parameters =
      quantizer.metadata_decode_parameters();
  const float inverse_sqrt_dimension =
      1.0f / std::sqrt(static_cast<float>(quantizer.dimension()));
  if (detail::exact_pq4_avx512_supported()) {
    emit_postprocessed_blocks_avx512<
        Metric, QuantizedDot, ComputeIntervals>(
        query, parameters, inverse_sqrt_dimension, block, exact_dots,
        quantized_dots, quantized_inverse_scale, quantized_bias, z_score,
        consume);
    return;
  }
#endif

  // Portable path preserves the identical block-SoA contract and endpoint
  // semantics. It is also the reference implementation for AVX-512 tests.
  for (size_t offset = 0; offset < block.size;
       offset += kFastScanBlockSize) {
    const size_t logical_count =
        std::min(kFastScanBlockSize, block.size - offset);
    const uint8_t* metadata_block =
        block.packed_metadata + (offset / kFastScanBlockSize) *
            kMetadataBlockBytes;
    PostprocessedBlock32 values;
    postprocess_block_scalar<Metric, QuantizedDot, ComputeIntervals>(
        quantizer, query, metadata_block,
        exact_dots == nullptr ? nullptr : exact_dots + offset,
        quantized_dots == nullptr ? nullptr : quantized_dots + offset,
        quantized_inverse_scale, quantized_bias, z_score, logical_count,
        &values);
    consume(offset, logical_count, values);
  }
}

template <DistanceMetric Metric, bool QuantizedDot>
void emit_selection_blocks(
    const RecastQuantizer& quantizer,
    const QueryContext& query,
    const CodeBlockView& block,
    const float* exact_dots,
    const uint16_t* quantized_dots,
    float quantized_inverse_scale,
    float quantized_bias,
    float z_score,
    FusedIntervalAccumulator& accumulator) {
#if RECAST_COMPILE_POSTPROCESS_AVX512
  // Dispatch once per IVF list, outside the bbs=32 tile loop. Metric and LUT
  // representation are template parameters, so the hot loop has no switch.
  if (detail::exact_pq4_avx512_supported()) {
    emit_selected_blocks_avx512<Metric, QuantizedDot>(
        query, quantizer.metadata_decode_parameters(),
        1.0f / std::sqrt(static_cast<float>(quantizer.dimension())),
        block, exact_dots, quantized_dots, quantized_inverse_scale,
        quantized_bias, z_score, accumulator);
    return;
  }
#endif

  // Non-AVX-512 hosts retain the portable metric decoder, but still use the
  // same bounded SoA selector and defer lower-bound construction to finish().
  emit_postprocessed_blocks<Metric, QuantizedDot, true>(
      quantizer, query, block, exact_dots, quantized_dots,
      quantized_inverse_scale, quantized_bias, z_score,
      [&](size_t offset,
          size_t count,
          const PostprocessedBlock32& values) {
        accumulator.add_scalar_block(
            block.ids == nullptr ? nullptr : block.ids + offset,
            static_cast<VectorId>(block.implicit_id_base + offset),
            values.estimated_distances, values.uncertainties,
            values.upper_bounds, count);
      });
}

template <DistanceMetric Metric, bool ComputeIntervals, typename Consumer>
void scan_block_metric(
    const RecastQuantizer& quantizer,
    const QueryContext& query,
    const CodeBlockView& block,
    float z_score,
    Consumer&& consume,
    ScanStats* stats,
    const ScanOptions& options) {
  if (block.packed_codes == nullptr || block.packed_metadata == nullptr ||
      block.size > block.padded_size ||
      block.block_size != kFastScanBlockSize ||
      block.padded_size % block.block_size != 0 ||
      query.dot_lut.size() != quantizer.subquantizers() * 16) {
    throw std::invalid_argument("invalid Recast code block or query context");
  }

  const size_t M = quantizer.subquantizers();
  // Faiss's PQ4 bbs=32 code layout is organized by pairs of subquantizers:
  //
  //   tile 0, (m=0,m=1): 32 packed bytes
  //   tile 0, (m=2,m=3): 32 packed bytes
  //   ...
  //
  // Each byte contains two 4-bit centroid IDs after Faiss's lane permutation.
  // Both the native exact kernel and pq4_accumulate_loop restore those nibbles
  // into one dot-product result per vector lane. Metadata uses the parallel
  // 96-byte tile described by pack_metadata_bbs32().
  faiss::AlignedTableTightAlloc<float> exact_values;
  faiss::AlignedTableTightAlloc<uint16_t> quantized_values;
  float quantized_scale = 1.0f;
  float quantized_bias = 0.0f;
  LutMode mode = choose_mode(options.lut_mode);
  bool scalar_fallback = false;
  if (mode != LutMode::QuantizedUint8) {
    exact_values.resize(block.padded_size);
  }

  const auto begin = std::chrono::steady_clock::now();
  if (mode == LutMode::ExactFloat &&
      detail::exact_pq4_avx512_supported()) {
    detail::accumulate_exact_pq4_lut_avx512(
        block.padded_size, block.block_size, M,
        block.packed_codes, query.dot_lut.data(), exact_values.get());
  } else if (mode == LutMode::ExactFloat || mode == LutMode::ScalarFloat) {
    scalar_fallback = mode == LutMode::ExactFloat;
    detail::accumulate_exact_pq4_lut_scalar(
        block.size, block.padded_size, block.block_size, M,
        block.packed_codes, query.dot_lut.data(), exact_values.get());
    mode = LutMode::ScalarFloat;
  } else {
    // Faiss quantizes each float LUT row to uint8, performs SIMD byte shuffles,
    // and accumulates into uint16 lanes. The postprocessor reverses this affine
    // transform before applying the metric-specific formula.
    QuantizedLut lut = quantize_lut(query.dot_lut, M);
    quantized_values.resize(block.padded_size);
    faiss::simd_result_handlers::StoreResultHandler handler(
        quantized_values.get(), block.padded_size);
    faiss::pq4_accumulate_loop(
        1, block.padded_size, static_cast<int>(block.block_size),
        static_cast<int>(M), block.packed_codes, lut.values.get(),
        handler, nullptr);
    quantized_scale = lut.scale;
    quantized_bias = lut.bias;
  }
  const auto end = std::chrono::steady_clock::now();

  const auto postprocess_begin = std::chrono::steady_clock::now();
  using ConsumerType =
      std::remove_cv_t<std::remove_reference_t<Consumer>>;
  if constexpr (
      ComputeIntervals &&
      std::is_same_v<ConsumerType, FusedIntervalAccumulator>) {
    // Only the production interval selector takes the register-fused path.
    // scan_all() and reference/materializing consumers retain the generic
    // PostprocessedBlock32 contract, which keeps correctness testing independent.
    if (mode == LutMode::QuantizedUint8) {
      emit_selection_blocks<Metric, true>(
          quantizer, query, block, nullptr, quantized_values.get(),
          1.0f / quantized_scale, quantized_bias, z_score, consume);
    } else {
      emit_selection_blocks<Metric, false>(
          quantizer, query, block, exact_values.get(), nullptr,
          1.0f, 0.0f, z_score, consume);
    }
  } else {
    if (mode == LutMode::QuantizedUint8) {
      emit_postprocessed_blocks<Metric, true, ComputeIntervals>(
          quantizer, query, block, nullptr, quantized_values.get(),
          1.0f / quantized_scale, quantized_bias, z_score,
          std::forward<Consumer>(consume));
    } else {
      emit_postprocessed_blocks<Metric, false, ComputeIntervals>(
          quantizer, query, block, exact_values.get(), nullptr,
          1.0f, 0.0f, z_score, std::forward<Consumer>(consume));
    }
  }
  const auto postprocess_end = std::chrono::steady_clock::now();

  if (stats != nullptr) {
    // Preserve the established timing contract: kernel_nanoseconds covers PQ4
    // LUT accumulation only, not metadata decoding, interval arithmetic, or the
    // top-K collector. End-to-end search timing includes all of them.
    stats->valid_codes += block.size;
    stats->padded_codes += block.padded_size;
    stats->kernel_nanoseconds += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    stats->postprocess_nanoseconds += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            postprocess_end - postprocess_begin).count());
    stats->exact_float_kernel =
        mode == LutMode::ExactFloat || mode == LutMode::ScalarFloat;
    stats->used_scalar_fallback = stats->used_scalar_fallback || scalar_fallback ||
        mode == LutMode::ScalarFloat;
  }
}

template <DistanceMetric Metric>
void scan_all_metric(const RecastQuantizer& quantizer,
                     const QueryContext& query,
                     const CodeBlockView& block,
                     std::vector<DistanceEstimate>* output,
                     ScanStats* stats,
                     const ScanOptions& options) {
  scan_block_metric<Metric, false>(
      quantizer, query, block, 0.0f,
      [&](size_t offset,
          size_t count,
          const PostprocessedBlock32& values) {
        for (size_t lane = 0; lane < count; ++lane) {
          const VectorId id = block.ids == nullptr
              ? static_cast<VectorId>(block.implicit_id_base + offset + lane)
              : block.ids[offset + lane];
          output->push_back(DistanceEstimate{
              id,
              values.estimated_distances[lane],
              values.uncertainties[lane]});
        }
      },
      stats, options);
}

template <DistanceMetric Metric>
SelectionResult scan_interval_candidates_metric(
    const RecastQuantizer& quantizer,
    const QueryContext& query,
    const std::vector<CodeBlockView>& blocks,
    size_t topk,
    float z_score,
    size_t max_candidates,
    ScanStats* stats,
    const ScanOptions& options) {
  FusedIntervalAccumulator accumulator(topk, z_score, max_candidates);

  // A single accumulator spans every probed IVF list. Creating one per list
  // would compute a different theta_U and a per-list top-R, violating the global
  // selector contract. Metric is fixed by this template instantiation before
  // entering the loop, so there is no switch per list or tile.
  for (const CodeBlockView& block : blocks) {
    scan_block_metric<Metric, true>(
        quantizer, query, block, z_score, accumulator, stats, options);
  }
  const auto finish_begin = std::chrono::steady_clock::now();
  SelectionResult result = accumulator.finish();
  const auto finish_end = std::chrono::steady_clock::now();
  if (stats != nullptr) {
    stats->postprocess_nanoseconds += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            finish_end - finish_begin).count());
  }
  return result;
}

}  // namespace

RecastFastScanner::RecastFastScanner(const RecastQuantizer& quantizer)
    : quantizer_(quantizer) {
  if (!quantizer_.is_trained()) {
    throw std::invalid_argument("scanner requires a trained quantizer");
  }
}

void RecastFastScanner::scan_all(
    const QueryContext& query,
    const CodeBlockView& block,
    std::vector<DistanceEstimate>* output,
    ScanStats* stats,
    const ScanOptions& options) const {
  if (output == nullptr) {
    throw std::invalid_argument("distance-estimate output must not be null");
  }
  output->clear();
  output->reserve(block.size);

  // QuantizerConfig keeps metric runtime-configurable, so exactly one dispatch
  // per public scan is unavoidable without templating the public index type.
  // Each branch enters a fully specialized implementation; production interval
  // scans perform this dispatch once per query, outside every list/block loop.
  switch (quantizer_.distance_metric()) {
    case DistanceMetric::SquaredL2:
      scan_all_metric<DistanceMetric::SquaredL2>(
          quantizer_, query, block, output, stats, options);
      return;
    case DistanceMetric::InnerProduct:
      scan_all_metric<DistanceMetric::InnerProduct>(
          quantizer_, query, block, output, stats, options);
      return;
    case DistanceMetric::Cosine:
      scan_all_metric<DistanceMetric::Cosine>(
          quantizer_, query, block, output, stats, options);
      return;
  }
  throw std::logic_error("unknown Recast distance metric");
}

SelectionResult RecastFastScanner::scan_interval_candidates(
    const QueryContext& query,
    const std::vector<CodeBlockView>& blocks,
    size_t topk,
    float z_score,
    size_t max_candidates,
    ScanStats* stats,
    const ScanOptions& options) const {
  // This switch is once per query. All list and 32-row tile loops live inside
  // the selected template instantiation, so metric branches are absent from the
  // hot path requested by the caller.
  switch (quantizer_.distance_metric()) {
    case DistanceMetric::SquaredL2:
      return scan_interval_candidates_metric<DistanceMetric::SquaredL2>(
          quantizer_, query, blocks, topk, z_score, max_candidates,
          stats, options);
    case DistanceMetric::InnerProduct:
      return scan_interval_candidates_metric<DistanceMetric::InnerProduct>(
          quantizer_, query, blocks, topk, z_score, max_candidates,
          stats, options);
    case DistanceMetric::Cosine:
      return scan_interval_candidates_metric<DistanceMetric::Cosine>(
          quantizer_, query, blocks, topk, z_score, max_candidates,
          stats, options);
  }
  throw std::logic_error("unknown Recast distance metric");
}

bool RecastFastScanner::exact_float_supported() noexcept {
  return detail::exact_pq4_avx512_supported();
}

}  // namespace recastlib
