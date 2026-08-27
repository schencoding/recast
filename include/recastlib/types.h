#ifndef RECASTLIB_TYPES_H
#define RECASTLIB_TYPES_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace recastlib {

/// Stable identifier assigned to a database vector by an index adapter.
using VectorId = uint32_t;

/**
 * Three-byte side information stored next to one PQ4 direction code.
 *
 * The norm and scale fields use 255 as an overflow sentinel. The error field
 * is a conservative upper quantization of the direction reconstruction error.
 */
struct Metadata {
  uint8_t norm;   ///< Quantized norm before direction normalization.
  uint8_t scale;  ///< Metric-specific decoded-reconstruction scale.
  uint8_t error;  ///< Quantized direction-error bound in the range [0, 2].
};

static_assert(sizeof(Metadata) == 3,
              "Recast metadata must occupy exactly three bytes");

/** Floating-point constants needed to decode the three metadata bytes. */
struct MetadataDecodeParameters {
  float norm_max = 0.0f;   ///< Replacement value for norm overflow code 255.
  float norm_step = 1.0f;  ///< Multiplicative step for norm codes 0..254.
  float scale_max = 0.0f;  ///< Replacement value for scale overflow code 255.
  float scale_step = 1.0f; ///< Multiplicative step for scale codes 0..254.
};

/** Query-dependent values shared by every encoded block in one search. */
struct QueryContext {
  std::vector<float> centered_query;     ///< Metric-specific query before rotation.
  std::vector<float> transformed_query;  ///< Metric-specific query after rotation.
  std::vector<float> dot_lut;            ///< Row-major M-by-16 centroid dot-product LUT.
  float query_norm = 0.0f;               ///< L2 norm used by the error model.
  float query_norm_squared = 0.0f;       ///< Squared norm used by squared-L2.
  float mean_dot_query = 0.0f;           ///< <q, mean> term for inner product.
};

/**
 * Non-owning view of one Faiss PQ4 fast-scan block sequence.
 *
 * packed_codes follows the Faiss bbs=32 layout and contains padded_size rows.
 * Metadata and explicit IDs contain only size logical rows. Every pointer must
 * remain valid for the duration of a scan.
 */
struct CodeBlockView {
  const uint8_t* packed_codes = nullptr;  ///< Packed PQ4 bytes for padded_size rows.
  /**
   * Metadata packed in 32-row block-SoA order.
   *
   * Every 96-byte block stores norm[32], then scale[32], then error[32].
   * Its padding lanes use byte 255 and are ignored beyond size logical rows.
   */
  const uint8_t* packed_metadata = nullptr;
  const VectorId* ids = nullptr;          ///< Optional explicit IDs for size rows.
  size_t size = 0;                        ///< Number of logical, non-padding rows.
  size_t padded_size = 0;                 ///< Row count rounded up to block_size.
  size_t block_size = 32;                 ///< Faiss packing block size; currently 32.
  VectorId implicit_id_base = 0;          ///< First ID used when ids is null.
};

/**
 * Scan-stage smaller-is-better distance estimate and additive error radius.
 *
 * A scanned vector does not become a refinement candidate until a selection
 * policy places its estimate in SelectionResult::candidates.
 */
struct DistanceEstimate {
  VectorId id = 0;  ///< Database vector identifier.
  float estimated_distance = std::numeric_limits<float>::infinity();  ///< Metric distance estimate.
  float uncertainty = std::numeric_limits<float>::infinity();  ///< Radius at z_score == 1.

  /**
   * @param z_score Uncertainty multiplier.
   * @return Conservative lower endpoint. An unbounded radius yields -infinity.
   */
  float lower(float z_score) const {
    if (z_score == 0.0f || uncertainty == 0.0f) {
      return estimated_distance;
    }
    if (std::isinf(uncertainty)) {
      return -std::numeric_limits<float>::infinity();
    }
    const float endpoint = estimated_distance - z_score * uncertainty;
    return std::isnan(endpoint)
        ? -std::numeric_limits<float>::infinity()
        : endpoint;
  }

  /**
   * @param z_score Uncertainty multiplier.
   * @return Conservative upper endpoint. An unbounded radius yields +infinity.
   */
  float upper(float z_score) const {
    if (z_score == 0.0f || uncertainty == 0.0f) {
      return estimated_distance;
    }
    if (std::isinf(uncertainty)) {
      return std::numeric_limits<float>::infinity();
    }
    const float endpoint = estimated_distance + z_score * uncertainty;
    return std::isnan(endpoint)
        ? std::numeric_limits<float>::infinity()
        : endpoint;
  }
};

/** Output of estimate-shortlist and interval candidate selection. */
struct SelectionResult {
  std::vector<DistanceEstimate> candidates;  ///< Refine candidates ordered by estimate, then ID.
  float kth_upper_bound = std::numeric_limits<float>::infinity();  ///< K-th smallest upper bound.
  /**
   * True when the cap discarded any scan-stage estimate.
   *
   * This describes shortlist truncation, not completeness of the uncapped
   * interval-selected set.
   */
  bool truncated = false;
};

/** Exact nearest-neighbor result. */
struct Neighbor {
  VectorId id = 0;  ///< Database vector identifier.
  float distance = std::numeric_limits<float>::infinity();  ///< Exact metric distance.
};

/** Additive counters and backend flags collected by scan operations. */
struct ScanStats {
  size_t valid_codes = 0;       ///< Logical vectors scanned across all calls.
  size_t padded_codes = 0;      ///< Backend-reported scan slots, including known padding.
  uint64_t kernel_nanoseconds = 0;  ///< Backend-reported encoded-distance time.
  /** Metadata decode, metric/interval reconstruction, and selection sink time. */
  uint64_t postprocess_nanoseconds = 0;
  bool exact_float_kernel = false;  ///< Whether the most recent scan used float accumulation.
  bool used_scalar_fallback = false;  ///< Whether any scan used the scalar float path.
};

}  // namespace recastlib

#endif  // RECASTLIB_TYPES_H
