#ifndef RECASTLIB_FASTSCAN_H
#define RECASTLIB_FASTSCAN_H

#include <cstddef>
#include <vector>

#include "recastlib/quantizer.h"
#include "recastlib/types.h"

namespace recastlib {

/** Backend used to accumulate a query's PQ4 lookup-table values. */
enum class LutMode {
  Auto,           ///< AVX-512 float when available, otherwise Faiss uint8.
  ExactFloat,     ///< AVX-512 float with a portable scalar fallback.
  QuantizedUint8, ///< Faiss FastScan after per-query LUT quantization.
  ScalarFloat,    ///< Portable reference implementation using float LUT values.
};

/** Per-scan backend selection options. */
struct ScanOptions {
  LutMode lut_mode = LutMode::Auto;  ///< Requested PQ4 LUT accumulation mode.
};

/**
 * Returns the bytes required by bbs=32 block-SoA metadata storage.
 *
 * @param padded_size Logical row count rounded up to a multiple of 32.
 * @return padded_size * 3 bytes.
 */
size_t packed_metadata_size_bbs32(size_t padded_size);

/**
 * Packs ordinary three-byte records into the FastScan metadata layout.
 *
 * For every 32-vector tile, output stores three contiguous byte planes:
 *
 * @code
 * byte  0..31: norm  for vectors 0..31
 * byte 32..63: scale for vectors 0..31
 * byte 64..95: error for vectors 0..31
 * @endcode
 *
 * Padding lanes are filled with 255. Full blocks therefore remain exactly
 * three bytes per vector while allowing each field to be loaded contiguously.
 *
 * @param metadata Input AoS records for size logical vectors.
 * @param size Number of logical vectors.
 * @param padded_size Row count rounded up to a multiple of 32.
 * @param output Output buffer with packed_metadata_size_bbs32() bytes.
 */
void pack_metadata_bbs32(const Metadata* metadata,
                         size_t size,
                         size_t padded_size,
                         uint8_t* output);

/** Converts packed Recast PQ4 records into distance estimates and intervals. */
class RecastFastScanner {
 public:
  /**
   * Binds the scanner to a trained quantizer.
   *
   * @param quantizer Non-owning reference that must outlive this scanner.
   */
  explicit RecastFastScanner(const RecastQuantizer& quantizer);

  /**
   * Scans every logical row in one packed block sequence.
   *
   * The output vector is cleared and then filled in block order. Statistics,
   * when supplied, are accumulated rather than reset.
   *
   * @param query Context prepared by the bound quantizer.
   * @param block Non-owning Faiss bbs=32 packed-code view.
   * @param output Receives exactly block.size estimates.
   * @param stats Optional aggregate scan statistics.
   * @param options LUT backend selection.
   */
  void scan_all(const QueryContext& query,
                const CodeBlockView& block,
                std::vector<DistanceEstimate>* output,
                ScanStats* stats = nullptr,
                const ScanOptions& options = {}) const;

  /**
   * Scans several blocks into a bounded estimate shortlist, then filters it.
   *
   * With a positive cap, selection working memory is O(topk + 2R), in addition
   * to one backend output buffer for the current block. A zero cap retains all
   * scanned estimates.
   *
   * @param query Context prepared by the bound quantizer.
   * @param blocks Non-owning packed-code views to concatenate logically.
   * @param topk Requested neighbor count K.
   * @param z_score Finite, non-negative multiplier for uncertainty radii.
   * @param max_candidates Shortlist cap, at least topk; zero keeps all estimates.
   * @param stats Optional aggregate scan statistics.
   * @param options LUT backend selection.
   * @return Selected candidates and the K-th upper-bound certificate.
   */
  SelectionResult scan_interval_candidates(
      const QueryContext& query,
      const std::vector<CodeBlockView>& blocks,
      size_t topk,
      float z_score,
      size_t max_candidates = 1000,
      ScanStats* stats = nullptr,
      const ScanOptions& options = {}) const;

  /** @return True when the host and OS can execute the AVX-512F float kernel. */
  static bool exact_float_supported() noexcept;

 private:
  /** Non-owning trained quantizer used to decode metadata and build LUTs. */
  const RecastQuantizer& quantizer_;
};

}  // namespace recastlib

#endif  // RECASTLIB_FASTSCAN_H
