#ifndef RECASTLIB_QUANTIZER_H
#define RECASTLIB_QUANTIZER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "recastlib/types.h"

namespace recastlib {

/** Strategy used to divide transformed dimensions among PQ4 subquantizers. */
enum class GroupingMode {
  Uniform,  ///< Equal-width groups; requires dimension divisible by M.
  Uneven,   ///< Variance-aware groups whose widths may differ.
};

/** Optional orthogonal transform applied before direction quantization. */
enum class TransformMode {
  None,          ///< Keep encoded directions in their original coordinates.
  Pca,           ///< Rotate with a full-dimensional PCA matrix.
  ParametricOpq, ///< PCA with Faiss balanced_bins=M; not iterative OPQ.
};

/** Smaller-is-better distance represented by each scan estimate. */
enum class DistanceMetric {
  SquaredL2,    ///< Squared Euclidean distance.
  InnerProduct, ///< Negative inner product, so larger similarity ranks first.
  Cosine,       ///< Cosine distance 1 - cos(q, x).
};

/** Optional translation applied before direction normalization and encoding. */
enum class CenteringMode {
  None, ///< Encode directions relative to the origin.
  Mean, ///< Subtract the learned training mean before encoding.
};

/** Construction parameters for RecastQuantizer. */
struct QuantizerConfig {
  size_t dimension = 0;      ///< Vector dimension D.
  size_t subquantizers = 0;  ///< Even number M of 4-bit direction groups.
  GroupingMode grouping = GroupingMode::Uneven;  ///< Dimension grouping policy.
  TransformMode transform = TransformMode::Pca;  ///< Pre-quantization rotation.
  DistanceMetric metric = DistanceMetric::SquaredL2;  ///< Scan distance semantics.
  CenteringMode centering = CenteringMode::Mean;  ///< Encoding translation.
};

/**
 * Trains and applies Recast's compact direction-plus-metadata encoding.
 *
 * Each encoded vector occupies M/2 PQ4 bytes followed by a separate
 * three-byte Metadata record. The quantizer owns all trained state but does
 * not own any caller-provided input or output buffers.
 */
class RecastQuantizer {
 public:
  /**
   * Creates an untrained quantizer.
   *
   * @param config Valid dimension, even M, grouping, and transform settings.
   * @throws std::invalid_argument if the configuration cannot form PQ4 groups.
   */
  explicit RecastQuantizer(QuantizerConfig config);

  /** Destroys the owned trained quantizer state. */
  ~RecastQuantizer();

  /** Quantizers are non-copyable because their Faiss state is uniquely owned. */
  RecastQuantizer(const RecastQuantizer&) = delete;

  /** Quantizers are non-copy-assignable because their state is uniquely owned. */
  RecastQuantizer& operator=(const RecastQuantizer&) = delete;

  /** Transfers ownership of the trained state. */
  RecastQuantizer(RecastQuantizer&&) noexcept;

  /** Transfers ownership of the trained state. */
  RecastQuantizer& operator=(RecastQuantizer&&) noexcept;

  /**
   * Learns the optional mean, rotation, direction codebooks, and metadata ranges.
   *
   * @param count Number of row-major training vectors.
   * @param vectors Contiguous buffer containing count * dimension() floats.
   */
  void train(size_t count, const float* vectors);

  /**
   * Atomically persists all learned state required by encode, decode, and scan.
   *
   * The model includes centering, the trained transform, PQ4 codebooks, uneven
   * grouping, and metadata ranges. Encoded database postings are not included.
   */
  void save(const std::string& path) const;

  /**
   * Restores a trained model without accessing training vectors.
   *
   * @throws std::runtime_error for truncated, corrupt, or incompatible files.
   */
  static RecastQuantizer load(const std::string& path);

  /**
   * Encodes row-major vectors into flat PQ4 bytes and metadata records.
   *
   * @param count Number of vectors to encode; zero is allowed.
   * @param vectors Input buffer containing count * dimension() floats.
   * @param codes Output buffer containing count * code_size() bytes.
   * @param metadata Output buffer containing count Metadata records.
   */
  void encode(size_t count,
              const float* vectors,
              uint8_t* codes,
              Metadata* metadata) const;

  /**
   * Reconstructs row-major vectors from flat PQ4 bytes and metadata.
   *
   * @param count Number of vectors to decode; zero is allowed.
   * @param codes Input buffer containing count * code_size() bytes.
   * @param metadata Input buffer containing count Metadata records.
   * @param vectors Output buffer containing count * dimension() floats.
   */
  void decode(size_t count,
              const uint8_t* codes,
              const Metadata* metadata,
              float* vectors) const;

  /**
   * Precomputes the metric-specific query, transformed query, and dot LUT.
   *
   * @param query One vector containing dimension() floats.
   * @return Self-contained context reusable across all blocks for this query.
   */
  QueryContext prepare_query(const float* query) const;

  /** @return Configured vector dimension D. */
  size_t dimension() const noexcept;

  /** @return Complete immutable construction and distance configuration. */
  QuantizerConfig configuration() const noexcept;

  /** @return Configured number M of PQ4 subquantizers. */
  size_t subquantizers() const noexcept;

  /** @return Flat PQ4 payload size M/2 in bytes per vector. */
  size_t code_size() const noexcept;

  /** @return Metadata payload size, fixed at three bytes per vector. */
  size_t metadata_size() const noexcept { return sizeof(Metadata); }

  /** @return Nominal encoded payload bytes per vector: M/2 + 3. */
  size_t record_size() const noexcept { return code_size() + metadata_size(); }

  /** @return True after train() has completed successfully. */
  bool is_trained() const noexcept;

  /** @return True when variance-aware uneven grouping is configured. */
  bool uses_uneven_grouping() const noexcept;

  /** @return Smaller-is-better metric represented by scan estimates. */
  DistanceMetric distance_metric() const noexcept;

  /** @return Configured vector-centering policy. */
  CenteringMode centering_mode() const noexcept;

  /**
   * Returns zeros when CenteringMode::None is configured.
   * @return Encoding mean in the original D-dimensional coordinate system.
   */
  const std::vector<float>& mean() const;

  /**
   * Fills an M-by-16 row-major inner-product LUT for the scanner.
   *
   * This scanner-only hook keeps Faiss codebook objects out of the public API.
   * @param transformed_query Metric-specific transformed query, D floats.
   * @param lut Output buffer containing subquantizers() * 16 floats.
   */
  void compute_dot_lut(const float* transformed_query, float* lut) const;

  /**
   * Returns constants used by scalar and SIMD metadata decoders.
   *
   * Exposing the four immutable scalars lets a FastScan block decode 16 byte
   * lanes at once without making 32 calls to decode_metadata().
   */
  MetadataDecodeParameters metadata_decode_parameters() const noexcept;

  /**
   * Decodes the three metadata bytes into the scanner's floating-point values.
   *
   * @param metadata Encoded side information for one vector.
   * @param norm Receives the decoded pre-normalization vector norm.
   * @param scale Receives the metric-specific reconstruction scale: r/||y'||
   *     for L2/IP and 1/||y'|| for cosine.
   * @param direction_error Receives the conservative direction-error bound.
   * @param overflow Receives true when metric-required metadata overflowed.
   */
  void decode_metadata(const Metadata& metadata,
                       float* norm,
                       float* scale,
                       float* direction_error,
                       bool* overflow) const;

 private:
  /** Opaque implementation containing Faiss objects and learned parameters. */
  struct Impl;

  /** Unique ownership of the opaque implementation. */
  std::unique_ptr<Impl> impl_;
};

}  // namespace recastlib

#endif  // RECASTLIB_QUANTIZER_H
