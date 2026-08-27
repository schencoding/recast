#ifndef RECASTLIB_DETAIL_UNEVEN_PRODUCT_QUANTIZER_H
#define RECASTLIB_DETAIL_UNEVEN_PRODUCT_QUANTIZER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>

namespace faiss {
struct IOReader;
struct IOWriter;
}  // namespace faiss

namespace recastlib::detail {

/**
 * A 4-bit product quantizer whose subspaces may have different dimensions.
 *
 * Subspaces with the same dimensionality are trained together using one
 * Faiss ProductQuantizer.  The number of subspaces in every such bucket is
 * even, so every bucket emits a whole number of bytes.  Concatenating the
 * bucket codes therefore produces exactly M / 2 bytes with the ordinary PQ4
 * nibble order; no query-time repacking is required.
 *
 * During training, reverse water-filling converts each PCA dimension's
 * log2-standard-deviation into an ideal continuous bit budget.  Dimensions are
 * assigned so that every 4-bit group has approximately four bits of aggregate
 * ideal demand.  A deterministic greedy planner and local improvement pass
 * choose both the paired subspace capacities and the dimension assignment.
 */
struct UnevenProductQuantizer {
  /** Public description of one equal-width bucket of adjacent PQ groups. */
  struct BucketInfo {
    size_t subspace_dimension = 0;  ///< Dimensions assigned to each group.
    size_t subspace_count = 0;      ///< Number of groups trained by this bucket.
    size_t first_subspace = 0;      ///< First global PQ group represented here.
    size_t code_offset = 0;         ///< Byte offset in a vector's flat PQ4 code.
  };

  /**
   * Creates an untrained uneven-width product quantizer.
   *
   * @param d Vector dimension; currently required to be even.
   * @param M Positive, even number of PQ groups no greater than d.
   * @param nbits Bits per group; currently required to equal four.
   */
  UnevenProductQuantizer(size_t d, size_t M, size_t nbits = 4);

  size_t d;          ///< Input vector dimension.
  size_t M;          ///< Number of uneven PQ subspaces.
  size_t nbits;      ///< Bits per subspace code; currently four.
  size_t ksub;       ///< Centroids per subspace, equal to 2^nbits.
  size_t code_size;  ///< Flat code bytes per vector, equal to M/2 for PQ4.

  /**
   * Upper bound on one subspace's dimensionality.  Zero selects
   * ceil(d / M) + 2.  The bound is applied while planning paired capacities.
   */
  size_t max_subspace_dimension = 0;

  /// Print per-dimension ideal bits and the learned group loads/layout.
  bool report_layout_after_training = true;

  bool verbose = false;     ///< Forward verbose logging to bucket quantizers.
  bool is_trained = false;  ///< True after every bucket has trained successfully.

  /**
   * Plans uneven groups and trains one Faiss PQ per equal-width bucket.
   * @param n Number of row-major training vectors.
   * @param x Input buffer containing n * d floats.
   */
  void train(size_t n, const float* x);

  /**
   * Encodes vectors into ordinary per-vector flat PQ4 bytes.
   *
   * The low and high nibbles contain consecutive global subspaces, matching
   * Faiss's flat-code convention; this is not the bbs=32 fast-scan layout.
   *
   * @param x Input buffer containing n * d floats.
   * @param codes Output buffer containing n * code_size bytes.
   * @param n Number of vectors to encode.
   */
  void compute_codes(const float* x, uint8_t* codes, size_t n) const;

  /**
   * Decodes one flat PQ4 code into one reconstructed vector.
   * @param code Input buffer containing code_size bytes.
   * @param x Output buffer containing d floats.
   */
  void decode(const uint8_t* code, float* x) const;

  /**
   * Decodes n flat PQ4 codes into a row-major n-by-d output buffer.
   * @param codes Input buffer containing n * code_size bytes.
   * @param x Output buffer containing n * d floats.
   * @param n Number of vectors to decode.
   */
  void decode(const uint8_t* codes, float* x, size_t n) const;

  /**
   * Builds the row-major M-by-ksub inner-product LUT for one query vector.
   * @param x Input vector containing d floats.
   * @param table Output buffer containing M * ksub floats.
   */
  void compute_inner_prod_table(const float* x, float* table) const;

  /**
   * Builds n consecutive M-by-ksub inner-product LUTs.
   * @param n Number of input vectors.
   * @param x Input buffer containing n * d floats.
   * @param tables Output buffer containing n * M * ksub floats.
   */
  void compute_inner_prod_tables(
      size_t n, const float* x, float* tables) const;

  /**
   * Builds the row-major M-by-ksub squared-L2 LUT for one query vector.
   * @param x Input vector containing d floats.
   * @param table Output buffer containing M * ksub floats.
   */
  void compute_distance_table(const float* x, float* table) const;

  /**
   * Builds n consecutive M-by-ksub squared-L2 LUTs.
   * @param n Number of input vectors.
   * @param x Input buffer containing n * d floats.
   * @param tables Output buffer containing n * M * ksub floats.
   */
  void compute_distance_tables(
      size_t n, const float* x, float* tables) const;

  /** @return Original dimension indices assigned to each global PQ group. */
  const std::vector<std::vector<size_t>>& subspace_dimensions() const {
    return subspace_dimensions_;
  }

  /** @return Reverse-water-filling bit demand for every input dimension. */
  const std::vector<double>& ideal_bit_budgets() const {
    return ideal_bit_budgets_;
  }

  /** @return Sum of ideal bit demands assigned to each PQ group. */
  const std::vector<double>& group_bit_loads() const {
    return group_bit_loads_;
  }

  /** @return Stable public descriptions of the trained equal-width buckets. */
  std::vector<BucketInfo> bucket_info() const;

  /** Writes the complete trained uneven-PQ state to a sequential Faiss writer. */
  void write(faiss::IOWriter* writer) const;

  /** Restores a trained uneven PQ without rerunning grouping or codebook training. */
  static std::unique_ptr<UnevenProductQuantizer> read(
      faiss::IOReader* reader);

 private:
  /** Internal bucket trained as one ordinary Faiss ProductQuantizer. */
  struct Bucket {
    size_t subspace_dimension = 0;  ///< Width of each subspace in this bucket.
    size_t subspace_count = 0;      ///< Even number of subspaces in the bucket.
    size_t first_subspace = 0;      ///< First global PQ group in this bucket.
    size_t code_offset = 0;         ///< Byte offset in each global flat code.
    size_t bucket_dimension = 0;    ///< subspace_dimension * subspace_count.
    size_t bucket_code_size = 0;    ///< PQ4 bytes emitted per vector by the bucket.
    std::vector<size_t> dimensions; ///< Flattened original dimensions in group order.
    std::unique_ptr<faiss::ProductQuantizer> pq;  ///< Equal-width bucket codebook.
  };

  /** Trial dimension assignment and its normalized imbalance score. */
  struct Assignment {
    std::vector<std::vector<size_t>> dimensions;  ///< Original dimensions by group.
    std::vector<double> loads;  ///< Aggregate ideal-bit demand by group.
    double score = 0.0;         ///< Normalized squared deviation from nbits/group.
  };

  /** Learned mapping from global PQ groups to original dimensions. */
  std::vector<std::vector<size_t>> subspace_dimensions_;

  /** Reverse-water-filling demand retained for diagnostics. */
  std::vector<double> ideal_bit_budgets_;

  /** Per-group demand retained for diagnostics. */
  std::vector<double> group_bit_loads_;

  /** Equal-width Faiss PQ buckets ordered by first_subspace. */
  std::vector<Bucket> buckets_;

  /** Computes per-dimension ideal bit demand with reverse water-filling. */
  std::vector<double> compute_ideal_bit_budgets(
      size_t n, const float* x) const;

  /** Chooses one capacity per pair of equal-width, byte-aligned PQ groups. */
  std::vector<size_t> plan_paired_capacities(
      const std::vector<double>& bit_budgets) const;

  /** Assigns dimensions to fixed capacities while balancing ideal-bit loads. */
  Assignment assign_dimensions(
      const std::vector<size_t>& paired_capacities,
      const std::vector<double>& bit_budgets) const;

  /** Creates byte-aligned Faiss PQ buckets from the chosen group assignment. */
  void build_buckets(const std::vector<std::vector<size_t>>& dimensions);

  /** Gathers a bucket's non-contiguous dimensions into row-major scratch space. */
  void gather_bucket(
      const Bucket& bucket, size_t n, const float* x,
      std::vector<float>& gathered) const;

  /** Verifies complete dimension coverage and padding-free code offsets. */
  void validate_layout() const;
};

} // namespace recastlib::detail

#endif // RECASTLIB_DETAIL_UNEVEN_PRODUCT_QUANTIZER_H
