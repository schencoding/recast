#ifndef RECASTLIB_ADAPTERS_FAISS_ROUTER_H
#define RECASTLIB_ADAPTERS_FAISS_ROUTER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexPQFastScan.h>

namespace recastlib::adapters {

/**
 * Coarse IVF router used by the external-list Recast index.
 *
 * Training first learns ordinary float32 IVF centroids. The router then builds
 * a four-bit PQFastScan view with M=D, preserving centroid insertion order as
 * the IVF list number, and releases the original centroids. The same compact
 * metric therefore assigns database vectors and selects query lists.
 */
class FaissIvfRouter {
 public:
  /** Creates an untrained squared-L2 router. */
  FaissIvfRouter(size_t dimension, size_t nlist);

  /** Learns nlist float32 centroids from count row-major vectors. */
  void train(size_t count, const float* vectors);

  /**
   * Converts trained centroids to their immutable M=D PQFastScan form and
   * releases the original float32 centroid representation.
   */
  void train_pq_router();

  /** Restores a previously persisted M=D PQFastScan centroid router. */
  void load_pq_router(const std::string& path);

  /** Assigns vectors with the compact metric also used by route_pq(). */
  std::vector<uint32_t> assign_pq(
      size_t count, const float* vectors) const;

  /** Returns up to nprobe compact-router list IDs for one query. */
  std::vector<uint32_t> route_pq(
      const float* query, size_t nprobe) const;

  /**
   * Copies a float32 centroid before train_pq_router() releases it.
   * ExternalListRecastIndex uses this only while publishing router.faiss.
   */
  void reconstruct_centroid(size_t list_no, float* output) const;

  /** True only while the temporary float32 centroids remain resident. */
  bool retains_original_centroids() const noexcept {
    return centroids_ != nullptr;
  }

  size_t dimension() const noexcept { return dimension_; }
  size_t nlist() const noexcept { return nlist_; }

 private:
  size_t dimension_;
  size_t nlist_;
  std::unique_ptr<faiss::IndexFlatL2> centroids_;
  std::unique_ptr<faiss::IndexIVFFlat> trainer_;
  std::unique_ptr<faiss::IndexPQFastScan> pq_router_;
};

}  // namespace recastlib::adapters

#endif  // RECASTLIB_ADAPTERS_FAISS_ROUTER_H
