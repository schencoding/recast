#include "faiss_router.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <faiss/index_io.h>

namespace recastlib::adapters {

FaissIvfRouter::FaissIvfRouter(size_t dimension, size_t nlist)
    : dimension_(dimension),
      nlist_(nlist),
      centroids_(std::make_unique<faiss::IndexFlatL2>(dimension)),
      trainer_(std::make_unique<faiss::IndexIVFFlat>(
          centroids_.get(), dimension, nlist, faiss::METRIC_L2)),
      pq_router_(std::make_unique<faiss::IndexPQFastScan>(
          static_cast<int>(dimension), dimension, 4,
          faiss::METRIC_L2, 32)) {
  if (dimension == 0 || nlist == 0) {
    throw std::invalid_argument("IVF dimension and nlist must be positive");
  }
  // IndexIVFFlat borrows the centroid index. Keep its ownership disabled so
  // member destruction cannot delete centroids_ twice.
  trainer_->own_fields = false;

  // M=D creates one scalar dimension per four-bit subquantizer. Every scalar
  // codebook sees all nlist centroids instead of Faiss's default subsample.
  pq_router_->pq.cp.min_points_per_centroid = 1;
  pq_router_->pq.cp.max_points_per_centroid = static_cast<int>(std::min(
      nlist, static_cast<size_t>(std::numeric_limits<int>::max())));
  pq_router_->implem = 0;
}

void FaissIvfRouter::train(size_t count, const float* vectors) {
  if (trainer_ == nullptr || centroids_ == nullptr) {
    throw std::logic_error("original IVF centroid state was already released");
  }
  if (count == 0 || vectors == nullptr) {
    throw std::invalid_argument("IVF training vectors must not be empty");
  }
  trainer_->train(static_cast<faiss::idx_t>(count), vectors);
  if (!trainer_->is_trained ||
      centroids_->ntotal != static_cast<faiss::idx_t>(nlist_)) {
    throw std::runtime_error("Faiss IVF router training failed");
  }
}

void FaissIvfRouter::train_pq_router() {
  if (trainer_ == nullptr || centroids_ == nullptr) {
    throw std::logic_error("original IVF centroid state was already released");
  }
  if (!trainer_->is_trained ||
      centroids_->ntotal != static_cast<faiss::idx_t>(nlist_)) {
    throw std::invalid_argument(
        "IVF centroids must be trained before the PQFastScan router");
  }
  if (nlist_ < 16) {
    throw std::invalid_argument(
        "four-bit PQFastScan centroid routing requires at least 16 lists");
  }

  // Centroids are added in original list-ID order. Faiss result labels can
  // consequently be consumed directly as Recast external-list numbers.
  std::vector<float> centroids(nlist_ * dimension_);
  centroids_->reconstruct_n(
      0, static_cast<faiss::idx_t>(nlist_), centroids.data());
  pq_router_->reset();
  pq_router_->train(static_cast<faiss::idx_t>(nlist_), centroids.data());
  pq_router_->add(static_cast<faiss::idx_t>(nlist_), centroids.data());
  if (!pq_router_->is_trained ||
      pq_router_->ntotal != static_cast<faiss::idx_t>(nlist_)) {
    throw std::runtime_error("Faiss PQFastScan centroid router training failed");
  }

  // The compact router is now sufficient for both insertion and querying.
  trainer_.reset();
  centroids_.reset();
}

void FaissIvfRouter::load_pq_router(const std::string& path) {
  std::unique_ptr<faiss::Index> loaded(faiss::read_index(path.c_str()));
  auto* typed = dynamic_cast<faiss::IndexPQFastScan*>(loaded.get());
  if (typed == nullptr || typed->d != static_cast<int>(dimension_) ||
      typed->ntotal != static_cast<faiss::idx_t>(nlist_) ||
      typed->pq.M != dimension_ || typed->pq.nbits != 4) {
    throw std::invalid_argument("persisted PQFastScan router is incompatible");
  }
  pq_router_.reset(static_cast<faiss::IndexPQFastScan*>(loaded.release()));
  trainer_.reset();
  centroids_.reset();
}

std::vector<uint32_t> FaissIvfRouter::assign_pq(
    size_t count, const float* vectors) const {
  if (!pq_router_->is_trained ||
      pq_router_->ntotal != static_cast<faiss::idx_t>(nlist_) ||
      (count > 0 && vectors == nullptr)) {
    throw std::invalid_argument("invalid PQFastScan IVF assignment input");
  }
  if (count == 0) return {};
  std::vector<float> distances(count);
  std::vector<faiss::idx_t> labels(count);
  pq_router_->search(
      static_cast<faiss::idx_t>(count), vectors, 1,
      distances.data(), labels.data());
  std::vector<uint32_t> result(count);
  for (size_t i = 0; i < count; ++i) {
    if (labels[i] < 0 || labels[i] >= static_cast<faiss::idx_t>(nlist_)) {
      throw std::runtime_error(
          "Faiss returned an invalid PQFastScan IVF assignment");
    }
    result[i] = static_cast<uint32_t>(labels[i]);
  }
  return result;
}

std::vector<uint32_t> FaissIvfRouter::route_pq(
    const float* query, size_t nprobe) const {
  if (!pq_router_->is_trained ||
      pq_router_->ntotal != static_cast<faiss::idx_t>(nlist_) ||
      query == nullptr || nprobe == 0) {
    throw std::invalid_argument("invalid PQFastScan IVF route input");
  }
  nprobe = std::min(nprobe, nlist_);
  std::vector<float> distances(nprobe);
  std::vector<faiss::idx_t> labels(nprobe);
  pq_router_->search(
      1, query, static_cast<faiss::idx_t>(nprobe),
      distances.data(), labels.data());
  std::vector<uint32_t> result;
  result.reserve(nprobe);
  for (faiss::idx_t label : labels) {
    if (label >= 0) result.push_back(static_cast<uint32_t>(label));
  }
  return result;
}

void FaissIvfRouter::reconstruct_centroid(
    size_t list_no, float* output) const {
  if (centroids_ == nullptr) {
    throw std::logic_error(
        "original IVF centroids were released after centroid compaction");
  }
  if (list_no >= nlist_ || output == nullptr) {
    throw std::invalid_argument("invalid centroid reconstruction request");
  }
  centroids_->reconstruct(static_cast<faiss::idx_t>(list_no), output);
}

}  // namespace recastlib::adapters
