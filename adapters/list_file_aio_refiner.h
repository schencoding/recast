#ifndef RECASTLIB_ADAPTERS_LIST_FILE_AIO_REFINER_H
#define RECASTLIB_ADAPTERS_LIST_FILE_AIO_REFINER_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "list_file_record_store.h"
#include "recastlib/types.h"

namespace recastlib::adapters {

/** Computes an exact distance from one external-list payload. */
using ExactDistanceFunction = float (*)(
    const float* query, const void* payload, size_t dimension);

/** A selected encoded row addressed only by its list and list-local offset. */
struct ListCandidate {
  uint32_t list_no = 0;
  uint32_t offset = 0;
  float estimated_distance = 0.0f;
  float uncertainty = 0.0f;
};

struct ListFileRefineOptions {
  size_t queue_depth = 1;
  size_t max_open_files = 64;
};

struct ListFileRefineStats {
  uint64_t refined_vectors = 0;
  uint64_t unique_pages = 0;
  uint64_t read_requests = 0;
  uint64_t requested_bytes = 0;
  uint64_t opened_files = 0;
  uint64_t open_nanoseconds = 0;
  uint64_t close_nanoseconds = 0;
  uint64_t wait_nanoseconds = 0;
  size_t peak_inflight = 0;
};

/** Exact refinement over many headerless list files using one worker pipeline. */
class ListFileAioRefiner {
 public:
  ListFileAioRefiner(
      const ListFileRecordStore& store,
      ListFileRefineOptions options = {});

  std::vector<Neighbor> refine(
      const float* query,
      size_t dimension,
      const std::vector<ListCandidate>& candidates,
      size_t topk,
      ExactDistanceFunction distance_function,
      ListFileRefineStats* stats = nullptr) const;

 private:
  const ListFileRecordStore& store_;
  ListFileRefineOptions options_;
};

}  // namespace recastlib::adapters

#endif  // RECASTLIB_ADAPTERS_LIST_FILE_AIO_REFINER_H
