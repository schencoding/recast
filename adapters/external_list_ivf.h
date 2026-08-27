#ifndef RECASTLIB_ADAPTERS_EXTERNAL_LIST_IVF_H
#define RECASTLIB_ADAPTERS_EXTERNAL_LIST_IVF_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include <faiss/utils/AlignedTable.h>

#include "external_location_directory.h"
#include "faiss_router.h"
#include "list_file_aio_refiner.h"
#include "list_file_record_store.h"
#include "recastlib/recastlib.h"

namespace recastlib::adapters {

enum class ExactPayloadFormat : uint64_t {
  kOpaque = 0,
  kFloat32 = 1,
  kUInt8 = 2,
  kInt8 = 3,
};

struct ExternalListIndexOptions {
  std::string storage_root;
  size_t payload_bytes = 0;
  ExactPayloadFormat payload_format = ExactPayloadFormat::kOpaque;
  size_t page_size = 4096;
  size_t lists_per_shard = 256;
  size_t queue_depth = 1;
  size_t max_open_files_per_query = 64;
  bool direct_io = false;
  bool sync_writes = false;
  bool create_storage = true;
};

struct ExternalListSearchStats {
  ScanStats scan;
  ListFileRefineStats refine;
  size_t probed_lists = 0;
  size_t candidate_lists = 0;
  uint64_t selection_nanoseconds = 0;
  uint64_t refinement_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;
};

/** Additive wall-time and page counters for one or more update batches. */
struct ExternalListUpdateStats {
  uint64_t routing_nanoseconds = 0;
  uint64_t encoding_nanoseconds = 0;
  uint64_t grouping_nanoseconds = 0;
  uint64_t record_io_nanoseconds = 0;
  uint64_t directory_io_nanoseconds = 0;
  uint64_t posting_update_nanoseconds = 0;
  uint64_t total_nanoseconds = 0;
  uint64_t opened_record_files = 0;
  uint64_t record_pages_read = 0;
  uint64_t record_pages_written = 0;
  uint64_t directory_pages_read = 0;
  uint64_t directory_pages_written = 0;
};

/**
 * IVF Recast adapter whose resident postings contain no IDs or locations.
 *
 * Scan-stage IDs are query-local tokens decoded to {list, offset}; the stable
 * Logical ID is materialized only after its external record is read.
 */
class ExternalListRecastIndex {
 public:
  ExternalListRecastIndex(
      FaissIvfRouter& router,
      QuantizerConfig config,
      ExternalListIndexOptions options);

  ExternalListRecastIndex(
      FaissIvfRouter& router,
      RecastQuantizer quantizer,
      ExternalListIndexOptions options);

  void train(size_t count, const float* vectors);

  /** Convenience overload for float32 exact payloads. */
  std::vector<uint32_t> insert_batch(
      size_t count,
      const float* vectors,
      ExternalListUpdateStats* stats = nullptr);

  std::vector<uint32_t> insert_batch(
      size_t count,
      const float* vectors,
      const void* exact_payloads,
      size_t payload_stride = 0,
      ExternalListUpdateStats* stats = nullptr);

  size_t erase_batch(
      size_t count,
      const uint32_t* logical_ids,
      ExternalListUpdateStats* stats = nullptr);

  /**
   * Atomically publishes adapter-owned checkpoint files, manifest last.
   * train() has already persisted the immutable PQFastScan router separately.
   */
  void checkpoint() const;

  /** Returns the model path consumed by RecastQuantizer::load(). */
  std::string quantizer_checkpoint_path() const;

  std::vector<Neighbor> search(
      const float* query,
      size_t topk,
      size_t nprobe,
      float z_score,
      size_t max_candidates,
      ExactDistanceFunction distance_function,
      ExternalListSearchStats* stats = nullptr,
      const ScanOptions& scan_options = {}) const;

  size_t size() const;
  uint64_t next_id() const;
  size_t list_size(uint32_t list_no) const;
  const RecastQuantizer& quantizer() const noexcept { return quantizer_; }

 private:
  struct MemoryList {
    faiss::AlignedTableTightAlloc<uint8_t> packed_codes;
    faiss::AlignedTableTightAlloc<uint8_t> packed_metadata;
    uint32_t logical_size = 0;
    uint32_t padded_size = 0;
  };

  struct LocatedSelection {
    std::vector<ListCandidate> candidates;
    size_t probed_lists = 0;
    size_t candidate_lists = 0;
  };

  LocatedSelection select_candidates_locked(
      const float* query,
      size_t topk,
      size_t nprobe,
      float z_score,
      size_t max_candidates,
      ScanStats* stats,
      const ScanOptions& options) const;

  void load_checkpoint_locked();

  FaissIvfRouter& router_;
  RecastQuantizer quantizer_;
  ExternalListIndexOptions options_;
  ListFileRecordStore record_store_;
  ExternalLocationDirectory directory_;
  std::vector<MemoryList> lists_;
  uint64_t next_id_ = 0;
  size_t active_count_ = 0;
  mutable uint64_t checkpoint_generation_ = 0;
  mutable std::shared_mutex mutex_;
};

}  // namespace recastlib::adapters

#endif  // RECASTLIB_ADAPTERS_EXTERNAL_LIST_IVF_H
