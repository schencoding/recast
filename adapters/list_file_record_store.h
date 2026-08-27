#ifndef RECASTLIB_ADAPTERS_LIST_FILE_RECORD_STORE_H
#define RECASTLIB_ADAPTERS_LIST_FILE_RECORD_STORE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace recastlib::adapters {

struct ListFileStoreOptions {
  size_t page_size = 4096;
  size_t payload_bytes = 0;
  size_t lists_per_shard = 256;
  bool direct_io = false;
  bool sync_writes = false;
};

struct RecordMove {
  uint32_t destination = 0;
  uint32_t source = 0;
};

/** Opaque, move-only page snapshot produced before a deletion commit. */
class PreparedRecordMutation {
 public:
  PreparedRecordMutation();
  ~PreparedRecordMutation();
  PreparedRecordMutation(PreparedRecordMutation&&) noexcept;
  PreparedRecordMutation& operator=(PreparedRecordMutation&&) noexcept;
  PreparedRecordMutation(const PreparedRecordMutation&) = delete;
  PreparedRecordMutation& operator=(const PreparedRecordMutation&) = delete;

  const std::vector<uint32_t>& observed_logical_ids() const;
  size_t pages_read() const noexcept;
  size_t pages_written() const noexcept;

 private:
  struct Impl;
  explicit PreparedRecordMutation(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class ListFileRecordStore;
};

/** Headerless page-contained {uint32 LogicalID, ExactVector} list files. */
class ListFileRecordStore {
 public:
  ListFileRecordStore(
      std::string root,
      size_t nlist,
      ListFileStoreOptions options,
      bool create);

  size_t nlist() const noexcept { return nlist_; }
  size_t page_size() const noexcept { return options_.page_size; }
  size_t payload_bytes() const noexcept { return options_.payload_bytes; }
  size_t record_stride() const noexcept { return record_stride_; }
  size_t records_per_page() const noexcept { return records_per_page_; }
  bool direct_io_enabled() const noexcept { return options_.direct_io; }
  const std::string& root() const noexcept { return root_; }

  /** Returns the deterministic path for one zero-based IVF list. */
  std::string list_path(uint32_t list_no) const;

  /**
   * Appends one list-major range at old_size. Payload rows are contiguous and
   * payload_stride may exceed payload_bytes; no record crosses a page.
   */
  void append_records(
      uint32_t list_no,
      uint32_t old_size,
      const uint32_t* logical_ids,
      const void* payloads,
      size_t payload_stride,
      size_t count);

  /** Reads only the Logical IDs at the requested list-local offsets. */
  std::vector<uint32_t> read_logical_ids(
      uint32_t list_no,
      uint32_t logical_size,
      const std::vector<uint32_t>& offsets) const;

  /** Validates that a nonempty list file has exactly its derived page count. */
  void validate_list_file(uint32_t list_no, uint32_t logical_size) const;

  /**
   * Reads each required record page once, snapshots sources, and prepares all
   * page images without changing the file. observed_offsets are returned in
   * input order through PreparedRecordMutation::observed_logical_ids().
   */
  PreparedRecordMutation prepare_moves_and_truncate(
      uint32_t list_no,
      uint32_t old_size,
      uint32_t new_size,
      const std::vector<RecordMove>& moves,
      const std::vector<uint32_t>& observed_offsets) const;

  /** Writes a fully validated prepared mutation and truncates its stale tail. */
  void commit_prepared_mutation(const PreparedRecordMutation& prepared);

  /**
   * Applies a prevalidated swap-with-last plan and truncates unused tail pages.
   * Every source record is snapshotted before any destination page is written.
   */
  void apply_moves_and_truncate(
      uint32_t list_no,
      uint32_t old_size,
      uint32_t new_size,
      const std::vector<RecordMove>& moves);

 private:
  std::string root_;
  size_t nlist_;
  ListFileStoreOptions options_;
  size_t record_stride_;
  size_t records_per_page_;
};

}  // namespace recastlib::adapters

#endif  // RECASTLIB_ADAPTERS_LIST_FILE_RECORD_STORE_H
