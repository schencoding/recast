#ifndef RECASTLIB_ADAPTERS_EXTERNAL_LOCATION_DIRECTORY_H
#define RECASTLIB_ADAPTERS_EXTERNAL_LOCATION_DIRECTORY_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace recastlib::adapters {

/** Eight-byte update-only location stored outside the resident index. */
struct DiskLocation {
  uint32_t list_no = UINT32_MAX;
  uint32_t offset = UINT32_MAX;

  bool active() const noexcept {
    return list_no != UINT32_MAX && offset != UINT32_MAX;
  }

  static DiskLocation tombstone() noexcept { return {}; }
};

static_assert(sizeof(DiskLocation) == 8,
              "external locations must occupy exactly eight bytes");

struct ExternalDirectoryOptions {
  size_t page_size = 4096;
  bool direct_io = false;
  bool sync_writes = false;
};

/**
 * O_DIRECT-compatible, page-batched LogicalID -> {list, offset} directory.
 *
 * The file has no header. Entry id is stored at byte id * 8. The object owns
 * no per-ID resident state; each operation uses only request-sized temporary
 * buffers and page-sized aligned I/O buffers.
 */
class ExternalLocationDirectory {
 public:
  ExternalLocationDirectory(
      std::string path,
      ExternalDirectoryOptions options,
      bool create);

  const std::string& path() const noexcept { return path_; }
  size_t entries_per_page() const noexcept;
  bool direct_io_enabled() const noexcept { return options_.direct_io; }

  /** Reads IDs in input order. Missing/unallocated entries are rejected. */
  std::vector<DiskLocation> read_batch(
      const std::vector<uint32_t>& ids) const;

  /**
   * Applies one set of unique ID updates with one read-modify-write per page.
   * New pages are initialized entirely to the tombstone byte pattern.
   */
  void write_batch(
      std::vector<std::pair<uint32_t, DiskLocation>> updates);

 private:
  std::string path_;
  ExternalDirectoryOptions options_;
};

}  // namespace recastlib::adapters

#endif  // RECASTLIB_ADAPTERS_EXTERNAL_LOCATION_DIRECTORY_H
