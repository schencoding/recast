#include "external_location_directory.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include "external_io_util.h"

namespace recastlib::adapters {
namespace {

using external_io_detail::AlignedBuffer;

struct Request {
  uint64_t page = 0;
  uint32_t id = 0;
  size_t input_index = 0;
};

off_t checked_offset(uint64_t page, size_t page_size) {
  if (page > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) /
          page_size) {
    throw std::overflow_error("external directory offset exceeds off_t");
  }
  return static_cast<off_t>(page * page_size);
}

}  // namespace

ExternalLocationDirectory::ExternalLocationDirectory(
    std::string path,
    ExternalDirectoryOptions options,
    bool create)
    : path_(std::move(path)), options_(options) {
  if (path_.empty() || !external_io_detail::is_power_of_two(options_.page_size) ||
      options_.page_size < sizeof(void*) ||
      options_.page_size % sizeof(DiskLocation) != 0 ||
      !external_io_detail::is_little_endian()) {
    throw std::invalid_argument("invalid external location directory layout");
  }
  external_io_detail::direct_flag(options_.direct_io);
  const std::filesystem::path parent = std::filesystem::path(path_).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  const int flags = O_RDWR | O_CLOEXEC |
      external_io_detail::direct_flag(options_.direct_io) |
      (create ? (O_CREAT | O_TRUNC) : 0);
  const int fd = ::open(path_.c_str(), flags, 0644);
  if (fd < 0) throw external_io_detail::system_error("open(" + path_ + ")");
  ::close(fd);
}

size_t ExternalLocationDirectory::entries_per_page() const noexcept {
  return options_.page_size / sizeof(DiskLocation);
}

std::vector<DiskLocation> ExternalLocationDirectory::read_batch(
    const std::vector<uint32_t>& ids) const {
  if (ids.empty()) return {};
  const int fd = ::open(
      path_.c_str(), O_RDONLY | O_CLOEXEC |
          external_io_detail::direct_flag(options_.direct_io));
  if (fd < 0) throw external_io_detail::system_error("open(" + path_ + ")");
  try {
    struct stat status {};
    if (::fstat(fd, &status) != 0) {
      throw external_io_detail::system_error("fstat(" + path_ + ")");
    }
    if (status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) % options_.page_size != 0) {
      throw std::runtime_error("external directory is not page aligned");
    }

    std::vector<Request> requests;
    requests.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      requests.push_back(Request{
          ids[i] / entries_per_page(), ids[i], i});
    }
    std::sort(requests.begin(), requests.end(), [](const Request& lhs,
                                                   const Request& rhs) {
      if (lhs.page != rhs.page) return lhs.page < rhs.page;
      return lhs.id < rhs.id;
    });

    std::vector<DiskLocation> result(ids.size());
    AlignedBuffer page(options_.page_size, options_.page_size);
    size_t begin = 0;
    while (begin < requests.size()) {
      size_t end = begin + 1;
      while (end < requests.size() &&
             requests[end].page == requests[begin].page) ++end;
      const off_t offset = checked_offset(requests[begin].page, options_.page_size);
      if (static_cast<uint64_t>(offset) + options_.page_size >
          static_cast<uint64_t>(status.st_size)) {
        throw std::out_of_range("Logical ID is outside locations.bin");
      }
      external_io_detail::pread_exact(
          fd, page.data(), options_.page_size, offset);
      const auto* entries = static_cast<const DiskLocation*>(page.data());
      for (size_t i = begin; i < end; ++i) {
        result[requests[i].input_index] =
            entries[requests[i].id % entries_per_page()];
      }
      begin = end;
    }
    ::close(fd);
    return result;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

void ExternalLocationDirectory::write_batch(
    std::vector<std::pair<uint32_t, DiskLocation>> updates) {
  if (updates.empty()) return;
  std::sort(updates.begin(), updates.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });
  for (size_t i = 1; i < updates.size(); ++i) {
    if (updates[i - 1].first == updates[i].first) {
      throw std::invalid_argument("duplicate Logical ID directory update");
    }
  }

  const int fd = ::open(
      path_.c_str(), O_RDWR | O_CLOEXEC |
          external_io_detail::direct_flag(options_.direct_io));
  if (fd < 0) throw external_io_detail::system_error("open(" + path_ + ")");
  try {
    struct stat status {};
    if (::fstat(fd, &status) != 0) {
      throw external_io_detail::system_error("fstat(" + path_ + ")");
    }
    if (status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) % options_.page_size != 0) {
      throw std::runtime_error("external directory is not page aligned");
    }
    uint64_t current_bytes = static_cast<uint64_t>(status.st_size);
    AlignedBuffer page(options_.page_size, options_.page_size);
    size_t begin = 0;
    while (begin < updates.size()) {
      const uint64_t page_id = updates[begin].first / entries_per_page();
      size_t end = begin + 1;
      while (end < updates.size() &&
             updates[end].first / entries_per_page() == page_id) ++end;
      const off_t offset = checked_offset(page_id, options_.page_size);
      if (static_cast<uint64_t>(offset) < current_bytes) {
        external_io_detail::pread_exact(
            fd, page.data(), options_.page_size, offset);
      } else {
        std::memset(page.data(), 0xff, options_.page_size);
      }
      auto* entries = static_cast<DiskLocation*>(page.data());
      for (size_t i = begin; i < end; ++i) {
        entries[updates[i].first % entries_per_page()] = updates[i].second;
      }
      external_io_detail::pwrite_exact(
          fd, page.data(), options_.page_size, offset);
      current_bytes = std::max<uint64_t>(
          current_bytes, static_cast<uint64_t>(offset) + options_.page_size);
      begin = end;
    }
    if (::ftruncate(fd, static_cast<off_t>(current_bytes)) != 0) {
      throw external_io_detail::system_error("ftruncate(" + path_ + ")");
    }
    external_io_detail::sync_file(fd, options_.sync_writes);
    ::close(fd);
  } catch (...) {
    ::close(fd);
    throw;
  }
}

}  // namespace recastlib::adapters
