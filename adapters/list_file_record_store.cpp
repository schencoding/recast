#include "list_file_record_store.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include "external_io_util.h"

namespace recastlib::adapters {
namespace {

using external_io_detail::AlignedBuffer;

off_t page_offset(uint64_t page, size_t page_size) {
  if (page > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) /
          page_size) {
    throw std::overflow_error("list record page offset exceeds off_t");
  }
  return static_cast<off_t>(page * page_size);
}

uint64_t page_count(uint64_t rows, size_t records_per_page) {
  return rows == 0 ? 0 : 1 + (rows - 1) / records_per_page;
}

struct PageBuffer {
  explicit PageBuffer(size_t alignment, size_t bytes)
      : storage(alignment, bytes) {}
  AlignedBuffer storage;
};

}  // namespace

struct PreparedRecordMutation::Impl {
  std::string path;
  uint32_t list_no = 0;
  uint32_t old_size = 0;
  uint32_t new_size = 0;
  uint64_t old_pages = 0;
  uint64_t new_pages = 0;
  size_t page_size = 0;
  size_t record_stride = 0;
  size_t records_per_page = 0;
  std::map<uint32_t, PageBuffer> pages;
  std::vector<uint32_t> write_pages;
  std::vector<uint32_t> observed_ids;
};

PreparedRecordMutation::PreparedRecordMutation() = default;
PreparedRecordMutation::~PreparedRecordMutation() = default;
PreparedRecordMutation::PreparedRecordMutation(
    PreparedRecordMutation&&) noexcept = default;
PreparedRecordMutation& PreparedRecordMutation::operator=(
    PreparedRecordMutation&&) noexcept = default;
PreparedRecordMutation::PreparedRecordMutation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

const std::vector<uint32_t>&
PreparedRecordMutation::observed_logical_ids() const {
  if (impl_ == nullptr) {
    throw std::logic_error("record mutation has not been prepared");
  }
  return impl_->observed_ids;
}

size_t PreparedRecordMutation::pages_read() const noexcept {
  return impl_ == nullptr ? 0 : impl_->pages.size();
}

size_t PreparedRecordMutation::pages_written() const noexcept {
  return impl_ == nullptr ? 0 : impl_->write_pages.size();
}

ListFileRecordStore::ListFileRecordStore(
    std::string root,
    size_t nlist,
    ListFileStoreOptions options,
    bool create)
    : root_(std::move(root)),
      nlist_(nlist),
      options_(options),
      record_stride_(sizeof(uint32_t) + options.payload_bytes),
      records_per_page_(
          record_stride_ == 0 ? 0 : options.page_size / record_stride_) {
  if (root_.empty() || nlist_ == 0 || options_.payload_bytes == 0 ||
      options_.lists_per_shard == 0 ||
      !external_io_detail::is_power_of_two(options_.page_size) ||
      options_.page_size < sizeof(void*) || record_stride_ > options_.page_size ||
      records_per_page_ == 0 || !external_io_detail::is_little_endian()) {
    throw std::invalid_argument("invalid list-file record-store layout");
  }
  external_io_detail::direct_flag(options_.direct_io);
  const std::filesystem::path records =
      std::filesystem::path(root_) / "records";
  if (create) {
    std::filesystem::create_directories(records);
  } else if (!std::filesystem::is_directory(records)) {
    throw std::invalid_argument("list-file record directory does not exist");
  }
}

std::string ListFileRecordStore::list_path(uint32_t list_no) const {
  if (list_no >= nlist_) throw std::out_of_range("IVF list number is invalid");
  const uint32_t shard =
      list_no / static_cast<uint32_t>(options_.lists_per_shard);
  std::ostringstream shard_name;
  shard_name << std::setw(6) << std::setfill('0') << shard;
  std::ostringstream file_name;
  file_name << "list_" << std::setw(8) << std::setfill('0') << list_no
            << ".bin";
  return (std::filesystem::path(root_) / "records" / shard_name.str() /
          file_name.str()).string();
}

void ListFileRecordStore::append_records(
    uint32_t list_no,
    uint32_t old_size,
    const uint32_t* logical_ids,
    const void* payloads,
    size_t payload_stride,
    size_t count) {
  if (count == 0) return;
  if (logical_ids == nullptr || payloads == nullptr ||
      payload_stride < options_.payload_bytes ||
      count > std::numeric_limits<uint32_t>::max() - old_size) {
    throw std::invalid_argument("invalid list record append");
  }
  const std::string path = list_path(list_no);
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  const int fd = ::open(
      path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC |
          external_io_detail::direct_flag(options_.direct_io),
      0644);
  if (fd < 0) throw external_io_detail::system_error("open(" + path + ")");
  try {
    struct stat status {};
    if (::fstat(fd, &status) != 0) {
      throw external_io_detail::system_error("fstat(" + path + ")");
    }
    const uint64_t old_pages = page_count(old_size, records_per_page_);
    const uint64_t expected_bytes = old_pages * options_.page_size;
    if (status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) < expected_bytes ||
        static_cast<uint64_t>(status.st_size) % options_.page_size != 0) {
      throw std::runtime_error("list file size disagrees with logical_size");
    }
    // A previous append can fail after writing externally visible pages but
    // before the in-memory logical_size is published. Such rows are outside
    // the index state. Restore the old page boundary so retry overwrites them.
    if (static_cast<uint64_t>(status.st_size) > expected_bytes &&
        ::ftruncate(fd, static_cast<off_t>(expected_bytes)) != 0) {
      throw external_io_detail::system_error(
          "truncate unpublished list tail(" + path + ")");
    }
    const uint32_t new_size = old_size + static_cast<uint32_t>(count);
    const uint64_t new_pages = page_count(new_size, records_per_page_);
    const auto* payload_bytes = static_cast<const uint8_t*>(payloads);
    AlignedBuffer page(options_.page_size, options_.page_size);
    const uint64_t first_page = old_size / records_per_page_;
    for (uint64_t page_id = first_page; page_id < new_pages; ++page_id) {
      if (page_id < old_pages) {
        external_io_detail::pread_exact(
            fd, page.data(), options_.page_size,
            page_offset(page_id, options_.page_size));
      } else {
        std::memset(page.data(), 0, options_.page_size);
      }
      const uint64_t page_first = page_id * records_per_page_;
      const uint64_t begin = std::max<uint64_t>(old_size, page_first);
      const uint64_t end = std::min<uint64_t>(
          new_size, page_first + records_per_page_);
      auto* bytes = static_cast<uint8_t*>(page.data());
      for (uint64_t row = begin; row < end; ++row) {
        const size_t input = static_cast<size_t>(row - old_size);
        const size_t in_page =
            static_cast<size_t>(row - page_first) * record_stride_;
        std::memcpy(bytes + in_page, logical_ids + input, sizeof(uint32_t));
        std::memcpy(
            bytes + in_page + sizeof(uint32_t),
            payload_bytes + input * payload_stride,
            options_.payload_bytes);
      }
      external_io_detail::pwrite_exact(
          fd, page.data(), options_.page_size,
          page_offset(page_id, options_.page_size));
    }
    const uint64_t final_bytes = new_pages * options_.page_size;
    if (final_bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        ::ftruncate(fd, static_cast<off_t>(final_bytes)) != 0) {
      throw external_io_detail::system_error("ftruncate(" + path + ")");
    }
    external_io_detail::sync_file(fd, options_.sync_writes);
    ::close(fd);
  } catch (...) {
    ::close(fd);
    throw;
  }
}

std::vector<uint32_t> ListFileRecordStore::read_logical_ids(
    uint32_t list_no,
    uint32_t logical_size,
    const std::vector<uint32_t>& offsets) const {
  if (offsets.empty()) return {};
  struct Request {
    uint32_t page = 0;
    uint32_t offset = 0;
    size_t input = 0;
  };
  std::vector<Request> requests;
  requests.reserve(offsets.size());
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (offsets[i] >= logical_size) {
      throw std::out_of_range("record offset exceeds list logical size");
    }
    requests.push_back(Request{
        offsets[i] / static_cast<uint32_t>(records_per_page_), offsets[i], i});
  }
  std::sort(requests.begin(), requests.end(), [](const Request& lhs,
                                                 const Request& rhs) {
    if (lhs.page != rhs.page) return lhs.page < rhs.page;
    return lhs.offset < rhs.offset;
  });

  const std::string path = list_path(list_no);
  const int fd = ::open(
      path.c_str(), O_RDONLY | O_CLOEXEC |
          external_io_detail::direct_flag(options_.direct_io));
  if (fd < 0) throw external_io_detail::system_error("open(" + path + ")");
  try {
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) !=
            page_count(logical_size, records_per_page_) * options_.page_size) {
      throw std::runtime_error("list file size disagrees with logical_size");
    }
    AlignedBuffer page(options_.page_size, options_.page_size);
    std::vector<uint32_t> result(offsets.size());
    size_t begin = 0;
    while (begin < requests.size()) {
      size_t end = begin + 1;
      while (end < requests.size() &&
             requests[end].page == requests[begin].page) ++end;
      external_io_detail::pread_exact(
          fd, page.data(), options_.page_size,
          page_offset(requests[begin].page, options_.page_size));
      const auto* bytes = static_cast<const uint8_t*>(page.data());
      for (size_t i = begin; i < end; ++i) {
        const size_t lane = requests[i].offset % records_per_page_;
        std::memcpy(
            &result[requests[i].input], bytes + lane * record_stride_,
            sizeof(uint32_t));
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

void ListFileRecordStore::validate_list_file(
    uint32_t list_no, uint32_t logical_size) const {
  const std::string path = list_path(list_no);
  const uint64_t expected =
      page_count(logical_size, records_per_page_) * options_.page_size;
  struct stat status {};
  if (::stat(path.c_str(), &status) != 0) {
    if (logical_size == 0 && errno == ENOENT) return;
    throw external_io_detail::system_error("stat(" + path + ")");
  }
  if (status.st_size < 0 || static_cast<uint64_t>(status.st_size) != expected) {
    throw std::runtime_error("list file size disagrees with checkpoint");
  }
}

PreparedRecordMutation ListFileRecordStore::prepare_moves_and_truncate(
    uint32_t list_no,
    uint32_t old_size,
    uint32_t new_size,
    const std::vector<RecordMove>& moves,
    const std::vector<uint32_t>& observed_offsets) const {
  if (new_size > old_size) {
    throw std::invalid_argument("record deletion cannot grow a list");
  }
  for (const RecordMove& move : moves) {
    if (move.destination >= new_size || move.source < new_size ||
        move.source >= old_size) {
      throw std::invalid_argument("invalid list record move");
    }
  }
  for (uint32_t offset : observed_offsets) {
    if (offset >= old_size) {
      throw std::invalid_argument("observed record offset exceeds old list size");
    }
  }

  const std::string path = list_path(list_no);
  auto prepared = std::make_unique<PreparedRecordMutation::Impl>();
  prepared->path = path;
  prepared->list_no = list_no;
  prepared->old_size = old_size;
  prepared->new_size = new_size;
  prepared->old_pages = page_count(old_size, records_per_page_);
  prepared->new_pages = page_count(new_size, records_per_page_);
  prepared->page_size = options_.page_size;
  prepared->record_stride = record_stride_;
  prepared->records_per_page = records_per_page_;

  std::set<uint32_t> needed_pages;
  std::set<uint32_t> write_pages;
  for (uint32_t offset : observed_offsets) {
    needed_pages.insert(offset / records_per_page_);
  }
  for (const RecordMove& move : moves) {
    needed_pages.insert(move.destination / records_per_page_);
    needed_pages.insert(move.source / records_per_page_);
    write_pages.insert(move.destination / records_per_page_);
  }
  if (new_size != 0 && new_size % records_per_page_ != 0) {
    const uint32_t tail_page = new_size / records_per_page_;
    needed_pages.insert(tail_page);
    write_pages.insert(tail_page);
  }

  const int fd = ::open(
      path.c_str(), O_RDONLY | O_CLOEXEC |
          external_io_detail::direct_flag(options_.direct_io));
  if (fd < 0) throw external_io_detail::system_error("open(" + path + ")");
  try {
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) !=
            prepared->old_pages * options_.page_size) {
      throw std::runtime_error("list file size disagrees with deletion input");
    }
    for (uint32_t page_id : needed_pages) {
      PageBuffer buffer(options_.page_size, options_.page_size);
      external_io_detail::pread_exact(
          fd, buffer.storage.data(), options_.page_size,
          page_offset(page_id, options_.page_size));
      prepared->pages.emplace(page_id, std::move(buffer));
    }

    prepared->observed_ids.resize(observed_offsets.size());
    for (size_t i = 0; i < observed_offsets.size(); ++i) {
      const uint32_t offset = observed_offsets[i];
      const uint32_t page_id = offset / records_per_page_;
      const size_t lane = offset % records_per_page_;
      const auto* record = static_cast<const uint8_t*>(
          prepared->pages.at(page_id).storage.data()) + lane * record_stride_;
      std::memcpy(&prepared->observed_ids[i], record, sizeof(uint32_t));
    }

    std::vector<uint8_t> snapshots(
        external_io_detail::checked_multiply(
            moves.size(), record_stride_, "record move snapshots overflow"));
    for (size_t i = 0; i < moves.size(); ++i) {
      const RecordMove& move = moves[i];
      const uint32_t source_page = move.source / records_per_page_;
      const size_t source_lane = move.source % records_per_page_;
      const auto* source = static_cast<const uint8_t*>(
          prepared->pages.at(source_page).storage.data()) +
          source_lane * record_stride_;
      std::memcpy(
          snapshots.data() + i * record_stride_, source, record_stride_);
    }
    for (size_t i = 0; i < moves.size(); ++i) {
      const RecordMove& move = moves[i];
      const uint32_t destination_page = move.destination / records_per_page_;
      const size_t destination_lane = move.destination % records_per_page_;
      auto* destination = static_cast<uint8_t*>(
          prepared->pages.at(destination_page).storage.data()) +
          destination_lane * record_stride_;
      std::memcpy(
          destination, snapshots.data() + i * record_stride_, record_stride_);
    }

    // Canonicalize every stale record lane in the retained tail page. Full
    // pages beyond new_pages are reclaimed by ftruncate and need no writes.
    if (new_size != 0 && new_size % records_per_page_ != 0) {
      const uint32_t tail_page = new_size / records_per_page_;
      auto* bytes = static_cast<uint8_t*>(
          prepared->pages.at(tail_page).storage.data());
      const size_t first_stale = new_size % records_per_page_;
      std::memset(
          bytes + first_stale * record_stride_, 0,
          (records_per_page_ - first_stale) * record_stride_);
    }
    for (uint32_t page_id : write_pages) {
      if (page_id < prepared->new_pages) {
        prepared->write_pages.push_back(page_id);
      }
    }
    ::close(fd);
    return PreparedRecordMutation(std::move(prepared));
  } catch (...) {
    ::close(fd);
    throw;
  }
}

void ListFileRecordStore::commit_prepared_mutation(
    const PreparedRecordMutation& mutation) {
  if (mutation.impl_ == nullptr ||
      mutation.impl_->page_size != options_.page_size ||
      mutation.impl_->record_stride != record_stride_ ||
      mutation.impl_->records_per_page != records_per_page_ ||
      mutation.impl_->path != list_path(mutation.impl_->list_no)) {
    throw std::invalid_argument("record mutation belongs to another store");
  }
  const PreparedRecordMutation::Impl& prepared = *mutation.impl_;
  const int fd = ::open(
      prepared.path.c_str(), O_RDWR | O_CLOEXEC |
          external_io_detail::direct_flag(options_.direct_io));
  if (fd < 0) {
    throw external_io_detail::system_error("open(" + prepared.path + ")");
  }
  try {
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) !=
            prepared.old_pages * options_.page_size) {
      throw std::runtime_error("list file changed after deletion preparation");
    }
    for (uint32_t page_id : prepared.write_pages) {
      external_io_detail::pwrite_exact(
          fd, prepared.pages.at(page_id).storage.data(), options_.page_size,
          page_offset(page_id, options_.page_size));
    }
    const uint64_t final_bytes = prepared.new_pages * options_.page_size;
    if (final_bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        ::ftruncate(fd, static_cast<off_t>(final_bytes)) != 0) {
      throw external_io_detail::system_error(
          "ftruncate(" + prepared.path + ")");
    }
    external_io_detail::sync_file(fd, options_.sync_writes);
    ::close(fd);
  } catch (...) {
    ::close(fd);
    throw;
  }
}

void ListFileRecordStore::apply_moves_and_truncate(
    uint32_t list_no,
    uint32_t old_size,
    uint32_t new_size,
    const std::vector<RecordMove>& moves) {
  PreparedRecordMutation prepared = prepare_moves_and_truncate(
      list_no, old_size, new_size, moves, {});
  commit_prepared_mutation(prepared);
}

}  // namespace recastlib::adapters
