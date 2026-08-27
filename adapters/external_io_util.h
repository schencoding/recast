#ifndef RECASTLIB_ADAPTERS_EXTERNAL_IO_UTIL_H
#define RECASTLIB_ADAPTERS_EXTERNAL_IO_UTIL_H

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <fcntl.h>

#include "recastlib/platform.h"

namespace recastlib::adapters::external_io_detail {

inline bool is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

inline bool is_little_endian() noexcept {
  const uint16_t value = 1;
  return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

inline size_t checked_add(size_t lhs, size_t rhs, const char* message) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

inline size_t checked_multiply(size_t lhs, size_t rhs, const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

inline std::runtime_error system_error(
    const std::string& operation, int error = errno) {
  return std::runtime_error(
      operation + " failed: " + std::string(std::strerror(error)));
}

inline int direct_flag(bool enabled) {
  if (!enabled) return 0;
#if RECASTLIB_HAS_LINUX_AIO
  return O_DIRECT;
#else
  throw std::invalid_argument(
      "O_DIRECT external-list storage is available only in Linux builds");
#endif
}

class AlignedBuffer {
 public:
  AlignedBuffer(size_t alignment, size_t bytes) : bytes_(bytes) {
    if (!is_power_of_two(alignment) || alignment < sizeof(void*) || bytes == 0) {
      throw std::invalid_argument("invalid aligned I/O buffer size");
    }
    const int result = ::posix_memalign(&data_, alignment, bytes);
    if (result != 0) throw system_error("posix_memalign", result);
  }

  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this == &other) return *this;
    std::free(data_);
    data_ = other.data_;
    bytes_ = other.bytes_;
    other.data_ = nullptr;
    other.bytes_ = 0;
    return *this;
  }

  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

inline void pread_exact(int fd, void* buffer, size_t bytes, off_t offset) {
  size_t completed = 0;
  while (completed < bytes) {
    const ssize_t result = ::pread(
        fd, static_cast<unsigned char*>(buffer) + completed,
        bytes - completed, offset + static_cast<off_t>(completed));
    if (result < 0 && errno == EINTR) continue;
    if (result < 0) throw system_error("pread");
    if (result == 0) throw std::runtime_error("unexpected end of external file");
    completed += static_cast<size_t>(result);
  }
}

inline void pwrite_exact(
    int fd, const void* buffer, size_t bytes, off_t offset) {
  size_t completed = 0;
  while (completed < bytes) {
    const ssize_t result = ::pwrite(
        fd, static_cast<const unsigned char*>(buffer) + completed,
        bytes - completed, offset + static_cast<off_t>(completed));
    if (result < 0 && errno == EINTR) continue;
    if (result < 0) throw system_error("pwrite");
    if (result == 0) throw std::runtime_error("pwrite made no progress");
    completed += static_cast<size_t>(result);
  }
}

inline void sync_file(int fd, bool enabled) {
  if (!enabled) return;
#if defined(__linux__)
  if (::fdatasync(fd) != 0) throw system_error("fdatasync");
#else
  if (::fsync(fd) != 0) throw system_error("fsync");
#endif
}

}  // namespace recastlib::adapters::external_io_detail

#endif  // RECASTLIB_ADAPTERS_EXTERNAL_IO_UTIL_H
