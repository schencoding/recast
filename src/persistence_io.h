#ifndef RECASTLIB_SRC_PERSISTENCE_IO_H
#define RECASTLIB_SRC_PERSISTENCE_IO_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <faiss/impl/io.h>

namespace recastlib::detail::persistence {

inline void write_bytes(
    faiss::IOWriter* writer, const void* data, size_t bytes) {
  if (writer == nullptr || (bytes != 0 && data == nullptr) ||
      (bytes != 0 && (*writer)(data, 1, bytes) != bytes)) {
    throw std::runtime_error("failed to write Recast quantizer state");
  }
}

inline void read_bytes(faiss::IOReader* reader, void* data, size_t bytes) {
  if (reader == nullptr || (bytes != 0 && data == nullptr) ||
      (bytes != 0 && (*reader)(data, 1, bytes) != bytes)) {
    throw std::runtime_error("truncated Recast quantizer state");
  }
}

template <typename T>
void write_scalar(faiss::IOWriter* writer, const T& value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "persistent scalars must be trivially copyable");
  write_bytes(writer, &value, sizeof(value));
}

template <typename T>
T read_scalar(faiss::IOReader* reader) {
  static_assert(std::is_trivially_copyable<T>::value,
                "persistent scalars must be trivially copyable");
  T value{};
  read_bytes(reader, &value, sizeof(value));
  return value;
}

inline size_t checked_size(uint64_t value, size_t maximum, const char* name) {
  if (value > maximum || value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(std::string("invalid persisted ") + name);
  }
  return static_cast<size_t>(value);
}

template <typename T>
void write_vector(faiss::IOWriter* writer, const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable<T>::value,
                "persistent vectors must be trivially copyable");
  write_scalar(writer, static_cast<uint64_t>(values.size()));
  if (!values.empty()) {
    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("persistent vector byte size overflows");
    }
    write_bytes(writer, values.data(), values.size() * sizeof(T));
  }
}

template <typename T>
std::vector<T> read_vector(
    faiss::IOReader* reader, size_t maximum, const char* name) {
  static_assert(std::is_trivially_copyable<T>::value,
                "persistent vectors must be trivially copyable");
  const size_t count = checked_size(
      read_scalar<uint64_t>(reader), maximum, name);
  std::vector<T> values(count);
  if (!values.empty()) {
    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("persistent vector byte size overflows");
    }
    read_bytes(reader, values.data(), values.size() * sizeof(T));
  }
  return values;
}

}  // namespace recastlib::detail::persistence

#endif  // RECASTLIB_SRC_PERSISTENCE_IO_H
