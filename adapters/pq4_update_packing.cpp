#include "pq4_update_packing.h"

#include <cstring>
#include <limits>
#include <stdexcept>

#include <faiss/Index.h>
#include <faiss/impl/pq4_fast_scan.h>

#include "recastlib/fastscan.h"

namespace recastlib::adapters::detail {
namespace {

static_assert(
    FAISS_VERSION_MAJOR == 1 && FAISS_VERSION_MINOR == 8 &&
        FAISS_VERSION_PATCH == 0,
    "Recast mutable PQ4 packing must be re-audited for this Faiss version");

size_t checked_multiply(size_t lhs, size_t rhs, const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

size_t flat_code_size(size_t subquantizers) {
  if (subquantizers == 0 || subquantizers % 2 != 0) {
    throw std::invalid_argument(
        "mutable PQ4 packing requires a positive even subquantizer count");
  }
  return subquantizers / 2;
}

bool ranges_overlap(
    const void* first,
    size_t first_bytes,
    const void* second,
    size_t second_bytes) {
  if (first_bytes == 0 || second_bytes == 0) return false;
  const uintptr_t first_begin = reinterpret_cast<uintptr_t>(first);
  const uintptr_t second_begin = reinterpret_cast<uintptr_t>(second);
  if (first_begin > std::numeric_limits<uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<uintptr_t>::max() - second_bytes) {
    throw std::overflow_error("pointer range overflows uintptr_t");
  }
  return first_begin < second_begin + second_bytes &&
      second_begin < first_begin + first_bytes;
}

}  // namespace

size_t packed_code_block_bytes(size_t subquantizers) {
  return checked_multiply(
      kUpdateBlockSize, flat_code_size(subquantizers),
      "packed PQ4 block size overflows");
}

void validate_empty_packed_code_range_bbs32(
    size_t subquantizers,
    size_t begin,
    size_t end,
    const uint8_t* packed_codes,
    size_t packed_code_bytes) {
  const size_t code_size = flat_code_size(subquantizers);
  if (begin > end || (end > begin && packed_codes == nullptr)) {
    throw std::invalid_argument("invalid packed PQ4 validation range");
  }
  if (begin == end) return;
  if (end > std::numeric_limits<size_t>::max() - (kUpdateBlockSize - 1)) {
    throw std::overflow_error("packed PQ4 validation row count overflows");
  }
  const size_t padded_end =
      (end + kUpdateBlockSize - 1) / kUpdateBlockSize * kUpdateBlockSize;
  const size_t required_packed_bytes = checked_multiply(
      padded_end, code_size, "packed PQ4 validation size overflows");
  if (packed_code_bytes < required_packed_bytes) {
    throw std::invalid_argument("packed PQ4 validation buffer is too small");
  }

  // Faiss 1.8.0 pq4_pack_codes_range() ORs nibbles into the destination.
  // Inspect individual lanes because the paired nibble in the same byte can
  // legitimately belong to an older vector.
  for (size_t row = begin; row < end; ++row) {
    for (size_t m = 0; m < subquantizers; ++m) {
      if (faiss::pq4_get_packed_element(
              packed_codes, kUpdateBlockSize, subquantizers, row, m) != 0) {
        throw std::logic_error(
            "pq4_pack_codes_range destination lane is not zero");
      }
    }
  }
}

void append_flat_codes_bbs32(
    const uint8_t* flat_codes,
    size_t flat_code_bytes,
    size_t subquantizers,
    size_t begin,
    size_t end,
    uint8_t* packed_codes,
    size_t packed_code_bytes) {
  const size_t code_size = flat_code_size(subquantizers);
  if (begin > end || (end > begin &&
      (flat_codes == nullptr || packed_codes == nullptr))) {
    throw std::invalid_argument("invalid packed PQ4 append range");
  }
  const size_t count = end - begin;
  const size_t expected_flat_bytes = checked_multiply(
      count, code_size, "flat PQ4 append size overflows");
  if (flat_code_bytes != expected_flat_bytes) {
    throw std::invalid_argument("flat PQ4 append byte count is inconsistent");
  }
  if (count == 0) return;

  if (ranges_overlap(
          flat_codes, flat_code_bytes, packed_codes, packed_code_bytes)) {
    throw std::invalid_argument(
        "flat and packed PQ4 append buffers must not overlap");
  }

  validate_empty_packed_code_range_bbs32(
      subquantizers, begin, end, packed_codes, packed_code_bytes);

  faiss::pq4_pack_codes_range(
      flat_codes, subquantizers, begin, end, kUpdateBlockSize,
      subquantizers, packed_codes);
}

void unpack_flat_block_bbs32(
    const uint8_t* packed_block,
    size_t packed_block_bytes_value,
    size_t subquantizers,
    uint8_t* flat_codes,
    size_t flat_code_bytes) {
  const size_t expected_block_bytes =
      packed_code_block_bytes(subquantizers);
  if (packed_block == nullptr || flat_codes == nullptr ||
      packed_block_bytes_value != expected_block_bytes ||
      flat_code_bytes != expected_block_bytes) {
    throw std::invalid_argument("invalid PQ4 block unpack buffers");
  }
  if (ranges_overlap(
          packed_block, packed_block_bytes_value,
          flat_codes, flat_code_bytes)) {
    throw std::invalid_argument(
        "packed and flat PQ4 unpack buffers must not overlap");
  }
  faiss::CodePackerPQ4 packer(subquantizers, kUpdateBlockSize);
  packer.unpack_all(packed_block, flat_codes);
}

void rewrite_flat_block_bbs32(
    const uint8_t* flat_codes,
    size_t flat_code_bytes,
    size_t subquantizers,
    uint8_t* packed_block,
    size_t packed_block_bytes_value) {
  const size_t expected_block_bytes =
      packed_code_block_bytes(subquantizers);
  if (flat_codes == nullptr || packed_block == nullptr ||
      flat_code_bytes != expected_block_bytes ||
      packed_block_bytes_value != expected_block_bytes) {
    throw std::invalid_argument("invalid PQ4 block rewrite buffers");
  }
  if (ranges_overlap(
          flat_codes, flat_code_bytes,
          packed_block, packed_block_bytes_value)) {
    throw std::invalid_argument(
        "flat and packed PQ4 rewrite buffers must not overlap");
  }
  faiss::CodePackerPQ4 packer(subquantizers, kUpdateBlockSize);
  packer.pack_all(flat_codes, packed_block);
}

void write_metadata_range_bbs32(
    const Metadata* metadata,
    size_t metadata_count,
    size_t begin,
    size_t end,
    uint8_t* packed_metadata,
    size_t packed_metadata_bytes) {
  if (begin > end || metadata_count != end - begin ||
      (end > begin && (metadata == nullptr || packed_metadata == nullptr))) {
    throw std::invalid_argument("invalid packed metadata append range");
  }
  if (metadata_count == 0) return;
  const size_t source_bytes = checked_multiply(
      metadata_count, sizeof(Metadata),
      "metadata append source size overflows");
  if (ranges_overlap(
          metadata, source_bytes,
          packed_metadata, packed_metadata_bytes)) {
    throw std::invalid_argument(
        "flat and packed metadata append buffers must not overlap");
  }
  validate_empty_packed_metadata_range_bbs32(
      begin, end, packed_metadata, packed_metadata_bytes);
  for (size_t i = 0; i < metadata_count; ++i) {
    const size_t row = begin + i;
    const size_t block_base =
        (row / kUpdateBlockSize) * kUpdateBlockSize * sizeof(Metadata);
    const size_t lane = row % kUpdateBlockSize;
    packed_metadata[block_base + lane] = metadata[i].norm;
    packed_metadata[block_base + kUpdateBlockSize + lane] = metadata[i].scale;
    packed_metadata[block_base + 2 * kUpdateBlockSize + lane] = metadata[i].error;
  }
}

void validate_empty_packed_metadata_range_bbs32(
    size_t begin,
    size_t end,
    const uint8_t* packed_metadata,
    size_t packed_metadata_bytes) {
  if (begin > end || (end > begin && packed_metadata == nullptr)) {
    throw std::invalid_argument("invalid packed metadata validation range");
  }
  if (begin == end) return;
  if (end > std::numeric_limits<size_t>::max() - (kUpdateBlockSize - 1)) {
    throw std::overflow_error("packed metadata validation row count overflows");
  }
  const size_t padded_end =
      (end + kUpdateBlockSize - 1) / kUpdateBlockSize * kUpdateBlockSize;
  const size_t required_bytes = checked_multiply(
      padded_end, sizeof(Metadata),
      "packed metadata validation size overflows");
  if (packed_metadata_bytes < required_bytes) {
    throw std::invalid_argument("packed metadata validation buffer is too small");
  }

  // Validate the complete destination before the first write so a failed
  // precondition cannot leave a partially populated logical range.
  for (size_t row = begin; row < end; ++row) {
    const size_t block_base =
        (row / kUpdateBlockSize) * kUpdateBlockSize * sizeof(Metadata);
    const size_t lane = row % kUpdateBlockSize;
    if (packed_metadata[block_base + lane] != 255 ||
        packed_metadata[block_base + kUpdateBlockSize + lane] != 255 ||
        packed_metadata[block_base + 2 * kUpdateBlockSize + lane] != 255) {
      throw std::logic_error("packed metadata destination lane is not empty");
    }
  }
}

void unpack_metadata_block_bbs32(
    const uint8_t* packed_block,
    size_t packed_block_bytes,
    Metadata* metadata,
    size_t metadata_count) {
  constexpr size_t kMetadataBlockBytes =
      kUpdateBlockSize * sizeof(Metadata);
  if (packed_block == nullptr || metadata == nullptr ||
      packed_block_bytes != kMetadataBlockBytes ||
      metadata_count != kUpdateBlockSize) {
    throw std::invalid_argument("invalid metadata block unpack buffers");
  }
  if (ranges_overlap(
          packed_block, packed_block_bytes,
          metadata, metadata_count * sizeof(Metadata))) {
    throw std::invalid_argument(
        "packed and flat metadata unpack buffers must not overlap");
  }
  for (size_t lane = 0; lane < kUpdateBlockSize; ++lane) {
    metadata[lane] = Metadata{
        packed_block[lane],
        packed_block[kUpdateBlockSize + lane],
        packed_block[2 * kUpdateBlockSize + lane]};
  }
}

void rewrite_metadata_block_bbs32(
    const Metadata* metadata,
    size_t metadata_count,
    uint8_t* packed_block,
    size_t packed_block_bytes) {
  constexpr size_t kMetadataBlockBytes =
      kUpdateBlockSize * sizeof(Metadata);
  if (metadata == nullptr || packed_block == nullptr ||
      metadata_count != kUpdateBlockSize ||
      packed_block_bytes != kMetadataBlockBytes) {
    throw std::invalid_argument("invalid metadata block rewrite buffers");
  }
  if (ranges_overlap(
          metadata, metadata_count * sizeof(Metadata),
          packed_block, packed_block_bytes)) {
    throw std::invalid_argument(
        "flat and packed metadata rewrite buffers must not overlap");
  }
  pack_metadata_bbs32(
      metadata, kUpdateBlockSize, kUpdateBlockSize, packed_block);
}

}  // namespace recastlib::adapters::detail
