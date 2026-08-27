#ifndef RECASTLIB_ADAPTERS_PQ4_UPDATE_PACKING_H
#define RECASTLIB_ADAPTERS_PQ4_UPDATE_PACKING_H

#include <cstddef>
#include <cstdint>

#include "recastlib/types.h"

namespace recastlib::adapters::detail {

constexpr size_t kUpdateBlockSize = 32;

/** Returns the packed PQ4 bytes occupied by one 32-row block. */
size_t packed_code_block_bytes(size_t subquantizers);

/** Validates that every PQ4 lane in [begin, end) is zero and appendable. */
void validate_empty_packed_code_range_bbs32(
    size_t subquantizers,
    size_t begin,
    size_t end,
    const uint8_t* packed_codes,
    size_t packed_code_bytes);

/**
 * Appends a contiguous flat-code range into zeroed lanes of a packed list.
 *
 * Faiss 1.8.0 implements pq4_pack_codes_range() with bitwise OR. This wrapper
 * therefore validates that every destination subquantizer is zero before
 * delegating to Faiss. It must not be used as an overwrite operation.
 */
void append_flat_codes_bbs32(
    const uint8_t* flat_codes,
    size_t flat_code_bytes,
    size_t subquantizers,
    size_t begin,
    size_t end,
    uint8_t* packed_codes,
    size_t packed_code_bytes);

/** Unpacks exactly one 32-row PQ4 block into ordinary row-major flat codes. */
void unpack_flat_block_bbs32(
    const uint8_t* packed_block,
    size_t packed_block_bytes,
    size_t subquantizers,
    uint8_t* flat_codes,
    size_t flat_code_bytes);

/** Overwrites exactly one 32-row PQ4 block from row-major flat codes. */
void rewrite_flat_block_bbs32(
    const uint8_t* flat_codes,
    size_t flat_code_bytes,
    size_t subquantizers,
    uint8_t* packed_block,
    size_t packed_block_bytes);

/** Writes ordinary Metadata records into a half-open packed metadata range. */
void write_metadata_range_bbs32(
    const Metadata* metadata,
    size_t metadata_count,
    size_t begin,
    size_t end,
    uint8_t* packed_metadata,
    size_t packed_metadata_bytes);

/** Validates that every metadata lane in [begin, end) has the empty sentinel. */
void validate_empty_packed_metadata_range_bbs32(
    size_t begin,
    size_t end,
    const uint8_t* packed_metadata,
    size_t packed_metadata_bytes);

/** Unpacks exactly one 96-byte metadata block into 32 ordinary records. */
void unpack_metadata_block_bbs32(
    const uint8_t* packed_block,
    size_t packed_block_bytes,
    Metadata* metadata,
    size_t metadata_count);

/** Overwrites exactly one metadata block from 32 ordinary records. */
void rewrite_metadata_block_bbs32(
    const Metadata* metadata,
    size_t metadata_count,
    uint8_t* packed_block,
    size_t packed_block_bytes);

}  // namespace recastlib::adapters::detail

#endif  // RECASTLIB_ADAPTERS_PQ4_UPDATE_PACKING_H
