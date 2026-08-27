#ifndef RECASTLIB_DETAIL_EXACT_PQ4_H
#define RECASTLIB_DETAIL_EXACT_PQ4_H

#include <cstddef>
#include <cstdint>

namespace recastlib::detail {

/** @return True when the compiled AVX-512F kernel is safe to execute. */
bool exact_pq4_avx512_supported() noexcept;

/**
 * Accumulates an exact float M-by-16 LUT over Faiss bbs=32 PQ4 codes.
 *
 * @param padded_size Number of packed rows; must be a multiple of 32.
 * @param block_size Packing block size; currently required to be 32.
 * @param subquantizers Even number M of PQ4 groups.
 * @param packed_codes Input containing padded_size * M/2 packed bytes.
 * @param lut Row-major M-by-16 float lookup table.
 * @param output Output buffer containing padded_size accumulated dot products.
 */
void accumulate_exact_pq4_lut_avx512(
    size_t padded_size,
    size_t block_size,
    size_t subquantizers,
    const uint8_t* packed_codes,
    const float* lut,
    float* output);

/**
 * Portable reference accumulation for Faiss packed PQ4 codes.
 *
 * Only the first count rows are decoded; remaining output rows through
 * padded_size are initialized to zero.
 *
 * @param count Number of logical rows to decode.
 * @param padded_size Number of packed rows in the input buffer.
 * @param block_size Packing block size used by Faiss.
 * @param subquantizers Even number M of PQ4 groups.
 * @param packed_codes Input containing padded_size * M/2 packed bytes.
 * @param lut Row-major M-by-16 float lookup table.
 * @param output Output buffer containing padded_size values.
 */
void accumulate_exact_pq4_lut_scalar(
    size_t count,
    size_t padded_size,
    size_t block_size,
    size_t subquantizers,
    const uint8_t* packed_codes,
    const float* lut,
    float* output);

}  // namespace recastlib::detail

#endif  // RECASTLIB_DETAIL_EXACT_PQ4_H
