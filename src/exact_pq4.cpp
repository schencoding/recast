#include "recastlib/detail/exact_pq4.h"

#include <algorithm>
#include <stdexcept>

#include <faiss/impl/pq4_fast_scan.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define RECAST_X86 1
#else
#define RECAST_X86 0
#endif

#if RECAST_X86 && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <immintrin.h>
#define RECAST_COMPILE_AVX512_KERNEL 1
#else
#define RECAST_COMPILE_AVX512_KERNEL 0
#endif

namespace {

#if RECAST_COMPILE_AVX512_KERNEL

uint64_t read_xcr0() noexcept {
  // CPUID reports CPU capabilities, but XCR0 is the authoritative check that
  // the operating system saves and restores the corresponding SIMD registers.
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
}

bool detect_avx512f() noexcept {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  // Leaf 7 contains AVX-512F, while leaf 1 contains the AVX/OSXSAVE gates
  // required before executing XGETBV.
  if (__get_cpuid_max(0, nullptr) < 7 ||
      !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return false;
  }

  constexpr uint32_t kOsxsave = 1u << 27;
  constexpr uint32_t kAvx = 1u << 28;
  if ((ecx & (kOsxsave | kAvx)) != (kOsxsave | kAvx)) {
    return false;
  }

  // XMM, YMM, opmask, upper-ZMM, and high-ZMM state must all be enabled.
  constexpr uint64_t kAvx512Xcr0Mask = 0xe6;
  if ((read_xcr0() & kAvx512Xcr0Mask) != kAvx512Xcr0Mask) {
    return false;
  }

  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  constexpr uint32_t kAvx512f = 1u << 16;
  return (ebx & kAvx512f) != 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f"), noinline))
#endif
void accumulate_exact_pq4_lut_avx512_impl(
    size_t padded_size,
    size_t M,
    size_t M2,
    const uint8_t* packed_codes,
    const float* lut,
    float* output) {
  // Within each nibble half, pq4_pack_codes uses Faiss's perm0 lane order
  // [0, 8, 1, 9, ..., 7, 15]. Restore sequential vector order before writing
  // vector lanes 0..15 and 16..31.
  const __m512i sequential_order = _mm512_setr_epi32(
      0, 2, 4, 6, 8, 10, 12, 14,
      1, 3, 5, 7, 9, 11, 13, 15);
  const __m128i nibble_mask = _mm_set1_epi8(0x0f);
  // One 32-vector tile stores 32 bytes for every pair of subquantizers.
  // For a pair, the first/second 16-byte vectors select LUT rows m/m+1; their
  // low nibbles encode vector lanes 0..15 and high nibbles encode 16..31.
  const size_t block_stride = (M2 / 2) * 32;

  for (size_t block_no = 0; block_no < padded_size / 32; ++block_no) {
    const uint8_t* block = packed_codes + block_no * block_stride;
    // "low" and "high" refer to vector-lane nibbles, not subquantizer numbers.
    __m512 low_sums = _mm512_setzero_ps();
    __m512 high_sums = _mm512_setzero_ps();

    for (size_t m = 0; m < M; m += 2) {
      const uint8_t* pair = block + (m / 2) * 32;
      const __m128i packed0 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(pair));
      const __m128i packed1 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(pair + 16));

      const __m128i low0 = _mm_and_si128(packed0, nibble_mask);
      const __m128i low1 = _mm_and_si128(packed1, nibble_mask);
      const __m128i high0 = _mm_and_si128(
          _mm_srli_epi16(packed0, 4), nibble_mask);
      const __m128i high1 = _mm_and_si128(
          _mm_srli_epi16(packed1, 4), nibble_mask);

      // PQ4 has exactly 16 centroids per subquantizer, so each LUT row fills
      // one AVX-512 register and each unpacked nibble is a direct lane index.
      const __m512 lut0 = _mm512_loadu_ps(lut + m * 16);
      const __m512 lut1 = _mm512_loadu_ps(lut + (m + 1) * 16);

      // Preserve scalar subquantizer order: add row m, then row m + 1. Besides
      // matching semantics, this minimizes avoidable float-rounding differences.
      low_sums = _mm512_add_ps(
          low_sums,
          _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(low0), lut0));
      low_sums = _mm512_add_ps(
          low_sums,
          _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(low1), lut1));
      high_sums = _mm512_add_ps(
          high_sums,
          _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(high0), lut0));
      high_sums = _mm512_add_ps(
          high_sums,
          _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(high1), lut1));
    }

    low_sums = _mm512_permutexvar_ps(sequential_order, low_sums);
    high_sums = _mm512_permutexvar_ps(sequential_order, high_sums);
    _mm512_storeu_ps(output + block_no * 32, low_sums);
    _mm512_storeu_ps(output + block_no * 32 + 16, high_sums);
  }
}

#endif // RECAST_COMPILE_AVX512_KERNEL

} // anonymous namespace

namespace recastlib::detail {

bool exact_pq4_avx512_supported() noexcept {
#if RECAST_COMPILE_AVX512_KERNEL
  // CPU and OS SIMD-state capabilities do not change during this process.
  static const bool supported = detect_avx512f();
  return supported;
#else
  return false;
#endif
}

void accumulate_exact_pq4_lut_avx512(
    size_t padded_size,
    size_t block_size,
    size_t subquantizers,
    const uint8_t* packed_codes,
    const float* lut,
    float* output) {
  if (!exact_pq4_avx512_supported()) {
    throw std::runtime_error(
        "AVX-512F exact-float PQ4 kernel is unavailable on this host");
  }
  if (block_size != 32 || padded_size % 32 != 0 ||
      subquantizers == 0 || subquantizers % 2 != 0 ||
      packed_codes == nullptr || lut == nullptr ||
      output == nullptr) {
    throw std::invalid_argument("invalid exact-float PQ4 kernel arguments");
  }
#if RECAST_COMPILE_AVX512_KERNEL
  // Recast requires an even M, so the packed layout needs no synthetic extra
  // subquantizer. The kernel processes padding lanes, but callers consume only
  // the logical prefix described by CodeBlockView::size.
  accumulate_exact_pq4_lut_avx512_impl(
      padded_size, subquantizers, subquantizers,
      packed_codes, lut, output);
#else
  (void)padded_size;
  (void)subquantizers;
  (void)packed_codes;
  (void)lut;
  (void)output;
#endif
}

void accumulate_exact_pq4_lut_scalar(
    size_t count,
    size_t padded_size,
    size_t block_size,
    size_t subquantizers,
    const uint8_t* packed_codes,
    const float* lut,
    float* output) {
  if (count > padded_size || block_size == 0 ||
      padded_size % block_size != 0 || subquantizers == 0 ||
      subquantizers % 2 != 0 || packed_codes == nullptr || lut == nullptr ||
      output == nullptr) {
    throw std::invalid_argument("invalid scalar PQ4 kernel arguments");
  }

  std::fill(output, output + padded_size, 0.0f);
  // Faiss's accessor hides the bbs-specific byte and nibble permutations. This
  // path is both the portable exact backend and the reference for SIMD checks.
  for (size_t i = 0; i < count; ++i) {
    float sum = 0.0f;
    for (size_t m = 0; m < subquantizers; ++m) {
      const uint8_t code = faiss::pq4_get_packed_element(
          packed_codes, block_size, subquantizers, i, m);
      sum += lut[m * 16 + code];
    }
    output[i] = sum;
  }
}

} // namespace recastlib::detail
