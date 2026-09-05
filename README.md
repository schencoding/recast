# RecastLib

RecastLib is the compact C++ research prototype accompanying Recast. This
repository contains only the current external-list implementation used by the
paper: memory-resident RecastQ postings contain PQ4 direction codes and three
metadata bytes per vector, while exact records are stored in page-aligned IVF
list files.

The repository intentionally excludes historical IVF adapters, competing
quantizers, experiment sweep drivers, plotting scripts, and paper-specific
benchmark harnesses.

## Current Design

- **RecastQ encoding.** A vector uses `M / 2` bytes of four-bit direction code
  plus three bytes of norm, reconstruction-scale, and error metadata.
- **Uneven grouping.** `M` need not divide the dimension. Recast learns PCA and
  variance-aware groups, then packs all groups into Faiss `bbs=32` PQ4 blocks.
- **FastScan.** The scanner uses exact float32 LUT accumulation with AVX-512
  when available, Faiss uint8 FastScan when selected, and a scalar float32
  fallback on machines without AVX-512.
- **Uncertainty-aware selection.** Recast forms a global upper-bound threshold
  and refines candidates whose lower endpoint can cross that boundary.
- **Compact routing.** Float32 IVF centroids are temporary. After training, an
  `M=D` PQFastScan router is used consistently for insertion and query list
  selection, and the original centroids are released.
- **External-list storage.** Resident postings contain neither Logical IDs nor
  exact-vector locations. A query-local `{list, offset}` token locates a
  headerless page-aligned record containing `{Logical ID, exact payload}`.
- **Updates.** Batched insertion and swap-with-last deletion update each touched
  packed list once. A disk-resident Logical-ID directory supports deletion
  without adding an O(N) resident location table.
- **Linux AIO.** Linux builds can issue page reads through `libaio`. The same
  source builds on macOS with the AIO path disabled.

The current implementation targets squared L2 search over float32 vectors in
memory. Exact payloads may use float32, uint8, int8, or an application-defined
opaque representation with a matching exact-distance callback.

## Dependencies

Required:

- C++17 compiler
- CMake 3.16 or newer
- Faiss 1.8.0, including PQ4 FastScan headers
- OpenMP
- BLAS/LAPACK

Linux additionally uses:

- OpenBLAS (preferred for Linux builds)
- `libaio` development headers and library

On macOS, Homebrew packages are sufficient:

```bash
brew install cmake faiss libomp
```

On a CentOS/RHEL-like Linux host, install or provide equivalent development
packages for OpenBLAS, libaio, OpenMP, and Faiss 1.8.0. `FAISS_ROOT` must point
to the Faiss installation prefix when it is not under `/usr/local`.

## Build

Linux:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ROOT=/usr/local
cmake --build build -j8
```

macOS:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ROOT=/opt/homebrew \
  -DRECASTLIB_WITH_LINUX_AIO=OFF
cmake --build build -j8
```

Release builds enable LTO by default. Pass
`-DRECASTLIB_ENABLE_LTO=OFF` only when the compiler or linker does not support
interprocedural optimization.

## SIFT1M Integration Test

The repository has one end-to-end test. It reads standard SIFT1M `fvecs` or
`bvecs` files and an `ivecs` ground-truth file, trains and builds an index with
eight threads, then runs all queries with eight threads. The exact-list payload
keeps the base file's float32 or uint8 representation.

The fixed test configuration is:

```text
distance       squared L2
top-k          10
nlist          1024
nprobe         64
Recast M       50 (28 bytes/vector including metadata)
z              1
max candidates 2000
query threads  8
```

Run it by supplying every dataset path explicitly:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./build/recastlib_test \
  --learn /path/to/sift_learn.fvecs \
  --base /path/to/sift_base.fvecs \
  --query /path/to/sift_query.fvecs \
  --groundtruth /path/to/sift_groundtruth.ivecs
```

The program reports training time, indexing time, Recall@10, QPS, scanned
vectors/query, refined vectors/query, QUIC time/query, pages/query, and read
requests/query.
It creates a temporary external-list store and removes it on exit. The portable
test uses ordinary page-aligned file reads rather than O_DIRECT, so its QPS is a
functional smoke-test number, not a paper benchmark.

`ctest` only checks that the test executable and command-line interface are
available; the full SIFT1M test is invoked explicitly because dataset paths are
machine-specific.

## Repository Layout

```text
include/recastlib/   public quantizer, scanner, and refinement APIs
src/                 RecastQ and FastScan implementation
adapters/            current PQFastScan router and external-list index/storage
tests/test.cpp       single SIFT1M end-to-end test
```

The adapter keeps list files headerless. Each page stores an integral number of
`{Logical ID, exact payload}` records; unused bytes at the end of a page are
padding. This makes a record address a pure function of `{list, offset}` while
preserving page-aligned reads.
