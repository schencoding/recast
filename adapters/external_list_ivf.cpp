#include "external_list_ivf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <faiss/index_io.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pq4_update_packing.h"

namespace recastlib::adapters {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_nanoseconds(Clock::time_point begin) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin)
          .count());
}

size_t round_up(size_t value, size_t multiple) {
  if (multiple == 0 ||
      value > std::numeric_limits<size_t>::max() - (multiple - 1)) {
    throw std::overflow_error("rounded size overflows");
  }
  return (value + multiple - 1) / multiple * multiple;
}

size_t checked_add(size_t lhs, size_t rhs, const char* message) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

size_t checked_multiply(size_t lhs, size_t rhs, const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

template <typename Function>
void parallel_for_static(
    size_t count, size_t minimum_items_per_worker, Function&& function) {
#ifdef _OPENMP
  const size_t useful_workers = count == 0 ? 0 :
      1 + (count - 1) / std::max<size_t>(minimum_items_per_worker, 1);
  const int worker_count = static_cast<int>(std::min<size_t>(
      static_cast<size_t>(omp_get_max_threads()), useful_workers));
  if (worker_count > 1 && !omp_in_parallel()) {
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::overflow_error("parallel batch stage is too large");
    }
    std::decay_t<Function> task(std::forward<Function>(function));
    std::exception_ptr first_error;
#pragma omp parallel for schedule(static) firstprivate(task) num_threads(worker_count)
    for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
      try {
        task(static_cast<size_t>(index));
      } catch (...) {
#pragma omp critical(recastlib_external_list_error)
        {
          if (first_error == nullptr) first_error = std::current_exception();
        }
      }
    }
    if (first_error != nullptr) std::rethrow_exception(first_error);
    return;
  }
#endif
  for (size_t index = 0; index < count; ++index) function(index);
}

template <typename Function>
void parallel_for_dynamic(
    size_t count, size_t minimum_items_per_worker, Function&& function) {
#ifdef _OPENMP
  const size_t useful_workers = count == 0 ? 0 :
      1 + (count - 1) / std::max<size_t>(minimum_items_per_worker, 1);
  const int worker_count = static_cast<int>(std::min<size_t>(
      static_cast<size_t>(omp_get_max_threads()), useful_workers));
  if (worker_count > 1 && !omp_in_parallel()) {
    if (count > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::overflow_error("parallel batch stage is too large");
    }
    std::decay_t<Function> task(std::forward<Function>(function));
    std::exception_ptr first_error;
#pragma omp parallel for schedule(dynamic, 8) firstprivate(task) num_threads(worker_count)
    for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
      try {
        task(static_cast<size_t>(index));
      } catch (...) {
#pragma omp critical(recastlib_external_list_error)
        {
          if (first_error == nullptr) first_error = std::current_exception();
        }
      }
    }
    if (first_error != nullptr) std::rethrow_exception(first_error);
    return;
  }
#endif
  for (size_t index = 0; index < count; ++index) function(index);
}

struct ListBatchPlan {
  std::vector<size_t> offsets;
  std::vector<size_t> order;
  std::vector<uint32_t> touched_lists;
};

ListBatchPlan group_by_list(
    const std::vector<uint32_t>& assignments, size_t nlist) {
  ListBatchPlan plan;
  plan.offsets.assign(nlist + 1, 0);
  plan.order.resize(assignments.size());
  for (uint32_t list_no : assignments) {
    if (list_no >= nlist) {
      throw std::invalid_argument("batch contains an invalid IVF list number");
    }
    ++plan.offsets[static_cast<size_t>(list_no) + 1];
  }
  for (size_t list_no = 0; list_no < nlist; ++list_no) {
    plan.offsets[list_no + 1] = checked_add(
        plan.offsets[list_no], plan.offsets[list_no + 1],
        "batch grouping prefix sum overflows");
    if (plan.offsets[list_no] != plan.offsets[list_no + 1]) {
      plan.touched_lists.push_back(static_cast<uint32_t>(list_no));
    }
  }
  std::vector<size_t> cursors = plan.offsets;
  for (size_t input = 0; input < assignments.size(); ++input) {
    plan.order[cursors[assignments[input]]++] = input;
  }
  return plan;
}

uint64_t page_count(uint64_t rows, size_t rows_per_page) {
  return rows == 0 ? 0 : 1 + (rows - 1) / rows_per_page;
}

uint64_t unique_pages(
    const std::vector<uint32_t>& offsets, size_t rows_per_page) {
  if (offsets.empty()) return 0;
  std::vector<uint32_t> pages;
  pages.reserve(offsets.size());
  for (uint32_t offset : offsets) {
    pages.push_back(offset / static_cast<uint32_t>(rows_per_page));
  }
  std::sort(pages.begin(), pages.end());
  return static_cast<uint64_t>(
      std::unique(pages.begin(), pages.end()) - pages.begin());
}

uint64_t unique_directory_pages(
    const std::vector<uint32_t>& ids, size_t entries_per_page) {
  return unique_pages(ids, entries_per_page);
}

ListFileStoreOptions store_options(const ExternalListIndexOptions& options) {
  return ListFileStoreOptions{
      options.page_size, options.payload_bytes, options.lists_per_shard,
      options.direct_io, options.sync_writes};
}

ExternalDirectoryOptions directory_options(
    const ExternalListIndexOptions& options) {
  return ExternalDirectoryOptions{
      options.page_size, options.direct_io, options.sync_writes};
}

bool valid_payload_format(ExactPayloadFormat format) {
  switch (format) {
    case ExactPayloadFormat::kOpaque:
    case ExactPayloadFormat::kFloat32:
    case ExactPayloadFormat::kUInt8:
    case ExactPayloadFormat::kInt8:
      return true;
  }
  return false;
}

std::string directory_path(const ExternalListIndexOptions& options) {
  return (std::filesystem::path(options.storage_root) / "locations.bin").string();
}

struct BlockScratch {
  size_t block_index = 0;
  std::vector<uint8_t> original_codes;
  std::vector<uint8_t> updated_codes;
  std::array<Metadata, detail::kUpdateBlockSize> original_metadata{};
  std::array<Metadata, detail::kUpdateBlockSize> updated_metadata{};
};

struct PreparedMove {
  uint32_t destination = 0;
  uint32_t source = 0;
  uint32_t source_id = 0;
};

struct PreparedDeletion {
  uint32_t list_no = 0;
  uint32_t old_size = 0;
  uint32_t new_size = 0;
  uint32_t new_padded_size = 0;
  std::vector<uint32_t> deleted_ids;
  std::vector<uint32_t> deleted_offsets;
  std::vector<uint32_t> expected_deleted_ids;
  std::vector<uint32_t> holes;
  std::vector<uint32_t> sources;
  std::vector<RecordMove> record_moves;
  std::vector<PreparedMove> moves;
  PreparedRecordMutation record_mutation;
  std::vector<BlockScratch> blocks;
};

struct CheckpointManifest {
  char magic[8] = {'R', 'C', 'X', 'L', 'I', 'S', 'T', '1'};
  uint64_t format_version = 1;
  uint64_t dimension = 0;
  uint64_t subquantizers = 0;
  uint64_t nlist = 0;
  uint64_t page_size = 0;
  uint64_t payload_bytes = 0;
  uint64_t payload_format = 0;
  uint64_t record_stride = 0;
  uint64_t records_per_page = 0;
  uint64_t next_id = 0;
  uint64_t active_count = 0;
  uint64_t posting_block_size = detail::kUpdateBlockSize;
  uint64_t metadata_layout_version = 1;
  uint64_t generation = 0;
};

static_assert(std::is_trivially_copyable<CheckpointManifest>::value,
              "checkpoint manifest must be trivially copyable");
static_assert(sizeof(CheckpointManifest) == 120,
              "checkpoint manifest layout changed unexpectedly");

std::filesystem::path checkpoint_path(
    const ExternalListIndexOptions& options, const char* name) {
  return std::filesystem::path(options.storage_root) / name;
}

std::filesystem::path generation_path(
    const ExternalListIndexOptions& options,
    const char* prefix,
    uint64_t generation) {
  return std::filesystem::path(options.storage_root) /
      (std::string(prefix) + "." + std::to_string(generation) + ".bin");
}

void replace_file(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination) {
  if (::rename(temporary.c_str(), destination.c_str()) != 0) {
    throw std::runtime_error(
        "cannot publish checkpoint file " + destination.string() + ": " +
        std::string(std::strerror(errno)));
  }
}

void persist_and_install_pq_router(
    FaissIvfRouter* router, const ExternalListIndexOptions& options) {
  const size_t dimension = router->dimension();
  const size_t nlist = router->nlist();
  if (nlist < 16 || dimension > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("invalid dimensions for persisted PQFastScan router");
  }
  std::vector<float> centroids(checked_multiply(
      nlist, dimension, "persisted router centroid buffer overflows"));
  for (size_t list_no = 0; list_no < nlist; ++list_no) {
    router->reconstruct_centroid(
        list_no, centroids.data() + list_no * dimension);
  }

  faiss::IndexPQFastScan persisted(
      static_cast<int>(dimension), dimension, 4, faiss::METRIC_L2, 32);
  persisted.pq.cp.min_points_per_centroid = 1;
  persisted.pq.cp.max_points_per_centroid = static_cast<int>(std::min(
      nlist, static_cast<size_t>(std::numeric_limits<int>::max())));
  persisted.implem = 0;
  persisted.train(static_cast<faiss::idx_t>(nlist), centroids.data());
  persisted.add(static_cast<faiss::idx_t>(nlist), centroids.data());
  if (!persisted.is_trained ||
      persisted.ntotal != static_cast<faiss::idx_t>(nlist)) {
    throw std::runtime_error("persisted PQFastScan router training failed");
  }

  std::filesystem::create_directories(options.storage_root);
  const auto temporary = checkpoint_path(options, "router.faiss.tmp");
  const auto destination = checkpoint_path(options, "router.faiss");
  faiss::write_index(&persisted, temporary.c_str());
  replace_file(temporary, destination);
  router->load_pq_router(destination.string());
}

template <typename T>
void write_binary(std::ofstream* output, const T* values, size_t count) {
  const size_t bytes = checked_multiply(
      count, sizeof(T), "checkpoint write size overflows");
  if (bytes == 0) return;
  output->write(
      reinterpret_cast<const char*>(values), static_cast<std::streamsize>(bytes));
  if (!*output) throw std::runtime_error("checkpoint write failed");
}

template <typename T>
void read_binary(std::ifstream* input, T* values, size_t count) {
  const size_t bytes = checked_multiply(
      count, sizeof(T), "checkpoint read size overflows");
  if (bytes == 0) return;
  input->read(
      reinterpret_cast<char*>(values), static_cast<std::streamsize>(bytes));
  if (input->gcount() != static_cast<std::streamsize>(bytes)) {
    throw std::runtime_error("checkpoint file is truncated");
  }
}

}  // namespace

ExternalListRecastIndex::ExternalListRecastIndex(
    FaissIvfRouter& router,
    QuantizerConfig config,
    ExternalListIndexOptions options)
    : router_(router),
      quantizer_(std::move(config)),
      options_(std::move(options)),
      record_store_(
          options_.storage_root, router_.nlist(), store_options(options_),
          options_.create_storage),
      directory_(
          directory_path(options_), directory_options(options_),
          options_.create_storage),
      lists_(router_.nlist()) {
  if (options_.storage_root.empty() || options_.payload_bytes == 0 ||
      options_.queue_depth == 0 ||
      options_.max_open_files_per_query < options_.queue_depth ||
      !valid_payload_format(options_.payload_format) ||
      quantizer_.dimension() != router_.dimension()) {
    throw std::invalid_argument("invalid external-list Recast configuration");
  }
  if (!options_.create_storage) {
    throw std::invalid_argument(
        "checkpoint reload requires a trained RecastQuantizer");
  }
}

ExternalListRecastIndex::ExternalListRecastIndex(
    FaissIvfRouter& router,
    RecastQuantizer quantizer,
    ExternalListIndexOptions options)
    : router_(router),
      quantizer_(std::move(quantizer)),
      options_(std::move(options)),
      record_store_(
          options_.storage_root, router_.nlist(), store_options(options_),
          options_.create_storage),
      directory_(
          directory_path(options_), directory_options(options_),
          options_.create_storage),
      lists_(router_.nlist()) {
  if (options_.storage_root.empty() || options_.payload_bytes == 0 ||
      options_.queue_depth == 0 ||
      options_.max_open_files_per_query < options_.queue_depth ||
      !valid_payload_format(options_.payload_format) ||
      !quantizer_.is_trained() ||
      quantizer_.dimension() != router_.dimension()) {
    throw std::invalid_argument("invalid trained external-list Recast state");
  }
  if (!options_.create_storage) {
    router_.load_pq_router(checkpoint_path(options_, "router.faiss").string());
    load_checkpoint_locked();
  }
}

std::string ExternalListRecastIndex::quantizer_checkpoint_path() const {
  return checkpoint_path(options_, "quantizer.recast").string();
}

void ExternalListRecastIndex::checkpoint() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (!quantizer_.is_trained()) {
    throw std::logic_error("cannot checkpoint an untrained external-list index");
  }
  if (!std::filesystem::is_regular_file(
          checkpoint_path(options_, "router.faiss"))) {
    throw std::logic_error("persisted external-list router is missing");
  }
  std::filesystem::create_directories(options_.storage_root);
  const auto quantizer_tmp = checkpoint_path(options_, "quantizer.recast.tmp");
  const auto quantizer_final = checkpoint_path(options_, "quantizer.recast");
  quantizer_.save(quantizer_tmp.string());
  replace_file(quantizer_tmp, quantizer_final);

  const uint64_t generation = checkpoint_generation_ + 1;
  if (generation == 0) throw std::overflow_error("checkpoint generation overflows");
  const auto sizes_final = generation_path(options_, "list_sizes", generation);
  const auto sizes_tmp = std::filesystem::path(sizes_final.string() + ".tmp");
  {
    std::ofstream output(sizes_tmp, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create list_sizes checkpoint");
    for (const MemoryList& list : lists_) {
      write_binary(&output, &list.logical_size, 1);
    }
    output.flush();
    if (!output) throw std::runtime_error("cannot flush list_sizes checkpoint");
  }
  replace_file(sizes_tmp, sizes_final);

  const auto postings_final =
      generation_path(options_, "postings_no_ids", generation);
  const auto postings_tmp =
      std::filesystem::path(postings_final.string() + ".tmp");
  {
    std::ofstream output(postings_tmp, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create postings checkpoint");
    for (const MemoryList& list : lists_) {
      const size_t logical_code_bytes = checked_multiply(
          list.padded_size, quantizer_.code_size(),
          "checkpoint logical code size overflows");
      const size_t logical_metadata_bytes =
          packed_metadata_size_bbs32(list.padded_size);
      if (logical_code_bytes > list.packed_codes.size() ||
          logical_metadata_bytes > list.packed_metadata.size()) {
        throw std::logic_error("packed list is shorter than its logical checkpoint");
      }
      write_binary(&output, list.packed_codes.get(), logical_code_bytes);
      write_binary(
          &output, list.packed_metadata.get(), logical_metadata_bytes);
    }
    output.flush();
    if (!output) throw std::runtime_error("cannot flush postings checkpoint");
  }
  replace_file(postings_tmp, postings_final);

  CheckpointManifest manifest;
  manifest.dimension = quantizer_.dimension();
  manifest.subquantizers = quantizer_.subquantizers();
  manifest.nlist = lists_.size();
  manifest.page_size = record_store_.page_size();
  manifest.payload_bytes = record_store_.payload_bytes();
  manifest.payload_format = static_cast<uint64_t>(options_.payload_format);
  manifest.record_stride = record_store_.record_stride();
  manifest.records_per_page = record_store_.records_per_page();
  manifest.next_id = next_id_;
  manifest.active_count = active_count_;
  manifest.generation = generation;
  const auto manifest_tmp = checkpoint_path(options_, "manifest.bin.tmp");
  const auto manifest_final = checkpoint_path(options_, "manifest.bin");
  {
    std::ofstream output(manifest_tmp, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create manifest checkpoint");
    write_binary(&output, &manifest, 1);
    output.flush();
    if (!output) throw std::runtime_error("cannot flush manifest checkpoint");
  }
  // Manifest is the publish point: a reader never accepts partially replaced
  // postings or sizes from a checkpoint whose manifest was not installed.
  replace_file(manifest_tmp, manifest_final);
  const uint64_t old_generation = checkpoint_generation_;
  checkpoint_generation_ = generation;
  if (old_generation != 0) {
    std::error_code ignored;
    std::filesystem::remove(
        generation_path(options_, "list_sizes", old_generation), ignored);
    std::filesystem::remove(
        generation_path(options_, "postings_no_ids", old_generation), ignored);
  }
}

void ExternalListRecastIndex::load_checkpoint_locked() {
  if (!std::filesystem::is_regular_file(
          checkpoint_path(options_, "router.faiss"))) {
    throw std::runtime_error("external-list PQFastScan router is missing");
  }
  CheckpointManifest manifest;
  {
    std::ifstream input(checkpoint_path(options_, "manifest.bin"), std::ios::binary);
    if (!input) throw std::runtime_error("external-list manifest is missing");
    read_binary(&input, &manifest, 1);
    char extra = 0;
    if (input.read(&extra, 1)) {
      throw std::runtime_error("external-list manifest has trailing bytes");
    }
  }
  const CheckpointManifest expected_magic;
  if (std::memcmp(manifest.magic, expected_magic.magic, sizeof(manifest.magic)) != 0 ||
      manifest.format_version != 1 ||
      manifest.dimension != quantizer_.dimension() ||
      manifest.subquantizers != quantizer_.subquantizers() ||
      manifest.nlist != lists_.size() ||
      manifest.page_size != record_store_.page_size() ||
      manifest.payload_bytes != record_store_.payload_bytes() ||
      manifest.payload_format != static_cast<uint64_t>(options_.payload_format) ||
      manifest.record_stride != record_store_.record_stride() ||
      manifest.records_per_page != record_store_.records_per_page() ||
      manifest.posting_block_size != detail::kUpdateBlockSize ||
      manifest.metadata_layout_version != 1 ||
      manifest.generation == 0 ||
      manifest.next_id > std::numeric_limits<uint32_t>::max() ||
      manifest.active_count > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("external-list manifest is incompatible");
  }

  std::vector<uint32_t> logical_sizes(lists_.size());
  {
    std::ifstream input(
        generation_path(options_, "list_sizes", manifest.generation),
        std::ios::binary);
    if (!input) throw std::runtime_error("external-list sizes checkpoint is missing");
    read_binary(&input, logical_sizes.data(), logical_sizes.size());
    char extra = 0;
    if (input.read(&extra, 1)) {
      throw std::runtime_error("external-list sizes have trailing bytes");
    }
  }
  size_t active_sum = 0;
  const size_t code_size = quantizer_.code_size();
  for (size_t list_no = 0; list_no < lists_.size(); ++list_no) {
    active_sum = checked_add(
        active_sum, logical_sizes[list_no], "checkpoint active count overflows");
    record_store_.validate_list_file(
        static_cast<uint32_t>(list_no), logical_sizes[list_no]);
  }
  if (active_sum != manifest.active_count) {
    throw std::runtime_error("manifest active count disagrees with list sizes");
  }

  std::ifstream postings(
      generation_path(options_, "postings_no_ids", manifest.generation),
      std::ios::binary);
  if (!postings) throw std::runtime_error("external-list postings are missing");
  for (size_t list_no = 0; list_no < lists_.size(); ++list_no) {
    MemoryList& list = lists_[list_no];
    list.logical_size = logical_sizes[list_no];
    list.padded_size = static_cast<uint32_t>(
        round_up(list.logical_size, detail::kUpdateBlockSize));
    list.packed_codes.resize(checked_multiply(
        list.padded_size, code_size, "checkpoint code size overflows"));
    list.packed_metadata.resize(packed_metadata_size_bbs32(list.padded_size));
    read_binary(&postings, list.packed_codes.get(), list.packed_codes.size());
    read_binary(
        &postings, list.packed_metadata.get(), list.packed_metadata.size());
  }
  char extra = 0;
  if (postings.read(&extra, 1)) {
    throw std::runtime_error("external-list postings have trailing bytes");
  }
  next_id_ = manifest.next_id;
  active_count_ = active_sum;
  checkpoint_generation_ = manifest.generation;
}

void ExternalListRecastIndex::train(size_t count, const float* vectors) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (active_count_ != 0 || next_id_ != 0) {
    throw std::logic_error("cannot retrain a nonempty external-list index");
  }
  router_.train(count, vectors);
  quantizer_.train(count, vectors);
  persist_and_install_pq_router(&router_, options_);
}

std::vector<uint32_t> ExternalListRecastIndex::insert_batch(
    size_t count,
    const float* vectors,
    ExternalListUpdateStats* stats) {
  const size_t expected_bytes = checked_multiply(
      quantizer_.dimension(), sizeof(float), "float32 payload size overflows");
  if (options_.payload_bytes != expected_bytes ||
      (options_.payload_format != ExactPayloadFormat::kOpaque &&
       options_.payload_format != ExactPayloadFormat::kFloat32)) {
    throw std::invalid_argument(
        "float32 convenience insertion disagrees with exact payload layout");
  }
  return insert_batch(count, vectors, vectors, expected_bytes, stats);
}

std::vector<uint32_t> ExternalListRecastIndex::insert_batch(
    size_t count,
    const float* vectors,
    const void* exact_payloads,
    size_t payload_stride,
    ExternalListUpdateStats* stats) {
  const auto total_begin = Clock::now();
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (count == 0) return {};
  if (!quantizer_.is_trained() || vectors == nullptr || exact_payloads == nullptr) {
    throw std::invalid_argument("invalid external-list insertion state or input");
  }
  if (payload_stride == 0) payload_stride = options_.payload_bytes;
  if (payload_stride < options_.payload_bytes) {
    throw std::invalid_argument("exact payload stride is too small");
  }
  const uint64_t id_limit = std::numeric_limits<uint32_t>::max();
  if (next_id_ > id_limit ||
      static_cast<uint64_t>(count) > id_limit - next_id_) {
    throw std::overflow_error("prototype exhausts its 32-bit Logical ID space");
  }
  const uint64_t new_next_id = next_id_ + count;
  const size_t new_active_count = checked_add(
      active_count_, count, "active external-list vector count overflows");

  const size_t code_size = quantizer_.code_size();
  const size_t code_bytes = checked_multiply(
      count, code_size, "inserted PQ4 code size overflows");
  const auto routing_begin = Clock::now();
  const std::vector<uint32_t> assignments = router_.assign_pq(count, vectors);
  if (stats != nullptr) {
    stats->routing_nanoseconds += elapsed_nanoseconds(routing_begin);
  }
  const auto encoding_begin = Clock::now();
  std::vector<uint8_t> input_codes(code_bytes);
  std::vector<Metadata> input_metadata(count);
  quantizer_.encode(
      count, vectors, input_codes.data(), input_metadata.data());
  if (stats != nullptr) {
    stats->encoding_nanoseconds += elapsed_nanoseconds(encoding_begin);
  }
  const auto grouping_begin = Clock::now();
  const ListBatchPlan plan = group_by_list(assignments, lists_.size());

  std::vector<uint32_t> generated_ids(count);
  for (size_t i = 0; i < count; ++i) {
    generated_ids[i] = static_cast<uint32_t>(next_id_ + i);
  }
  std::vector<uint32_t> grouped_ids(count);
  std::vector<uint8_t> grouped_codes(code_bytes);
  std::vector<Metadata> grouped_metadata(count);
  std::vector<uint8_t> grouped_payloads(checked_multiply(
      count, options_.payload_bytes, "grouped exact payload size overflows"));
  const auto* payload_bytes = static_cast<const uint8_t*>(exact_payloads);
  parallel_for_static(count, 4096, [&](size_t grouped_index) {
    const size_t input = plan.order[grouped_index];
    grouped_ids[grouped_index] = generated_ids[input];
    grouped_metadata[grouped_index] = input_metadata[input];
    std::memcpy(
        grouped_codes.data() + grouped_index * code_size,
        input_codes.data() + input * code_size, code_size);
    std::memcpy(
        grouped_payloads.data() + grouped_index * options_.payload_bytes,
        payload_bytes + input * payload_stride, options_.payload_bytes);
  });
  if (stats != nullptr) {
    stats->grouping_nanoseconds += elapsed_nanoseconds(grouping_begin);
  }

  // Reserve and initialize every packed destination before the first external
  // append. Logical sizes remain unchanged, so concurrent readers (excluded by
  // mutex_) could never observe the newly allocated empty tail.
  parallel_for_dynamic(plan.touched_lists.size(), 8, [&](size_t index) {
    const uint32_t list_no = plan.touched_lists[index];
    MemoryList& list = lists_[list_no];
    const size_t append_count =
        plan.offsets[static_cast<size_t>(list_no) + 1] - plan.offsets[list_no];
    const size_t new_size = checked_add(
        list.logical_size, append_count, "external IVF list size overflows");
    if (new_size > std::numeric_limits<uint32_t>::max()) {
      throw std::overflow_error("external IVF list exceeds uint32 offsets");
    }
    const size_t new_padded = round_up(new_size, detail::kUpdateBlockSize);
    const size_t code_capacity = checked_multiply(
        new_padded, code_size, "packed external list code size overflows");
    const size_t old_code_capacity = list.packed_codes.size();
    list.packed_codes.resize(code_capacity);
    if (code_capacity > old_code_capacity) {
      std::memset(
          list.packed_codes.get() + old_code_capacity, 0,
          code_capacity - old_code_capacity);
    }
    const size_t metadata_capacity = packed_metadata_size_bbs32(new_padded);
    const size_t old_metadata_capacity = list.packed_metadata.size();
    list.packed_metadata.resize(metadata_capacity);
    if (metadata_capacity > old_metadata_capacity) {
      std::memset(
          list.packed_metadata.get() + old_metadata_capacity, 255,
          metadata_capacity - old_metadata_capacity);
    }
    detail::validate_empty_packed_code_range_bbs32(
        quantizer_.subquantizers(), list.logical_size, new_size,
        list.packed_codes.get(), list.packed_codes.size());
    detail::validate_empty_packed_metadata_range_bbs32(
        list.logical_size, new_size, list.packed_metadata.get(),
        list.packed_metadata.size());
  });

  // Build every fallible in-memory commit descriptor before changing either
  // external file. The actual commit path below then only performs file I/O and
  // fixed-capacity packed writes.
  std::vector<std::pair<uint32_t, DiskLocation>> directory_updates;
  directory_updates.reserve(count);
  uint64_t record_pages_read = 0;
  uint64_t record_pages_written = 0;
  for (uint32_t list_no : plan.touched_lists) {
    const uint32_t old_size = lists_[list_no].logical_size;
    const size_t begin = plan.offsets[list_no];
    const size_t end = plan.offsets[static_cast<size_t>(list_no) + 1];
    for (size_t i = begin; i < end; ++i) {
      directory_updates.emplace_back(
          grouped_ids[i],
          DiskLocation{list_no, old_size + static_cast<uint32_t>(i - begin)});
    }
    const uint64_t old_pages =
        page_count(old_size, record_store_.records_per_page());
    const uint64_t new_pages = page_count(
        old_size + end - begin, record_store_.records_per_page());
    const uint64_t rewrites_partial =
        old_size != 0 && old_size % record_store_.records_per_page() != 0;
    record_pages_read += rewrites_partial;
    record_pages_written += new_pages - old_pages + rewrites_partial;
  }
  const uint64_t first_directory_page =
      next_id_ / directory_.entries_per_page();
  const uint64_t last_directory_page =
      (new_next_id - 1) / directory_.entries_per_page();
  const uint64_t directory_pages_written =
      last_directory_page - first_directory_page + 1;
  const uint64_t old_directory_pages =
      page_count(next_id_, directory_.entries_per_page());
  const uint64_t directory_pages_read =
      first_directory_page < old_directory_pages
      ? std::min(last_directory_page + 1, old_directory_pages) -
          first_directory_page
      : 0;

  // Different list files are independent, so one batch appends each touched
  // list exactly once and can use the caller's OpenMP team without repacking.
  const auto record_begin = Clock::now();
  parallel_for_dynamic(plan.touched_lists.size(), 1, [&](size_t index) {
    const uint32_t list_no = plan.touched_lists[index];
    const size_t begin = plan.offsets[list_no];
    const size_t end = plan.offsets[static_cast<size_t>(list_no) + 1];
    record_store_.append_records(
        list_no, lists_[list_no].logical_size, grouped_ids.data() + begin,
        grouped_payloads.data() + begin * options_.payload_bytes,
        options_.payload_bytes, end - begin);
  });
  if (stats != nullptr) {
    stats->record_io_nanoseconds += elapsed_nanoseconds(record_begin);
    stats->opened_record_files += plan.touched_lists.size();
    stats->record_pages_read += record_pages_read;
    stats->record_pages_written += record_pages_written;
  }

  const auto directory_begin = Clock::now();
  directory_.write_batch(std::move(directory_updates));
  if (stats != nullptr) {
    stats->directory_io_nanoseconds += elapsed_nanoseconds(directory_begin);
    stats->directory_pages_read += directory_pages_read;
    stats->directory_pages_written += directory_pages_written;
  }

  // All capacity and empty-lane checks were completed before external writes.
  // These fixed-size packed writes therefore introduce no per-vector repack.
  const auto posting_begin = Clock::now();
  parallel_for_dynamic(plan.touched_lists.size(), 8, [&](size_t index) {
    const uint32_t list_no = plan.touched_lists[index];
    MemoryList& list = lists_[list_no];
    const size_t begin = plan.offsets[list_no];
    const size_t end = plan.offsets[static_cast<size_t>(list_no) + 1];
    const size_t append_count = end - begin;
    const size_t old_size = list.logical_size;
    const size_t new_size = old_size + append_count;
    detail::append_flat_codes_bbs32(
        grouped_codes.data() + begin * code_size, append_count * code_size,
        quantizer_.subquantizers(), old_size, new_size,
        list.packed_codes.get(), list.packed_codes.size());
    detail::write_metadata_range_bbs32(
        grouped_metadata.data() + begin, append_count, old_size, new_size,
        list.packed_metadata.get(), list.packed_metadata.size());
    list.logical_size = static_cast<uint32_t>(new_size);
    list.padded_size = static_cast<uint32_t>(
        round_up(new_size, detail::kUpdateBlockSize));
  });

  next_id_ = new_next_id;
  active_count_ = new_active_count;
  if (stats != nullptr) {
    stats->posting_update_nanoseconds += elapsed_nanoseconds(posting_begin);
    stats->total_nanoseconds += elapsed_nanoseconds(total_begin);
  }
  return generated_ids;
}

size_t ExternalListRecastIndex::erase_batch(
    size_t count,
    const uint32_t* logical_ids,
    ExternalListUpdateStats* stats) {
  const auto total_begin = Clock::now();
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (count == 0) return 0;
  if (logical_ids == nullptr || count > active_count_) {
    throw std::invalid_argument("invalid external-list deletion batch");
  }
  std::vector<uint32_t> ids(logical_ids, logical_ids + count);
  std::unordered_set<uint32_t> unique(ids.begin(), ids.end());
  if (unique.size() != count) {
    throw std::invalid_argument("deletion batch contains duplicate Logical IDs");
  }
  const auto initial_directory_begin = Clock::now();
  const std::vector<DiskLocation> locations = directory_.read_batch(ids);
  if (stats != nullptr) {
    stats->directory_io_nanoseconds +=
        elapsed_nanoseconds(initial_directory_begin);
    stats->directory_pages_read +=
        unique_directory_pages(ids, directory_.entries_per_page());
  }
  const auto grouping_begin = Clock::now();
  std::vector<uint32_t> assignments(count);
  for (size_t i = 0; i < count; ++i) {
    if (!locations[i].active() || locations[i].list_no >= lists_.size() ||
        locations[i].offset >= lists_[locations[i].list_no].logical_size) {
      throw std::invalid_argument("deletion references an inactive location");
    }
    assignments[i] = locations[i].list_no;
  }
  const ListBatchPlan plan = group_by_list(assignments, lists_.size());
  const size_t M = quantizer_.subquantizers();
  const size_t code_size = quantizer_.code_size();
  const size_t packed_block_bytes = detail::packed_code_block_bytes(M);
  constexpr size_t kMetadataBlockBytes =
      detail::kUpdateBlockSize * sizeof(Metadata);
  std::vector<PreparedDeletion> prepared(plan.touched_lists.size());

  parallel_for_dynamic(plan.touched_lists.size(), 8, [&](size_t index) {
    const uint32_t list_no = plan.touched_lists[index];
    const MemoryList& list = lists_[list_no];
    const size_t begin = plan.offsets[list_no];
    const size_t end = plan.offsets[static_cast<size_t>(list_no) + 1];
    const size_t delete_count = end - begin;
    if (delete_count > list.logical_size) {
      throw std::invalid_argument("too many rows deleted from one list");
    }
    PreparedDeletion mutation;
    mutation.list_no = list_no;
    mutation.old_size = list.logical_size;
    mutation.new_size = list.logical_size - static_cast<uint32_t>(delete_count);
    mutation.new_padded_size = static_cast<uint32_t>(
        round_up(mutation.new_size, detail::kUpdateBlockSize));
    mutation.deleted_ids.reserve(delete_count);
    mutation.deleted_offsets.reserve(delete_count);
    std::vector<std::pair<uint32_t, uint32_t>> deleted_rows;
    deleted_rows.reserve(delete_count);
    for (size_t i = begin; i < end; ++i) {
      const size_t input = plan.order[i];
      mutation.deleted_ids.push_back(ids[input]);
      deleted_rows.emplace_back(locations[input].offset, ids[input]);
    }
    std::sort(deleted_rows.begin(), deleted_rows.end());
    for (size_t i = 0; i < deleted_rows.size(); ++i) {
      if (i != 0 && deleted_rows[i - 1].first == deleted_rows[i].first) {
        throw std::invalid_argument("two Logical IDs resolve to one list slot");
      }
      mutation.deleted_offsets.push_back(deleted_rows[i].first);
      mutation.expected_deleted_ids.push_back(deleted_rows[i].second);
    }
    for (uint32_t offset : mutation.deleted_offsets) {
      if (offset < mutation.new_size) mutation.holes.push_back(offset);
    }
    for (uint32_t offset = mutation.old_size; offset-- > mutation.new_size;) {
      if (!std::binary_search(
              mutation.deleted_offsets.begin(), mutation.deleted_offsets.end(),
              offset)) {
        mutation.sources.push_back(offset);
      }
    }
    if (mutation.holes.size() != mutation.sources.size()) {
      throw std::logic_error("invalid external swap-with-last move plan");
    }
    mutation.record_moves.reserve(mutation.holes.size());
    for (size_t i = 0; i < mutation.holes.size(); ++i) {
      mutation.record_moves.push_back(
          RecordMove{mutation.holes[i], mutation.sources[i]});
    }
    prepared[index] = std::move(mutation);
  });
  const uint64_t grouping_plan_nanoseconds = elapsed_nanoseconds(grouping_begin);

  const auto record_prepare_begin = Clock::now();
  parallel_for_dynamic(prepared.size(), 1, [&](size_t index) {
    PreparedDeletion& mutation = prepared[index];
    std::vector<uint32_t> observed_offsets = mutation.deleted_offsets;
    observed_offsets.insert(
        observed_offsets.end(), mutation.sources.begin(), mutation.sources.end());
    mutation.record_mutation = record_store_.prepare_moves_and_truncate(
        mutation.list_no, mutation.old_size, mutation.new_size,
        mutation.record_moves,
        observed_offsets);
    const std::vector<uint32_t>& observed_ids =
        mutation.record_mutation.observed_logical_ids();
    for (size_t i = 0; i < mutation.deleted_offsets.size(); ++i) {
      if (observed_ids[i] != mutation.expected_deleted_ids[i]) {
        throw std::logic_error(
            "external record Logical ID disagrees with its directory entry");
      }
    }
    mutation.moves.reserve(mutation.holes.size());
    for (size_t i = 0; i < mutation.holes.size(); ++i) {
      mutation.moves.push_back(PreparedMove{
          mutation.holes[i], mutation.sources[i],
          observed_ids[mutation.deleted_offsets.size() + i]});
    }
  });
  if (stats != nullptr) {
    stats->record_io_nanoseconds += elapsed_nanoseconds(record_prepare_begin);
    for (const PreparedDeletion& mutation : prepared) {
      stats->opened_record_files += 1;
      stats->record_pages_read += mutation.record_mutation.pages_read();
    }
  }

  const auto packed_prepare_begin = Clock::now();
  parallel_for_dynamic(prepared.size(), 8, [&](size_t index) {
    PreparedDeletion& mutation = prepared[index];
    const MemoryList& list = lists_[mutation.list_no];
    std::vector<size_t> affected_blocks;
    for (uint32_t hole : mutation.holes) {
      affected_blocks.push_back(hole / detail::kUpdateBlockSize);
    }
    if (mutation.new_size < mutation.old_size) {
      const size_t first = mutation.new_size / detail::kUpdateBlockSize;
      const size_t last = (mutation.old_size - 1) / detail::kUpdateBlockSize;
      for (size_t block = first; block <= last; ++block) {
        affected_blocks.push_back(block);
      }
    }
    std::sort(affected_blocks.begin(), affected_blocks.end());
    affected_blocks.erase(
        std::unique(affected_blocks.begin(), affected_blocks.end()),
        affected_blocks.end());
    std::unordered_map<size_t, size_t> block_to_scratch;
    for (size_t block_index : affected_blocks) {
      const size_t code_offset = checked_multiply(
          block_index, packed_block_bytes,
          "packed deletion code offset overflows");
      const size_t metadata_offset = checked_multiply(
          block_index, kMetadataBlockBytes,
          "packed deletion metadata offset overflows");
      if (code_offset > list.packed_codes.size() ||
          packed_block_bytes > list.packed_codes.size() - code_offset ||
          metadata_offset > list.packed_metadata.size() ||
          kMetadataBlockBytes > list.packed_metadata.size() - metadata_offset) {
        throw std::logic_error("packed list is shorter than its logical size");
      }
      BlockScratch scratch;
      scratch.block_index = block_index;
      scratch.original_codes.resize(packed_block_bytes);
      detail::unpack_flat_block_bbs32(
          list.packed_codes.get() + code_offset, packed_block_bytes, M,
          scratch.original_codes.data(), scratch.original_codes.size());
      scratch.updated_codes = scratch.original_codes;
      detail::unpack_metadata_block_bbs32(
          list.packed_metadata.get() + metadata_offset, kMetadataBlockBytes,
          scratch.original_metadata.data(), scratch.original_metadata.size());
      scratch.updated_metadata = scratch.original_metadata;
      block_to_scratch.emplace(block_index, mutation.blocks.size());
      mutation.blocks.push_back(std::move(scratch));
    }

    for (PreparedMove& move : mutation.moves) {
      BlockScratch& destination =
          mutation.blocks[block_to_scratch.at(
              move.destination / detail::kUpdateBlockSize)];
      const BlockScratch& source =
          mutation.blocks[block_to_scratch.at(
              move.source / detail::kUpdateBlockSize)];
      const size_t destination_lane =
          move.destination % detail::kUpdateBlockSize;
      const size_t source_lane = move.source % detail::kUpdateBlockSize;
      std::memcpy(
          destination.updated_codes.data() + destination_lane * code_size,
          source.original_codes.data() + source_lane * code_size, code_size);
      destination.updated_metadata[destination_lane] =
          source.original_metadata[source_lane];
    }
    for (uint32_t offset = mutation.new_size; offset < mutation.old_size;
         ++offset) {
      BlockScratch& block = mutation.blocks[block_to_scratch.at(
          offset / detail::kUpdateBlockSize)];
      const size_t lane = offset % detail::kUpdateBlockSize;
      std::memset(block.updated_codes.data() + lane * code_size, 0, code_size);
      block.updated_metadata[lane] = Metadata{255, 255, 255};
    }
  });
  if (stats != nullptr) {
    stats->grouping_nanoseconds +=
        grouping_plan_nanoseconds + elapsed_nanoseconds(packed_prepare_begin);
  }

  // Validate that every moved external ID still points to its snapshotted
  // source before any list file is changed.
  size_t total_move_count = 0;
  for (const PreparedDeletion& mutation : prepared) {
    total_move_count = checked_add(
        total_move_count, mutation.moves.size(), "deletion move count overflows");
  }
  std::vector<uint32_t> moved_ids;
  moved_ids.reserve(total_move_count);
  for (const PreparedDeletion& mutation : prepared) {
    for (const PreparedMove& move : mutation.moves) moved_ids.push_back(move.source_id);
  }
  const auto moved_directory_begin = Clock::now();
  const std::vector<DiskLocation> moved_locations = directory_.read_batch(moved_ids);
  if (stats != nullptr && !moved_ids.empty()) {
    stats->directory_io_nanoseconds += elapsed_nanoseconds(moved_directory_begin);
    stats->directory_pages_read += unique_directory_pages(
        moved_ids, directory_.entries_per_page());
  }
  size_t moved_index = 0;
  for (const PreparedDeletion& mutation : prepared) {
    for (const PreparedMove& move : mutation.moves) {
      const DiskLocation location = moved_locations[moved_index++];
      if (!location.active() || location.list_no != mutation.list_no ||
          location.offset != move.source) {
        throw std::logic_error("external directory disagrees with a move source");
      }
    }
  }

  // Precompute every descriptor and diagnostic count before the first record
  // page is rewritten. This avoids post-commit vector growth on the normal path.
  uint64_t record_commit_pages_written = 0;
  for (const PreparedDeletion& mutation : prepared) {
    record_commit_pages_written += mutation.record_mutation.pages_written();
  }

  std::vector<std::pair<uint32_t, DiskLocation>> directory_updates;
  const size_t directory_update_count = checked_add(
      count, moved_ids.size(), "deletion directory update count overflows");
  directory_updates.reserve(directory_update_count);
  std::vector<uint32_t> directory_update_ids;
  if (stats != nullptr) directory_update_ids.reserve(directory_update_count);
  for (const PreparedDeletion& mutation : prepared) {
    for (uint32_t id : mutation.deleted_ids) {
      directory_updates.emplace_back(id, DiskLocation::tombstone());
      if (stats != nullptr) directory_update_ids.push_back(id);
    }
    for (const PreparedMove& move : mutation.moves) {
      directory_updates.emplace_back(
          move.source_id, DiskLocation{mutation.list_no, move.destination});
      if (stats != nullptr) directory_update_ids.push_back(move.source_id);
    }
  }
  const uint64_t directory_commit_pages = stats != nullptr
      ? unique_directory_pages(
            directory_update_ids, directory_.entries_per_page())
      : 0;

  const auto record_commit_begin = Clock::now();
  parallel_for_dynamic(prepared.size(), 1, [&](size_t index) {
    const PreparedDeletion& mutation = prepared[index];
    record_store_.commit_prepared_mutation(mutation.record_mutation);
  });
  if (stats != nullptr) {
    stats->record_io_nanoseconds += elapsed_nanoseconds(record_commit_begin);
    stats->opened_record_files += prepared.size();
    stats->record_pages_written += record_commit_pages_written;
  }

  const auto directory_commit_begin = Clock::now();
  directory_.write_batch(std::move(directory_updates));
  if (stats != nullptr) {
    stats->directory_io_nanoseconds +=
        elapsed_nanoseconds(directory_commit_begin);
    stats->directory_pages_read += directory_commit_pages;
    stats->directory_pages_written += directory_commit_pages;
  }

  const auto posting_begin = Clock::now();
  parallel_for_dynamic(prepared.size(), 8, [&](size_t index) {
    const PreparedDeletion& mutation = prepared[index];
    MemoryList& list = lists_[mutation.list_no];
    for (const BlockScratch& block : mutation.blocks) {
      const size_t code_offset = block.block_index * packed_block_bytes;
      const size_t metadata_offset = block.block_index * kMetadataBlockBytes;
      detail::rewrite_flat_block_bbs32(
          block.updated_codes.data(), block.updated_codes.size(), M,
          list.packed_codes.get() + code_offset, packed_block_bytes);
      detail::rewrite_metadata_block_bbs32(
          block.updated_metadata.data(), block.updated_metadata.size(),
          list.packed_metadata.get() + metadata_offset, kMetadataBlockBytes);
    }
    list.logical_size = mutation.new_size;
    list.padded_size = mutation.new_padded_size;
  });
  active_count_ -= count;
  if (stats != nullptr) {
    stats->posting_update_nanoseconds += elapsed_nanoseconds(posting_begin);
    stats->total_nanoseconds += elapsed_nanoseconds(total_begin);
  }
  return count;
}

ExternalListRecastIndex::LocatedSelection
ExternalListRecastIndex::select_candidates_locked(
    const float* query,
    size_t topk,
    size_t nprobe,
    float z_score,
    size_t max_candidates,
    ScanStats* stats,
    const ScanOptions& options) const {
  if (query == nullptr || topk == 0 || active_count_ < topk) {
    throw std::invalid_argument("invalid external-list search request");
  }
  const std::vector<uint32_t> selected = router_.route_pq(query, nprobe);
  struct TokenSpan {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t list_no = 0;
  };
  std::vector<TokenSpan> spans;
  std::vector<CodeBlockView> blocks;
  spans.reserve(selected.size());
  blocks.reserve(selected.size());
  uint64_t token_base = 0;
  for (uint32_t list_no : selected) {
    const MemoryList& list = lists_[list_no];
    if (list.logical_size == 0) continue;
    if (token_base + list.logical_size >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::overflow_error("query-local candidate token space overflows");
    }
    const uint32_t begin = static_cast<uint32_t>(token_base);
    token_base += list.logical_size;
    const uint32_t end = static_cast<uint32_t>(token_base);
    spans.push_back(TokenSpan{begin, end, list_no});
    blocks.push_back(CodeBlockView{
        list.packed_codes.get(), list.packed_metadata.get(), nullptr,
        list.logical_size, list.padded_size, detail::kUpdateBlockSize, begin});
  }
  if (token_base < topk) {
    throw std::invalid_argument("probed IVF lists contain fewer than topk rows");
  }
  RecastFastScanner scanner(quantizer_);
  const QueryContext context = quantizer_.prepare_query(query);
  const SelectionResult selected_candidates = scanner.scan_interval_candidates(
      context, blocks, topk, z_score, max_candidates, stats, options);
  if (selected_candidates.candidates.size() < topk) {
    throw std::logic_error("interval selection returned fewer than topk rows");
  }

  LocatedSelection output;
  output.probed_lists = spans.size();
  output.candidates.reserve(selected_candidates.candidates.size());
  std::vector<uint32_t> candidate_lists;
  candidate_lists.reserve(selected_candidates.candidates.size());
  for (const DistanceEstimate& candidate : selected_candidates.candidates) {
    const auto span = std::lower_bound(
        spans.begin(), spans.end(), candidate.id,
        [](const TokenSpan& lhs, uint32_t token) { return lhs.end <= token; });
    if (span == spans.end() || candidate.id < span->begin) {
      throw std::logic_error("FastScan returned an invalid query-local token");
    }
    output.candidates.push_back(ListCandidate{
        span->list_no, candidate.id - span->begin,
        candidate.estimated_distance, candidate.uncertainty});
    candidate_lists.push_back(span->list_no);
  }
  std::sort(candidate_lists.begin(), candidate_lists.end());
  candidate_lists.erase(
      std::unique(candidate_lists.begin(), candidate_lists.end()),
      candidate_lists.end());
  output.candidate_lists = candidate_lists.size();
  return output;
}

std::vector<Neighbor> ExternalListRecastIndex::search(
    const float* query,
    size_t topk,
    size_t nprobe,
    float z_score,
    size_t max_candidates,
    ExactDistanceFunction distance_function,
    ExternalListSearchStats* stats,
    const ScanOptions& scan_options) const {
  const auto total_begin = Clock::now();
  std::shared_lock<std::shared_mutex> lock(mutex_);
  ScanStats* scan_stats = stats == nullptr ? nullptr : &stats->scan;
  const auto selection_begin = Clock::now();
  LocatedSelection selection = select_candidates_locked(
      query, topk, nprobe, z_score, max_candidates, scan_stats, scan_options);
  if (stats != nullptr) {
    stats->probed_lists += selection.probed_lists;
    stats->candidate_lists += selection.candidate_lists;
    stats->selection_nanoseconds += elapsed_nanoseconds(selection_begin);
  }
  ListFileAioRefiner refiner(
      record_store_, ListFileRefineOptions{
          options_.queue_depth, options_.max_open_files_per_query});
  const auto refinement_begin = Clock::now();
  std::vector<Neighbor> result = refiner.refine(
      query, quantizer_.dimension(), selection.candidates, topk,
      distance_function, stats == nullptr ? nullptr : &stats->refine);
  if (stats != nullptr) {
    stats->refinement_nanoseconds += elapsed_nanoseconds(refinement_begin);
    stats->total_nanoseconds += elapsed_nanoseconds(total_begin);
  }
  return result;
}

size_t ExternalListRecastIndex::size() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return active_count_;
}

uint64_t ExternalListRecastIndex::next_id() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return next_id_;
}

size_t ExternalListRecastIndex::list_size(uint32_t list_no) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (list_no >= lists_.size()) {
    throw std::out_of_range("IVF list number is invalid");
  }
  return lists_[list_no].logical_size;
}

}  // namespace recastlib::adapters
