#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#include <unistd.h>

#include "external_list_ivf.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDimension = 128;
constexpr size_t kNlist = 1024;
constexpr size_t kNprobe = 64;
constexpr size_t kSubquantizers = 50;
constexpr size_t kTopK = 10;
constexpr size_t kMaxCandidates = 2000;
constexpr float kZScore = 1.0f;
constexpr int kThreads = 8;

struct Arguments {
  std::string learn;
  std::string base;
  std::string query;
  std::string groundtruth;
  bool help = false;
};

enum class VectorFormat {
  Float32,
  UInt8,
};

struct Vectors {
  size_t rows = 0;
  size_t dimension = 0;
  VectorFormat format = VectorFormat::Float32;
  std::vector<float> values;
  std::vector<uint8_t> byte_values;

  const void* exact_payload() const {
    return format == VectorFormat::UInt8
        ? static_cast<const void*>(byte_values.data())
        : static_cast<const void*>(values.data());
  }

  size_t payload_bytes() const {
    return dimension *
        (format == VectorFormat::UInt8 ? sizeof(uint8_t) : sizeof(float));
  }
};

struct GroundTruth {
  size_t rows = 0;
  size_t stored_k = 0;
  std::vector<uint32_t> ids;
};

class TemporaryIndexDirectory {
 public:
  TemporaryIndexDirectory()
      : path_(std::filesystem::temp_directory_path() /
              ("recastlib_sift1m_" + std::to_string(::getpid()))) {
    std::filesystem::remove_all(path_);
  }

  ~TemporaryIndexDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void print_usage(const char* executable) {
  std::cout
      << "Usage:\n  " << executable
      << " --learn <sift_learn.{fvecs,bvecs}>"
         " --base <sift_base.{fvecs,bvecs}>"
         " --query <sift_query.{fvecs,bvecs}>"
         " --groundtruth <sift_groundtruth.ivecs>\n\n"
      << "The integration test always uses 8 threads, top-10, nlist=1024, "
         "nprobe=64, M=50, z=1, and max_candidates=2000.\n";
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      arguments.help = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value after " + option);
    }
    const std::string value = argv[++i];
    if (option == "--learn") {
      arguments.learn = value;
    } else if (option == "--base") {
      arguments.base = value;
    } else if (option == "--query") {
      arguments.query = value;
    } else if (option == "--groundtruth") {
      arguments.groundtruth = value;
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (!arguments.help &&
      (arguments.learn.empty() || arguments.base.empty() ||
       arguments.query.empty() || arguments.groundtruth.empty())) {
    throw std::invalid_argument("all four dataset paths are required");
  }
  return arguments;
}

int32_t read_i32(std::istream* input, const std::string& path) {
  int32_t value = 0;
  input->read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!*input) throw std::runtime_error("truncated vector file: " + path);
  return value;
}

Vectors read_bvecs(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open bvecs file: " + path);
  const std::streamoff file_bytes = input.tellg();
  input.seekg(0);
  if (file_bytes < static_cast<std::streamoff>(sizeof(int32_t))) {
    throw std::runtime_error("empty bvecs file: " + path);
  }
  const int32_t dimension = read_i32(&input, path);
  if (dimension <= 0) throw std::runtime_error("invalid bvecs dimension");
  const uint64_t row_bytes = sizeof(int32_t) + static_cast<uint64_t>(dimension);
  if (static_cast<uint64_t>(file_bytes) % row_bytes != 0) {
    throw std::runtime_error("bvecs file has a partial row: " + path);
  }
  const size_t rows = static_cast<size_t>(
      static_cast<uint64_t>(file_bytes) / row_bytes);
  Vectors result;
  result.rows = rows;
  result.dimension = static_cast<size_t>(dimension);
  result.format = VectorFormat::UInt8;
  result.byte_values.resize(rows * result.dimension);
  result.values.resize(rows * result.dimension);
  input.seekg(0);
  for (size_t row = 0; row < rows; ++row) {
    const int32_t row_dimension = read_i32(&input, path);
    if (row_dimension != dimension) {
      throw std::runtime_error("bvecs rows have inconsistent dimensions");
    }
    input.read(
        reinterpret_cast<char*>(
            result.byte_values.data() + row * result.dimension),
        static_cast<std::streamsize>(result.dimension));
    if (!input) throw std::runtime_error("truncated bvecs payload: " + path);
  }
#pragma omp parallel for schedule(static) num_threads(kThreads)
  for (int64_t i = 0;
       i < static_cast<int64_t>(result.byte_values.size()); ++i) {
    result.values[static_cast<size_t>(i)] =
        static_cast<float>(result.byte_values[static_cast<size_t>(i)]);
  }
  return result;
}

Vectors read_fvecs(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open fvecs file: " + path);
  const std::streamoff file_bytes = input.tellg();
  input.seekg(0);
  if (file_bytes < static_cast<std::streamoff>(sizeof(int32_t))) {
    throw std::runtime_error("empty fvecs file: " + path);
  }
  const int32_t dimension = read_i32(&input, path);
  if (dimension <= 0) throw std::runtime_error("invalid fvecs dimension");
  const uint64_t row_bytes = sizeof(int32_t) +
      static_cast<uint64_t>(dimension) * sizeof(float);
  if (static_cast<uint64_t>(file_bytes) % row_bytes != 0) {
    throw std::runtime_error("fvecs file has a partial row: " + path);
  }
  Vectors result;
  result.rows = static_cast<size_t>(
      static_cast<uint64_t>(file_bytes) / row_bytes);
  result.dimension = static_cast<size_t>(dimension);
  result.format = VectorFormat::Float32;
  result.values.resize(result.rows * result.dimension);
  input.seekg(0);
  for (size_t row = 0; row < result.rows; ++row) {
    const int32_t row_dimension = read_i32(&input, path);
    if (row_dimension != dimension) {
      throw std::runtime_error("fvecs rows have inconsistent dimensions");
    }
    input.read(
        reinterpret_cast<char*>(result.values.data() + row * result.dimension),
        static_cast<std::streamsize>(result.dimension * sizeof(float)));
    if (!input) throw std::runtime_error("truncated fvecs payload: " + path);
  }
  return result;
}

Vectors read_vectors(const std::string& path) {
  const std::filesystem::path file(path);
  if (file.extension() == ".bvecs") return read_bvecs(path);
  if (file.extension() == ".fvecs") return read_fvecs(path);
  throw std::invalid_argument(
      "vector path must end in .fvecs or .bvecs: " + path);
}

GroundTruth read_ivecs_topk(const std::string& path, size_t topk) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open ivecs file: " + path);
  GroundTruth result;
  while (input.peek() != std::char_traits<char>::eof()) {
    const int32_t row_k = read_i32(&input, path);
    if (row_k < static_cast<int32_t>(topk)) {
      throw std::runtime_error("ground-truth row contains fewer than top-k IDs");
    }
    if (result.rows == 0) result.stored_k = static_cast<size_t>(row_k);
    result.ids.resize((result.rows + 1) * topk);
    for (int32_t rank = 0; rank < row_k; ++rank) {
      const int32_t id = read_i32(&input, path);
      if (id < 0) throw std::runtime_error("ground truth contains a negative ID");
      if (rank < static_cast<int32_t>(topk)) {
        result.ids[result.rows * topk + static_cast<size_t>(rank)] =
            static_cast<uint32_t>(id);
      }
    }
    ++result.rows;
  }
  return result;
}

float squared_l2_uint8(
    const float* query, const void* payload, size_t dimension) {
  const auto* vector = static_cast<const uint8_t*>(payload);
  float distance = 0.0f;
  for (size_t d = 0; d < dimension; ++d) {
    const float delta = query[d] - static_cast<float>(vector[d]);
    distance += delta * delta;
  }
  return distance;
}

float squared_l2_float32(
    const float* query, const void* payload, size_t dimension) {
  const auto* bytes = static_cast<const uint8_t*>(payload);
  float distance = 0.0f;
  for (size_t d = 0; d < dimension; ++d) {
    // External records begin with a uint32 Logical ID. Use memcpy rather than
    // imposing an alignment contract on the payload address supplied by the
    // storage adapter.
    float value = 0.0f;
    std::memcpy(&value, bytes + d * sizeof(float), sizeof(float));
    const float delta = query[d] - value;
    distance += delta * delta;
  }
  return distance;
}

size_t recall_hits(
    const std::vector<recastlib::Neighbor>& result,
    const uint32_t* groundtruth) {
  size_t hits = 0;
  for (const recastlib::Neighbor& neighbor : result) {
    for (size_t rank = 0; rank < kTopK; ++rank) {
      if (neighbor.id == groundtruth[rank]) {
        ++hits;
        break;
      }
    }
  }
  return hits;
}

double seconds_since(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

void validate_dataset(
    const Vectors& learn,
    const Vectors& base,
    const Vectors& query,
    const GroundTruth& groundtruth) {
  if (learn.dimension != kDimension || base.dimension != kDimension ||
      query.dimension != kDimension) {
    throw std::runtime_error(
        "this integration test requires 128-D SIFT fvecs or bvecs");
  }
  if (learn.rows < kNlist || base.rows < kTopK || query.rows == 0 ||
      groundtruth.rows < query.rows) {
    throw std::runtime_error("dataset does not contain enough rows");
  }
  for (uint32_t id : groundtruth.ids) {
    if (id >= base.rows) {
      throw std::runtime_error("ground-truth ID is outside the base dataset");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    if (arguments.help) {
      print_usage(argv[0]);
      return 0;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(kThreads);
    int observed_threads = 0;
#pragma omp parallel num_threads(kThreads)
    {
#pragma omp single
      observed_threads = omp_get_num_threads();
    }
    if (observed_threads != kThreads) {
      throw std::runtime_error("OpenMP did not create the required 8 threads");
    }
    std::cout << "Configuration: threads=8, topk=10, nlist=1024, nprobe=64, "
                 "M=50, z=1, max_candidates=2000\n";

    const auto load_begin = Clock::now();
    Vectors learn = read_vectors(arguments.learn);
    Vectors base = read_vectors(arguments.base);
    Vectors query = read_vectors(arguments.query);
    GroundTruth groundtruth =
        read_ivecs_topk(arguments.groundtruth, kTopK);
    validate_dataset(learn, base, query, groundtruth);
    std::cout << "Loaded learn=" << learn.rows << ", base=" << base.rows
              << ", query=" << query.rows << " in "
              << std::fixed << std::setprecision(3)
              << seconds_since(load_begin) << " s\n";

    TemporaryIndexDirectory storage;
    recastlib::adapters::FaissIvfRouter router(kDimension, kNlist);
    recastlib::QuantizerConfig config;
    config.dimension = kDimension;
    config.subquantizers = kSubquantizers;
    config.grouping = recastlib::GroupingMode::Uneven;
    config.transform = recastlib::TransformMode::Pca;
    config.metric = recastlib::DistanceMetric::SquaredL2;
    config.centering = recastlib::CenteringMode::Mean;

    recastlib::adapters::ExternalListIndexOptions index_options;
    index_options.storage_root = storage.path().string();
    index_options.payload_bytes = base.payload_bytes();
    index_options.payload_format = base.format == VectorFormat::UInt8
        ? recastlib::adapters::ExactPayloadFormat::kUInt8
        : recastlib::adapters::ExactPayloadFormat::kFloat32;
    index_options.page_size = 4096;
    index_options.lists_per_shard = 256;
    index_options.queue_depth = 1;
    index_options.max_open_files_per_query = kNprobe;
    // The portable integration test exercises the same page layout and
    // refinement logic without claiming an SSD/O_DIRECT benchmark.
    index_options.direct_io = false;
    index_options.sync_writes = false;
    index_options.create_storage = true;

    recastlib::adapters::ExternalListRecastIndex index(
        router, config, index_options);
    const auto train_begin = Clock::now();
    index.train(learn.rows, learn.values.data());
    const double train_seconds = seconds_since(train_begin);
    std::cout << "Training: " << train_seconds << " s\n";

    const auto add_begin = Clock::now();
    const std::vector<uint32_t> ids = index.insert_batch(
        base.rows, base.values.data(), base.exact_payload(),
        base.payload_bytes());
    const double add_seconds = seconds_since(add_begin);
    if (ids.size() != base.rows || ids.front() != 0 ||
        ids.back() != base.rows - 1 || index.size() != base.rows) {
      throw std::runtime_error("base insertion produced incorrect Logical IDs");
    }
    std::cout << "Indexing: " << add_seconds << " s\n";

    // Exact payloads now reside in the list files, and Recast postings own the
    // compressed representation. Release build-time copies before measuring.
    base.values.clear();
    base.values.shrink_to_fit();
    base.byte_values.clear();
    base.byte_values.shrink_to_fit();
    learn.values.clear();
    learn.values.shrink_to_fit();
    learn.byte_values.clear();
    learn.byte_values.shrink_to_fit();

    const recastlib::adapters::ExactDistanceFunction distance_function =
        base.format == VectorFormat::UInt8
        ? squared_l2_uint8
        : squared_l2_float32;

    const size_t warmup = std::min<size_t>(100, query.rows);
#pragma omp parallel for schedule(dynamic, 1) num_threads(kThreads)
    for (int64_t qi = 0; qi < static_cast<int64_t>(warmup); ++qi) {
      index.search(
          query.values.data() + static_cast<size_t>(qi) * kDimension,
          kTopK, kNprobe, kZScore, kMaxCandidates, distance_function);
    }

    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string first_error;
    uint64_t hits = 0;
    uint64_t scanned = 0;
    uint64_t refined = 0;
    uint64_t pages = 0;
    uint64_t requests = 0;
    uint64_t quic_nanoseconds = 0;
    const auto search_begin = Clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(kThreads) \
    reduction(+ : hits, scanned, refined, pages, requests, quic_nanoseconds)
    for (int64_t qi = 0; qi < static_cast<int64_t>(query.rows); ++qi) {
      if (failed.load(std::memory_order_relaxed)) continue;
      try {
        recastlib::adapters::ExternalListSearchStats stats;
        const std::vector<recastlib::Neighbor> result = index.search(
            query.values.data() + static_cast<size_t>(qi) * kDimension,
            kTopK, kNprobe, kZScore, kMaxCandidates,
            distance_function, &stats);
        hits += recall_hits(
            result, groundtruth.ids.data() + static_cast<size_t>(qi) * kTopK);
        scanned += stats.scan.valid_codes;
        refined += stats.refine.refined_vectors;
        pages += stats.refine.unique_pages;
        requests += stats.refine.read_requests;
        quic_nanoseconds += stats.scan.postprocess_nanoseconds;
      } catch (const std::exception& error) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error.empty()) first_error = error.what();
      }
    }
    const double search_seconds = seconds_since(search_begin);
    if (failed.load()) {
      throw std::runtime_error("parallel query failed: " + first_error);
    }

    const double recall = static_cast<double>(hits) /
        static_cast<double>(query.rows * kTopK);
    const double qps = static_cast<double>(query.rows) / search_seconds;
    const double denominator = static_cast<double>(query.rows);
    std::cout << std::setprecision(6)
              << "Recall@10: " << recall << '\n'
              << std::setprecision(2)
              << "QPS: " << qps << '\n'
              << "Mean scanned/query: " << scanned / denominator << '\n'
              << "Mean refined/query: " << refined / denominator << '\n'
              << "Mean QUIC time/query (us): "
              << quic_nanoseconds / denominator / 1000.0 << '\n'
              << "Mean pages/query: " << pages / denominator << '\n'
              << "Mean read requests/query: " << requests / denominator << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "recastlib_test failed: " << error.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
