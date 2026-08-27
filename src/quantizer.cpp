#include "recastlib/quantizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <faiss/VectorTransform.h>
#include <faiss/index_io.h>
#include <faiss/impl/io.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/distances.h>

#include "recastlib/detail/uneven_product_quantizer.h"
#include "persistence_io.h"

namespace recastlib {
namespace {

constexpr float kMinNorm = 1e-12f;
constexpr std::array<char, 8> kModelMagic{
    'R', 'C', 'Q', 'M', 'O', 'D', 'L', '1'};
constexpr uint32_t kModelVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304U;
constexpr uint64_t kMaximumModelBytes = uint64_t{1} << 30;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t checksum(const std::vector<uint8_t>& bytes) {
  uint64_t hash = kFnvOffset;
  for (uint8_t byte : bytes) {
    hash ^= byte;
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
void write_file_scalar(std::ofstream* output, const T& value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "model file scalars must be trivially copyable");
  output->write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!*output) throw std::runtime_error("failed to write quantizer model");
}

template <typename T>
T read_file_scalar(std::ifstream* input) {
  static_assert(std::is_trivially_copyable<T>::value,
                "model file scalars must be trivially copyable");
  T value{};
  input->read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!*input) throw std::runtime_error("truncated quantizer model header");
  return value;
}

GroupingMode decode_grouping(uint32_t value) {
  switch (value) {
    case static_cast<uint32_t>(GroupingMode::Uniform):
      return GroupingMode::Uniform;
    case static_cast<uint32_t>(GroupingMode::Uneven):
      return GroupingMode::Uneven;
    default:
      throw std::runtime_error("invalid persisted grouping mode");
  }
}

TransformMode decode_transform(uint32_t value) {
  switch (value) {
    case static_cast<uint32_t>(TransformMode::None):
      return TransformMode::None;
    case static_cast<uint32_t>(TransformMode::Pca):
      return TransformMode::Pca;
    case static_cast<uint32_t>(TransformMode::ParametricOpq):
      return TransformMode::ParametricOpq;
    default:
      throw std::runtime_error("invalid persisted transform mode");
  }
}

DistanceMetric decode_metric(uint32_t value) {
  switch (value) {
    case static_cast<uint32_t>(DistanceMetric::SquaredL2):
      return DistanceMetric::SquaredL2;
    case static_cast<uint32_t>(DistanceMetric::InnerProduct):
      return DistanceMetric::InnerProduct;
    case static_cast<uint32_t>(DistanceMetric::Cosine):
      return DistanceMetric::Cosine;
    default:
      throw std::runtime_error("invalid persisted distance metric");
  }
}

CenteringMode decode_centering(uint32_t value) {
  switch (value) {
    case static_cast<uint32_t>(CenteringMode::None):
      return CenteringMode::None;
    case static_cast<uint32_t>(CenteringMode::Mean):
      return CenteringMode::Mean;
    default:
      throw std::runtime_error("invalid persisted centering mode");
  }
}

float normalize(float* vector, size_t dimension) {
  const float norm =
      std::sqrt(faiss::fvec_norm_L2sqr(vector, dimension));
  if (norm <= kMinNorm) {
    // Treat numerically zero centered vectors as having no direction.
    std::fill(vector, vector + dimension, 0.0f);
    return 0.0f;
  }
  const float inverse = 1.0f / norm;
  for (size_t j = 0; j < dimension; ++j) vector[j] *= inverse;
  return norm;
}

uint8_t quantize_linear(float value, float maximum, float step, bool* overflow) {
  // Codes 0..254 cover the trained range; 255 is reserved as an overflow
  // sentinel so the scanner can widen the corresponding interval to infinity.
  *overflow = !std::isfinite(value) || value > maximum + 0.5f * step;
  if (*overflow) return 255;
  if (maximum <= 0.0f) return 0;
  long code = std::lround(value / step);
  code = std::max(0L, std::min(254L, code));
  return static_cast<uint8_t>(code);
}

uint8_t quantize_error_upper(float error) {
  // Round upward so decoding never understates the measured direction error.
  // Unlike norm and scale, code 255 is the valid endpoint 2, not a sentinel.
  error = std::max(0.0f, std::min(2.0f, error));
  long code = static_cast<long>(
      std::ceil(error * 255.0f / 2.0f - 1e-7f));
  code = std::max(0L, std::min(255L, code));
  return static_cast<uint8_t>(code);
}

}  // namespace

struct RecastQuantizer::Impl {
  explicit Impl(QuantizerConfig input) : config(input), mean(input.dimension) {
    // PQ4 stores two consecutive 4-bit group indices per byte, hence even M.
    // Uniform Faiss PQ also requires D % M == 0; the uneven backend obtains
    // byte alignment by planning group capacities in identical pairs.
    if (config.dimension == 0 || config.subquantizers == 0 ||
        config.subquantizers > config.dimension ||
        config.subquantizers % 2 != 0) {
      throw std::invalid_argument(
          "dimension and an even M in [2, dimension] are required");
    }
    if (config.grouping == GroupingMode::Uniform &&
        config.dimension % config.subquantizers != 0) {
      throw std::invalid_argument("uniform PQ requires dimension divisible by M");
    }
    if (config.metric == DistanceMetric::Cosine &&
        config.centering != CenteringMode::None) {
      throw std::invalid_argument(
          "cosine distance requires origin-centered encoding");
    }

    if (config.grouping == GroupingMode::Uniform) {
      uniform = std::make_unique<faiss::ProductQuantizer>(
          config.dimension, config.subquantizers, 4);
    } else {
      uneven = std::make_unique<detail::UnevenProductQuantizer>(
          config.dimension, config.subquantizers);
      uneven->report_layout_after_training = false;
    }
  }

  void transform(size_t count, const float* input, float* output) const {
    // Buffers are row-major [count][D]. With no configured rotation this is an
    // identity operation and also permits input and output to alias.
    if (rotation) {
      rotation->apply_noalloc(
          static_cast<faiss::idx_t>(count), input, output);
    } else if (input != output) {
      std::memcpy(
          output, input, count * config.dimension * sizeof(float));
    }
  }

  void inverse_transform(size_t count, const float* input, float* output) const {
    // Training accepts only an orthonormal, bias-free square transform, so the
    // inverse is the transpose R^T used by Faiss here.
    if (rotation) {
      rotation->transform_transpose(
          static_cast<faiss::idx_t>(count), input, output);
    } else if (input != output) {
      std::memcpy(
          output, input, count * config.dimension * sizeof(float));
    }
  }

  void train_direction_quantizer(size_t count, const float* directions) {
    if (uniform) {
      uniform->train(count, directions);
    } else {
      uneven->train(count, directions);
    }
  }

  void compute_codes(size_t count, const float* vectors, uint8_t* codes) const {
    if (uniform) {
      uniform->compute_codes(vectors, codes, count);
    } else {
      uneven->compute_codes(vectors, codes, count);
    }
  }

  void decode_codes(size_t count, const uint8_t* codes, float* vectors) const {
    if (uniform) {
      uniform->decode(codes, vectors, count);
    } else {
      uneven->decode(codes, vectors, count);
    }
  }

  void compute_dot_lut(const float* query, float* lut) const {
    if (uniform) {
      uniform->compute_inner_prod_table(query, lut);
    } else {
      uneven->compute_inner_prod_table(query, lut);
    }
  }

  QuantizerConfig config;
  std::vector<float> mean;
  std::unique_ptr<faiss::LinearTransform> rotation;
  std::unique_ptr<faiss::ProductQuantizer> uniform;
  std::unique_ptr<detail::UnevenProductQuantizer> uneven;
  float norm_max = 0.0f;
  float norm_step = 1.0f;
  float scale_max = 0.0f;
  float scale_step = 1.0f;
  bool trained = false;
};

RecastQuantizer::RecastQuantizer(QuantizerConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

RecastQuantizer::~RecastQuantizer() = default;
RecastQuantizer::RecastQuantizer(RecastQuantizer&&) noexcept = default;
RecastQuantizer& RecastQuantizer::operator=(RecastQuantizer&&) noexcept = default;

void RecastQuantizer::train(size_t count, const float* vectors) {
  if (count == 0 || vectors == nullptr) {
    throw std::invalid_argument("training vectors must not be empty");
  }
  const size_t dimension = impl_->config.dimension;

  // Stage 1: optionally learn the global translation and materialize vectors
  // in the configured encoding coordinate system.
  std::fill(impl_->mean.begin(), impl_->mean.end(), 0.0f);
  if (impl_->config.centering == CenteringMode::Mean) {
    for (size_t i = 0; i < count; ++i) {
      for (size_t j = 0; j < dimension; ++j) {
        impl_->mean[j] += vectors[i * dimension + j];
      }
    }
    const float inverse_count = 1.0f / static_cast<float>(count);
    for (float& value : impl_->mean) value *= inverse_count;
  }

  std::vector<float> centered(count * dimension);
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      centered[i * dimension + j] =
          vectors[i * dimension + j] - impl_->mean[j];
    }
  }

  // Stage 2: split each translated vector into its norm and unit direction.
  // Learn the orthogonal transform from translated vectors before normalization.
  // This matches the parametric-OPQ construction used by an outer Faiss
  // PCAMatrix: PCA captures the original covariance, while the direction
  // quantizer still receives unit vectors.
  std::vector<float> directions = centered;
  std::vector<float> norms(count);
  impl_->norm_max = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    norms[i] = normalize(directions.data() + i * dimension, dimension);
    impl_->norm_max = std::max(impl_->norm_max, norms[i]);
  }

  if (impl_->config.transform == TransformMode::Pca) {
    impl_->rotation = std::make_unique<faiss::PCAMatrix>(
        static_cast<int>(dimension), static_cast<int>(dimension), 0.0f, false);
    impl_->rotation->train(
        static_cast<faiss::idx_t>(count), centered.data());
  } else if (impl_->config.transform == TransformMode::ParametricOpq) {
    // This is PCA with variance-balanced bins, not iterative faiss::OPQMatrix
    // training. Faiss assigns PCA axes to M equal-width groups by eigenvalue.
    auto parametric_opq = std::make_unique<faiss::PCAMatrix>(
        static_cast<int>(dimension), static_cast<int>(dimension), 0.0f, false);
    parametric_opq->balanced_bins =
        static_cast<int>(impl_->config.subquantizers);
    parametric_opq->train(
        static_cast<faiss::idx_t>(count), centered.data());
    impl_->rotation = std::move(parametric_opq);
  }

  if (impl_->rotation && !impl_->rotation->is_orthonormal) {
    throw std::runtime_error("Recast requires an orthonormal PCA transform");
  }
  if (impl_->rotation) {
    // Translation is owned by Recast's optional centering stage. Drop the PCA
    // object's own residual bias so the stored transform is purely orthogonal.
    std::fill(impl_->rotation->b.begin(), impl_->rotation->b.end(), 0.0f);
  }

  // Stage 3: rotate only the unit directions, then train either the ordinary
  // equal-width PQ4 codebook or the variance-aware uneven implementation.
  std::vector<float> transformed(count * dimension);
  impl_->transform(count, directions.data(), transformed.data());
  // R is orthonormal, so nonzero transformed directions remain unit length and
  // both dot products and direction errors are preserved.

  impl_->train_direction_quantizer(count, transformed.data());
  impl_->norm_step =
      impl_->norm_max > 0.0f ? impl_->norm_max / 254.0f : 1.0f;

  // Stage 4: run the trained direction codec over the training set. For L2 and
  // inner product, byte 1 restores translated-vector magnitude. For cosine it
  // normalizes only the raw PQ reconstruction.
  std::vector<uint8_t> codes(count * code_size());
  std::vector<float> reconstructed(count * dimension);
  impl_->compute_codes(count, transformed.data(), codes.data());
  impl_->decode_codes(count, codes.data(), reconstructed.data());

  impl_->scale_max = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const float reconstruction_norm = std::sqrt(faiss::fvec_norm_L2sqr(
        reconstructed.data() + i * dimension, dimension));
    if (reconstruction_norm > kMinNorm &&
        (impl_->config.metric != DistanceMetric::Cosine ||
         norms[i] > kMinNorm)) {
      const float scale = impl_->config.metric == DistanceMetric::Cosine
          ? 1.0f / reconstruction_norm
          : norms[i] / reconstruction_norm;
      impl_->scale_max = std::max(
          impl_->scale_max, scale);
    }
  }
  impl_->scale_step =
      impl_->scale_max > 0.0f ? impl_->scale_max / 254.0f : 1.0f;
  impl_->trained = true;
}

void RecastQuantizer::save(const std::string& path) const {
  using detail::persistence::write_scalar;
  using detail::persistence::write_vector;
  if (!is_trained() || path.empty()) {
    throw std::invalid_argument(
        "a trained quantizer and nonempty model path are required");
  }

  faiss::VectorIOWriter payload;
  write_scalar(&payload, static_cast<uint64_t>(impl_->config.dimension));
  write_scalar(&payload, static_cast<uint64_t>(impl_->config.subquantizers));
  write_scalar(
      &payload, static_cast<uint32_t>(impl_->config.grouping));
  write_scalar(
      &payload, static_cast<uint32_t>(impl_->config.transform));
  write_scalar(&payload, static_cast<uint32_t>(impl_->config.metric));
  write_scalar(
      &payload, static_cast<uint32_t>(impl_->config.centering));
  write_vector(&payload, impl_->mean);

  const uint8_t has_rotation = impl_->rotation != nullptr;
  write_scalar(&payload, has_rotation);
  if (has_rotation != 0) {
    faiss::write_VectorTransform(impl_->rotation.get(), &payload);
  }

  const uint8_t backend =
      impl_->config.grouping == GroupingMode::Uniform ? 0 : 1;
  write_scalar(&payload, backend);
  if (backend == 0) {
    faiss::write_ProductQuantizer(impl_->uniform.get(), &payload);
  } else {
    impl_->uneven->write(&payload);
  }
  write_scalar(&payload, impl_->norm_max);
  write_scalar(&payload, impl_->norm_step);
  write_scalar(&payload, impl_->scale_max);
  write_scalar(&payload, impl_->scale_step);

  const uint64_t payload_bytes = payload.data.size();
  const uint64_t payload_checksum = checksum(payload.data);
  const std::filesystem::path destination(path);
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  const std::filesystem::path temporary = path + ".tmp." + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot create quantizer model: " + path);
    }
    output.write(kModelMagic.data(), kModelMagic.size());
    write_file_scalar(&output, kModelVersion);
    write_file_scalar(&output, kEndianMarker);
    write_file_scalar(&output, payload_bytes);
    write_file_scalar(&output, payload_checksum);
    if (!payload.data.empty()) {
      output.write(
          reinterpret_cast<const char*>(payload.data.data()),
          static_cast<std::streamsize>(payload.data.size()));
    }
    output.flush();
    if (!output) throw std::runtime_error("quantizer model write failed");
    output.close();
    std::filesystem::rename(temporary, destination);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

RecastQuantizer RecastQuantizer::load(const std::string& path) {
  using detail::persistence::checked_size;
  using detail::persistence::read_scalar;
  using detail::persistence::read_vector;
  if (path.empty()) {
    throw std::invalid_argument("quantizer model path must not be empty");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open quantizer model: " + path);

  std::array<char, kModelMagic.size()> magic{};
  input.read(magic.data(), magic.size());
  if (!input || magic != kModelMagic) {
    throw std::runtime_error("invalid Recast quantizer model magic");
  }
  if (read_file_scalar<uint32_t>(&input) != kModelVersion) {
    throw std::runtime_error("unsupported Recast quantizer model version");
  }
  if (read_file_scalar<uint32_t>(&input) != kEndianMarker) {
    throw std::runtime_error("Recast quantizer model has incompatible endianness");
  }
  const uint64_t persisted_bytes = read_file_scalar<uint64_t>(&input);
  const uint64_t persisted_checksum = read_file_scalar<uint64_t>(&input);
  if (persisted_bytes > kMaximumModelBytes ||
      persisted_bytes > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Recast quantizer model payload is too large");
  }
  std::vector<uint8_t> payload(static_cast<size_t>(persisted_bytes));
  if (!payload.empty()) {
    input.read(
        reinterpret_cast<char*>(payload.data()),
        static_cast<std::streamsize>(payload.size()));
  }
  if (!input || input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("Recast quantizer model length is inconsistent");
  }
  if (checksum(payload) != persisted_checksum) {
    throw std::runtime_error("Recast quantizer model checksum mismatch");
  }

  faiss::VectorIOReader reader;
  reader.data = std::move(payload);
  const size_t dimension = checked_size(
      read_scalar<uint64_t>(&reader), 1U << 20, "quantizer dimension");
  const size_t subquantizers = checked_size(
      read_scalar<uint64_t>(&reader), dimension, "quantizer subquantizers");
  QuantizerConfig config{
      dimension,
      subquantizers,
      decode_grouping(read_scalar<uint32_t>(&reader)),
      decode_transform(read_scalar<uint32_t>(&reader)),
      decode_metric(read_scalar<uint32_t>(&reader)),
      decode_centering(read_scalar<uint32_t>(&reader))};
  RecastQuantizer result(config);

  result.impl_->mean = read_vector<float>(
      &reader, dimension, "quantizer mean");
  if (result.impl_->mean.size() != dimension) {
    throw std::runtime_error("persisted quantizer mean has wrong dimension");
  }

  const bool has_rotation = read_scalar<uint8_t>(&reader) != 0;
  if (has_rotation != (config.transform != TransformMode::None)) {
    throw std::runtime_error("persisted transform presence is inconsistent");
  }
  if (has_rotation) {
    std::unique_ptr<faiss::VectorTransform> transform(
        faiss::read_VectorTransform(&reader));
    auto* linear = dynamic_cast<faiss::LinearTransform*>(transform.get());
    if (linear == nullptr || !linear->is_trained ||
        linear->d_in != static_cast<int>(dimension) ||
        linear->d_out != static_cast<int>(dimension) ||
        linear->A.size() != dimension * dimension ||
        !linear->is_orthonormal) {
      throw std::runtime_error("persisted Recast transform is incompatible");
    }
    transform.release();
    result.impl_->rotation.reset(linear);
  }

  const uint8_t backend = read_scalar<uint8_t>(&reader);
  if (backend != (config.grouping == GroupingMode::Uniform ? 0 : 1)) {
    throw std::runtime_error("persisted direction-quantizer backend is inconsistent");
  }
  if (backend == 0) {
    std::unique_ptr<faiss::ProductQuantizer> pq(
        faiss::read_ProductQuantizer(&reader));
    const size_t expected_centroids = dimension * 16;
    if (pq == nullptr || static_cast<size_t>(pq->d) != dimension ||
        static_cast<size_t>(pq->M) != subquantizers || pq->nbits != 4 ||
        static_cast<size_t>(pq->ksub) != 16 ||
        static_cast<size_t>(pq->code_size) != subquantizers / 2 ||
        pq->centroids.size() != expected_centroids) {
      throw std::runtime_error("persisted uniform PQ4 codebook is incompatible");
    }
    result.impl_->uniform = std::move(pq);
  } else {
    std::unique_ptr<detail::UnevenProductQuantizer> uneven =
        detail::UnevenProductQuantizer::read(&reader);
    if (uneven->d != dimension || uneven->M != subquantizers ||
        uneven->nbits != 4) {
      throw std::runtime_error("persisted uneven PQ4 codebook is incompatible");
    }
    result.impl_->uneven = std::move(uneven);
  }

  result.impl_->norm_max = read_scalar<float>(&reader);
  result.impl_->norm_step = read_scalar<float>(&reader);
  result.impl_->scale_max = read_scalar<float>(&reader);
  result.impl_->scale_step = read_scalar<float>(&reader);
  if (!std::isfinite(result.impl_->norm_max) || result.impl_->norm_max < 0.0f ||
      !std::isfinite(result.impl_->norm_step) || result.impl_->norm_step <= 0.0f ||
      !std::isfinite(result.impl_->scale_max) || result.impl_->scale_max < 0.0f ||
      !std::isfinite(result.impl_->scale_step) || result.impl_->scale_step <= 0.0f) {
    throw std::runtime_error("persisted metadata ranges are invalid");
  }
  if (reader.rp != reader.data.size()) {
    throw std::runtime_error("Recast quantizer model contains trailing state");
  }
  result.impl_->trained = true;
  return result;
}

void RecastQuantizer::encode(size_t count,
                             const float* vectors,
                             uint8_t* codes,
                             Metadata* metadata) const {
  if (!is_trained() || (count > 0 &&
      (vectors == nullptr || codes == nullptr || metadata == nullptr))) {
    throw std::invalid_argument("invalid encode input or untrained quantizer");
  }
  if (count == 0) return;

  const size_t dimension = impl_->config.dimension;

  // Reproduce the training pipeline: optionally translate, retain magnitude,
  // normalize to a direction, and apply the learned orthogonal transform.
  std::vector<float> directions(count * dimension);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      directions[i * dimension + j] =
          vectors[i * dimension + j] - impl_->mean[j];
    }
  }

  std::vector<float> norms(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
  for (size_t i = 0; i < count; ++i) {
    norms[i] = normalize(directions.data() + i * dimension, dimension);
  }
  if (impl_->rotation) {
    std::vector<float> transformed(count * dimension);
    impl_->transform(count, directions.data(), transformed.data());
    directions.swap(transformed);
  }

  // Encode flat PQ4 codes first, then decode them once to derive the two pieces
  // of side information that depend on the actual reconstructed direction.
  impl_->compute_codes(count, directions.data(), codes);
  std::vector<float> reconstructed(count * dimension);
  impl_->decode_codes(count, codes, reconstructed.data());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
  for (size_t i = 0; i < count; ++i) {
    bool norm_overflow = false;
    bool scale_overflow = false;
    const uint8_t norm_code = quantize_linear(
        norms[i], impl_->norm_max, impl_->norm_step, &norm_overflow);

    float* reconstruction = reconstructed.data() + i * dimension;
    const float reconstruction_norm = normalize(reconstruction, dimension);
    uint8_t scale_code = 255;
    uint8_t error_code = 255;
    if (reconstruction_norm > kMinNorm &&
        (impl_->config.metric != DistanceMetric::Cosine ||
         norms[i] > kMinNorm)) {
      // L2/IP scale restores translated-vector magnitude. Cosine scale only
      // normalizes the raw PQ reconstruction; source magnitude cancels.
      const float scale = impl_->config.metric == DistanceMetric::Cosine
          ? 1.0f / reconstruction_norm
          : norms[i] / reconstruction_norm;
      scale_code = quantize_linear(
          scale,
          impl_->scale_max,
          impl_->scale_step,
          &scale_overflow);
      const float error = std::sqrt(faiss::fvec_L2sqr(
          directions.data() + i * dimension, reconstruction, dimension));
      error_code = quantize_error_upper(error);
    } else {
      scale_overflow = true;
    }
    // quantize_linear has already encoded either overflow as sentinel 255; no
    // separate flag is needed in the three-byte record.
    (void)norm_overflow;
    (void)scale_overflow;
    metadata[i] = Metadata{norm_code, scale_code, error_code};
  }
}

void RecastQuantizer::decode(size_t count,
                             const uint8_t* codes,
                             const Metadata* metadata,
                             float* vectors) const {
  if (!is_trained() || (count > 0 &&
      (codes == nullptr || metadata == nullptr || vectors == nullptr))) {
    throw std::invalid_argument("invalid decode input or untrained quantizer");
  }
  if (count == 0) return;

  const size_t dimension = impl_->config.dimension;

  // Decode the transformed direction and restore its trained magnitude scale.
  // Cosine stores inverse reconstruction norm in scale, so source norm is also
  // needed to reconstruct the original vector magnitude.
  std::vector<float> transformed(count * dimension);
  impl_->decode_codes(count, codes, transformed.data());
  for (size_t i = 0; i < count; ++i) {
    float norm = 0.0f;
    float scale = 0.0f;
    float error = 0.0f;
    bool overflow = false;
    decode_metadata(metadata[i], &norm, &scale, &error, &overflow);
    (void)error;
    (void)overflow;
    const float reconstruction_scale =
        impl_->config.metric == DistanceMetric::Cosine
        ? norm * scale
        : scale;
    for (size_t j = 0; j < dimension; ++j) {
      transformed[i * dimension + j] *= reconstruction_scale;
    }
  }

  // An orthonormal transform is inverted by its transpose. Adding the global
  // mean then returns each reconstruction to the original coordinate system.
  std::vector<float> centered(count * dimension);
  impl_->inverse_transform(count, transformed.data(), centered.data());
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      vectors[i * dimension + j] = centered[i * dimension + j] + impl_->mean[j];
    }
  }
}

QueryContext RecastQuantizer::prepare_query(const float* query) const {
  if (!is_trained() || query == nullptr) {
    throw std::invalid_argument("invalid query or untrained quantizer");
  }
  QueryContext context;
  const size_t dimension = impl_->config.dimension;
  context.centered_query.resize(dimension);
  if (impl_->config.metric == DistanceMetric::SquaredL2) {
    // L2 is translation invariant only when the same mean is removed from the
    // query and every database vector.
    for (size_t j = 0; j < dimension; ++j) {
      context.centered_query[j] = query[j] - impl_->mean[j];
    }
  } else {
    // Inner product needs q itself: q.x = q.mean + q.(x - mean). Cosine uses
    // no translation and normalizes q before building its dot-product LUT.
    std::memcpy(
        context.centered_query.data(), query, dimension * sizeof(float));
  }
  if (impl_->config.metric == DistanceMetric::Cosine) {
    if (normalize(context.centered_query.data(), dimension) <= kMinNorm) {
      throw std::invalid_argument("cosine query must have nonzero norm");
    }
    context.query_norm = 1.0f;
    context.query_norm_squared = 1.0f;
  } else {
    context.query_norm_squared = faiss::fvec_norm_L2sqr(
        context.centered_query.data(), dimension);
    context.query_norm = std::sqrt(context.query_norm_squared);
  }
  if (impl_->config.metric == DistanceMetric::InnerProduct &&
      impl_->config.centering == CenteringMode::Mean) {
    context.mean_dot_query = faiss::fvec_inner_product(
        query, impl_->mean.data(), dimension);
  }

  // The metric-specific query is now ready: translated for L2, unchanged for
  // inner product, and unit-normalized for cosine.
  context.transformed_query.resize(dimension);
  impl_->transform(
      1, context.centered_query.data(), context.transformed_query.data());
  context.dot_lut.resize(impl_->config.subquantizers * 16);
  impl_->compute_dot_lut(
      context.transformed_query.data(), context.dot_lut.data());
  return context;
}

size_t RecastQuantizer::dimension() const noexcept {
  return impl_->config.dimension;
}

QuantizerConfig RecastQuantizer::configuration() const noexcept {
  return impl_->config;
}

size_t RecastQuantizer::subquantizers() const noexcept {
  return impl_->config.subquantizers;
}

size_t RecastQuantizer::code_size() const noexcept {
  return impl_->config.subquantizers / 2;
}

bool RecastQuantizer::is_trained() const noexcept { return impl_->trained; }

bool RecastQuantizer::uses_uneven_grouping() const noexcept {
  return impl_->config.grouping == GroupingMode::Uneven;
}

DistanceMetric RecastQuantizer::distance_metric() const noexcept {
  return impl_->config.metric;
}

CenteringMode RecastQuantizer::centering_mode() const noexcept {
  return impl_->config.centering;
}

const std::vector<float>& RecastQuantizer::mean() const {
  return impl_->mean;
}

void RecastQuantizer::compute_dot_lut(
    const float* transformed_query, float* lut) const {
  if (!is_trained() || transformed_query == nullptr || lut == nullptr) {
    throw std::invalid_argument("invalid LUT input or untrained quantizer");
  }
  impl_->compute_dot_lut(transformed_query, lut);
}

MetadataDecodeParameters
RecastQuantizer::metadata_decode_parameters() const noexcept {
  return MetadataDecodeParameters{
      impl_->norm_max,
      impl_->norm_step,
      impl_->scale_max,
      impl_->scale_step};
}

void RecastQuantizer::decode_metadata(
    const Metadata& metadata,
    float* norm,
    float* scale,
    float* direction_error,
    bool* overflow) const {
  if (norm == nullptr || scale == nullptr || direction_error == nullptr ||
      overflow == nullptr) {
    throw std::invalid_argument("metadata outputs must not be null");
  }
  // Norm and scale reserve 255 as overflow. Substitute their learned maxima to
  // keep a usable numerical estimate, then tell the scanner to use an infinite
  // uncertainty. Error 255 decodes normally to the valid endpoint 2.
  *norm = metadata.norm == 255
      ? impl_->norm_max
      : static_cast<float>(metadata.norm) * impl_->norm_step;
  *scale = metadata.scale == 255
      ? impl_->scale_max
      : static_cast<float>(metadata.scale) * impl_->scale_step;
  *direction_error = static_cast<float>(metadata.error) * (2.0f / 255.0f);
  // Cosine does not use source norm during scanning, so only scale overflow
  // invalidates its estimate. L2 and inner product require both fields.
  *overflow = metadata.scale == 255 ||
      (impl_->config.metric != DistanceMetric::Cosine && metadata.norm == 255);
}

}  // namespace recastlib
