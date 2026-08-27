#include "recastlib/detail/uneven_product_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <faiss/index_io.h>
#include <faiss/impl/FaissAssert.h>

#include "persistence_io.h"

namespace {

constexpr double kRelativeVarianceFloor = 1e-12;
constexpr double kScoreTolerance = 1e-12;
constexpr uint32_t kUnevenPersistenceVersion = 1;
constexpr size_t kMaximumPersistedDimension = 1U << 20;

bool almost_equal(double a, double b) {
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= 1e-12 * scale;
}

} // anonymous namespace

namespace recastlib::detail {

UnevenProductQuantizer::UnevenProductQuantizer(
    size_t d, size_t M, size_t nbits)
    : d(d), M(M), nbits(nbits), ksub(size_t{1} << nbits),
      code_size(M / 2) {
  // PQ4 stores one nibble per group. Planning capacities in identical pairs
  // makes every equal-width bucket byte-aligned and keeps the payload at M/2.
  FAISS_THROW_IF_NOT_MSG(d > 0, "vector dimension must be positive");
  FAISS_THROW_IF_NOT_MSG(M > 0 && M <= d,
                         "the number of PQ groups must be in [1, d]");
  FAISS_THROW_IF_NOT_MSG(nbits == 4,
                         "UnevenProductQuantizer currently supports PQ4 only");
  FAISS_THROW_IF_NOT_MSG(M % 2 == 0,
                         "an even number of PQ groups is required");
  FAISS_THROW_IF_NOT_MSG(d % 2 == 0,
                         "even-sized dimension buckets require even d");
}

std::vector<double> UnevenProductQuantizer::compute_ideal_bit_budgets(
    size_t n, const float* x) const {
  FAISS_THROW_IF_NOT_MSG(n > 0 && x != nullptr, "training set is empty");

  // x is row-major [n][d]. Population variance is sufficient because only the
  // relative coding demand between transformed coordinates drives the planner.
  std::vector<double> means(d, 0.0);
  std::vector<double> variances(d, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const float* row = x + i * d;
    for (size_t j = 0; j < d; ++j) means[j] += (double)row[j];
  }
  const double inv_n = 1.0 / (double)n;
  for (double& mean : means) mean *= inv_n;

  for (size_t i = 0; i < n; ++i) {
    const float* row = x + i * d;
    for (size_t j = 0; j < d; ++j) {
      const double delta = (double)row[j] - means[j];
      variances[j] += delta * delta;
    }
  }
  for (double& variance : variances) variance *= inv_n;

  const double max_variance =
      *std::max_element(variances.begin(), variances.end());
  FAISS_THROW_IF_NOT_MSG(
      max_variance > 0.0,
      "cannot train uneven PQ on a zero-variance data set");

  // Under the high-rate proxy D_j proportional to sigma_j^2 * 2^(-2*b_j), use
  // h_j = log2(sigma_j) and b_j = max(0, h_j - water_level). Reverse
  // water-filling chooses the level so sum_j b_j = M * nbits; the relative
  // variance floor only prevents log2(0) on constant coordinates.
  const double variance_floor =
      max_variance * kRelativeVarianceFloor;
  std::vector<double> heights(d);
  for (size_t j = 0; j < d; ++j) {
    const double bounded_variance =
        std::max(variance_floor, variances[j]);
    heights[j] = 0.5 * std::log2(bounded_variance);
  }

  std::vector<size_t> order(d);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (heights[a] != heights[b]) return heights[a] > heights[b];
    return a < b;
  });

  double remaining_bits = (double)M * (double)nbits;
  double water_level = heights[order[0]];
  size_t active_dimensions = 1;
  // Lower the common level through descending heights. Reaching the next
  // height activates another coordinate; exhaustion within a gap stops early.
  for (size_t next = 1; next < d; ++next) {
    const double next_height = heights[order[next]];
    const double gap = std::max(0.0, water_level - next_height);
    const double cost = (double)active_dimensions * gap;
    if (remaining_bits + kScoreTolerance >= cost) {
      remaining_bits = std::max(0.0, remaining_bits - cost);
      water_level = next_height;
      ++active_dimensions;
    } else {
      water_level -= remaining_bits / (double)active_dimensions;
      remaining_bits = 0.0;
      break;
    }
  }
  if (remaining_bits > 0.0) {
    water_level -= remaining_bits / (double)d;
  }

  std::vector<double> bit_budgets(d);
  for (size_t j = 0; j < d; ++j) {
    bit_budgets[j] = std::max(0.0, heights[j] - water_level);
  }

  const double assigned_bits =
      std::accumulate(bit_budgets.begin(), bit_budgets.end(), 0.0);
  const double target_bits = (double)M * (double)nbits;
  // Downstream balancing assumes an average ideal load of exactly nbits for
  // each of the M physical subquantizers.
  FAISS_THROW_IF_NOT_MSG(
      std::abs(assigned_bits - target_bits) <=
          1e-8 * std::max(1.0, target_bits),
      "reverse water-filling did not preserve the total bit budget");
  return bit_budgets;
}

UnevenProductQuantizer::Assignment
UnevenProductQuantizer::assign_dimensions(
    const std::vector<size_t>& paired_capacities,
    const std::vector<double>& bit_budgets) const {
  FAISS_ASSERT(paired_capacities.size() == M / 2);
  FAISS_ASSERT(bit_budgets.size() == d);

  // Each entry is the capacity of both groups in one pair. Duplicating and
  // sorting produces M capacities with complete coverage and contiguous runs
  // of equal widths for bucket construction.
  std::vector<size_t> capacities;
  capacities.reserve(M);
  for (size_t capacity : paired_capacities) {
    capacities.push_back(capacity);
    capacities.push_back(capacity);
  }
  std::sort(capacities.begin(), capacities.end());
  FAISS_ASSERT(
      std::accumulate(capacities.begin(), capacities.end(), size_t{0}) == d);

  Assignment result;
  result.dimensions.resize(M);
  result.loads.assign(M, 0.0);
  for (size_t group = 0; group < M; ++group) {
    result.dimensions[group].reserve(capacities[group]);
  }

  std::vector<size_t> dimension_order(d);
  std::iota(dimension_order.begin(), dimension_order.end(), size_t{0});
  std::stable_sort(
      dimension_order.begin(), dimension_order.end(),
      [&](size_t a, size_t b) {
        if (bit_budgets[a] != bit_budgets[b]) {
          return bit_budgets[a] > bit_budgets[b];
        }
        return a < b;
      });

  // Largest-budget-first scheduling places each PCA dimension into the
  // currently lightest non-full group.  Capacity ties prefer the smaller
  // subspace because it has fewer later opportunities to adjust.
  for (size_t dimension : dimension_order) {
    size_t best_group = M;
    for (size_t group = 0; group < M; ++group) {
      if (result.dimensions[group].size() >= capacities[group]) continue;
      if (best_group == M ||
          result.loads[group] < result.loads[best_group] ||
          (almost_equal(result.loads[group], result.loads[best_group]) &&
           capacities[group] < capacities[best_group]) ||
          (almost_equal(result.loads[group], result.loads[best_group]) &&
           capacities[group] == capacities[best_group] &&
           group < best_group)) {
        best_group = group;
      }
    }
    FAISS_ASSERT(best_group < M);
    result.dimensions[best_group].push_back(dimension);
    result.loads[best_group] += bit_budgets[dimension];
  }

  // Evaluate cross-group one-for-one swaps and apply the best strict decrease
  // in sum(load^2). Each swap preserves capacities, coverage, and total load.
  for (size_t pass = 0; pass < 8; ++pass) {
    double best_delta = -kScoreTolerance;
    size_t best_a = M, best_b = M, best_ia = 0, best_ib = 0;
    for (size_t a = 0; a < M; ++a) {
      for (size_t b = a + 1; b < M; ++b) {
        for (size_t ia = 0; ia < result.dimensions[a].size(); ++ia) {
          const double wa =
              bit_budgets[result.dimensions[a][ia]];
          for (size_t ib = 0; ib < result.dimensions[b].size(); ++ib) {
            const double wb =
                bit_budgets[result.dimensions[b][ib]];
            const double new_a = result.loads[a] - wa + wb;
            const double new_b = result.loads[b] - wb + wa;
            const double before = result.loads[a] * result.loads[a] +
                result.loads[b] * result.loads[b];
            const double after = new_a * new_a + new_b * new_b;
            const double delta = after - before;
            if (delta < best_delta) {
              best_delta = delta;
              best_a = a;
              best_b = b;
              best_ia = ia;
              best_ib = ib;
            }
          }
        }
      }
    }
    if (best_a == M) break;

    const size_t dim_a = result.dimensions[best_a][best_ia];
    const size_t dim_b = result.dimensions[best_b][best_ib];
    const double wa = bit_budgets[dim_a];
    const double wb = bit_budgets[dim_b];
    std::swap(result.dimensions[best_a][best_ia],
              result.dimensions[best_b][best_ib]);
    result.loads[best_a] += wb - wa;
    result.loads[best_b] += wa - wb;
  }

  const double target = (double)nbits;
  double squared_error = 0.0;
  for (double load : result.loads) {
    const double error = load - target;
    squared_error += error * error;
  }
  // Every physical group still stores exactly nbits. This dimensionless score
  // measures how unevenly the continuous ideal demand is mapped onto them.
  result.score = squared_error /
      ((double)M * target * target + kRelativeVarianceFloor);
  return result;
}

std::vector<size_t> UnevenProductQuantizer::plan_paired_capacities(
    const std::vector<double>& bit_budgets) const {
  // Work in pair units whose capacities sum to d/2. Duplicating them later
  // recovers M groups covering all d dimensions; the initial profile is nearly
  // uniform before the data-dependent hill climb.
  const size_t pair_count = M / 2;
  const size_t paired_dimensions = d / 2;
  const size_t base_capacity = paired_dimensions / pair_count;
  const size_t larger_pair_count = paired_dimensions % pair_count;
  std::vector<size_t> paired_capacities(pair_count, base_capacity);
  for (size_t i = pair_count - larger_pair_count; i < pair_count; ++i) {
    ++paired_capacities[i];
  }

  const size_t automatic_max = (d + M - 1) / M + 2;
  const size_t maximum = max_subspace_dimension == 0
      ? automatic_max
      : max_subspace_dimension;
  FAISS_THROW_IF_NOT_MSG(maximum >= (d + M - 1) / M,
                         "max_subspace_dimension is too small");

  // A unit transfer shrinks both groups of one pair and grows both groups of
  // another. It preserves total dimensionality, nonempty groups, and even
  // multiplicity of every width. Sorting canonicalizes equivalent identities.
  double current_score =
      assign_dimensions(paired_capacities, bit_budgets).score;
  // Bounded best-improvement hill climbing rescans the complete dimension
  // assignment, including its local swap pass, for every trial profile.
  for (size_t pass = 0; pass < 16; ++pass) {
    double best_score = current_score;
    std::vector<size_t> best_capacities;
    size_t previous_from_capacity = std::numeric_limits<size_t>::max();
    for (size_t from = 0; from < pair_count; ++from) {
      const size_t from_capacity = paired_capacities[from];
      if (from_capacity == previous_from_capacity) continue;
      previous_from_capacity = from_capacity;
      if (from_capacity <= 1) continue;
      size_t previous_to_capacity = std::numeric_limits<size_t>::max();
      for (size_t to = 0; to < pair_count; ++to) {
        const size_t to_capacity = paired_capacities[to];
        if (to_capacity == previous_to_capacity) continue;
        previous_to_capacity = to_capacity;
        if (to_capacity >= maximum) continue;

        size_t actual_to = to;
        if (actual_to == from) {
          // A transfer between two equal-sized pairs is the only way to move
          // from a uniform profile (for example, all 2-D groups) to a 1-D/3-D
          // profile.  Locate the second representative explicitly.
          ++actual_to;
          while (actual_to < pair_count &&
                 paired_capacities[actual_to] != to_capacity) {
            ++actual_to;
          }
          if (actual_to == pair_count) continue;
        }
        std::vector<size_t> trial = paired_capacities;
        --trial[from];
        ++trial[actual_to];
        std::sort(trial.begin(), trial.end());
        if (trial == paired_capacities) continue;
        const double score =
            assign_dimensions(trial, bit_budgets).score;
        if (score + kScoreTolerance < best_score) {
          best_score = score;
          best_capacities = std::move(trial);
        }
      }
    }
    if (best_capacities.empty()) break;
    paired_capacities = std::move(best_capacities);
    current_score = best_score;
  }
  return paired_capacities;
}

void UnevenProductQuantizer::build_buckets(
    const std::vector<std::vector<size_t>>& dimensions) {
  subspace_dimensions_ = dimensions;
  buckets_.clear();

  size_t first = 0;
  // Sorted capacities make equal-width groups contiguous; paired planning
  // guarantees every run contains an even number of groups.
  while (first < M) {
    const size_t subspace_dimension = dimensions[first].size();
    size_t end = first + 1;
    while (end < M && dimensions[end].size() == subspace_dimension) ++end;
    const size_t count = end - first;
    FAISS_THROW_IF_NOT_MSG(
        count % 2 == 0,
        "each subspace-dimension bucket must contain an even group count");

    Bucket bucket;
    // Global group g occupies nibble g, so an even first group and even group
    // count map this bucket to one contiguous whole-byte range.
    bucket.subspace_dimension = subspace_dimension;
    bucket.subspace_count = count;
    bucket.first_subspace = first;
    bucket.code_offset = first / 2;
    bucket.bucket_dimension = subspace_dimension * count;
    bucket.bucket_code_size = count / 2;
    // Flatten in group-major order. Faiss then sees count equal-width,
    // contiguous subspaces even though their source dimensions were scattered.
    bucket.dimensions.reserve(bucket.bucket_dimension);
    for (size_t group = first; group < end; ++group) {
      bucket.dimensions.insert(
          bucket.dimensions.end(), dimensions[group].begin(),
          dimensions[group].end());
    }
    // One Faiss PQ handles one width bucket; its subquantizers retain separate
    // 16-centroid codebooks despite sharing this container object.
    bucket.pq = std::make_unique<faiss::ProductQuantizer>(
        bucket.bucket_dimension, count, nbits);
    buckets_.push_back(std::move(bucket));
    first = end;
  }
  validate_layout();
}

void UnevenProductQuantizer::validate_layout() const {
  // Check contiguous group/byte ranges, a permutation of [0,d), and exactly
  // M/2 bytes without padding or partially occupied bucket boundaries.
  FAISS_THROW_IF_NOT_MSG(subspace_dimensions_.size() == M,
                         "uneven PQ layout has the wrong group count");
  std::vector<uint8_t> seen(d, 0);
  size_t dimension_count = 0;
  size_t byte_count = 0;
  size_t expected_first = 0;
  for (const Bucket& bucket : buckets_) {
    FAISS_THROW_IF_NOT_MSG(bucket.subspace_count % 2 == 0,
                           "bucket group count is not even");
    FAISS_THROW_IF_NOT_MSG(bucket.first_subspace == expected_first,
                           "bucket group offsets are not contiguous");
    FAISS_THROW_IF_NOT_MSG(bucket.code_offset == expected_first / 2,
                           "bucket code offsets are not byte aligned");
    dimension_count += bucket.bucket_dimension;
    byte_count += bucket.bucket_code_size;
    expected_first += bucket.subspace_count;
    for (size_t dimension : bucket.dimensions) {
      FAISS_THROW_IF_NOT_MSG(dimension < d, "dimension index is out of range");
      FAISS_THROW_IF_NOT_MSG(!seen[dimension],
                             "dimension assigned to more than one group");
      seen[dimension] = 1;
    }
  }
  FAISS_THROW_IF_NOT_MSG(expected_first == M, "layout omits PQ groups");
  FAISS_THROW_IF_NOT_MSG(dimension_count == d, "layout omits dimensions");
  FAISS_THROW_IF_NOT_MSG(byte_count == code_size,
                         "uneven PQ code contains padding bytes");
}

void UnevenProductQuantizer::gather_bucket(
    const Bucket& bucket, size_t n, const float* x,
    std::vector<float>& gathered) const {
  // Convert [n][d] into Faiss's contiguous [n][bucket_dimension] layout. The
  // same dimension map is later used in reverse to scatter decoded values.
  gathered.resize(n * bucket.bucket_dimension);
  for (size_t i = 0; i < n; ++i) {
    float* output = gathered.data() + i * bucket.bucket_dimension;
    const float* input = x + i * d;
    for (size_t j = 0; j < bucket.bucket_dimension; ++j) {
      output[j] = input[bucket.dimensions[j]];
    }
  }
}

void UnevenProductQuantizer::train(size_t n, const float* x) {
  // Pipeline: continuous per-coordinate demand -> paired capacities -> balanced
  // dimension assignment -> equal-width buckets -> one Faiss PQ per bucket.
  is_trained = false;
  ideal_bit_budgets_.clear();
  group_bit_loads_.clear();
  const std::vector<double> bit_budgets =
      compute_ideal_bit_budgets(n, x);
  const std::vector<size_t> paired_capacities =
      plan_paired_capacities(bit_budgets);
  Assignment assignment =
      assign_dimensions(paired_capacities, bit_budgets);
  build_buckets(assignment.dimensions);

  std::vector<float> gathered;
  for (Bucket& bucket : buckets_) {
    // A bucket's source coordinates need not be contiguous, so materialize the
    // exact order expected by its ordinary equal-width ProductQuantizer.
    gather_bucket(bucket, n, x, gathered);
    bucket.pq->verbose = verbose;
    bucket.pq->train(n, gathered.data());
  }
  ideal_bit_budgets_ = bit_budgets;
  group_bit_loads_ = assignment.loads;

  FAISS_THROW_IF_NOT_MSG(
      group_bit_loads_.size() == M,
      "uneven PQ diagnostics have the wrong group count");
  const double total_ideal_bits = std::accumulate(
      ideal_bit_budgets_.begin(), ideal_bit_budgets_.end(), 0.0);
  const double total_group_bits = std::accumulate(
      group_bit_loads_.begin(), group_bit_loads_.end(), 0.0);
  FAISS_THROW_IF_NOT_MSG(
      almost_equal(total_group_bits, total_ideal_bits),
      "final group loads do not preserve the ideal bit budget");
  for (size_t group = 0; group < M; ++group) {
    double recomputed_load = 0.0;
    for (size_t dimension : subspace_dimensions_[group]) {
      recomputed_load += ideal_bit_budgets_[dimension];
    }
    FAISS_THROW_IF_NOT_MSG(
        almost_equal(recomputed_load, group_bit_loads_[group]),
        "stored group load does not match its assigned dimensions");
  }
  is_trained = true;

  if (report_layout_after_training) {
    size_t active_dimensions = 0;
    size_t over_budget_dimensions = 0;
    std::printf("UnevenProductQuantizer ideal bit budgets by PCA dimension:\n");
    for (size_t dimension = 0; dimension < d; ++dimension) {
      const double bits = ideal_bit_budgets_[dimension];
      if (bits > kScoreTolerance) ++active_dimensions;
      if (bits > (double)nbits + kScoreTolerance) {
        ++over_budget_dimensions;
      }
      std::printf("  dimension %zu: %.9f bits\n", dimension, bits);
    }
    std::printf(
        "Ideal-bit summary: total=%.9f, target=%.9f, active=%zu, "
        "dimensions_above_%zu_bits=%zu\n",
        total_ideal_bits, (double)M * (double)nbits,
        active_dimensions, nbits, over_budget_dimensions);

    std::printf("UnevenProductQuantizer final group bit loads:\n");
    for (size_t group = 0; group < M; ++group) {
      std::printf(
          "  group %zu: dimensions=%zu, sum_ideal_bits=%.9f\n",
          group, subspace_dimensions_[group].size(),
          group_bit_loads_[group]);
    }

    size_t covered_dimensions = 0;
    std::printf("UnevenProductQuantizer group-dimension distribution:\n");
    for (const Bucket& bucket : buckets_) {
      const size_t bucket_dimensions =
          bucket.subspace_count * bucket.subspace_dimension;
      covered_dimensions += bucket_dimensions;
      std::printf(
          "  %zu dimensions/group: %zu groups (%zu dimensions total)\n",
          bucket.subspace_dimension, bucket.subspace_count,
          bucket_dimensions);
    }
    std::printf(
        "UnevenProductQuantizer summary: M=%zu, covered_dimensions=%zu, "
        "code_size=%zu bytes/vector\n",
        M, covered_dimensions, code_size);
    std::fflush(stdout);
  }

  if (verbose) {
    std::printf("UnevenProductQuantizer load_cv2=%.6g\n", assignment.score);
  }
}

void UnevenProductQuantizer::compute_codes(
    const float* x, uint8_t* codes, size_t n) const {
  FAISS_THROW_IF_NOT_MSG(is_trained, "uneven PQ is not trained");
  FAISS_THROW_IF_NOT_MSG(n == 0 || (x != nullptr && codes != nullptr),
                         "invalid encoding input");
  if (n == 0) return;
  // Output is row-major flat PQ4 [n][M/2]: byte m/2 holds even group m in the
  // low nibble and group m+1 in the high nibble. This is not the bbs=32 layout;
  // callers convert it later with faiss::pq4_pack_codes().
  std::memset(codes, 0, n * code_size);

  std::vector<float> gathered;
  std::vector<uint8_t> bucket_codes;
  for (const Bucket& bucket : buckets_) {
    gather_bucket(bucket, n, x, gathered);
    bucket_codes.resize(n * bucket.bucket_code_size);
    bucket.pq->compute_codes(gathered.data(), bucket_codes.data(), n);
    for (size_t i = 0; i < n; ++i) {
      // Bucket boundaries are whole-byte aligned, so no nibble repacking is
      // required when splicing this range into each vector's global code.
      std::memcpy(
          codes + i * code_size + bucket.code_offset,
          bucket_codes.data() + i * bucket.bucket_code_size,
          bucket.bucket_code_size);
    }
  }
}

void UnevenProductQuantizer::decode(const uint8_t* code, float* x) const {
  decode(code, x, 1);
}

void UnevenProductQuantizer::decode(
    const uint8_t* codes, float* x, size_t n) const {
  FAISS_THROW_IF_NOT_MSG(is_trained, "uneven PQ is not trained");
  FAISS_THROW_IF_NOT_MSG(n == 0 || (codes != nullptr && x != nullptr),
                         "invalid decoding input");
  if (n == 0) return;
  std::fill(x, x + n * d, 0.0f);

  // Reverse the byte-range splice, let Faiss decode contiguous bucket rows,
  // then scatter coordinates back to their transformed-dimension positions.
  std::vector<uint8_t> bucket_codes;
  std::vector<float> reconstructed;
  for (const Bucket& bucket : buckets_) {
    bucket_codes.resize(n * bucket.bucket_code_size);
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(
          bucket_codes.data() + i * bucket.bucket_code_size,
          codes + i * code_size + bucket.code_offset,
          bucket.bucket_code_size);
    }
    reconstructed.resize(n * bucket.bucket_dimension);
    bucket.pq->decode(bucket_codes.data(), reconstructed.data(), n);
    for (size_t i = 0; i < n; ++i) {
      const float* input =
          reconstructed.data() + i * bucket.bucket_dimension;
      float* output = x + i * d;
      for (size_t j = 0; j < bucket.bucket_dimension; ++j) {
        output[bucket.dimensions[j]] = input[j];
      }
    }
  }
}

void UnevenProductQuantizer::compute_inner_prod_table(
    const float* x, float* table) const {
  FAISS_THROW_IF_NOT_MSG(is_trained, "uneven PQ is not trained");
  std::vector<float> gathered;
  // The global layout is [M][ksub]. Contiguous group numbering lets each Faiss
  // bucket write directly at first_subspace * ksub without a second copy.
  for (const Bucket& bucket : buckets_) {
    gather_bucket(bucket, 1, x, gathered);
    bucket.pq->compute_inner_prod_table(
        gathered.data(), table + bucket.first_subspace * ksub);
  }
}

void UnevenProductQuantizer::compute_inner_prod_tables(
    size_t n, const float* x, float* tables) const {
  // Batched lookup tables are contiguous in row-major [n][M][ksub] order.
  for (size_t i = 0; i < n; ++i) {
    compute_inner_prod_table(x + i * d, tables + i * M * ksub);
  }
}

void UnevenProductQuantizer::compute_distance_table(
    const float* x, float* table) const {
  FAISS_THROW_IF_NOT_MSG(is_trained, "uneven PQ is not trained");
  std::vector<float> gathered;
  // Squared-L2 tables use the same [M][ksub] layout as inner-product tables.
  for (const Bucket& bucket : buckets_) {
    gather_bucket(bucket, 1, x, gathered);
    bucket.pq->compute_distance_table(
        gathered.data(), table + bucket.first_subspace * ksub);
  }
}

void UnevenProductQuantizer::compute_distance_tables(
    size_t n, const float* x, float* tables) const {
  for (size_t i = 0; i < n; ++i) {
    compute_distance_table(x + i * d, tables + i * M * ksub);
  }
}

std::vector<UnevenProductQuantizer::BucketInfo>
UnevenProductQuantizer::bucket_info() const {
  std::vector<BucketInfo> result;
  result.reserve(buckets_.size());
  for (const Bucket& bucket : buckets_) {
    result.push_back(BucketInfo{
        bucket.subspace_dimension, bucket.subspace_count,
        bucket.first_subspace, bucket.code_offset});
  }
  return result;
}

void UnevenProductQuantizer::write(faiss::IOWriter* writer) const {
  using persistence::write_scalar;
  using persistence::write_vector;
  if (!is_trained || writer == nullptr) {
    throw std::invalid_argument("cannot persist an untrained uneven PQ");
  }
  validate_layout();

  write_scalar(writer, kUnevenPersistenceVersion);
  write_scalar(writer, static_cast<uint64_t>(d));
  write_scalar(writer, static_cast<uint64_t>(M));
  write_scalar(writer, static_cast<uint64_t>(nbits));
  write_scalar(writer, static_cast<uint64_t>(max_subspace_dimension));
  write_scalar(writer, static_cast<uint8_t>(report_layout_after_training));
  write_scalar(writer, static_cast<uint8_t>(verbose));

  write_scalar(writer, static_cast<uint64_t>(subspace_dimensions_.size()));
  for (const std::vector<size_t>& group : subspace_dimensions_) {
    std::vector<uint64_t> persisted(group.begin(), group.end());
    write_vector(writer, persisted);
  }
  write_vector(writer, ideal_bit_budgets_);
  write_vector(writer, group_bit_loads_);

  write_scalar(writer, static_cast<uint64_t>(buckets_.size()));
  for (const Bucket& bucket : buckets_) {
    write_scalar(writer, static_cast<uint64_t>(bucket.subspace_dimension));
    write_scalar(writer, static_cast<uint64_t>(bucket.subspace_count));
    write_scalar(writer, static_cast<uint64_t>(bucket.first_subspace));
    write_scalar(writer, static_cast<uint64_t>(bucket.code_offset));
    faiss::write_ProductQuantizer(bucket.pq.get(), writer);
  }
}

std::unique_ptr<UnevenProductQuantizer> UnevenProductQuantizer::read(
    faiss::IOReader* reader) {
  using persistence::checked_size;
  using persistence::read_scalar;
  using persistence::read_vector;
  if (reader == nullptr) {
    throw std::invalid_argument("uneven-PQ reader must not be null");
  }
  if (read_scalar<uint32_t>(reader) != kUnevenPersistenceVersion) {
    throw std::runtime_error("unsupported uneven-PQ persistence version");
  }
  const size_t loaded_d = checked_size(
      read_scalar<uint64_t>(reader), kMaximumPersistedDimension, "dimension");
  const size_t loaded_M = checked_size(
      read_scalar<uint64_t>(reader), loaded_d, "subquantizer count");
  const size_t loaded_nbits = checked_size(
      read_scalar<uint64_t>(reader), 64, "bits per subquantizer");
  auto result = std::make_unique<UnevenProductQuantizer>(
      loaded_d, loaded_M, loaded_nbits);
  result->max_subspace_dimension = checked_size(
      read_scalar<uint64_t>(reader), loaded_d, "maximum subspace dimension");
  result->report_layout_after_training = read_scalar<uint8_t>(reader) != 0;
  result->verbose = read_scalar<uint8_t>(reader) != 0;

  const size_t group_count = checked_size(
      read_scalar<uint64_t>(reader), loaded_M, "uneven group count");
  if (group_count != loaded_M) {
    throw std::runtime_error("persisted uneven PQ has the wrong group count");
  }
  std::vector<std::vector<size_t>> dimensions(group_count);
  size_t total_dimensions = 0;
  for (size_t group = 0; group < group_count; ++group) {
    const std::vector<uint64_t> persisted = read_vector<uint64_t>(
        reader, loaded_d, "group dimension list");
    dimensions[group].reserve(persisted.size());
    for (uint64_t dimension : persisted) {
      dimensions[group].push_back(checked_size(
          dimension, loaded_d - 1, "group dimension index"));
    }
    if (dimensions[group].empty() ||
        total_dimensions > loaded_d - dimensions[group].size()) {
      throw std::runtime_error("invalid persisted uneven group dimensions");
    }
    total_dimensions += dimensions[group].size();
  }
  if (total_dimensions != loaded_d) {
    throw std::runtime_error("persisted uneven groups do not cover dimension");
  }

  result->ideal_bit_budgets_ = read_vector<double>(
      reader, loaded_d, "ideal bit budgets");
  result->group_bit_loads_ = read_vector<double>(
      reader, loaded_M, "group bit loads");
  if (result->ideal_bit_budgets_.size() != loaded_d ||
      result->group_bit_loads_.size() != loaded_M) {
    throw std::runtime_error("persisted uneven diagnostics have wrong sizes");
  }

  result->build_buckets(dimensions);
  const size_t bucket_count = checked_size(
      read_scalar<uint64_t>(reader), loaded_M, "uneven bucket count");
  if (bucket_count != result->buckets_.size()) {
    throw std::runtime_error("persisted uneven bucket count is inconsistent");
  }
  for (Bucket& bucket : result->buckets_) {
    const size_t subspace_dimension = checked_size(
        read_scalar<uint64_t>(reader), loaded_d, "bucket subspace dimension");
    const size_t subspace_count = checked_size(
        read_scalar<uint64_t>(reader), loaded_M, "bucket subspace count");
    const size_t first_subspace = checked_size(
        read_scalar<uint64_t>(reader), loaded_M, "bucket first subspace");
    const size_t code_offset = checked_size(
        read_scalar<uint64_t>(reader), result->code_size, "bucket code offset");
    if (subspace_dimension != bucket.subspace_dimension ||
        subspace_count != bucket.subspace_count ||
        first_subspace != bucket.first_subspace ||
        code_offset != bucket.code_offset) {
      throw std::runtime_error("persisted uneven bucket layout is inconsistent");
    }

    std::unique_ptr<faiss::ProductQuantizer> pq(
        faiss::read_ProductQuantizer(reader));
    const size_t expected_centroids = bucket.bucket_dimension * result->ksub;
    if (pq == nullptr || static_cast<size_t>(pq->d) != bucket.bucket_dimension ||
        static_cast<size_t>(pq->M) != bucket.subspace_count ||
        static_cast<size_t>(pq->nbits) != result->nbits ||
        static_cast<size_t>(pq->ksub) != result->ksub ||
        static_cast<size_t>(pq->code_size) != bucket.bucket_code_size ||
        pq->centroids.size() != expected_centroids) {
      throw std::runtime_error("persisted uneven PQ codebook is incompatible");
    }
    bucket.pq = std::move(pq);
  }

  result->validate_layout();
  for (size_t group = 0; group < loaded_M; ++group) {
    double recomputed = 0.0;
    for (size_t dimension : result->subspace_dimensions_[group]) {
      recomputed += result->ideal_bit_budgets_[dimension];
    }
    if (!almost_equal(recomputed, result->group_bit_loads_[group])) {
      throw std::runtime_error("persisted uneven group load is inconsistent");
    }
  }
  result->is_trained = true;
  return result;
}

} // namespace recastlib::detail
