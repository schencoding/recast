#ifndef RECASTLIB_INTERVAL_SELECTION_INTERNAL_H
#define RECASTLIB_INTERVAL_SELECTION_INTERNAL_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "recastlib/types.h"

namespace recastlib::detail {

struct FuzzyRankThreshold {
  float estimate = 0.0f;
  VectorId id = 0;
  size_t output_size = 0;
  size_t equal_to_keep = 0;
};

inline float fuzzy_median3(float a, float b, float c) {
  if (a > b) std::swap(a, b);
  if (c > b) return b;
  return c > a ? c : a;
}

inline float sample_fuzzy_threshold(const float* values,
                                    size_t count,
                                    float lower,
                                    float upper) {
  constexpr size_t kBigPrime = 6700417;
  std::array<float, 3> sample{};
  size_t sample_size = 0;
  for (size_t i = 0; i < count; ++i) {
    const float value = values[(i * kBigPrime) % count];
    if (value > lower && value < upper) {
      sample[sample_size++] = value;
      if (sample_size == sample.size()) break;
    }
  }
  if (sample_size == sample.size()) {
    return fuzzy_median3(sample[0], sample[1], sample[2]);
  }
  return sample_size == 0 ? lower : sample[0];
}

inline void count_fuzzy_threshold(const float* values,
                                  size_t count,
                                  float threshold,
                                  size_t* less,
                                  size_t* equal) {
  *less = 0;
  *equal = 0;
  for (size_t i = 0; i < count; ++i) {
    *less += values[i] < threshold;
    *equal += values[i] == threshold;
  }
}

/** Finds an exact rank prefix whose size is in [q_min, q_max]. */
inline FuzzyRankThreshold find_fuzzy_rank_threshold(
    const float* estimates,
    const VectorId* ids,
    size_t count,
    size_t q_min,
    size_t q_max,
    std::vector<float>* value_scratch,
    std::vector<VectorId>* id_scratch) {
  if (estimates == nullptr || ids == nullptr || value_scratch == nullptr ||
      id_scratch == nullptr || q_min == 0 || q_min > q_max || q_max >= count) {
    throw std::invalid_argument("invalid fuzzy rank partition input");
  }

  float threshold = estimates[0];
  size_t less = 0;
  size_t equal = 0;
  size_t output = 0;
  bool found = false;

  if (count >= 3) {
    float lower = -std::numeric_limits<float>::infinity();
    float upper = std::numeric_limits<float>::infinity();
    threshold = fuzzy_median3(
        estimates[0], estimates[count / 2], estimates[count - 1]);
    for (size_t iteration = 0; iteration < 200; ++iteration) {
      count_fuzzy_threshold(estimates, count, threshold, &less, &equal);
      if (less <= q_min) {
        if (less + equal >= q_min) {
          output = q_min;
          found = true;
          break;
        }
        lower = threshold;
      } else if (less <= q_max) {
        output = less;
        found = true;
        break;
      } else {
        upper = threshold;
      }

      const float next =
          sample_fuzzy_threshold(estimates, count, lower, upper);
      if (next == lower) break;
      threshold = next;
    }
  }

  if (!found) {
    value_scratch->assign(estimates, estimates + count);
    auto middle = value_scratch->begin() +
        static_cast<std::ptrdiff_t>(q_min - 1);
    std::nth_element(value_scratch->begin(), middle, value_scratch->end());
    threshold = *middle;
    count_fuzzy_threshold(estimates, count, threshold, &less, &equal);
    output = q_min;
  }

  if (less > output || output - less > equal) {
    throw std::logic_error("fuzzy rank partition produced an invalid rank");
  }

  const size_t equal_to_keep = output - less;
  VectorId threshold_id = 0;
  if (equal_to_keep > 0) {
    id_scratch->clear();
    id_scratch->reserve(equal);
    for (size_t i = 0; i < count; ++i) {
      if (estimates[i] == threshold) id_scratch->push_back(ids[i]);
    }
    auto middle = id_scratch->begin() +
        static_cast<std::ptrdiff_t>(equal_to_keep - 1);
    std::nth_element(id_scratch->begin(), middle, id_scratch->end());
    threshold_id = *middle;
  }
  return FuzzyRankThreshold{
      threshold, threshold_id, output, equal_to_keep};
}

/** One shortlist record with its already computed lower endpoint. */
struct IntervalCandidateRecord {
  DistanceEstimate estimate;
  float lower_bound = 0.0f;
};

/**
 * Streaming implementation of Recast's estimate-shortlist policy.
 *
 * Every scanned estimate contributes its upper endpoint to the global top-K
 * certificate. With a positive cap, only the globally best estimates are kept
 * in a Faiss-style bounded reservoir; interval filtering happens in finish().
 */
class IntervalSelectionAccumulator {
 public:
  IntervalSelectionAccumulator(
      size_t topk, float z_score, size_t max_candidates);

  /** Adds one scan-stage distance estimate. */
  void add(const DistanceEstimate& estimate);

  /**
   * Adds one already postprocessed FastScan tile.
   *
   * The four float arrays are contiguous 32-lane scratch planes. IDs may be
   * explicit or derived from implicit_id_base. Upper-bound admission is first
   * reduced with a SIMD mask; only possible top-K entrants touch the scalar
   * global heap. Every estimate still participates in the bounded shortlist.
   */
  void add_block(const VectorId* ids,
                 VectorId implicit_id_base,
                 const float* estimated_distances,
                 const float* uncertainties,
                 const float* lower_bounds,
                 const float* upper_bounds,
                 size_t count);

  /** Finalizes the global threshold and returns the refinement candidates. */
  SelectionResult finish();

 private:
  static bool estimate_less(const IntervalCandidateRecord& lhs,
                            const IntervalCandidateRecord& rhs);

  void update_upper(float upper);

  void add_shortlist(const IntervalCandidateRecord& record);

  void shrink_shortlist(size_t keep);

  size_t topk_;
  float z_score_;
  size_t max_candidates_;
  size_t reservoir_capacity_ = 0;
  size_t seen_ = 0;
  std::priority_queue<float> smallest_uppers_;
  std::vector<IntervalCandidateRecord> shortlist_;
  IntervalCandidateRecord admission_threshold_;
  bool has_admission_threshold_ = false;
};

}  // namespace recastlib::detail

#endif  // RECASTLIB_INTERVAL_SELECTION_INTERNAL_H
