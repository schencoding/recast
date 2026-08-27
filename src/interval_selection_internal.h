#ifndef RECASTLIB_INTERVAL_SELECTION_INTERNAL_H
#define RECASTLIB_INTERVAL_SELECTION_INTERNAL_H

#include <cstddef>
#include <queue>
#include <vector>

#include "recastlib/types.h"

namespace recastlib::detail {

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
