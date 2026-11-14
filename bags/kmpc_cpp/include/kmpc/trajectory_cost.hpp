#pragma once
// One posture/motion cost. It has no target position, so a 3 cm drop cannot
// switch it. Weights are not a calibrated human model until marked so.

#include "kmpc/constraint_policy.hpp"
#include "kmpc/trajectory_package.hpp"

namespace kmpc {

struct TrajectoryCostWeights {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Per-joint scale on Δq, typically 1/dq_max. A slow joint costs more per rad.
  Vec18 motion_scale = Vec18::Ones();
  double neutral = 0.02;
  double limit = 1.0;
  bool calibrated = false;
};

struct PathCostBreakdown {
  double motion = 0.0;
  double smooth = 0.0;
  double neutral = 0.0;
  double limit = 0.0;
  double total = 0.0;
  bool limit_outside_hard_box = false;
  bool calibrated = false;
  // Kept so a caller cannot quietly turn the height switch back on.
  bool depends_on_target_height = false;
};

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights);

}  // namespace kmpc
