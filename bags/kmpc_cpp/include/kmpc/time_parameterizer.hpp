#pragma once
// Monotone scalar time law s(t) for an already-certified joint curve.
// Velocity / accel / jerk caps come from the MotionConstraintPolicy.
// When device a/j limits are not calibrated the table is marked
// software_limited and carries the policy source strings.

#include <vector>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

class TimeParameterizer {
 public:
  // samples: certified geometric curve. policy: dq/ddq/dddq caps.
  // stage_boundaries: stop_required entries force zero rate at that sample.
  // Result: s monotone nondecreasing in [0,1], t_s monotone nondecreasing.
  TimeLawTable compute(
      const JointCurve& samples, const MotionConstraintPolicy& policy,
      const std::vector<StageBoundary>& stage_boundaries) const;
};

}  // namespace kmpc
