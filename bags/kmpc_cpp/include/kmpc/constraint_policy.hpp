#pragma once
// One hard-box meaning for stream, the final gate, and the new checker.
// q_limit_margin is not a second joint hard wall.

#include <algorithm>

#include "kmpc/config.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

// Same slack the r40 final gate used (1e-7 rad). Not a 0.3° soft wall.
inline constexpr double kHardBoxSlackRad = 1e-7;

inline double joint_hard_margin_rad(
    const Vec18& q, const Vec18& q_min, const Vec18& q_max) {
  return std::min((q - q_min).minCoeff(), (q_max - q).minCoeff());
}

inline bool joint_violates_hard_box(
    const Vec18& q, const Vec18& q_min, const Vec18& q_max) {
  if (!q.allFinite() || !q_min.allFinite() || !q_max.allFinite()) {
    return true;
  }
  return joint_hard_margin_rad(q, q_min, q_max) < -kHardBoxSlackRad;
}

// Shared numbers. q_limit_margin_rad is recorded so a caller can see the old
// planner preference. It is not an argument to joint_violates_hard_box.
struct MotionConstraintPolicy {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_min = Vec18::Zero();
  Vec18 q_max = Vec18::Zero();
  double torso_pitch_max_rad = 32.0 * kPi / 180.0;
  double q_limit_margin_rad = 0.0;
};

inline MotionConstraintPolicy policy_from_config(const Config& cfg) {
  MotionConstraintPolicy policy;
  policy.q_min = cfg.limits.q_min;
  policy.q_max = cfg.limits.q_max;
  policy.torso_pitch_max_rad = cfg.params.torso_pitch_max_deg * kPi / 180.0;
  policy.q_limit_margin_rad = cfg.params.q_limit_margin;
  return policy;
}

}  // namespace kmpc
