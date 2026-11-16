#pragma once
// One hard-box meaning for stream, the final gate, and the new checker.
// q_limit_margin is not a second joint hard wall.

#include <algorithm>
#include <string>

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

struct MotionConstraintPolicy {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_min = Vec18::Zero();
  Vec18 q_max = Vec18::Zero();
  Vec18 dq_max = Vec18::Ones();
  Vec18 ddq_max = Vec18::Ones();
  Vec18 dddq_max = Vec18::Ones();
  double torso_pitch_max_rad = 32.0 * kPi / 180.0;
  // Preference scale only. Never a second hard wall.
  double q_limit_margin_rad = 0.0;
  double yaw_abs_max_rad = 0.0;
  bool yaw_limited = false;
  bool a_max_calibrated = false;
  bool j_max_calibrated = false;
  std::string accel_source = "software_limited_from_step_cap";
  std::string jerk_source = "software_limited";
  std::string collision_coverage = "NOT_AVAILABLE";
  std::string com_model = "xy_work_rectangle";
};

inline MotionConstraintPolicy policy_from_config(const Config& cfg,
                                                   double velocity_scale = 1.0) {
  MotionConstraintPolicy policy;
  policy.q_min = cfg.limits.q_min;
  policy.q_max = cfg.limits.q_max;
  policy.dq_max = cfg.limits.dq_max * std::clamp(velocity_scale, 0.0, 1.0);
  // Conservative stand-in until device a/j are calibrated.
  for (int j = 0; j < kNDof; ++j) {
    const double dq = std::max(policy.dq_max[j], 1e-3);
    policy.ddq_max[j] = 8.0 * dq;
    policy.dddq_max[j] = 40.0 * dq;
  }
  policy.torso_pitch_max_rad = cfg.params.torso_pitch_max_deg * kPi / 180.0;
  policy.q_limit_margin_rad = cfg.params.q_limit_margin;
  policy.a_max_calibrated = false;
  policy.j_max_calibrated = false;
  return policy;
}

}  // namespace kmpc
