#pragma once
// Physical hard box (q_min/q_max) is the send/certify wall.
// q_limit_margin (~0.3°) shrinks the planner work box; it is not a send clip.

#include <algorithm>
#include <cmath>
#include <string>

#include "kmpc/config.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

// Same slack the r40 final gate used (1e-7 rad). Not a 0.3° soft wall.
inline constexpr double kHardBoxSlackRad = 1e-7;
// Task TCP endpoint tolerance (hard): stage endpoints + StageAck.
inline constexpr double kDefaultTaskPositionTolM = 0.005;
// LIN runway: soft lateral band (m). Mid-chord may leave then densify/GN
// pulls back; cert does not fail cross_track inside this band.
inline constexpr double kDefaultLinRunwaySoftM = 0.010;
// Coarse LIN IK / seed may temporarily exceed endpoint tol up to runway width.
inline constexpr double kCoarseLinMidAdmitM = kDefaultLinRunwaySoftM;

inline double lin_cross_track_soft_limit_m(bool stage_endpoint,
                                             double stage_position_tol_m) {
  return stage_endpoint ? stage_position_tol_m : kDefaultLinRunwaySoftM;
}
// Descent tasks: planner + cert share this inset below ±torso_pitch_max (not send clip).
inline constexpr double kPitchDescentWorkMarginRad = 2.0 * kPi / 180.0;
inline constexpr double kMinTcpDropForDescentPitchM = 0.02;
inline constexpr double kChanGainCap = 50.0;

// Unified pitch cap for planning and certification (descent → leave margin, not rail).
inline double pitch_planning_cap_rad(double torso_hard_rad, double start_pitch_abs,
                                     double task_tcp_dz_m) {
  if (!(torso_hard_rad > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  double cap = torso_hard_rad;
  if (task_tcp_dz_m < -kMinTcpDropForDescentPitchM) {
    cap = std::max(start_pitch_abs,
                   torso_hard_rad - kPitchDescentWorkMarginRad);
  }
  return std::max(cap, start_pitch_abs);
}

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

// Regular planner work box: hard ± q_limit_margin on all 18 axes.
inline void joint_work_limits(const Vec18& q_min, const Vec18& q_max,
                              double margin, Vec18& work_min, Vec18& work_max) {
  const double m = margin > 0.0 ? margin : 0.0;
  work_min = q_min;
  work_max = q_max;
  for (int j = 0; j < kNDof; ++j) {
    const double lo = q_min[j] + m;
    const double hi = q_max[j] - m;
    if (hi > lo) {
      work_min[j] = lo;
      work_max[j] = hi;
    }
  }
}

// Start in the 0.3° band may stay (no-growth) and recover inward. Never
// project q0 onto the inner box; never grow further into the margin.
inline void joint_recovery_limits(const Vec18& q0, const Vec18& q_min,
                                  const Vec18& q_max, double margin, Vec18& lo,
                                  Vec18& hi) {
  joint_work_limits(q_min, q_max, margin, lo, hi);
  for (int j = 0; j < kNDof; ++j) {
    if (std::isfinite(q0[j])) {
      if (q0[j] < lo[j] && q0[j] >= q_min[j] - kHardBoxSlackRad) {
        lo[j] = q0[j];
      }
      if (q0[j] > hi[j] && q0[j] <= q_max[j] + kHardBoxSlackRad) {
        hi[j] = q0[j];
      }
    }
    lo[j] = std::max(lo[j], q_min[j]);
    hi[j] = std::min(hi[j], q_max[j]);
    if (lo[j] > hi[j]) {
      const double mid = 0.5 * (q_min[j] + q_max[j]);
      lo[j] = hi[j] =
          std::isfinite(q0[j]) ? std::clamp(q0[j], q_min[j], q_max[j]) : mid;
    }
  }
}

inline double joint_work_margin_rad(const Vec18& q, const Vec18& q0,
                                    const Vec18& q_min, const Vec18& q_max,
                                    double margin) {
  Vec18 lo;
  Vec18 hi;
  joint_recovery_limits(q0, q_min, q_max, margin, lo, hi);
  return joint_hard_margin_rad(q, lo, hi);
}

inline bool joint_violates_work_box(const Vec18& q, const Vec18& q0,
                                    const Vec18& q_min, const Vec18& q_max,
                                    double margin) {
  if (!q.allFinite()) {
    return true;
  }
  return joint_work_margin_rad(q, q0, q_min, q_max, margin) < -kHardBoxSlackRad;
}

// Old line_path chan_gain: extra motion cost only when walking toward a
// work-box wall. Inward exit stays at 1. Iterative reweight (freeze gain).
inline double joint_chan_gain(double q, double lo, double hi, double dq) {
  const double span = hi - lo;
  if (!(span > 1e-4) || !std::isfinite(dq)) {
    return 1.0;
  }
  constexpr double kPad = 1e-4;
  const double qc = std::min(hi - kPad, std::max(lo + kPad, q));
  const double den = (hi - qc) * (qc - lo);
  if (!(den > 1e-8)) {
    return kChanGainCap;
  }
  const double dH = 0.25 * span * span * (2.0 * qc - hi - lo) / (den * den);
  if (!(dH * dq > 0.0)) {
    return 1.0;
  }
  return std::min(1.0 + std::abs(dH), kChanGainCap);
}

struct MotionConstraintPolicy {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_min = Vec18::Zero();
  Vec18 q_max = Vec18::Zero();
  Vec18 dq_max = Vec18::Ones();
  Vec18 ddq_max = Vec18::Ones();
  Vec18 dddq_max = Vec18::Ones();
  double torso_pitch_max_rad = 32.0 * kPi / 180.0;
  // Planner work-box inset (~0.3°). Send/stream still use the physical box.
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
  // Task yaw: absolute box from mechanical zero (not per-segment relative).
  constexpr double kDefaultTaskYawMaxRad = 30.0 * kPi / 180.0;
  const double yaml_yaw =
      cfg.params.yaw_max_deg > 0.0
          ? cfg.params.yaw_max_deg * kPi / 180.0
          : kDefaultTaskYawMaxRad;
  policy.yaw_abs_max_rad = std::min(kDefaultTaskYawMaxRad, yaml_yaw);
  policy.yaw_limited = policy.yaw_abs_max_rad > 1e-6;
  policy.a_max_calibrated = false;
  policy.j_max_calibrated = false;
  return policy;
}

// Intersect planner work/recovery boxes with the task yaw box (±yaw_abs_max).
inline void joint_apply_yaw_task_limits(const MotionConstraintPolicy& policy,
                                        Vec18& lo, Vec18& hi) {
  if (!policy.yaw_limited || policy.yaw_abs_max_rad <= 0.0) {
    return;
  }
  const double ylo = -policy.yaw_abs_max_rad;
  const double yhi = policy.yaw_abs_max_rad;
  lo[kIdxYaw] = std::max(lo[kIdxYaw], ylo);
  hi[kIdxYaw] = std::min(hi[kIdxYaw], yhi);
  if (lo[kIdxYaw] > hi[kIdxYaw]) {
    const double mid = 0.5 * (ylo + yhi);
    lo[kIdxYaw] = hi[kIdxYaw] = mid;
  }
}

inline void joint_effective_work_limits(const Vec18& /*q0*/,
                                        const MotionConstraintPolicy& policy,
                                        Vec18& work_lo, Vec18& work_hi) {
  joint_work_limits(policy.q_min, policy.q_max, policy.q_limit_margin_rad,
                    work_lo, work_hi);
  joint_apply_yaw_task_limits(policy, work_lo, work_hi);
}

inline void joint_effective_recovery_limits(
    const Vec18& q0, const MotionConstraintPolicy& policy, Vec18& rec_lo,
    Vec18& rec_hi) {
  joint_recovery_limits(q0, policy.q_min, policy.q_max,
                        policy.q_limit_margin_rad, rec_lo, rec_hi);
  joint_apply_yaw_task_limits(policy, rec_lo, rec_hi);
}

inline bool yaw_start_outside_task_box(const Vec18& q0,
                                       const MotionConstraintPolicy& policy) {
  if (!policy.yaw_limited || policy.yaw_abs_max_rad <= 0.0) {
    return false;
  }
  const double y = q0[kIdxYaw];
  return y < -policy.yaw_abs_max_rad - kHardBoxSlackRad ||
         y > policy.yaw_abs_max_rad + kHardBoxSlackRad;
}

inline bool joint_violates_yaw_task_box(const Vec18& q,
                                        const MotionConstraintPolicy& policy) {
  if (!policy.yaw_limited || policy.yaw_abs_max_rad <= 0.0) {
    return false;
  }
  if (!q.allFinite()) {
    return true;
  }
  const double y = q[kIdxYaw];
  return y < -policy.yaw_abs_max_rad - kHardBoxSlackRad ||
         y > policy.yaw_abs_max_rad + kHardBoxSlackRad;
}

inline bool joint_violates_effective_work_box(
    const Vec18& q, const Vec18& q0, const MotionConstraintPolicy& policy) {
  if (!q.allFinite()) {
    return true;
  }
  Vec18 lo;
  Vec18 hi;
  joint_effective_recovery_limits(q0, policy, lo, hi);
  if (joint_hard_margin_rad(q, lo, hi) < -kHardBoxSlackRad) {
    return true;
  }
  return joint_violates_yaw_task_box(q, policy);
}

}  // namespace kmpc
