#pragma once
// Production final-send geometric gate shared by kmpc_node and offline chain.
// Matches stream_filtered_fault: joint, pitch, CoM, progress, ori.
// TCP chord tube (stream_filtered_path) removed — planning/QP soft tube remains.
// Direction envelope remains a separate call (direction_envelope_accept).

#include <algorithm>
#include <cmath>
#include <string>

#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct LineFinalGateGeo {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 start = Vec3::Zero();
  Vec3 target = Vec3::Zero();
  Vec4 quat0 = Vec4(0, 0, 0, 1);
  Vec4 quat1 = Vec4(0, 0, 0, 1);
  std::string frame;
  bool ori_on = false;
  double previous_along = 0.0;
};

struct LineFinalGateLimits {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_min = Vec18::Zero();
  Vec18 q_max = Vec18::Zero();
  double q_limit_margin = 0.0;
  double torso_pitch_max_rad = 0.0;
  double arrived_ori_rad = 0.05;
};

// Returns nullptr if OK, else a stable fault tag (same strings as production).
inline const char* line_final_filtered_fault(
    const Vec18& q, double pitch_extra, const LineFinalGateLimits& lim,
    Kinematics& kin, const WholeBodyGuards& guards, LineFinalGateGeo* geo) {
  if (!q.allFinite()) {
    return "stream_filtered_nonfinite";
  }
  // Hard joint box only — soft margin must not stop a stream that is already
  // past soft while moving away from the hard wall (r39 s0_qlim_j0 / rise).
  if ((q.array() < lim.q_min.array() - 1e-7).any() ||
      (q.array() > lim.q_max.array() + 1e-7).any()) {
    return "stream_filtered_joint_limit";
  }
  if (lim.torso_pitch_max_rad > 0.0 &&
      std::abs(q.head<3>().sum()) >
          lim.torso_pitch_max_rad + pitch_extra + kPitchRoundEpsRad) {
    return "stream_filtered_pitch";
  }
  if (!guards.com_xy_inside(kin.com(q))) {
    return "stream_filtered_com";
  }
  if (geo == nullptr || geo->frame.empty()) {
    return nullptr;
  }
  const auto pose = kin.fk(q, geo->frame);
  const Vec3 ab = geo->target - geo->start;
  const double along = (pose.col(3).head<3>() - geo->start).dot(ab);
  const double t =
      std::clamp(along / std::max(ab.squaredNorm(), 1e-12), 0.0, 1.0);
  if (along < geo->previous_along -
                 kStreamFilteredProgressSlackM * std::max(ab.norm(), 1e-6)) {
    return "stream_filtered_progress";
  }
  if (geo->ori_on &&
      ori_error_xyzw(quat_slerp_xyzw(geo->quat0, geo->quat1, t),
                     rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
              .norm() > lim.arrived_ori_rad + 1e-6) {
    return "stream_filtered_orientation";
  }
  geo->previous_along = along;
  return nullptr;
}

inline LineFinalGateLimits make_final_gate_limits(const Config& cfg) {
  LineFinalGateLimits lim;
  lim.q_min = cfg.limits.q_min;
  lim.q_max = cfg.limits.q_max;
  lim.q_limit_margin = cfg.params.q_limit_margin;
  lim.torso_pitch_max_rad = cfg.params.torso_pitch_max_deg * kPi / 180.0;
  lim.arrived_ori_rad = cfg.params.arrived_ori_rad;
  return lim;
}

}  // namespace kmpc
