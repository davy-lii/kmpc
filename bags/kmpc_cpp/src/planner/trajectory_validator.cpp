#include "kmpc/trajectory_validator.hpp"

#include <algorithm>
#include <cmath>

#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr double kLegacyHardTubeM = 0.20;

void note(const char** fault, const char* why) {
  if (*fault == nullptr) {
    *fault = why;
  }
}

double pitch_abs(const Vec18& q) { return std::abs(q[0] + q[1] + q[2]); }

}  // namespace

TrajectoryCertificate validate_discrete_trajectory(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards) {
  TrajectoryCertificate out;
  out.coverage = PathCoverage::DiscreteSamples;
  out.continuous_certified = false;
  out.com_model = "not_checked";
  out.min_joint_margin_rad = 0.0;
  out.min_pitch_margin_rad = 0.0;
  const char* fault = nullptr;

  if (samples.size() < 2) {
    out.first_fault = "too_few_samples";
    return out;
  }
  if (task.stages.size() != 1) {
    out.first_fault = "stage_count_not_supported";
    return out;
  }
  const TaskStage& stage = task.stages.front();
  if (stage.geometry == TaskGeometry::FollowCurve) {
    out.first_fault = "follow_curve_not_implemented";
    return out;
  }
  if (stage.has_orientation) {
    out.first_fault = "orientation_check_not_implemented";
    return out;
  }
  if (!(stage.position_tol_m > 0.0) || !std::isfinite(stage.position_tol_m)) {
    out.first_fault = "bad_task_tol";
    return out;
  }
  if (stage.geometry == TaskGeometry::Lin &&
      stage.position_tol_m >= kLegacyHardTubeM) {
    out.first_fault = "lin_tol_is_legacy_hard_tube";
    return out;
  }

  out.hard_box_ok = true;
  out.pitch_ok = true;
  out.min_joint_margin_rad = joint_hard_margin_rad(
      samples.front(), policy.q_min, policy.q_max);
  const double pitch_limit = policy.torso_pitch_max_rad + kPitchRoundEpsRad;
  out.min_pitch_margin_rad = pitch_limit - pitch_abs(samples.front());

  for (const Vec18& q : samples) {
    const double joint_margin =
        joint_hard_margin_rad(q, policy.q_min, policy.q_max);
    out.min_joint_margin_rad = std::min(out.min_joint_margin_rad, joint_margin);
    if (joint_violates_hard_box(q, policy.q_min, policy.q_max)) {
      out.hard_box_ok = false;
      note(&fault, "joint");
    }
    const double pitch_margin = pitch_limit - pitch_abs(q);
    out.min_pitch_margin_rad = std::min(out.min_pitch_margin_rad, pitch_margin);
    if (!(pitch_margin >= 0.0) || !q.head<3>().allFinite()) {
      out.pitch_ok = false;
      note(&fault, "pitch");
    }
  }

  out.com_checked = guards != nullptr;
  if (guards == nullptr) {
    out.com_ok = false;
    out.com_model = "not_checked";
    note(&fault, "com_not_checked");
  } else {
    out.com_ok = true;
    out.com_model = "xy_work_rectangle";
    for (const Vec18& q : samples) {
      if (!q.allFinite() || !guards->com_xy_inside(kin.com(q))) {
        out.com_ok = false;
        note(&fault, "com");
        break;
      }
    }
  }

  const std::string_view frame = task.left ? kEeFrameLeft : kEeFrameRight;
  const Vec3 start = kin.fk_pos(samples.front(), frame);
  const Vec3 target = stage.position;
  const Vec3 ab = target - start;
  const double ab2 = ab.squaredNorm();
  out.task_path_ok = true;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const Vec3 p = kin.fk_pos(samples[i], frame);
    double cross = 0.0;
    if (stage.geometry == TaskGeometry::Ptp) {
      cross = 0.0;
    } else if (ab2 < 1e-16) {
      cross = (p - target).norm();
    } else {
      double t = (p - start).dot(ab) / ab2;
      if (t < 0.0) {
        t = 0.0;
      } else if (t > 1.0) {
        t = 1.0;
      }
      cross = (p - (start + t * ab)).norm();
    }
    out.worst_tcp_cross_track_m = std::max(out.worst_tcp_cross_track_m, cross);
    if (stage.geometry == TaskGeometry::Lin && cross > stage.position_tol_m) {
      out.task_path_ok = false;
      note(&fault, "tcp_cross_track");
    }
  }
  out.worst_tcp_endpoint_m = (kin.fk_pos(samples.back(), frame) - target).norm();
  if (out.worst_tcp_endpoint_m > stage.position_tol_m) {
    out.task_path_ok = false;
    note(&fault, "tcp_endpoint");
  }

  out.passed = fault == nullptr && out.hard_box_ok && out.pitch_ok &&
               out.com_checked && out.com_ok && out.task_path_ok;
  out.first_fault = fault == nullptr ? "ok" : fault;
  return out;
}

}  // namespace kmpc
