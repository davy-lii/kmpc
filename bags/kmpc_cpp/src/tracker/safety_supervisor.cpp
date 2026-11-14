#include "kmpc/safety_supervisor.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

double pitch_of(const Vec18& q) { return q[0] + q[1] + q[2]; }

std::string_view ee_frame(bool left) {
  return left ? kEeFrameLeft : kEeFrameRight;
}

}  // namespace

SafetySupervisor::SafetySupervisor(const Kinematics& kin,
                                   const WholeBodyGuards* guards,
                                   MotionConstraintPolicy policy)
    : kin_(kin), guards_(guards), policy_(std::move(policy)) {}

void SafetySupervisor::seed(const Vec18& q_safe) {
  last_safe_ = q_safe;
  has_safe_ = q_safe.allFinite();
  last_vel_.setZero();
  has_vel_ = false;
}

SafetyDecision SafetySupervisor::check(const SafetySupervisorInput& in) {
  SafetyDecision out;
  out.q_send = has_safe_ ? last_safe_ : in.q_prev;
  out.accept = false;
  out.arrived = false;

  auto reject = [&](const char* reason) -> SafetyDecision {
    out.reason = reason;
    out.accept = false;
    out.arrived = false;  // never fake arrived on reject
    return out;
  };

  if (!in.q_cmd.allFinite() || !in.q_prev.allFinite() || !in.q_meas.allFinite()) {
    return reject("safety_nonfinite");
  }
  if (joint_violates_hard_box(in.q_cmd, policy_.q_min, policy_.q_max)) {
    return reject("safety_hard_box");
  }
  const double hard = policy_.torso_pitch_max_rad;
  if (hard > 0.0 &&
      std::abs(pitch_of(in.q_cmd)) > hard + kPitchRoundEpsRad) {
    return reject("safety_pitch");
  }

  const double dt = std::max(in.dt, 1e-4);
  Vec18 v_cmd_all = Vec18::Zero();
  for (int j = 0; j < kNDof; ++j) {
    const double dq_max = std::max(policy_.dq_max[j], 1e-6);
    const double step = std::abs(in.q_cmd[j] - in.q_prev[j]);
    if (step > dq_max * dt + 1e-9) {
      return reject("safety_step");
    }
    // Velocity of this command step; compare to last *accepted* velocity.
    // Do not recompute from (q_prev - last_safe_): after accept those are equal
    // so the proxy was always ~0 and falsely tripped safety_accel at 250 Hz.
    const double v_prev = has_vel_ ? last_vel_[j] : 0.0;
    const double v_cmd = (in.q_cmd[j] - in.q_prev[j]) / dt;
    v_cmd_all[j] = v_cmd;
    // Accel budget: calibrated device a_max when available; otherwise a software
    // stand-in that can reach dq_max within one 50 Hz control step (not a hard
    // joint-box relaxation).
    const double ddq_pol = std::max(policy_.ddq_max[j], 1e-6);
    const double ddq_soft =
        policy_.a_max_calibrated ? ddq_pol : std::max(ddq_pol, dq_max / 0.02);
    const double dv = std::abs(v_cmd - v_prev);
    if (dv > 4.0 * ddq_soft * dt + 1e-6) {
      return reject("safety_accel");
    }
  }

  if (guards_ != nullptr && !guards_->com_xy_inside(kin_.com(in.q_cmd))) {
    return reject("safety_com");
  }

  if (in.now_s > 0.0 && in.meas_timestamp_s > 0.0 &&
      (in.now_s - in.meas_timestamp_s) > freshness_s_) {
    return reject("safety_stale_meas");
  }
  if (in.now_s > 0.0 && in.cmd_timestamp_s > 0.0 &&
      (in.now_s - in.cmd_timestamp_s) > freshness_s_) {
    return reject("safety_stale_cmd");
  }

  if (in.package != nullptr) {
    if (in.package->generation != in.expected_generation) {
      return reject("safety_generation");
    }
    if (in.expected_path_id != 0 &&
        in.package->path_id != in.expected_path_id) {
      return reject("safety_path_id");
    }
    if (!in.package->certificate.passed) {
      return reject("safety_uncertified");
    }
    const auto frame = ee_frame(in.package->left);
    const Vec3 p_cmd = kin_.fk_pos(in.q_cmd, frame);
    const Vec3 p_plan = kin_.fk_pos(in.q_plan, frame);
    if ((p_cmd - p_plan).norm() > tcp_tube_m_ + 1e-9) {
      return reject("safety_tcp_tube");
    }
    const Vec3 p_meas = kin_.fk_pos(in.q_meas, frame);
    if ((p_meas - p_plan).norm() > track_err_m_ + 1e-9) {
      return reject("safety_track_err");
    }
  }

  if ((in.q_cmd - in.q_follow).cwiseAbs().maxCoeff() > follow_err_rad_ + 1e-9) {
    // 250 Hz must not invent a geometrically different command than 50 Hz.
    return reject("safety_follow_mismatch");
  }

  // Never silent-clamp: accept only the exact q_cmd that passed.
  out.accept = true;
  out.reason = "ok";
  out.q_send = in.q_cmd;

  if (in.claim_arrived) {
    if (in.package == nullptr || in.package->samples.empty()) {
      out.arrived = false;
      out.accept = false;
      out.reason = "safety_fake_arrived";
      out.q_send = has_safe_ ? last_safe_ : in.q_prev;
      return out;
    }
    const Vec18& q_end = in.package->samples.back();
    const bool near =
        (in.q_meas - q_end).cwiseAbs().maxCoeff() < 0.02 &&
        (in.q_cmd - q_end).cwiseAbs().maxCoeff() < 0.02;
    if (!near) {
      out.arrived = false;
      out.accept = false;
      out.reason = "safety_fake_arrived";
      out.q_send = has_safe_ ? last_safe_ : in.q_prev;
      return out;
    }
    out.arrived = true;
  }

  last_safe_ = in.q_cmd;
  has_safe_ = true;
  last_vel_ = v_cmd_all;
  has_vel_ = true;
  return out;
}

}  // namespace kmpc
