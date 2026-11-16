#include "kmpc/safety_supervisor.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

double pitch_of(const Vec18& q) { return q[0] + q[1] + q[2]; }

double pitch_excess(const Vec18& q, double hard) {
  if (!(hard > 0.0) || !q.allFinite()) {
    return 0.0;
  }
  return std::max(0.0, std::abs(pitch_of(q)) - hard);
}

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
  q_last_send_ = q_safe;
  has_safe_ = q_safe.allFinite();
  has_last_send_ = has_safe_;
  last_vel_.setZero();
  last_acc_.setZero();
  has_vel_ = false;
  has_acc_ = false;
  t_last_send_s_ = -1.0;
}

SafetyDecision SafetySupervisor::check(const SafetySupervisorInput& in,
                                       bool commit) {
  SafetyDecision out;
  out.q_send = has_safe_ ? last_safe_ : (has_last_send_ ? q_last_send_ : in.q_cmd);
  out.accept = false;
  out.arrived = false;

  auto reject = [&](const char* reason) -> SafetyDecision {
    out.reason = reason;
    out.accept = false;
    out.arrived = false;
    out.q_send = has_safe_ ? last_safe_ : q_last_send_;
    return out;
  };

  if (!in.q_cmd.allFinite()) {
    return reject("safety_nonfinite");
  }
  if (!in.has_meas || !in.q_meas.allFinite() || !(in.meas_timestamp_s > 0.0)) {
    return reject("safety_no_real_meas");
  }

  // Hard joint box never widened (including Recovering).
  if (joint_violates_hard_box(in.q_cmd, policy_.q_min, policy_.q_max)) {
    return reject("safety_hard_box");
  }

  const double hard = policy_.torso_pitch_max_rad;
  const double ex_cmd = pitch_excess(in.q_cmd, hard);
  if (in.mode == SafetyExecMode::Recovering) {
    const double ceiling =
        std::max(in.pitch_excess_ceiling, pitch_excess(in.q_meas, hard)) +
        kPitchRoundEpsRad;
    if (ex_cmd > ceiling + 1e-12) {
      return reject("safety_recovery_excess_growth");
    }
  } else if (hard > 0.0 && ex_cmd > kPitchRoundEpsRad) {
    return reject("safety_pitch");
  }

  // Adjacent-send kinematics: ignore caller q_prev.
  const Vec18 q_prev_send = has_last_send_ ? q_last_send_ : in.q_cmd;
  double dt = std::max(in.dt, 1e-4);
  const double send_s = in.send_timestamp_s > 0.0
                            ? in.send_timestamp_s : in.now_s;
  if (has_last_send_ && send_s > 0.0 && t_last_send_s_ > 0.0) {
    dt = std::max(send_s - t_last_send_s_, 1e-4);
  }
  out.dt_s = dt;

  Vec18 a_cmd_all = Vec18::Zero();
  for (int j = 0; j < kNDof; ++j) {
    out.joint = j;
    const double dq_max = std::max(policy_.dq_max[j], 1e-6);
    const double step = std::abs(in.q_cmd[j] - q_prev_send[j]);
    if (step > dq_max * dt + 1e-9) {
      out.actual = step;
      out.limit = dq_max * dt + 1e-9;
      return reject("safety_step");
    }
    const double v_prev = has_vel_ ? last_vel_[j] : 0.0;
    const double v_cmd = (in.q_cmd[j] - q_prev_send[j]) / dt;
    const double ddq_pol = std::max(policy_.ddq_max[j], 1e-6);
    const double ddq_soft =
        policy_.a_max_calibrated ? ddq_pol : std::max(ddq_pol, dq_max / 0.02);
    const double dv = std::abs(v_cmd - v_prev);
    out.velocity = v_cmd;
    out.acceleration = (v_cmd - v_prev) / dt;
    if (dv > 4.0 * ddq_soft * dt + 1e-6) {
      out.actual = dv;
      out.limit = 4.0 * ddq_soft * dt + 1e-6;
      return reject("safety_accel");
    }
    a_cmd_all[j] = (v_cmd - v_prev) / dt;
    // Jerk gate only after we have a prior accepted accel, and only with a
    // generous software band (uncalibrated device jerk is not hardware fact).
    if (has_acc_) {
      const double j_pol = std::max(policy_.dddq_max[j], 1e-6);
      const double j_soft =
          policy_.j_max_calibrated ? j_pol : std::max(j_pol, ddq_soft / 0.02);
      if (std::abs(a_cmd_all[j] - last_acc_[j]) > 16.0 * j_soft * dt + 1e-3) {
        out.actual = std::abs(a_cmd_all[j] - last_acc_[j]);
        out.limit = 16.0 * j_soft * dt + 1e-3;
        return reject("safety_jerk");
      }
    }
  }

  out.joint = -1;
  if (guards_ != nullptr && !guards_->com_xy_inside(kin_.com(in.q_cmd))) {
    return reject("safety_com");
  }

  if (in.now_s > 0.0 &&
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
    if (in.expected_task_digest != 0 &&
        in.package->task_digest != 0 &&
        in.package->task_digest != in.expected_task_digest) {
      return reject("safety_task_digest");
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

  // Optional follow hint: only when caller marks has_follow.
  if (in.has_follow && in.q_follow.allFinite() &&
      (in.q_cmd - in.q_follow).cwiseAbs().maxCoeff() > follow_err_rad_ + 1e-9) {
    if (in.mode != SafetyExecMode::Recovering) {
      return reject("safety_follow_mismatch");
    }
  }

  out.accept = true;
  out.joint = -1;
  out.reason = "ok";
  out.q_send = in.q_cmd;

  if (in.claim_arrived) {
    if (in.package == nullptr || in.package->samples.empty()) {
      return reject("safety_fake_arrived");
    }
    const Vec18& q_end = in.package->samples.back();
    const bool near =
        (in.q_meas - q_end).cwiseAbs().maxCoeff() < 0.02 &&
        (in.q_cmd - q_end).cwiseAbs().maxCoeff() < 0.02;
    if (!near) {
      return reject("safety_fake_arrived");
    }
    out.arrived = true;
  }

  if (commit) note_sent(in.q_cmd, send_s, in.dt);
  return out;
}

void SafetySupervisor::note_sent(const Vec18& q, double send_s,
                                  double nominal_dt) {
  if (!q.allFinite()) return;
  const double dt = (send_s > 0.0 && t_last_send_s_ > 0.0)
                        ? std::max(send_s - t_last_send_s_, 1e-4)
                        : std::max(nominal_dt, 1e-4);
  const Vec18 vel = has_last_send_ ? Vec18((q - q_last_send_) / dt)
                                  : Vec18::Zero();
  const Vec18 acc = has_vel_ ? Vec18((vel - last_vel_) / dt)
                            : Vec18::Zero();
  last_safe_ = q;
  has_safe_ = true;
  if (has_vel_) {
    last_acc_ = acc;
    has_acc_ = true;
  }
  last_vel_ = vel;
  has_vel_ = true;
  q_last_send_ = q;
  has_last_send_ = true;
  t_last_send_s_ = send_s > 0.0 ? send_s : t_last_send_s_;
}

}  // namespace kmpc
