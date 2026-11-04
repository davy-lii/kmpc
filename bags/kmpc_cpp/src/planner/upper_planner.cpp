// 状态机：CALIBRATING 臂均值误差 < calib_tol 才切 TRACKING。不装 QP。

#include "kmpc/upper_planner.hpp"

namespace kmpc {

UpperPlanner::UpperPlanner(
    Kinematics& kin, double calib_tol_rad, double com_xmin, double com_xmax,
    double com_margin)
    : kin_(kin),
      calib_tol_rad_(calib_tol_rad),
      com_xmin_(com_xmin),
      com_xmax_(com_xmax),
      com_margin_(com_margin) {}

bool UpperPlanner::com_safe(const Vec18& q) const {
  const Vec3 com = kin_.com(q);
  double lo = 0.0;
  double hi = 0.0;
  com_x_inner_bounds(com_xmin_, com_xmax_, com_margin_, lo, hi);
  return com[0] >= lo && com[0] <= hi;
}

bool UpperPlanner::request_calibrate(const Vec18& q_calib) {
  if (!com_safe(q_calib)) {
    q_calib_rejected_ = true;
    return false;
  }
  q_calib_rejected_ = false;
  q_calib_ = q_calib;
  has_q_calib_ = true;
  tracking_state_ = TrackingState::Calibrating;
  return true;
}

void UpperPlanner::request_tracking() {
  tracking_state_ = TrackingState::Tracking;
  has_q_calib_ = false;
  q_calib_.setZero();
  q_calib_rejected_ = false;
}

void UpperPlanner::set_waist_locked(bool locked) {
  waist_locked_ = locked;
}

PlannerStepResult UpperPlanner::step(const Vec18& q0) {
  bool state_changed = false;
  if (tracking_state_ == TrackingState::Calibrating && has_q_calib_) {
    if (arm_mean_calib_error(q0, q_calib_) < calib_tol_rad_) {
      tracking_state_ = TrackingState::Tracking;
      has_q_calib_ = false;
      q_calib_.setZero();
      state_changed = true;
    }
  }
  UpperCommand cmd;
  cmd.tracking_state = tracking_state_;
  cmd.waist_locked = waist_locked_;
  cmd.has_q_calib = has_q_calib_;
  cmd.q_calib = q_calib_;
  PlannerStepResult out;
  out.command = cmd;
  out.state_changed = state_changed;
  return out;
}

}  // namespace kmpc
