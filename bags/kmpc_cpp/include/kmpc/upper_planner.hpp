#pragma once
// 对应 Python kmpc/upper_planner.py；对照 09-04 §1.1。
// 状态机不进 horizon_qp / qp_solver。active_arm 现码未用。

#include "kmpc/kinematics.hpp"
#include "kmpc/types.hpp"
#include "kmpc/upper_command.hpp"

namespace kmpc {

inline double arm_mean_calib_error(const Vec18& q0, const Vec18& q_calib) {
  const Vec18 d = (q0 - q_calib).cwiseAbs();
  double sum = 0.0;
  for (int i = kSliceRight.begin; i < kSliceLeft.end; ++i) {
    sum += d[i];
  }
  return sum / static_cast<double>(kNArm * 2);
}

inline void com_x_inner_bounds(
    double com_xmin, double com_xmax, double com_margin, double& lo,
    double& hi) {
  lo = com_xmin + com_margin;
  hi = com_xmax - com_margin;
}

struct PlannerStepResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  UpperCommand command;
  bool state_changed = false;
};

class UpperPlanner {
 public:
  UpperPlanner(
      Kinematics& kin, double calib_tol_rad, double com_xmin, double com_xmax,
      double com_margin);

  TrackingState tracking_state() const { return tracking_state_; }
  bool waist_locked() const { return waist_locked_; }
  bool has_q_calib() const { return has_q_calib_; }
  Vec18 q_calib() const { return q_calib_; }
  bool q_calib_rejected() const { return q_calib_rejected_; }

  bool com_safe(const Vec18& q) const;
  bool request_calibrate(const Vec18& q_calib);
  void request_tracking();
  void set_waist_locked(bool locked);

  PlannerStepResult step(const Vec18& q0);

 private:
  Kinematics& kin_;
  double calib_tol_rad_ = 0.02;
  double com_xmin_ = -0.322;
  double com_xmax_ = 0.138;
  double com_margin_ = 0.025;
  TrackingState tracking_state_ = TrackingState::Tracking;
  bool waist_locked_ = false;
  bool has_q_calib_ = false;
  Vec18 q_calib_ = Vec18::Zero();
  bool q_calib_rejected_ = false;
};

}  // namespace kmpc
