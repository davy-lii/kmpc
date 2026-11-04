#pragma once
// 关节空间 P0：对角 W,R,S + 全身硬盒 ProxQP。不进 36 维 Hessian。
// 肘下墙只在这里；盒外 min(...,0) 只许返回。

#include "kmpc/config.hpp"
#include "kmpc/constraints.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/qp_solver.hpp"
#include "kmpc/types.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

class JointSpaceTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JointSpaceTracker(Kinematics& kin, const Config& cfg);

  void reset();

  Vec18 compute(const Vec18& q0, const Vec18& q_des, const Vec18* r);

  const WholeBodyGuards& guards() const { return guards_; }
  QpStatus last_qp_status() const { return last_qp_status_; }
  int last_n_ineq() const { return last_n_ineq_; }
  Vec3 last_com() const { return last_com_; }
  double last_pitch() const { return last_pitch_; }
  double last_theta_star() const { return last_theta_star_; }
  double last_pitch_seg_lo() const { return last_pitch_seg_lo_; }
  double last_pitch_seg_hi() const { return last_pitch_seg_hi_; }

 private:
  Kinematics& kin_;
  ConstraintBuilder constraints_;
  WholeBodyGuards guards_;
  QpSolver18 qp_;
  Vec18 w_ = Vec18::Zero();
  Vec18 r_default_ = Vec18::Zero();
  Vec18 s_ = Vec18::Zero();
  Vec18 dq_prev_ = Vec18::Zero();
  QpStatus last_qp_status_ = QpStatus::Success;
  int last_n_ineq_ = 0;
  Vec3 last_com_ = Vec3::Zero();
  double last_pitch_ = 0.0;
  double last_theta_star_ = 0.0;
  double last_pitch_seg_lo_ = 0.0;
  double last_pitch_seg_hi_ = 0.0;
};

}  // namespace kmpc
