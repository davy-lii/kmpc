#include "kmpc/joint_space_tracker.hpp"

namespace kmpc {

JointSpaceTracker::JointSpaceTracker(Kinematics& kin, const Config& cfg)
    : kin_(kin),
      constraints_(cfg.limits, cfg.params.dt),
      guards_(cfg.params, cfg.q_pref[kIdxYaw]),
      w_(Vec18::Constant(cfg.params.w_joint)),
      r_default_(Vec18::Constant(cfg.params.r_arm)),
      s_(Vec18::Constant(cfg.params.s_joint)) {}

void JointSpaceTracker::reset() {
  dq_prev_.setZero();
  last_qp_status_ = QpStatus::Success;
  last_n_ineq_ = 0;
  last_com_.setZero();
  last_pitch_ = 0.0;
  last_theta_star_ = 0.0;
  last_pitch_seg_lo_ = 0.0;
  last_pitch_seg_hi_ = 0.0;
}

Vec18 JointSpaceTracker::compute(
    const Vec18& q0, const Vec18& q_des, const Vec18* r) {
  last_pitch_ = WholeBodyGuards::pitch(q0);
  last_theta_star_ = WholeBodyGuards::pitch(q_des);
  const InertialQuantities inr = kin_.inertial_quantities(q0);
  last_com_ = inr.p_com;

  const DqBox box = constraints_.box_dq(q0);
  Vec18 p_diag = w_ + s_;
  if (r != nullptr) {
    p_diag += *r;
  } else {
    p_diag += r_default_;
  }
  const Vec18 g = w_.cwiseProduct(q0 - q_des) - s_.cwiseProduct(dq_prev_);
  Mat18 P = p_diag.asDiagonal();
  P.diagonal().array() += kHessRidge;

  MatIneq18 A = MatIneq18::Zero();
  VecIneq b = VecIneq::Zero();
  int n_ineq = 0;
  guards_.append_com(inr.p_com, inr.J_com, A, b, n_ineq);
  double dlo = 0.0;
  double dhi = 0.0;
  guards_.pitch_toward_bounds(q0, last_theta_star_, dlo, dhi);
  last_pitch_seg_lo_ = last_pitch_ + dlo;
  last_pitch_seg_hi_ = last_pitch_ + dhi;
  guards_.append_pitch_toward(q0, last_theta_star_, A, b, n_ineq);
  guards_.append_yaw(q0, A, b, n_ineq);
  guards_.append_elbow(q0, A, b, n_ineq);
  last_n_ineq_ = n_ineq;

  Vec18 dq = dq_prev_;
  last_qp_status_ = qp_.solve_box_ineq(P, g, box.lb, box.ub, A, b, n_ineq, dq);
  if (last_qp_status_ != QpStatus::Success || !dq.allFinite()) {
    dq.setZero();
    dq_prev_.setZero();
    return dq;
  }
  dq_prev_ = dq;
  return dq;
}

}  // namespace kmpc
