// 对应 cartesian_tracker.py。对照 09-04 §1 / §3：路径只在 k=0，球只进终端 V_o。

#include "kmpc/cartesian_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kmpc/horizon_qp.hpp"
#include "kmpc/jparse.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {

CartesianTracker::CartesianTracker(Kinematics& kin, const Config& cfg)
    : kin_(kin),
      limits_(cfg.limits),
      constraints_(limits_, cfg.params.dt),
      params_(cfg.params),
      guards_(cfg.params, cfg.q_pref[kIdxYaw]) {
  if (params_.n_horizon < 2) {
    throw std::invalid_argument("CartesianTracker needs n_horizon >= 2 (36-DoF QP)");
  }
  q_pos_ = params_.q_pos;
  q_offset_ = params_.q_offset;
  q_offset_y_ = params_.q_offset_y;
  w_idle_arm_ = params_.w_idle_arm;
  w_o_arm_ = params_.w_o_arm;
  track_gamma_ = params_.track_gamma;
  vo_clamp_m_ = params_.vo_clamp_m;
  vo_clamp_y_m_ = params_.vo_clamp_y_m;
  vo_arm_gamma_ = params_.vo_arm_gamma;
  waist_gamma_ = params_.waist_gamma;
  n_horizon_ = params_.n_horizon;
  q_limit_margin_ = params_.q_limit_margin;
  track_alpha_ = params_.track_alpha;
  arrived_err_ = params_.arrived_err;
  arrived_exit_ = params_.arrived_exit;
  q_com_ = params_.q_com;
  com_margin_ = params_.com_margin;
  com_xmin_ = params_.com_xmin;
  com_xmax_ = params_.com_xmax;
  com_ymin_ = params_.com_ymin;
  com_ymax_ = params_.com_ymax;
  c_sup_ = Eigen::Vector2d(
      0.5 * (com_xmin_ + com_xmax_), 0.5 * (com_ymin_ + com_ymax_));
  q_torso_ = params_.q_torso;
  q_syn_ = params_.q_syn;
  alpha_syn_ = params_.alpha_syn;
  q_yaw_ = params_.q_yaw;
  torso_pitch_max_ = deg_to_rad(params_.torso_pitch_max_deg);
  torso_drop_m_ = params_.torso_drop_m;
  torso_rise_hold_ = params_.torso_rise_hold;
  torso_reach_pitch_follow_ = params_.torso_reach_pitch_follow;
  barrier_joints_ = params_.barrier_joints;
  barrier_warn_rad_ = deg_to_rad(params_.barrier_warn_deg);
  barrier_floor_rad_ = deg_to_rad(params_.barrier_floor_deg);
  barrier_alpha_ = std::max(params_.barrier_alpha, 0.0);
  barrier_w_max_ratio_ = params_.barrier_w_max_ratio;
  com_hold_ez_m_ = std::max(params_.com_hold_ez_m, 0.0);
  if (params_.torso_k_per_m <= 0.0) {
    torso_k_per_m_ = torso_pitch_max_ > 0.0 ? torso_pitch_max_ / 0.80 : 0.0;
  } else {
    torso_k_per_m_ = params_.torso_k_per_m;
  }
  yaw_max_ = deg_to_rad(params_.yaw_max_deg);
  arrived_stall_m_ = std::max(params_.arrived_stall_mm, 0.0) * 1e-3;
  arrived_stall_ticks_ = std::max(params_.arrived_stall_ticks, 1);
  e0_hist_cap_ = std::min(arrived_stall_ticks_, kE0HistCap);
  waist_lock_eps_ = params_.waist_lock_eps;
  waist_pred_lambda_ = std::max(params_.waist_pred_lambda, 0.0);
  vo_arm_floor_m_ = params_.vo_arm_floor_m;
  q_ori_ = params_.q_ori;
  vo_clamp_ori_rad_ = params_.vo_clamp_ori_rad;
  arrived_ori_rad_ = params_.arrived_ori_rad;
  s_joint_ = params_.s_joint;
  w_joint_ = params_.w_joint;
  r_arm_ = params_.r_arm;

  j_torso_.head<3>().setOnes();
  h_syn_[0] = alpha_syn_;
  h_syn_[1] = 1.0;
  qo_ = Mat3::Zero();
  qo_(0, 0) = q_offset_;
  qo_(1, 1) = q_offset_y_;
  qo_(2, 2) = q_offset_;

  const Vec18 q_cal = cfg.q_pref;
  q_yaw_pref_ = q_cal[kIdxYaw];
  r_lambda_ = kin_.inertia_lambda(q_cal, r_arm_);
  z_home_ = kin_.fk_pos(q_cal, kEeFrameLeft)[2];
  z_star_ = z_home_;
  upper_ = default_upper_command();
}

void CartesianTracker::set_upper_commands(const UpperCommand& cmd) {
  upper_ = cmd;
}

void CartesianTracker::reset_warm() {
  clear_hold_warm();
  e0_hist_clear();
  e0_prev_ = 0.0;
}

void CartesianTracker::clear_hold_warm() {
  dq_prev_.setZero();
  y_prev_.setZero();
  has_pitch_pin_ = false;
}

void CartesianTracker::on_tracking_state_changed() {
  reset_warm();
}

void CartesianTracker::reset() {
  dq_prev_.setZero();
  y_prev_.setZero();
  last_e0_ = 0.0;
  last_e_ori_.setZero();
  last_ori_on_ = false;
  last_arrived_ = false;
  last_com_.setZero();
  last_pitch_ = 0.0;
  last_theta_des_ = 0.0;
  last_c_des_x_ = 0.0;
  last_com_hold_ = false;
  last_com_work_wall_ = false;
  last_com_work_xmin_ = 0.0;
  last_com_work_xmax_ = 0.0;
  last_com_slack_x_ = 0.0;
  has_pitch_pin_ = false;
  has_prev_left_ = false;
  has_prev_right_ = false;
  e_ori_prev_left_.setZero();
  e_ori_prev_right_.setZero();
  e0_hist_clear();
  e0_prev_ = 0.0;
  z_star_ = z_home_;
  deep_on_ = false;
  upper_ = default_upper_command();
  tracking_state_prev_ = TrackingState::Tracking;
  last_P_.setZero();
  last_g_.setZero();
  last_n_ineq_ = 0;
  line_clear();
  line_tip_.reset();
  line_reference_.reset();
}

Vec18 CartesianTracker::effective_dq_max() const {
  return kmpc::effective_dq_max(limits_, upper_.waist_locked, waist_lock_eps_);
}

Vec18 CartesianTracker::inertia_r(const Vec18& q0) const {
  const Vec18 mdiag = kin_.mass_matrix_diag(q0);
  return (r_lambda_ * mdiag).cwiseMax(1e-8);
}

Vec18 CartesianTracker::offset_weights(
    int n_active, bool left_active, bool calibrating) const {
  Vec18 w_o = Vec18::Constant(w_o_arm_);
  w_o.segment<kNWaist>(0).setZero();
  if (calibrating) {
    w_o.segment<kNArm>(kSliceLeft.begin).array() += w_idle_arm_;
    w_o.segment<kNArm>(kSliceRight.begin).array() += w_idle_arm_;
  } else if (n_active == 1) {
    if (left_active) {
      w_o.segment<kNArm>(kSliceRight.begin).array() += w_idle_arm_;
    } else {
      w_o.segment<kNArm>(kSliceLeft.begin).array() += w_idle_arm_;
    }
  }
  return w_o;
}

Vec18 CartesianTracker::idle_offset_grad(
    const Vec18& q0, const Vec18& q_pref, const Vec18& w_o, int n_active,
    bool left_active, bool calibrating) const {
  Vec18 g = Vec18::Zero();
  if (calibrating) {
    g.segment<kNArm>(kSliceLeft.begin) =
        w_o.segment<kNArm>(kSliceLeft.begin).cwiseProduct(
            q0.segment<kNArm>(kSliceLeft.begin) -
            q_pref.segment<kNArm>(kSliceLeft.begin));
    g.segment<kNArm>(kSliceRight.begin) =
        w_o.segment<kNArm>(kSliceRight.begin).cwiseProduct(
            q0.segment<kNArm>(kSliceRight.begin) -
            q_pref.segment<kNArm>(kSliceRight.begin));
    return g;
  }
  if (n_active != 1) {
    return g;
  }
  const int begin = left_active ? kSliceRight.begin : kSliceLeft.begin;
  g.segment<kNArm>(begin) = w_o.segment<kNArm>(begin).cwiseProduct(
      q0.segment<kNArm>(begin) - q_pref.segment<kNArm>(begin));
  return g;
}

TrackerStep CartesianTracker::compute(
    const Vec18& q0, const Vec18& q_pref, const Vec3* target_left,
    const Vec3* target_right, const Vec4* quat_left, const Vec4* quat_right,
    std::string_view frame_left, std::string_view frame_right) {
  auto fail = [this](QpStatus st) {
    last_qp_status_ = st;
    last_e0_ = 0.0;
    last_arrived_ = false;
    TrackerStep out;
    return out;
  };
  if (!q0.allFinite() || !q_pref.allFinite()) {
    return fail(QpStatus::NonFinite);
  }
  if ((target_left != nullptr && !target_left->allFinite()) ||
      (target_right != nullptr && !target_right->allFinite()) ||
      (quat_left != nullptr && !quat_left->allFinite()) ||
      (quat_right != nullptr && !quat_right->allFinite())) {
    return fail(QpStatus::NonFinite);
  }

  const TrackingState state = upper_.tracking_state;
  if (state != tracking_state_prev_) {
    on_tracking_state_changed();
    tracking_state_prev_ = state;
  }
  const bool calibrating = state == TrackingState::Calibrating;
  const Vec3* xyz_left = calibrating ? nullptr : target_left;
  const Vec3* xyz_right = calibrating ? nullptr : target_right;
  const Vec4* q_left = calibrating ? nullptr : quat_left;
  const Vec4* q_right = calibrating ? nullptr : quat_right;

  TrackerStep lined;
  if (follow_line(
          q0, xyz_left, xyz_right, q_left, q_right, frame_left, frame_right,
          lined)) {
    return lined;
  }
  if (line_.refused) {
    return lined;
  }

  Vec18 w_diag = Vec18::Constant(w_joint_);
  Vec6 err = Vec6::Zero();
  Vec6 e_ori6 = Vec6::Zero();
  const InertialQuantities inr = kin_.inertial_quantities(q0);
  const Vec18 r = (r_lambda_ * inr.mdiag).cwiseMax(1e-8);
  last_com_ = inr.p_com;
  last_pitch_ = q0[0] + q0[1] + q0[2];

  JacTerm terms[2];
  int n_terms = 0;
  int n_active = 0;
  bool left_active = false;
  double best_n = -1.0;
  double z_star = z_star_;
  bool any_ori = false;
  Vec3 e_ori_left = Vec3::Zero();
  Vec3 e_ori_right = Vec3::Zero();

  const auto add_arm = [&](const Vec3* xyz, const Vec4* quat,
                           std::string_view frame, bool left) {
    if (xyz == nullptr) {
      return;
    }
    ++n_active;
    if (left) {
      left_active = true;
    }
    const std::string_view ee = frame.empty()
        ? (left ? kEeFrameLeft : kEeFrameRight)
        : frame;
    const Eigen::Matrix4d t_now = kin_.fk(q0, ee);
    const Vec3 p_now = t_now.col(3).head<3>();
    const Vec3 e = *xyz - p_now;
    if (left) {
      err.head<3>() = e;
    } else {
      err.tail<3>() = e;
    }
    const Mat6x18 jac6 = kin_.jacobian_spatial(q0, ee);
    JacTerm term;
    term.left = left;
    term.jac_v = jac6.topRows<3>();
    term.jac_w = jac6.bottomRows<3>();
    term.e = e;
    term.jac_v_pad = pad_tracking_jacobian(term.jac_v, track_gamma_);
    term.ori_on = q_ori_ > 0.0 && !quat_is_identity_xyzw(quat);
    term.jac_w_pad = term.jac_w;
    if (term.ori_on) {
      any_ori = true;
      const Mat3 rot = t_now.block<3, 3>(0, 0);
      const Vec4 q_curr = rot_to_quat_xyzw(rot);
      term.e_ori = ori_error_xyzw(*quat, q_curr);
      term.jac_w_pad = pad_tracking_jacobian(term.jac_w, track_gamma_);
      if (left) {
        e_ori6.head<3>() = term.e_ori;
        e_ori_left = term.e_ori;
      } else {
        e_ori6.tail<3>() = term.e_ori;
        e_ori_right = term.e_ori;
      }
    }
    const double nrm = e.norm();
    if (nrm > best_n) {
      best_n = nrm;
      z_star = (*xyz)[2];
    }
    terms[n_terms++] = term;
  };

  add_arm(xyz_left, q_left, frame_left, true);
  add_arm(xyz_right, q_right, frame_right, false);
  if (best_n >= 0.0) {
    z_star_ = z_star;
  }
  last_e_ori_ = e_ori6;
  last_ori_on_ = any_ori;

  bool jumped = targets_jumped(xyz_left, xyz_right);
  jumped = ori_jumped(e_ori_left, e_ori_right) || jumped;
  if (jumped) {
    has_pitch_pin_ = false;
    y_prev_.setZero();
    dq_prev_.setZero();
  }

  if (n_active == 1 && !calibrating) {
    if (left_active) {
      w_diag.segment<kNArm>(kSliceRight.begin).array() += w_idle_arm_;
    } else {
      w_diag.segment<kNArm>(kSliceLeft.begin).array() += w_idle_arm_;
    }
  }

  const Vec18 w_o = offset_weights(n_active, left_active, calibrating);
  const Vec18 g_wo =
      idle_offset_grad(q0, q_pref, w_o, n_active, left_active, calibrating);
  // 只够一只手时，另一只钉在这一拍的 q0。软权重会把它往 q_pref 拉，真机步距会顶满。
  const int idle_begin = (n_active == 1 && !calibrating)
      ? (left_active ? kSliceRight.begin : kSliceLeft.begin)
      : -1;

  Vec18 dq = solve_horizon(
      q0, w_diag, w_o, g_wo, r, terms, n_terms, inr.p_com, inr.J_com,
      idle_begin);
  const double e0 = active_norm(err, xyz_left != nullptr, xyz_right != nullptr);
  last_e0_ = e0;
  if (calibrating) {
    last_arrived_ = false;
  } else {
    const bool ori_ok = ori_arrived_ok(terms, n_terms);
    dq = apply_arrived(dq, e0, jumped, q0, working_ez(terms, n_terms), ori_ok);
  }
  if (idle_begin >= 0) {
    dq.segment<kNArm>(idle_begin).setZero();
  }
  dq_prev_ = dq;
  TrackerStep out;
  out.dq = dq;
  out.ee_error = err;
  return out;
}

Vec18 CartesianTracker::solve_horizon(
    const Vec18& q0, const Vec18& w_diag, const Vec18& w_o, const Vec18& g_wo,
    const Vec18& r, const JacTerm* terms, int n_terms, const Vec3& p_com,
    const Mat3x18& j_com, int idle_begin) {
  Mat18 h_track = Mat18::Zero();
  for (int i = 0; i < n_terms; ++i) {
    h_track += q_pos_ * (terms[i].jac_v_pad.transpose() * terms[i].jac_v_pad);
    if (terms[i].ori_on && q_ori_ > 0.0) {
      h_track += q_ori_ * (terms[i].jac_w_pad.transpose() * terms[i].jac_w_pad);
    }
  }
  const double e_z = working_ez(terms, n_terms);
  update_pitch_pin(q0, e_z);

  const Vec18 dq_max = effective_dq_max();
  Vec36 lo;
  Vec36 hi;
  endpoint_bounds(
      q0, limits_, dq_max, constraints_.dt(), n_horizon_, q_limit_margin_, lo,
      hi);
  if (idle_begin >= 0 && idle_begin + kNArm <= kNDof) {
    lo.segment(idle_begin, kNArm).setZero();
    hi.segment(idle_begin, kNArm).setZero();
    lo.segment(kNDof + idle_begin, kNArm).setZero();
    hi.segment(kNDof + idle_begin, kNArm).setZero();
  }
  const Vec3 pitch_lo = lo.tail<kNDof>().head<3>();
  const Vec3 pitch_hi = hi.tail<kNDof>().head<3>();

  Mat18 h_vo = Mat18::Zero();
  Vec18 g_vo = Vec18::Zero();
  double sum_lo = 0.0;
  double sum_hi = 0.0;
  pitch_sum_bounds(q0, sum_lo, sum_hi);
  for (int i = 0; i < n_terms; ++i) {
    Mat18 h_i;
    Vec18 g_i;
    assemble_vo(
        terms[i].jac_v, terms[i].e, &pitch_lo, &pitch_hi, e_z, sum_lo, sum_hi,
        &terms[i].jac_w, &terms[i].e_ori, terms[i].ori_on, h_i, g_i);
    h_vo += h_i;
    g_vo += g_i;
  }

  MatIneq A = MatIneq::Zero();
  VecIneq b = VecIneq::Zero();
  int n_ineq = 0;
  Mat18 h_com;
  Vec18 g_com;
  com_terminal(p_com, j_com, e_z, h_com, g_com, A, b, n_ineq);
  Mat18 h_ts;
  Vec18 g_ts;
  torso_synergy_terminal(q0, e_z, h_ts, g_ts);
  pitch_ineq(q0, A, b, n_ineq);
  yaw_ineq(q0, A, b, n_ineq);
  last_n_ineq_ = n_ineq;
  h_vo += h_com + h_ts;
  g_vo += g_com + g_ts;
  h_vo.diagonal().head<4>() += r.head<4>();

  const Mat18 w_stage = path_stage_w(w_diag, q_yaw_, q_syn_, alpha_syn_);
  Vec18 s = Vec18::Constant(s_joint_);
  assemble_endpoints(
      n_horizon_, h_track, w_stage, track_alpha_, h_vo, g_vo, w_o, g_wo, r, s,
      dq_prev_, last_P_, last_g_);
  apply_joint_barrier(q0, r);

  Vec36 y = y_prev_;
  last_qp_status_ = qp_.solve_box_ineq(last_P_, last_g_, lo, hi, A, b, n_ineq, y);
  if (last_qp_status_ != QpStatus::Success) {
    y.setZero();
  }
  y_prev_ = shift_warm_endpoints(y, n_horizon_);
  return y.head<kNDof>();
}

void CartesianTracker::assemble_vo(
    const Mat3x18& jac_f, const Vec3& err, const Vec3* pitch_lo,
    const Vec3* pitch_hi, double e_z, double pitch_sum_lo, double pitch_sum_hi,
    const Mat3x18* jac_omega, const Vec3* e_ori, bool ori_on, Mat18& h_vo,
    Vec18& g_vo) {
  const bool deep = is_deep_drop(e_z);
  const Vec3 e_cap = clamp_arm_residual(err, vo_clamp_m_, vo_clamp_y_m_);
  const Eigen::Matrix<double, 3, 3> j_pitch_raw = jac_f.leftCols<3>();
  Vec3 y_hat = damped_waist_yhat(j_pitch_raw, e_cap, e_z);
  if (pitch_lo != nullptr && pitch_hi != nullptr) {
    y_hat = y_hat.cwiseMax(*pitch_lo).cwiseMin(*pitch_hi);
  }
  y_hat = clip_yhat_pitch_sum(
      y_hat, pitch_sum_lo, pitch_sum_hi, deep,
      deep && pitch_at_lower_wall(last_pitch_));

  Mat3x18 j_w_raw = jac_f;
  j_w_raw.middleCols<kNArm>(kSliceLeft.begin).setZero();
  j_w_raw.middleCols<kNArm>(kSliceRight.begin).setZero();
  j_w_raw.col(kIdxYaw).setZero();
  Vec18 y_w = Vec18::Zero();
  y_w.head<3>() = y_hat;
  const Vec3 e_w = j_w_raw * y_w;
  Vec3 e_tilde_raw = err;
  e_tilde_raw[2] = err[2] - e_w[2];
  const double d_xz = arm_residual_cap(e_w[2], err[2], e_z);
  Mat3x18 j_a = jac_f;
  j_a.leftCols<3>().setZero();
  Vec3 e_tilde = clamp_arm_residual(e_tilde_raw, d_xz, vo_clamp_y_m_);
  e_tilde = jparse_scale_task(j_a, e_tilde, vo_arm_gamma_);
  j_a = pad_tracking_jacobian(j_a, vo_arm_gamma_);

  if (deep) {
    const Vec3 m_hat = waist_yhat_metric(e_z);
    h_vo = Mat18::Zero();
    g_vo = Vec18::Zero();
    for (int i = 0; i < 3; ++i) {
      h_vo(i, i) = m_hat[i];
      g_vo[i] = -m_hat[i] * y_hat[i];
    }
  } else {
    Mat3x18 j_w = j_w_raw;
    j_w.leftCols<3>() = pad_tracking_jacobian(j_w.leftCols<3>().eval(), waist_gamma_);
    h_vo = j_w.transpose() * (qo_ * j_w);
    g_vo = -(j_w.transpose() * (qo_ * e_w));
  }
  h_vo += j_a.transpose() * (qo_ * j_a);
  g_vo -= j_a.transpose() * (qo_ * e_tilde);
  if (ori_on && q_ori_ > 0.0 && jac_omega != nullptr && e_ori != nullptr) {
    Mat3x18 j_aw = *jac_omega;
    j_aw.leftCols<3>().setZero();
    Vec3 e_w_ori = clamp_mag(*e_ori, vo_clamp_ori_rad_);
    e_w_ori = jparse_scale_task(j_aw, e_w_ori, vo_arm_gamma_);
    j_aw = pad_tracking_jacobian(j_aw, vo_arm_gamma_);
    const Mat3 qo_w = q_ori_ * Mat3::Identity();
    h_vo += j_aw.transpose() * (qo_w * j_aw);
    g_vo -= j_aw.transpose() * (qo_w * e_w_ori);
  }
}

void CartesianTracker::com_terminal(
    const Vec3& p_com, const Mat3x18& j_com, double e_z, Mat18& h_com,
    Vec18& g_com, MatIneq& A, VecIneq& b, int& n_ineq) {
  const Mat2x18 j_xy = j_com.topRows<2>();
  Mat2x18 j_xy_cost = j_xy;
  j_xy_cost.leftCols<kNWaist>() =
      pad_tracking_jacobian(j_xy.leftCols<kNWaist>().eval(), waist_gamma_);
  const Eigen::Vector2d p_xy(p_com[0], p_com[1]);
  Eigen::Vector2d c_des = c_sup_;
  last_com_hold_ = com_hold_ez_m_ > 0.0 && std::abs(e_z) >= com_hold_ez_m_;
  if (last_com_hold_) {
    c_des[0] = p_com[0];
  }
  last_c_des_x_ = c_des[0];
  h_com = q_com_ * (j_xy_cost.transpose() * j_xy_cost);
  const Eigen::Vector2d pull = c_des - p_xy;
  g_com = -q_com_ * (j_xy_cost.transpose() * pull);
  last_com_work_xmin_ = guards_.com_work_xmin();
  last_com_work_xmax_ = guards_.com_work_xmax();
  last_com_slack_x_ = last_com_work_xmax_ - p_com[0];
  last_com_work_wall_ = guards_.com_at_work_wall(p_com);
  guards_.stamp_com_both_ends(p_com, j_com, A, b, n_ineq);
}

void CartesianTracker::torso_synergy_terminal(
    const Vec18& q0, double e_z, Mat18& h, Vec18& g) {
  h.setZero();
  g.setZero();
  const double pitch0 = q0[0] + q0[1] + q0[2];
  double theta_des = std::clamp(pitch0, -torso_pitch_max_, torso_pitch_max_);
  if (!torso_rise_hold_ && e_z > 0.0) {
    theta_des = std::clamp(
        std::max(theta_des, torso_k_per_m_ * e_z), -torso_pitch_max_,
        torso_pitch_max_);
  } else if (torso_reach_pitch_follow_ && e_z < 0.0 && !is_deep_drop(e_z)) {
    // 20261015 第二刀：下降方向 θ_des 跟 κ·e_z 走（只许更深、不往回推），
    // 华成实测够点需要俯仰和 -20.2°。深下降钉死语义一行不动。
    theta_des = std::clamp(
        std::min(theta_des, torso_k_per_m_ * e_z), -torso_pitch_max_,
        torso_pitch_max_);
  }
  last_theta_des_ = theta_des;
  if (q_torso_ > 0.0) {
    h += q_torso_ * (j_torso_ * j_torso_.transpose());
    g += (q_torso_ * (pitch0 - theta_des)) * j_torso_;
  }
  if (q_yaw_ > 0.0) {
    const double yaw0 = q0[kIdxYaw] - q_yaw_pref_;
    h(kIdxYaw, kIdxYaw) += q_yaw_;
    g[kIdxYaw] += q_yaw_ * yaw0;
  }
}

void CartesianTracker::apply_joint_barrier(const Vec18& q0, const Vec18& r) {
  // 20261015 第一刀：腕状态屏障。d = 距近侧限位；进警戒区后该轴价格按
  // 分段二次上抬（封顶 w_max_ratio×r_i 保条件数），弹簧打在状态 q0+y 上
  // 往 d_warn 边界回拉；只压 y0 块的 g（执行步），yN 块只加 P 防规划推深。
  // 回拉目标单拍限幅 0.5×cmd_reset_rad，防触发 re-seed 清热启动（红线 §5）。
  if (barrier_alpha_ <= 0.0 || barrier_warn_rad_ <= barrier_floor_rad_) {
    return;
  }
  const double span = barrier_warn_rad_ - barrier_floor_rad_;
  const double pull_max = 0.5 * params_.cmd_reset_rad;
  for (const int idx : barrier_joints_) {
    if (idx < 0 || idx >= kNDof) {
      continue;
    }
    const double d_hi = limits_.q_max[idx] - q0[idx];
    const double d_lo = q0[idx] - limits_.q_min[idx];
    const double d = std::min(d_hi, d_lo);
    if (d >= barrier_warn_rad_) {
      continue;
    }
    const double s = (barrier_warn_rad_ - d) / span;
    const double w_b =
        std::min(barrier_alpha_ * s * s, barrier_w_max_ratio_ * r[idx]);
    const double need = barrier_warn_rad_ - d;
    const double y_pull = std::clamp(
        d_hi < d_lo ? -need : need, -pull_max, pull_max);
    last_P_(idx, idx) += w_b;
    last_P_(kNDof + idx, kNDof + idx) += w_b;
    last_g_(idx) += -w_b * y_pull;
  }
}

double CartesianTracker::working_ez(const JacTerm* terms, int n_terms) {
  double best_n = -1.0;
  double best_z = 0.0;
  for (int i = 0; i < n_terms; ++i) {
    const double nrm = terms[i].e.norm();
    if (nrm > best_n) {
      best_n = nrm;
      best_z = terms[i].e[2];
    }
  }
  return best_z;
}

void CartesianTracker::update_pitch_pin(const Vec18& q0, double e_z) {
  if (!is_deep_drop(e_z)) {
    has_pitch_pin_ = false;
    return;
  }
  if (!has_pitch_pin_) {
    pitch_pin_ = q0[0] + q0[1] + q0[2];
    has_pitch_pin_ = true;
  }
}

void CartesianTracker::pitch_sum_bounds(
    const Vec18& q0, double& lo, double& hi) const {
  guards_.pitch_sum_bounds(q0, lo, hi);
}

void CartesianTracker::pitch_ineq(
    const Vec18& q0, MatIneq& A, VecIneq& b, int& n_ineq) const {
  guards_.stamp_pitch_both_ends(q0, A, b, n_ineq);
}

void CartesianTracker::yaw_ineq(
    const Vec18& q0, MatIneq& A, VecIneq& b, int& n_ineq) const {
  guards_.stamp_yaw_both_ends(q0, A, b, n_ineq);
}

double CartesianTracker::active_norm(
    const Vec6& err, bool has_left, bool has_right) const {
  double best = 0.0;
  bool any = false;
  if (has_left) {
    best = err.head<3>().norm();
    any = true;
  }
  if (has_right) {
    const double n = err.tail<3>().norm();
    best = any ? std::max(best, n) : n;
    any = true;
  }
  return any ? best : 0.0;
}

bool CartesianTracker::one_target_jumped(
    const Vec3* cur, bool has_prev, const Vec3& prev) {
  if (cur == nullptr && !has_prev) {
    return false;
  }
  if (cur == nullptr || !has_prev) {
    return true;
  }
  return (*cur - prev).norm() > kTargetJumpM;
}

bool CartesianTracker::targets_jumped(const Vec3* left, const Vec3* right) {
  const bool jumped =
      one_target_jumped(left, has_prev_left_, prev_left_) ||
      one_target_jumped(right, has_prev_right_, prev_right_);
  has_prev_left_ = left != nullptr;
  has_prev_right_ = right != nullptr;
  if (left != nullptr) {
    prev_left_ = *left;
  }
  if (right != nullptr) {
    prev_right_ = *right;
  }
  return jumped;
}

bool CartesianTracker::ori_jumped(const Vec3& e_left, const Vec3& e_right) {
  const bool jumped =
      (e_left - e_ori_prev_left_).norm() > kTargetJumpOriRad ||
      (e_right - e_ori_prev_right_).norm() > kTargetJumpOriRad;
  e_ori_prev_left_ = e_left;
  e_ori_prev_right_ = e_right;
  return jumped;
}

bool CartesianTracker::ori_arrived_ok(const JacTerm* terms, int n_terms) const {
  if (arrived_ori_rad_ <= 0.0) {
    return true;
  }
  for (int i = 0; i < n_terms; ++i) {
    if (!terms[i].ori_on) {
      continue;
    }
    if (terms[i].e_ori.norm() >= arrived_ori_rad_) {
      return false;
    }
  }
  return true;
}

void CartesianTracker::e0_hist_clear() {
  e0_hist_n_ = 0;
  e0_hist_i_ = 0;
}

void CartesianTracker::e0_hist_append(double e0) {
  e0_hist_[static_cast<std::size_t>(e0_hist_i_)] = e0;
  e0_hist_i_ = (e0_hist_i_ + 1) % e0_hist_cap_;
  if (e0_hist_n_ < e0_hist_cap_) {
    ++e0_hist_n_;
  }
}

bool CartesianTracker::e0_hist_stalled() const {
  if (e0_hist_n_ < e0_hist_cap_ || arrived_stall_m_ <= 0.0) {
    return false;
  }
  const int cap = e0_hist_cap_;
  const int oldest = e0_hist_i_;
  auto at = [&](int k) {
    return e0_hist_[static_cast<std::size_t>((oldest + k) % cap)];
  };
  double sum_abs = 0.0;
  double mn = at(0);
  double mx = at(0);
  for (int k = 0; k < cap; ++k) {
    const double v = at(k);
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    if (k + 1 < cap) {
      sum_abs += std::abs(at(k + 1) - v);
    }
  }
  const double mean_step = sum_abs / static_cast<double>(cap - 1);
  const double span = mx - mn;
  return mean_step < arrived_stall_m_ && span < arrived_stall_m_;
}

Vec18 CartesianTracker::apply_arrived(
    const Vec18& dq, double e0, bool jumped, const Vec18& q0, double e_z,
    bool ori_ok) {
  if (arrived_err_ <= 0.0) {
    last_arrived_ = false;
    return dq;
  }
  if (jumped) {
    last_arrived_ = false;
    e0_hist_clear();
    e0_hist_append(e0);
    e0_prev_ = e0;
    return dq;
  }
  const double improvement = e0_prev_ - e0;
  e0_hist_append(e0);
  bool stalled = false;
  if (e0_hist_n_ >= e0_hist_cap_ && e0 > arrived_err_ && arrived_stall_m_ > 0.0) {
    stalled = e0_hist_stalled();
  }
  if (stalled && stall_blocked_by_wall_slide(q0, e_z, e0)) {
    stalled = false;
  }
  const bool stay_close = last_arrived_ && e0 < arrived_exit_ && ori_ok;
  const bool stay_stall = last_arrived_ && stalled && improvement < 0.002;
  const bool enter_close = (!last_arrived_) && e0 < arrived_err_ && ori_ok;
  const bool enter_stall = (!last_arrived_) && stalled;
  e0_prev_ = e0;
  if (stay_close || stay_stall || enter_close || enter_stall) {
    last_arrived_ = true;
    clear_hold_warm();
    return Vec18::Zero();
  }
  last_arrived_ = false;
  return dq;
}

bool CartesianTracker::stall_blocked_by_wall_slide(
    const Vec18& q0, double e_z, double e0) const {
  if (e0 <= kStallWallE0M) {
    return false;
  }
  if (last_com_work_wall_) {
    return true;
  }
  if (!pitch_at_lower_wall(q0[0] + q0[1] + q0[2])) {
    return false;
  }
  if (std::abs(e_z) <= kStallWallEzM) {
    return false;
  }
  const double slack = q0[1] - limits_.q_min[1];
  return slack > deg_to_rad(kStallJ2SlackDeg);
}

bool CartesianTracker::pitch_at_lower_wall(double pitch0) const {
  const double eps = deg_to_rad(kPitchWallEpsDeg);
  return pitch0 <= (-torso_pitch_max_ + eps);
}

bool CartesianTracker::is_deep_drop(double e_z) {
  refresh_deep_drop(e_z, z_star_);
  return deep_on_;
}

void CartesianTracker::refresh_deep_drop(double e_z, double z_star) {
  const double z_deep = z_home_ - torso_drop_m_;
  if (e_z > 0.0 || z_star > z_deep + kDeepExitM) {
    deep_on_ = false;
  } else if (z_star < z_deep && e_z < 0.0) {
    deep_on_ = true;
  }
}

Vec3 CartesianTracker::damped_waist_yhat(
    const Eigen::Matrix<double, 3, 3>& j_pitch, const Vec3& e_cap, double e_z) {
  const double lam = waist_pred_lambda_;
  if (is_deep_drop(e_z)) {
    const Eigen::Matrix<double, 3, 2> j12 = j_pitch.leftCols<2>();
    const Eigen::Matrix<double, 2, 1> y12 = damped_joint_ls<2>(j12, e_cap, lam);
    return Vec3(y12[0], std::min(y12[1], 0.0), 0.0);
  }
  return damped_pitch_ls(j_pitch, e_cap, lam);
}

Vec3 CartesianTracker::clip_yhat_pitch_sum(
    const Vec3& y_hat, double lo, double hi, bool keep_j3_zero,
    bool dump_j3) const {
  Vec3 out = y_hat;
  const double ps = out[0] + out[1] + out[2];
  double delta = 0.0;
  if (ps < lo) {
    delta = lo - ps;
  } else if (ps > hi) {
    delta = hi - ps;
  } else {
    return out;
  }
  if (keep_j3_zero && dump_j3 && ps < lo) {
    out[1] = std::min(out[1], 0.0);
    out[2] = lo - out[0] - out[1];
    return out;
  }
  if (keep_j3_zero) {
    out[0] += 0.5 * delta;
    out[1] += 0.5 * delta;
    out[2] = 0.0;
  } else {
    out += Vec3::Constant(delta / 3.0);
  }
  return out;
}

Vec3 CartesianTracker::waist_yhat_metric(double e_z) {
  const double m1 = q_offset_;
  if (is_deep_drop(e_z)) {
    const double m3 = pitch_at_lower_wall(last_pitch_) ? m1 : 20.0 * m1;
    return Vec3(m1, 2.0 * m1, m3);
  }
  return Vec3(m1, m1, m1);
}

double CartesianTracker::arm_residual_cap(
    double e_w_z, double e0_z, double e_z) {
  const double d_xz = vo_clamp_m_;
  if (!is_deep_drop(e_z)) {
    return d_xz;
  }
  const double d_floor =
      d_xz > 0.0 ? std::min(vo_arm_floor_m_, d_xz) : 0.0;
  const double eta = std::clamp(
      std::abs(e_w_z) / std::max(std::abs(e0_z), 1e-3), 0.0, 1.0);
  return d_floor + (d_xz - d_floor) * eta;
}

}  // namespace kmpc
