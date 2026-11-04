// 与 cartesian_tracker 抽出前 pitch_sum_bounds / com_terminal / yaw_ineq 同式。

#include "kmpc/whole_body_guards.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace kmpc {

void WholeBodyGuards::add_row36(
    MatIneq& A, VecIneq& b, int& n, const Vec36& row, double rhs) {
  if (n >= kMaxIneq) {
    std::fprintf(stderr, "kmpc Guards: inequality overflow\n");
    return;
  }
  A.row(n) = row.transpose();
  b[n] = rhs;
  ++n;
}

void WholeBodyGuards::add_row18(
    MatIneq18& A, VecIneq& b, int& n, const Vec18& a, double rhs) {
  if (n >= kMaxIneq) {
    std::fprintf(stderr, "kmpc Guards: inequality overflow\n");
    return;
  }
  A.row(n) = a.transpose();
  b[n] = rhs;
  ++n;
}

WholeBodyGuards::WholeBodyGuards(const KmpcParams& params, double yaw_pref)
    : q_yaw_pref_(yaw_pref) {
  j_torso_.head<3>().setOnes();
  torso_pitch_max_ = deg_to_rad(params.torso_pitch_max_deg);
  yaw_max_ = deg_to_rad(params.yaw_max_deg);
  com_margin_ = params.com_margin;
  com_xmin_ = params.com_xmin;
  com_xmax_ = params.com_xmax;
  com_ymin_ = params.com_ymin;
  com_ymax_ = params.com_ymax;
  com_work_m_ = std::max(params.com_work_m, 0.0);
}

void WholeBodyGuards::pitch_sum_bounds(
    const Vec18& q0, double& lo, double& hi) const {
  const double pitch0 = pitch(q0);
  lo = std::min(-torso_pitch_max_ - pitch0, 0.0);
  hi = std::max(torso_pitch_max_ - pitch0, 0.0);
}

void WholeBodyGuards::pitch_toward_bounds(
    const Vec18& q0, double theta_star, double& lo, double& hi) const {
  double lo15 = 0.0;
  double hi15 = 0.0;
  pitch_sum_bounds(q0, lo15, hi15);
  const double pitch0 = pitch(q0);
  const double th_lo =
      std::max(-torso_pitch_max_, std::min(pitch0, theta_star));
  const double th_hi =
      std::min(torso_pitch_max_, std::max(pitch0, theta_star));
  const double dlo = std::min(th_lo - pitch0, 0.0);
  const double dhi = std::max(th_hi - pitch0, 0.0);
  lo = std::max(lo15, dlo);
  hi = std::min(hi15, dhi);
}

void WholeBodyGuards::append_pitch(
    const Vec18& q0, MatIneq18& A, VecIneq& b, int& n) const {
  if (torso_pitch_max_ <= 0.0) {
    return;
  }
  double lo = 0.0;
  double hi = 0.0;
  pitch_sum_bounds(q0, lo, hi);
  add_row18(A, b, n, j_torso_, lo);
  add_row18(A, b, n, (-j_torso_).eval(), -hi);
}

void WholeBodyGuards::append_pitch_toward(
    const Vec18& q0, double theta_star, MatIneq18& A, VecIneq& b,
    int& n) const {
  if (torso_pitch_max_ <= 0.0) {
    return;
  }
  double lo = 0.0;
  double hi = 0.0;
  pitch_toward_bounds(q0, theta_star, lo, hi);
  add_row18(A, b, n, j_torso_, lo);
  add_row18(A, b, n, (-j_torso_).eval(), -hi);
}

void WholeBodyGuards::append_yaw(
    const Vec18& q0, MatIneq18& A, VecIneq& b, int& n) const {
  if (yaw_max_ <= 0.0) {
    return;
  }
  const double yaw0 = q0[kIdxYaw] - q_yaw_pref_;
  const double lo = std::min(-yaw_max_ - yaw0, 0.0);
  const double hi = std::max(yaw_max_ - yaw0, 0.0);
  Vec18 j = Vec18::Zero();
  j[kIdxYaw] = 1.0;
  add_row18(A, b, n, j, lo);
  add_row18(A, b, n, (-j).eval(), -hi);
}

void WholeBodyGuards::append_com(
    const Vec3& p_com, const Mat3x18& j_com, MatIneq18& A, VecIneq& b,
    int& n) const {
  const Mat2x18 j_xy = j_com.topRows<2>();
  Eigen::Vector2d dlo = Eigen::Vector2d::Zero();
  Eigen::Vector2d dhi = Eigen::Vector2d::Zero();
  com_step_bounds(p_com, dlo, dhi);
  for (int i = 0; i < 2; ++i) {
    add_row18(A, b, n, j_xy.row(i).transpose(), dlo[i]);
    add_row18(A, b, n, (-j_xy.row(i).transpose()).eval(), -dhi[i]);
  }
}

void WholeBodyGuards::append_elbow(
    const Vec18& /*q0*/, MatIneq18& /*A*/, VecIneq& /*b*/, int& /*n*/) const {
  // 肘下墙已取消。关节目标只受 joint_limits 的 J4（-140°～57°）约束。
}

void WholeBodyGuards::stamp_both_ends(
    MatIneq& A, VecIneq& b, int& n, const Vec18& a, double lo, double hi) const {
  Vec36 r0 = Vec36::Zero();
  r0.head<kNDof>() = a;
  Vec36 rN = Vec36::Zero();
  rN.tail<kNDof>() = a;
  add_row36(A, b, n, r0, lo);
  add_row36(A, b, n, (-r0).eval(), -hi);
  add_row36(A, b, n, rN, lo);
  add_row36(A, b, n, (-rN).eval(), -hi);
}

void WholeBodyGuards::stamp_pitch_both_ends(
    const Vec18& q0, MatIneq& A, VecIneq& b, int& n) const {
  if (torso_pitch_max_ <= 0.0) {
    return;
  }
  double lo = 0.0;
  double hi = 0.0;
  pitch_sum_bounds(q0, lo, hi);
  stamp_both_ends(A, b, n, j_torso_, lo, hi);
}

void WholeBodyGuards::stamp_yaw_both_ends(
    const Vec18& q0, MatIneq& A, VecIneq& b, int& n) const {
  if (yaw_max_ <= 0.0) {
    return;
  }
  const double yaw0 = q0[kIdxYaw] - q_yaw_pref_;
  const double lo = std::min(-yaw_max_ - yaw0, 0.0);
  const double hi = std::max(yaw_max_ - yaw0, 0.0);
  Vec18 j = Vec18::Zero();
  j[kIdxYaw] = 1.0;
  stamp_both_ends(A, b, n, j, lo, hi);
}

void WholeBodyGuards::stamp_com_both_ends(
    const Vec3& p_com, const Mat3x18& j_com, MatIneq& A, VecIneq& b,
    int& n) const {
  const Mat2x18 j_xy = j_com.topRows<2>();
  Eigen::Vector2d dlo = Eigen::Vector2d::Zero();
  Eigen::Vector2d dhi = Eigen::Vector2d::Zero();
  com_step_bounds(p_com, dlo, dhi);
  for (int i = 0; i < 2; ++i) {
    stamp_both_ends(A, b, n, j_xy.row(i).transpose(), dlo[i], dhi[i]);
  }
}

double WholeBodyGuards::com_work_xmin() const {
  Eigen::Vector2d lo = Eigen::Vector2d::Zero();
  Eigen::Vector2d hi = Eigen::Vector2d::Zero();
  com_work_xy(lo, hi);
  return lo[0];
}

double WholeBodyGuards::com_work_xmax() const {
  Eigen::Vector2d lo = Eigen::Vector2d::Zero();
  Eigen::Vector2d hi = Eigen::Vector2d::Zero();
  com_work_xy(lo, hi);
  return hi[0];
}

double WholeBodyGuards::com_work_ymin() const {
  Eigen::Vector2d lo = Eigen::Vector2d::Zero();
  Eigen::Vector2d hi = Eigen::Vector2d::Zero();
  com_work_xy(lo, hi);
  return lo[1];
}

double WholeBodyGuards::com_work_ymax() const {
  Eigen::Vector2d lo = Eigen::Vector2d::Zero();
  Eigen::Vector2d hi = Eigen::Vector2d::Zero();
  com_work_xy(lo, hi);
  return hi[1];
}

void WholeBodyGuards::com_work_xy(
    Eigen::Vector2d& lo, Eigen::Vector2d& hi) const {
  lo[0] = com_xmin_ + com_margin_ + com_work_m_;
  lo[1] = com_ymin_ + com_margin_ + com_work_m_;
  hi[0] = com_xmax_ - com_margin_ - com_work_m_;
  hi[1] = com_ymax_ - com_margin_ - com_work_m_;
}

void WholeBodyGuards::com_step_bounds(
    const Vec3& p_com, Eigen::Vector2d& dlo, Eigen::Vector2d& dhi) const {
  Eigen::Vector2d lo = Eigen::Vector2d::Zero();
  Eigen::Vector2d hi = Eigen::Vector2d::Zero();
  com_work_xy(lo, hi);
  for (int i = 0; i < 2; ++i) {
    dlo[i] = std::min(lo[i] - p_com[i], 0.0);
    dhi[i] = std::max(hi[i] - p_com[i], 0.0);
  }
}

bool WholeBodyGuards::com_at_work_wall(const Vec3& p_com) const {
  Eigen::Vector2d dlo = Eigen::Vector2d::Zero();
  Eigen::Vector2d dhi = Eigen::Vector2d::Zero();
  com_step_bounds(p_com, dlo, dhi);
  return dhi[0] <= 0.0 || dlo[0] >= 0.0;
}

bool WholeBodyGuards::com_xy_inside(const Vec3& p_com) const {
  const double lo_x = com_xmin_ + com_margin_;
  const double hi_x = com_xmax_ - com_margin_;
  const double lo_y = com_ymin_ + com_margin_;
  const double hi_y = com_ymax_ - com_margin_;
  return p_com[0] >= lo_x && p_com[0] <= hi_x && p_com[1] >= lo_y &&
         p_com[1] <= hi_y;
}

Vec18 WholeBodyGuards::project_q_des(const Vec18& q_raw, QDesFlags& flags) const {
  Vec18 q = q_raw;
  flags = QDesFlags{};
  if (torso_pitch_max_ > 0.0) {
    const double th = pitch(q);
    if (th > torso_pitch_max_ || th < -torso_pitch_max_) {
      const double target =
          std::clamp(th, -torso_pitch_max_, torso_pitch_max_);
      const double d = (target - th) / 3.0;
      q[0] += d;
      q[1] += d;
      q[2] += d;
      flags.pitch_clip = true;
    }
  }
  flags.projected = flags.pitch_clip;
  return q;
}

}  // namespace kmpc
