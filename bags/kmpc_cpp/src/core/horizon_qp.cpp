// 只抄 assemble_endpoints / y_bounds；禁止 270 维 assemble。

#include "kmpc/horizon_qp.hpp"

#include "kmpc/config.hpp"

namespace kmpc {
namespace {

void joint_soft_box(
    const Vec18& q0,
    const JointVelocityLimits& limits,
    double margin,
    Vec18& joint_lo,
    Vec18& joint_hi) {
  const Vec18 hard_lo = limits.q_min - q0;
  const Vec18 hard_hi = limits.q_max - q0;
  const double m = margin > 0.0 ? margin : 0.0;
  const Vec18 soft_lo = (limits.q_min.array() + m - q0.array()).min(0.0).matrix();
  const Vec18 soft_hi = (limits.q_max.array() - m - q0.array()).max(0.0).matrix();
  joint_lo = soft_lo.cwiseMax(hard_lo);
  joint_hi = soft_hi.cwiseMin(hard_hi);
  joint_lo = joint_lo.cwiseMin(joint_hi);
}

}  // namespace

YStageBox y_stage_bounds(
    const Vec18& q0,
    const JointVelocityLimits& limits,
    const Vec18& dq_max,
    double dt,
    int k,
    double margin) {
  Vec18 joint_lo;
  Vec18 joint_hi;
  joint_soft_box(q0, limits, margin, joint_lo, joint_hi);
  const Vec18 vdt = dq_max * dt;
  const double travel = static_cast<double>(k + 1);
  YStageBox box;
  box.lo = joint_lo.cwiseMax((-vdt) * travel);
  box.hi = joint_hi.cwiseMin(vdt * travel);
  box.lo = box.lo.cwiseMin(box.hi);
  return box;
}

void endpoint_bounds(
    const Vec18& q0,
    const JointVelocityLimits& limits,
    const Vec18& dq_max,
    double dt,
    int n_horizon,
    double margin,
    Vec36& lb,
    Vec36& ub) {
  const int n = n_horizon < 1 ? 1 : n_horizon;
  const YStageBox y0 = y_stage_bounds(q0, limits, dq_max, dt, 0, margin);
  const YStageBox ya = y_stage_bounds(q0, limits, dq_max, dt, n - 1, margin);
  lb.head<kNDof>() = y0.lo;
  ub.head<kNDof>() = y0.hi;
  lb.tail<kNDof>() = ya.lo;
  ub.tail<kNDof>() = ya.hi;
}

Mat18 path_stage_w(
    const Vec18& w_diag,
    double q_yaw,
    double q_syn,
    double alpha_syn) {
  Mat18 w = w_diag.asDiagonal();
  if (q_yaw > 0.0) {
    w(3, 3) += q_yaw;
  }
  if (q_syn <= 0.0) {
    return w;
  }
  Vec18 h_syn = Vec18::Zero();
  h_syn[0] = alpha_syn;
  h_syn[1] = 1.0;
  w += q_syn * (h_syn * h_syn.transpose());
  return w;
}

void assemble_endpoints(
    int n_horizon,
    const Mat18& h_track,
    const Mat18& w_stage,
    double beta0,
    const Mat18& h_vo,
    const Vec18& g_vo,
    const Vec18& w_o,
    const Vec18& g_wo,
    const Vec18& r,
    const Vec18& s,
    const Vec18& dq_prev,
    Mat36& P,
    Vec36& g) {
  P.setZero();
  g.setZero();
  const Mat18 w_d = 0.5 * (w_stage + w_stage.transpose());
  const Mat18 h_w = h_track + w_d;
  const double b = beta0;
  P.block<kNDof, kNDof>(0, 0) += h_w;
  P.block<kNDof, kNDof>(kNDof, kNDof) += (b * b) * h_w;
  const Mat18 cross = (-b) * h_w;
  P.block<kNDof, kNDof>(0, kNDof) += cross;
  P.block<kNDof, kNDof>(kNDof, 0) += cross;

  P.block<kNDof, kNDof>(kNDof, kNDof) += h_vo;
  P.block<kNDof, kNDof>(kNDof, kNDof).diagonal() += w_o;
  g.segment<kNDof>(kNDof) += g_vo;
  g.segment<kNDof>(kNDof) += g_wo;

  P.block<kNDof, kNDof>(0, 0).diagonal() += r + s;
  g.head<kNDof>() -= s.cwiseProduct(dq_prev);

  const int n_h = n_horizon;
  if (n_h > 1) {
    Mat18 rch = Mat18::Zero();
    rch.diagonal() = r / static_cast<double>(n_h - 1);
    P.block<kNDof, kNDof>(0, 0) += rch;
    P.block<kNDof, kNDof>(kNDof, kNDof) += rch;
    P.block<kNDof, kNDof>(0, kNDof) -= rch;
    P.block<kNDof, kNDof>(kNDof, 0) -= rch;
  }
  P.diagonal().array() += kHessRidge;
  // 腰锁 / dq_max 不进本函数：盒在 endpoint_bounds，对照 09-04。
}

Vec36 shift_warm_endpoints(const Vec36& y, int n_horizon) {
  const Vec18 remain = y.tail<kNDof>() - y.head<kNDof>();
  const int n_h = n_horizon < 2 ? 2 : n_horizon;
  Vec36 out = Vec36::Zero();
  out.head<kNDof>() = remain / static_cast<double>(n_h - 1);
  out.tail<kNDof>() = remain;
  return out;
}

}  // namespace kmpc
