#pragma once
// 对应 Python kmpc/horizon_qp.py 的 y_bounds / assemble_endpoints。
// 对照 09-04：x=(y0, y_{N-1})∈R^{36}；路径只在 k=0；W_syn 走 path_stage_w，全程开。

#include "kmpc/config.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct YStageBox {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 lo = Vec18::Zero();
  Vec18 hi = Vec18::Zero();
};

YStageBox y_stage_bounds(
    const Vec18& q0,
    const JointVelocityLimits& limits,
    const Vec18& dq_max,
    double dt,
    int k,
    double margin);

void endpoint_bounds(
    const Vec18& q0,
    const JointVelocityLimits& limits,
    const Vec18& dq_max,
    double dt,
    int n_horizon,
    double margin,
    Vec36& lb,
    Vec36& ub);

// 路径度量 diag(W)+Q_yaw 在 J4，加上 Q_syn H_syn^T H_syn。e_z 不关 W_syn。
Mat18 path_stage_w(
    const Vec18& w_diag,
    double q_yaw,
    double q_syn,
    double alpha_syn);

// 0.5 x^T P x + g^T x，x=(y0,y_{N-1})。与 assemble_endpoints 同式同顺序。
// w_stage 可以是对角（传入 diag 矩阵）或已含 W_syn 的满矩阵。
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
    Vec36& g);

Vec36 shift_warm_endpoints(const Vec36& y, int n_horizon);

}  // namespace kmpc
