// Safety 永远是出口裁剪；基础限值来自构造时那份 JointVelocityLimits。

#include "kmpc/safety.hpp"

#include <cmath>

namespace kmpc {

SafetyFilter::SafetyFilter(const JointVelocityLimits& limits, double dt, double margin_rad)
    : limits_(limits), dt_(dt), margin_rad_(margin_rad > 0.0 ? margin_rad : 0.0) {}

namespace {

void box_limits(
    const JointVelocityLimits& limits, double margin, Vec18& lo, Vec18& hi) {
  lo = limits.q_min;
  hi = limits.q_max;
  if (!(margin > 0.0)) {
    return;
  }
  for (int i = 0; i < kNDof; ++i) {
    const double inner_lo = limits.q_min[i] + margin;
    const double inner_hi = limits.q_max[i] - margin;
    if (inner_hi > inner_lo) {
      lo[i] = inner_lo;
      hi[i] = inner_hi;
    }
  }
}

}  // namespace

void SafetyFilter::seed(const Vec18& q) {
  Vec18 lo;
  Vec18 hi;
  box_limits(limits_, margin_rad_, lo, hi);
  last_cmd_ = q.allFinite() ? q.cwiseMax(lo).cwiseMin(hi) : q;
  has_last_ = true;
}

bool SafetyFilter::diverged(const Vec18& q_meas, double tol) const {
  if (!has_last_) {
    return false;
  }
  return (q_meas - last_cmd_).cwiseAbs().maxCoeff() > tol;
}

Vec18 SafetyFilter::clamp(const Vec18& q_cmd) {
  return clamp_impl(q_cmd, nullptr, nullptr);
}

Vec18 SafetyFilter::clamp(const Vec18& q_cmd, const Vec18& q0) {
  return clamp_impl(q_cmd, &q0, nullptr);
}

Vec18 SafetyFilter::clamp(const Vec18& q_cmd, const Vec18& q0, const Vec18& dq_max) {
  return clamp_impl(q_cmd, &q0, &dq_max);
}

Vec18 SafetyFilter::failsafe() const {
  if (!has_last_) {
    return Vec18::Zero();
  }
  return last_cmd_;
}

Vec18 SafetyFilter::clamp_impl(
    const Vec18& q_cmd, const Vec18* q0, const Vec18* dq_max) {
  if (!q_cmd.allFinite()) {
    return failsafe();
  }
  Vec18 lo;
  Vec18 hi;
  box_limits(limits_, margin_rad_, lo, hi);
  Vec18 q = q_cmd.cwiseMax(lo).cwiseMin(hi);
  if (q0 != nullptr) {
    const Vec18& vmax = dq_max != nullptr ? *dq_max : limits_.dq_max;
    const Vec18 max_step = vmax * dt_;
    q = q.cwiseMax(*q0 - max_step).cwiseMin(*q0 + max_step);
    q = q.cwiseMax(lo).cwiseMin(hi);
  }
  last_cmd_ = q;
  has_last_ = true;
  return q;
}

}  // namespace kmpc
