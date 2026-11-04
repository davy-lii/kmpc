#pragma once
// 对应 Python kmpc/safety_filter.py。最终裁剪层；限值不得自己再读 yaml。
// dq_max / q_min / q_max 与 QP 同一 JointVelocityLimits。腰锁覆盖走 optional dq_max。

#include "kmpc/config.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

class SafetyFilter {
 public:
  // margin_rad>0 时夹到硬限位内侧。0 只夹硬限位。
  SafetyFilter(const JointVelocityLimits& limits, double dt, double margin_rad = 0.0);

  void seed(const Vec18& q);
  bool has_last() const { return has_last_; }
  Vec18 last_cmd() const { return last_cmd_; }

  // |q_meas - last_cmd|_∞ > tol → 需要 re-seed（对照 cmd_reset_rad）。
  bool diverged(const Vec18& q_meas, double tol) const;

  // dq_max=nullptr 用 yaml；与 QP 同拍腰锁时传入同一份 effective_dq_max。
  Vec18 clamp(const Vec18& q_cmd);
  Vec18 clamp(const Vec18& q_cmd, const Vec18& q0);
  Vec18 clamp(const Vec18& q_cmd, const Vec18& q0, const Vec18& dq_max);

  Vec18 failsafe() const;

 private:
  Vec18 clamp_impl(const Vec18& q_cmd, const Vec18* q0, const Vec18* dq_max);

  const JointVelocityLimits& limits_;
  double dt_;
  double margin_rad_;
  Vec18 last_cmd_ = Vec18::Zero();
  bool has_last_ = false;
};

}  // namespace kmpc
