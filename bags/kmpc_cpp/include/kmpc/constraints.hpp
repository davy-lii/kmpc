#pragma once
// 对应 Python kmpc/constraint_builder.py；对照 09-04 腰锁只改盒。
// 腰锁只覆盖 dq_max 副本，禁止写 P。限值来自 Config::limits，不再读 yaml。

#include "kmpc/config.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct DqBox {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 lb = Vec18::Zero();
  Vec18 ub = Vec18::Zero();
};

class ConstraintBuilder {
 public:
  ConstraintBuilder(const JointVelocityLimits& limits, double dt);

  const JointVelocityLimits& limits() const { return limits_; }
  double dt() const { return dt_; }

  // 与 Python box_dq 同式。dq_max=nullptr 用 yaml；腰锁传入 effective_dq_max。
  DqBox box_dq(const Vec18& q0) const;
  DqBox box_dq(const Vec18& q0, const Vec18& dq_max) const;

 private:
  DqBox box_dq_impl(const Vec18& q0, const Vec18& vmax) const;

  const JointVelocityLimits& limits_;
  double dt_;
};

}  // namespace kmpc
