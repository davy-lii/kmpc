// 对应 Python constraint_builder.box_dq。N=1 走这一盒；N>1 终端盒见 y_bounds。

#include "kmpc/constraints.hpp"

#include <stdexcept>

namespace kmpc {

ConstraintBuilder::ConstraintBuilder(const JointVelocityLimits& limits, double dt)
    : limits_(limits), dt_(dt) {
  if (dt_ <= 0.0) {
    throw ConfigError("dt must be > 0");
  }
}

DqBox ConstraintBuilder::box_dq(const Vec18& q0) const {
  return box_dq_impl(q0, limits_.dq_max);
}

DqBox ConstraintBuilder::box_dq(const Vec18& q0, const Vec18& dq_max) const {
  return box_dq_impl(q0, dq_max);
}

DqBox ConstraintBuilder::box_dq_impl(const Vec18& q0, const Vec18& vmax) const {
  const Vec18 max_step = vmax * dt_;
  DqBox box;
  box.lb = (limits_.q_min - q0).cwiseMax(-max_step);
  box.ub = (limits_.q_max - q0).cwiseMin(max_step);
  // q0 已越界时仍给一个可行盒（Python: lb = min(lb, ub)）。
  box.lb = box.lb.cwiseMin(box.ub);
  return box;
}

}  // namespace kmpc
