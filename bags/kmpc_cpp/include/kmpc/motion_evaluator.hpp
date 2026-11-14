#pragma once
// One q, then TCP / body / leg / CoM / Jacobians. Does not call CRBA.

#include "kmpc/kinematics.hpp"

namespace kmpc {

struct MotionSnapshot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 tcp = Vec3::Zero();
  Mat3x18 tcp_jac = Mat3x18::Zero();
  Vec3 body = Vec3::Zero();
  Vec3 leg_y = Vec3::Zero();
  Vec3 com = Vec3::Zero();
  Mat3x18 j_com = Mat3x18::Zero();
};

class MotionEvaluator {
 public:
  explicit MotionEvaluator(const Kinematics& kin) : kin_(kin) {}

  void evaluate(const Vec18& q, bool left, MotionSnapshot* out) const;

 private:
  const Kinematics& kin_;
};

}  // namespace kmpc
