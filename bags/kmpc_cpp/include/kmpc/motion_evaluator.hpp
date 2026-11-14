#pragma once
// One q, then TCP / body / leg / CoM / Jacobians. Does not call CRBA.

#include <string_view>
#include <vector>

#include "kmpc/kinematics.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct MotionSnapshot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 tcp = Vec3::Zero();
  Mat3x18 tcp_jac = Mat3x18::Zero();
  Vec3 body = Vec3::Zero();
  Mat3x18 body_jac = Mat3x18::Zero();
  Vec3 leg_y = Vec3::Zero();
  Mat3x18 leg_jac = Mat3x18::Zero();
  Vec3 com = Vec3::Zero();
  Mat3x18 j_com = Mat3x18::Zero();
  double body_minus_leg_x = 0.0;
  double pitch = 0.0;
};

class MotionEvaluator {
 public:
  explicit MotionEvaluator(const Kinematics& kin) : kin_(kin) {}

  void evaluate(const Vec18& q, bool left, MotionSnapshot* out) const;

  void batch(
      const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& knots, bool left,
      std::vector<MotionSnapshot, Eigen::aligned_allocator<MotionSnapshot>>* out)
      const;

 private:
  const Kinematics& kin_;
};

}  // namespace kmpc
