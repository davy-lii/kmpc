#pragma once
// Production pose selector. One cost, real FK/Jacobian, known stages only.
// A missing later stage is not invented.

#include <string>
#include <vector>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct UnifiedStageGoal {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 xyz = Vec3::Zero();
  Vec4 quat = Vec4(0, 0, 0, 1);
  bool ori_on = false;
};

struct UnifiedPlanRequest {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q0 = Vec18::Zero();
  Vec18 q_neutral = Vec18::Zero();
  Vec18 dq_max = Vec18::Ones();
  bool left = true;
  bool waist_locked = false;
  std::string frame;
  UnifiedStageGoal goal;
  bool has_next = false;
  UnifiedStageGoal next;
  MotionConstraintPolicy policy;
  double pos_tol_m = 0.003;
  double sample_max_m = 0.005;
  int max_samples = 400;
};

struct UnifiedPlanResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool ok = false;
  std::string reason;
  std::string report;
  JointCurve segment;
  JointCurve successor;
  bool has_successor = false;
  DirectionContract contract{};
  DirectionCheckResult direction{};
  DirectionContract successor_contract{};
  DirectionCheckResult successor_direction{};
  int iterations = 0;
  double optimize_ms = 0.0;
  double certify_ms = 0.0;
  double tcp_worst_m = 0.0;
  double min_hard_margin_rad = 0.0;
  double body_drop_m = 0.0;
  double waist_tcp_m = 0.0;
  double arm_tcp_m = 0.0;
  uint32_t path_id = 0;
};

UnifiedPlanResult plan_unified_multistage(
    const Kinematics& kin, const WholeBodyGuards& guards,
    const UnifiedPlanRequest& req);

// Replacing samples.front() is a new curve. This does not rewrite it.
// False means the caller must drop the package and plan again from q_meas.
bool recertify_anchored_start(
    const Kinematics& kin, const WholeBodyGuards& guards,
    const MotionConstraintPolicy& policy, const JointCurve& samples,
    const Vec18& q_meas, const Vec3& target, const std::string& frame,
    double pos_tol_m, std::string* why);

}  // namespace kmpc
