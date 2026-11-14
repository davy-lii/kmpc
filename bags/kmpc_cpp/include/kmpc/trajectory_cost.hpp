#pragma once
// One posture/motion cost used for solve, score, and logs.

#include <vector>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct TrajectoryCostWeights {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Per-joint scale on Δq, typically 1/dq_max. A slow joint costs more per rad.
  Vec18 motion_scale = Vec18::Ones();
  double move = 1.0;
  double smooth = 1.0;
  double body_forward = 1.0;
  double neutral = 0.1;
  double joint_limit = 1.0;
  double pitch_comfort = 0.5;
  double com_margin = 0.5;
  double arm_manip = 0.25;
  double body_forward_len_m = 0.05;
  double pitch_scale_rad = 32.0 * kPi / 180.0;
  bool calibrated = false;
  std::string calibration_version = "unset";
};

struct PathCostBreakdown {
  double motion = 0.0;
  double smooth = 0.0;
  double body_forward = 0.0;
  double neutral = 0.0;
  double limit = 0.0;
  double pitch = 0.0;
  double com_margin = 0.0;
  double arm_manip = 0.0;
  double total = 0.0;
  bool limit_outside_hard_box = false;
  bool calibrated = false;
  // Kept so a caller cannot quietly turn the height switch back on.
  bool depends_on_target_height = false;
  double body_dx_m = 0.0;
  double body_dz_m = 0.0;
  double body_minus_leg_dx_m = 0.0;
};

struct CostLinearization {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PathCostBreakdown value;
  // Gradient w.r.t. active joints stacked knot-major. Empty if unused.
  std::vector<double> grad;
  // Diagonal Hessian approximation, same layout as grad.
  std::vector<double> hess_diag;
};

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights);

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards);

// Same weights and features as evaluate_path_cost. Linearizes about knots for SQP.
CostLinearization value_and_linearize(
    const JointCurve& knots, const Vec18& q_start, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards,
    const std::vector<int>& active_joints);

TrajectoryCostWeights default_cost_weights(const Vec18& dq_max);

// Load bags/kmpc_cpp/config/trajectory_quality.yaml (or absolute path).
// On failure returns default_cost_weights and sets *why. Never marks
// calibrated=true unless the yaml says so.
TrajectoryCostWeights load_trajectory_quality_weights(
    const std::string& yaml_path, const Vec18& dq_max, std::string* why);

}  // namespace kmpc
