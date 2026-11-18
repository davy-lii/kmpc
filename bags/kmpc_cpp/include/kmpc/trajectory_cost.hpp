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
  // Soft quality terms. Foldback is path-length excess (densify-stable).
  // j4_twist is mean (q4-q0)^2 along the path, not a peak-only pull.
  double j4_twist = 0.0;       // mean (q4-q0)^2; 0 = use foldback only
  double foldback = 0.6;       // excess travel beyond net displacement
  double wrist_foldback = 1.2; // extra on wrist J6/J7
  double wrist_span = 1.0;    // softplus on J6/J7 peak-to-peak (mild in GN)
  double height_coop = 0.4;    // body-drop vs task TCP drop mismatch
  double pitch_near_barrier = 0.6;  // softplus near ±pitch_scale
  double body_forward_len_m = 0.05;
  double pitch_scale_rad = 32.0 * kPi / 180.0;
  double height_coop_len_m = 0.08;
  // Soft comfort span for wrist J6/J7 (offline gate is 0.50 rad on J7).
  double wrist_span_soft_rad = 0.45;
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
  double j4_twist = 0.0;
  double foldback = 0.0;
  double wrist_span = 0.0;
  double height_coop = 0.0;
  double pitch_near_barrier = 0.0;
  double total = 0.0;
  bool limit_outside_hard_box = false;
  bool calibrated = false;
  // Kept so a caller cannot quietly turn the height switch back on.
  bool depends_on_target_height = false;
  double body_dx_m = 0.0;
  double body_dz_m = 0.0;
  double body_minus_leg_dx_m = 0.0;
  // Continuous task geometry input for height_coop (m). Negative = need drop.
  double task_tcp_dz_m = 0.0;
};

// Comparable multi-stage path quality (lower score is better).
struct PathQualityMetrics {
  double score = 1e9;
  double j4_peak_from_start_rad = 0.0;
  double j4_travel_rad = 0.0;
  double wrist_j6_travel_rad = 0.0;
  double wrist_j7_travel_rad = 0.0;
  double wrist_j6_fold_rad = 0.0;
  double wrist_j7_fold_rad = 0.0;
  double wrist_j6_span_rad = 0.0;
  double wrist_j7_span_rad = 0.0;
  double total_travel_rad = 0.0;
  double pitch_peak_abs_rad = 0.0;
  double pitch_min_margin_rad = 0.0;
  double body_dz_m = 0.0;
  double body_dx_m = 0.0;
  double soft_cost = 0.0;
  bool severe_foldback = false;
};

// Whole-path foldback ruler (travel − |net|): GN J_foldback, severe, logs, gtests.
double joint_path_foldback_rad(const JointCurve& samples, int joint_index);

PathQualityMetrics evaluate_path_quality(
    const JointCurve& samples, const Vec18& q0, bool left,
    const MotionConstraintPolicy& policy, const PathCostBreakdown& soft,
    double pitch_cap_rad,
    const std::vector<int>* stage_knot_ends = nullptr);

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
    const Kinematics& kin, bool left, const WholeBodyGuards* guards,
    const std::vector<int>* stage_knot_ends = nullptr);

// Same weights and features as evaluate_path_cost. Linearizes about knots for SQP.
CostLinearization value_and_linearize(
    const JointCurve& knots, const Vec18& q_start, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards,
    const std::vector<int>& active_joints,
    const std::vector<int>* stage_knot_ends = nullptr);

TrajectoryCostWeights default_cost_weights(const Vec18& dq_max);

// Load bags/kmpc_cpp/config/trajectory_quality.yaml (or absolute path).
// On failure returns default_cost_weights and sets *why. Never marks
// calibrated=true unless the yaml says so.
TrajectoryCostWeights load_trajectory_quality_weights(
    const std::string& yaml_path, const Vec18& dq_max, std::string* why);

}  // namespace kmpc
