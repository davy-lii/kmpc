#pragma once
// unified_multistage_v2 offline/runtime optimizer.
// Multi-stage knots (K<=16), active 11 DoF (waist + task arm), Gauss-Newton
// iterations whose subproblem carries hard joint box / pitch / TCP stage
// target rows. The solved knots are never re-posed afterwards: densify is
// joint interpolation only, and the certified package is what validation
// sees. No direction sign hard gate, no smooth_reversal patch.

#include <string>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/trajectory_validator.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct FastPlanResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool ok = false;
  std::string reason;
  std::string report;
  TrajectoryPackage package;
  TrajectoryCertificate certificate;
  int iterations = 0;
  double optimize_ms = 0.0;
  double certify_ms = 0.0;
  double tcp_worst_m = 0.0;
  double min_hard_margin_rad = 0.0;
  double body_dx_m = 0.0;
  double body_dz_m = 0.0;
  double body_minus_leg_dx_m = 0.0;
  double duration_s = 0.0;
  bool deadline_hit = false;
  bool software_limited = true;
  // PlannerWorker submit sequence that produced this result.
  uint64_t submit_seq = 0;
  uint64_t task_digest = 0;
  uint64_t model_digest = 0;
  uint64_t limits_digest = 0;
  uint64_t generation = 0;
};

class FastTrajectoryOptimizer {
 public:
  // guards == nullptr is allowed, but the certificate then cannot pass
  // (CoM reports not_checked), so ok is false by construction.
  FastTrajectoryOptimizer(const Kinematics& kin, const WholeBodyGuards* guards)
      : kin_(kin), guards_(guards) {}

  // deadline_ms <= 0 means no wall-clock deadline.
  FastPlanResult plan(const MotionTask& task, const RobotState& state,
                      const MotionConstraintPolicy& policy,
                      const TrajectoryCostWeights& weights,
                      double deadline_ms) const;

 private:
  const Kinematics& kin_;
  const WholeBodyGuards* guards_;
};

}  // namespace kmpc
