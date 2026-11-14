#pragma once
// Multi-stage path + timed trajectory certification.

#include <string>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

enum class PathCoverage {
  DiscreteSamples = 0,
  ConservativeSplineBounds = 1,
  AdaptiveSampledWithOnlineGuard = 2,
};

struct TrajectoryCertificate {
  bool passed = false;
  bool hard_box_ok = false;
  bool pitch_ok = false;
  bool com_checked = false;
  bool com_ok = false;
  bool task_path_ok = false;
  bool continuous_certified = false;
  PathCoverage coverage = PathCoverage::DiscreteSamples;
  double min_joint_margin_rad = 0.0;
  double min_pitch_margin_rad = 0.0;
  double worst_tcp_cross_track_m = 0.0;
  double worst_tcp_endpoint_m = 0.0;
  const char* first_fault = "not_checked";
  // What CoM check this certificate actually ran.
  const char* com_model = "not_checked";
  std::string coverage_note = "sampled_only";
};

// guards == nullptr is not a pass. CoM is then "not checked", not "inside".
TrajectoryCertificate validate_discrete_trajectory(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards);

// Multi-stage geometry validation on the spline samples that will execute.
TrajectoryCertificate validate_path(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards);

// Re-check the timed package that execution will interpolate.
TrajectoryCertificate validate_actual_q_of_t(
    const TrajectoryPackage& package, const MotionTask& task,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards);

}  // namespace kmpc
