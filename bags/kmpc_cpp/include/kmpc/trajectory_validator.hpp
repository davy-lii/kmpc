#pragma once
// Discrete checker. continuous_certified stays false until a bound exists.

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

enum class PathCoverage {
  DiscreteSamples = 0,
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
};

// guards == nullptr is not a pass. CoM is then "not checked", not "inside".
// One stage only. FollowCurve and orientation return an explicit fault.
TrajectoryCertificate validate_discrete_trajectory(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards);

}  // namespace kmpc
