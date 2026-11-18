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
  bool work_box_ok = false;
  bool yaw_task_ok = true;
  bool j4_stage_mono_ok = true;
  bool pitch_ok = false;
  bool com_checked = false;
  bool com_ok = false;
  bool task_path_ok = false;
  bool continuous_certified = false;
  PathCoverage coverage = PathCoverage::DiscreteSamples;
  double min_joint_margin_rad = 0.0;
  double min_work_margin_rad = 0.0;
  double min_pitch_margin_rad = 0.0;
  double worst_tcp_cross_track_m = 0.0;
  double worst_tcp_endpoint_m = 0.0;
  const char* first_fault = "not_checked";
  // What CoM check this certificate actually ran.
  const char* com_model = "not_checked";
  std::string coverage_note = "sampled_only";
  // Populated on tcp_cross_track / related path faults.
  int tcp_fault_stage = -1;
  double tcp_fault_t_s = -1.0;
  double tcp_fault_s = -1.0;
  // "knot_samples" | "hermite_exec" | "" 
  const char* tcp_fault_source = "";
  std::string fault_detail;
};

// guards == nullptr is not a pass. CoM is then "not checked", not "inside".
TrajectoryCertificate validate_discrete_trajectory(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards,
    const std::vector<StageBoundary>* stage_boundaries = nullptr);

// Multi-stage geometry validation on the densified samples.
// When stage_boundaries is set, each sample is checked against its own stage
// chord (StageBoundary ranges) instead of the nearest polyline segment.
TrajectoryCertificate validate_path(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards,
    const std::vector<StageBoundary>* stage_boundaries = nullptr);

// Re-check the timed package that execution will interpolate, including
// Hermite q(t) TCP cross-track on the 250 Hz send grid.
TrajectoryCertificate validate_actual_q_of_t(
    const TrajectoryPackage& package, const MotionTask& task,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards);

// Sample index → stage for a densified point (endpoint at boundary b stays
// with the ending stage).
std::size_t stage_for_sample_point(
    std::size_t sample_i, std::size_t nstage,
    const std::vector<StageBoundary>* boundaries);

// Span between sample i and i+1 belongs to the stage of the right endpoint.
std::size_t stage_for_sample_span(
    std::size_t span_left_i, std::size_t nstage,
    const std::vector<StageBoundary>* boundaries);

// Continuous progress r∈[0,1] on densified samples → owning stage.
std::size_t stage_for_curve_progress(
    double s, std::size_t n_samples, std::size_t nstage,
    const std::vector<StageBoundary>* boundaries);

}  // namespace kmpc
