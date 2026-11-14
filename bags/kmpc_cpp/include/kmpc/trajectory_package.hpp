#pragma once
// Immutable certified trajectory package. Never rewrite samples in place.

#include <cstdint>
#include <string>
#include <vector>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

using JointCurve = std::vector<Vec18, Eigen::aligned_allocator<Vec18>>;

struct StageBoundary {
  std::size_t sample_index = 0;
  uint32_t stage_id = 0;
  bool stop_required = false;
  bool wait_for_external_ack = false;
};

struct TimeLawTable {
  // Monotone s in [0,1] and wall time samples for the certified curve.
  std::vector<double> s;
  std::vector<double> t_s;
  // Optional node rates ds/dt (same length as s). Empty → inferred from t_s/s.
  std::vector<double> ds_dt;
  double duration_s = 0.0;
  bool software_limited = true;
  std::string accel_source = "software_limited";
  std::string jerk_source = "software_limited";
};

// Actual executable sample at wall time along the certified package.
// q is Hermite (C1) on geometric knots; dq/ddq from time-consistent derivatives.
struct TrajectoryTimeSample {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q = Vec18::Zero();
  Vec18 dq = Vec18::Zero();
  Vec18 ddq = Vec18::Zero();
  double s = 0.0;
  double t_s = 0.0;
  bool ok = false;
  const char* reason = "not_sampled";
};

struct TrajectoryCertificateSummary {
  bool passed = false;
  bool continuous_certified = false;
  bool sampled_only = true;
  std::string coverage = "sampled_only";
  double min_joint_margin_rad = 0.0;
  double min_pitch_margin_rad = 0.0;
  double worst_tcp_cross_track_m = 0.0;
  double worst_tcp_endpoint_m = 0.0;
  const char* first_fault = "not_checked";
};

struct TrajectoryPackage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t generation = 0;
  uint64_t task_digest = 0;
  uint64_t model_digest = 0;
  uint64_t limits_digest = 0;
  uint32_t path_id = 0;
  Vec18 start_q = Vec18::Zero();
  Vec18 start_dq = Vec18::Zero();
  bool start_dq_known = false;
  bool left = true;
  JointCurve samples;
  TimeLawTable time_law;
  std::vector<StageBoundary> stage_boundaries;
  TrajectoryCertificateSummary certificate;
  std::string planner = "unified_multistage_v2";
  std::string quality_report;
};

struct AdmitDecision {
  bool ok = false;
  const char* reason = "not_checked";
};

// FNV-1a over the fields adopt is allowed to trust. Not a cryptographic hash.
uint64_t package_digest(const TrajectoryPackage& package);

uint64_t limits_digest_of(const MotionConstraintPolicy& policy);

// Immutable model identity from actual URDF bytes + limits digest.
// Format: "urdf:<content_fnv>:limits:<limits_digest_hex>"
std::string model_identity_string(const std::string& urdf, bool is_xml,
                                  uint64_t limits_digest);

// Seal task.model_digest before PlannerWorker submit. Returns motion_task_digest.
// Worker must only verify this string; it must not rewrite identity fields.
uint64_t seal_motion_task_identity(MotionTask* task, const std::string& urdf,
                                   bool is_xml,
                                   const MotionConstraintPolicy& policy);

// Wall time along the certified time law at a densified sample index.
double time_at_sample_index(const TrajectoryPackage& package,
                            std::size_t sample_index);

// Rejects a measured start that disagrees with the certified start.
// Does not write package.samples.front(). A bridge segment is not built here.
AdmitDecision admit_package_start(
    const TrajectoryPackage& package, const Vec18& q, const Vec18& dq,
    bool dq_known, double q_tol_rad, double dq_tol);

// Sample the *executed* continuous trajectory at execution time t_s (clamped to
// [0, duration]). Uses Hermite on samples with s(t) from the time law.
// This is the single source of truth for 50 Hz monitor and 250 Hz send.
bool sample_trajectory_at(const TrajectoryPackage& package, double t_s,
                          TrajectoryTimeSample* out);

// 250 Hz may extrapolate exec time between 50 Hz publishes. Clamp so it cannot
// cross an unreleased TIP/STOP StageBoundary before Follower freezes / ACK.
inline double clamp_exec_t_to_unreleased_barrier(double t_s,
                                                 bool barrier_pending,
                                                 bool barrier_released,
                                                 double barrier_t_s) {
  if (!(t_s >= 0.0)) {
    return 0.0;
  }
  if (barrier_pending && !barrier_released) {
    if (!(barrier_t_s >= 0.0)) {
      return t_s;
    }
    return t_s < barrier_t_s ? t_s : barrier_t_s;
  }
  return t_s;
}

}  // namespace kmpc
