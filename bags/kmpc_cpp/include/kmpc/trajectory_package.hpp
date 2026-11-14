#pragma once
// Immutable certified trajectory package. Never rewrite samples in place.

#include <cstdint>
#include <string>
#include <vector>

#include "kmpc/constraint_policy.hpp"
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
  double duration_s = 0.0;
  bool software_limited = true;
  std::string accel_source = "software_limited";
  std::string jerk_source = "software_limited";
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

// Rejects a measured start that disagrees with the certified start.
// Does not write package.samples.front(). A bridge segment is not built here.
AdmitDecision admit_package_start(
    const TrajectoryPackage& package, const Vec18& q, const Vec18& dq,
    bool dq_known, double q_tol_rad, double dq_tol);

}  // namespace kmpc
