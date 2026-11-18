#pragma once
// Immutable certified trajectory package. Never rewrite samples in place.

#include <cmath>
#include <cstdint>
#include <limits>
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
  // Task-stage TCP settle contract (from MotionStage / TaskStage). ACK and
  // WaitAck settle must use these — never a fixed 10 mm substitute.
  double position_tol_m = kDefaultTaskPositionTolM;
  double orientation_tol_rad = 0.05;
  bool has_orientation = false;
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
  // Explicit time-law result (never silently return an over-limit table).
  bool ok = true;
  std::string reason = "ok";
  double duration_before_s = 0.0;
  double max_vel_ratio = 0.0;
  double max_acc_ratio = 0.0;
  double max_jerk_ratio = 0.0;
  int worst_joint = -1;
  int scale_iters = 0;
  // Software estimate vs calibrated: accel/jerk checks use 4× / 16× soft
  // multipliers when a_max/j_max are not device-calibrated (see TimeParameterizer).
  std::string limit_source = "none";
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
  double min_work_margin_rad = 0.0;
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

// Offline trial defaults for 18-axis feedback stability / fast adopt.
// Not a verified field safety envelope — calibrate from live sampling before
// treating these as hard device limits.
inline constexpr double kFeedbackMaxAgeS = 0.100;
inline constexpr double kFeedbackStableWindowS = 0.200;
inline constexpr int kFeedbackMinSamples = 3;
inline constexpr double kFeedbackStableRangeRad = 0.002;
inline constexpr double kPlanMeasDeltaMaxRad = 0.002;
// Certified plan.start_q vs current measured at adopt (shared with follower).
inline constexpr double kAdoptStartQTolRad = 0.002;
inline constexpr double kAdoptStableWaitMaxS = 1.0;

struct FeedbackSample18 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q = Vec18::Zero();
  // Collection time from joint_states.header when present; NaN = unknown.
  // Never fill this with receive time.
  double source_stamp_s = std::numeric_limits<double>::quiet_NaN();
  // Node receive time on the same clock as now().seconds() (ROS time).
  double recv_stamp_s = 0.0;
  bool has_source = false;
  bool valid = false;
};

// Compact receive-side cache for stability / adopt diagnostics (not 250 Hz log).
struct FeedbackRing18 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr int kCap = 64;
  FeedbackSample18 buf[kCap]{};
  int head = 0;
  int count = 0;

  void clear() {
    head = 0;
    count = 0;
  }

  // recv_stamp_s: ROS receive clock. source_stamp_s: header collection if known
  // (NaN/!has_source → unknown; do not pass recv as source).
  void push(const Vec18& q, double recv_stamp_s, double source_stamp_s =
                                    std::numeric_limits<double>::quiet_NaN()) {
    if (!q.allFinite() || !std::isfinite(recv_stamp_s)) {
      return;
    }
    buf[head].q = q;
    buf[head].recv_stamp_s = recv_stamp_s;
    buf[head].has_source = std::isfinite(source_stamp_s);
    buf[head].source_stamp_s =
        buf[head].has_source ? source_stamp_s
                             : std::numeric_limits<double>::quiet_NaN();
    buf[head].valid = true;
    head = (head + 1) % kCap;
    if (count < kCap) {
      ++count;
    }
  }
};

struct FeedbackStabilityReport {
  bool ok = false;
  const char* reason = "not_checked";
  int sample_count = 0;
  int worst_joint = -1;
  double worst_range_rad = 0.0;
  double oldest_age_s = 0.0;
  double newest_age_s = 0.0;
};

struct PlanStartSnapshot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_plan_start = Vec18::Zero();
  Vec18 q_meas_start = Vec18::Zero();
  Vec18 q_cmd_start = Vec18::Zero();
  bool has_meas_start = false;
  bool has_cmd_start = false;
  // submit_ros_s: node now() at submit (ROS time). Not wall/steady.
  double submit_ros_s = std::numeric_limits<double>::quiet_NaN();
  // Feedback at submit: collection (header) vs receive — never conflate.
  double meas_source_s = std::numeric_limits<double>::quiet_NaN();
  double meas_recv_s = std::numeric_limits<double>::quiet_NaN();
  bool has_meas_source = false;
  bool has_meas_recv = false;
  // Legacy alias of submit_ros_s (kept for existing callers).
  double stamp_s = 0.0;
  uint64_t generation = 0;
};

struct FeedbackWindowStats {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool pre_ok = false;
  bool during_ok = false;
  const char* pre_reason = "not_checked";
  const char* during_reason = "not_checked";
  int pre_samples = 0;
  int during_samples = 0;
  int pre_worst_joint = -1;
  double pre_worst_range_rad = 0.0;
  int during_worst_joint = -1;
  double during_worst_delta_rad = 0.0;
  Vec18 pre_range_rad = Vec18::Zero();
  Vec18 during_delta_rad = Vec18::Zero();
};

struct FastAdoptDecision {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool ok = false;
  bool fast_path = false;
  bool wait_for_stable = false;
  bool request_replan = false;
  const char* reason = "not_checked";
  // Continuity base for follower.admit / interpolator (not raw meas reset).
  Vec18 q_continuity = Vec18::Zero();
  int worst_plan_delta_joint = -1;
  double worst_plan_delta_rad = 0.0;
  double meas_vs_plan_start_rad = 0.0;
  double meas_vs_cmd_rad = 0.0;
  double cmd_vs_plan_start_rad = 0.0;
  std::string detail;
};

// Event-only diagnostics (submit/adopt/fail) — not a 250 Hz stream.
std::string format_vec18_compact(const Vec18& q, int precision = 5);
FeedbackWindowStats summarize_feedback_windows(
    const FeedbackRing18& ring, double submit_recv_s, double adopt_recv_s,
    const Vec18& q_meas_start, bool has_meas_start,
    double pre_window_s = kFeedbackStableWindowS);
std::string format_plan_adopt_diag(
    const PlanStartSnapshot& start, double adopt_ros_s,
    double adopt_meas_source_s, bool has_adopt_meas_source,
    double adopt_meas_recv_s, bool has_adopt_meas_recv,
    const Vec18& q_meas_adopt, bool has_meas_adopt, const Vec18& q_cmd_now,
    bool has_cmd_now, const FeedbackWindowStats& windows,
    const char* result, const char* reason, double elapsed_s);

// Shortest signed angular delta in (-pi, pi].
inline double wrap_angle_delta(double a, double b) {
  double d = a - b;
  while (d > kPi) {
    d -= 2.0 * kPi;
  }
  while (d < -kPi) {
    d += 2.0 * kPi;
  }
  return d;
}

FeedbackStabilityReport evaluate_feedback_stability(
    const FeedbackRing18& ring, double now_s,
    double max_age_s = kFeedbackMaxAgeS,
    double window_s = kFeedbackStableWindowS,
    int min_samples = kFeedbackMinSamples,
    double max_range_rad = kFeedbackStableRangeRad);

// Fast adopt: stable 18-axis feedback, planning-window meas change ≤ 0.002,
// and command continuity with package.start_q. Does not widen hard gates and
// does not build a short bridge / mid-trajectory splice.
FastAdoptDecision evaluate_fast_adopt(
    const TrajectoryPackage& package, const PlanStartSnapshot& start,
    const Vec18& q_meas_now, bool meas_valid, double meas_stamp_s,
    const Vec18& q_cmd_now, bool has_cmd_now, const FeedbackRing18& ring,
    double now_s, double track_err_max_rad, const Vec18& dq_max,
    double dt_step);

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
// Sending and curve time differ while waiting for ACK. The shared probe
// includes two rest sends at every stop and resumes from that frozen curve time.
struct TrajectorySendProbe {
  double send_s;
  double curve_s;
};
std::vector<TrajectorySendProbe> trajectory_send_probe_grid(
    const TrajectoryPackage& package, double send_dt = 0.004);

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
// need_derivatives=false returns q/s/t only (certify Hermite grid uses this;
// dq/ddq then come from the send-period finite differences).
bool sample_trajectory_at(const TrajectoryPackage& package, double t_s,
                          TrajectoryTimeSample* out,
                          bool need_derivatives = true);

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
