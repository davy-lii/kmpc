#pragma once
// 250 Hz final gate before send for unified_multistage_v2.
// Hard box, pitch, step/accel/jerk from *actual published sends*, CoM, freshness,
// TCP tube, package generation, tracking error. No direction_envelope_accept.
// Rejects unsafe cmds; never silent-clamps geometry; never fakes arrived.
// Step/accel/jerk use internal q_last_send — never trust caller q_prev.

#include <algorithm>
#include <cstdint>
#include <string>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/types.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

enum class SafetyExecMode {
  Normal = 0,
  Recovering = 1,
};

struct SafetySupervisorInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_cmd = Vec18::Zero();
  // Deprecated for step/accel: ignored when supervisor has q_last_send.
  Vec18 q_prev = Vec18::Zero();
  Vec18 q_meas = Vec18::Zero();
  Vec18 q_follow = Vec18::Zero();  // optional 50 Hz hint / offset target
  bool has_follow = false;         // only then enforce follow_err
  Vec18 q_plan = Vec18::Zero();
  double dt = 0.004;  // nominal 250 Hz; actual dt from last_send timestamp
  double cmd_timestamp_s = 0.0;
  double meas_timestamp_s = 0.0;  // real joint_states receive/source time
  double now_s = 0.0;
  // Monotonic clock for sending; ROS time above is only for freshness.
  double send_timestamp_s = 0.0;
  bool has_meas = false;  // must be true with finite q_meas + timestamp
  uint64_t expected_generation = 0;
  uint32_t expected_path_id = 0;
  uint64_t expected_task_digest = 0;
  bool claim_arrived = false;
  SafetyExecMode mode = SafetyExecMode::Normal;
  // Recovering: measured excess ceiling; cmd excess must not grow.
  double pitch_excess_ceiling = 0.0;
  const TrajectoryPackage* package = nullptr;
};

struct SafetyDecision {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool accept = false;
  bool arrived = false;  // true only when claim is geometrically true
  const char* reason = "not_checked";
  Vec18 q_send = Vec18::Zero();  // last safe hold when rejected
  int joint = -1;
  double dt_s = 0.0;
  double actual = 0.0;
  double limit = 0.0;
  double velocity = 0.0;
  double acceleration = 0.0;
};

class SafetySupervisor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SafetySupervisor(const Kinematics& kin, const WholeBodyGuards* guards,
                   MotionConstraintPolicy policy);

  // Offline/tests: mark q as sent history in one step.
  void seed(const Vec18& q_safe);
  // New task session: drop publish history; optional anchor for the first packet
  // only (not counted as last_send until note_sent succeeds).
  void reset_session();
  void set_start_anchor(const Vec18& q_start);
  bool has_start_anchor() const { return has_start_anchor_; }
  const Vec18& start_anchor() const { return start_anchor_; }
  void set_freshness_s(double s) { freshness_s_ = std::max(s, 1e-3); }
  void set_tcp_tube_m(double m) { tcp_tube_m_ = std::max(m, 1e-4); }
  void set_track_err_m(double m) { track_err_m_ = std::max(m, 1e-4); }
  void set_follow_err_rad(double r) { follow_err_rad_ = std::max(r, 1e-6); }

  // Offline default commits accepted checks. Production validates without commit
  // and records the actual output with note_sent(), including fault holds.
  SafetyDecision check(const SafetySupervisorInput& in, bool commit = true);
  // Production calls this only after publish succeeds, for EVERY output,
  // including holds. A rejected candidate must never enter send history.
  void note_sent(const Vec18& q, double send_timestamp_s,
                 double nominal_dt = 0.004);

  const Vec18& last_safe() const { return last_safe_; }
  const Vec18& q_last_send() const { return q_last_send_; }
  bool has_last_send() const { return has_last_send_; }

 private:
  const Kinematics& kin_;
  const WholeBodyGuards* guards_;
  MotionConstraintPolicy policy_;
  Vec18 last_safe_ = Vec18::Zero();
  Vec18 start_anchor_ = Vec18::Zero();
  Vec18 q_last_send_ = Vec18::Zero();
  bool has_start_anchor_ = false;
  Vec18 last_vel_ = Vec18::Zero();
  Vec18 last_acc_ = Vec18::Zero();
  double t_last_send_s_ = -1.0;
  bool has_safe_ = false;
  bool has_last_send_ = false;
  bool has_vel_ = false;
  bool has_acc_ = false;
  double freshness_s_ = 0.05;
  double tcp_tube_m_ = 0.02;
  double track_err_m_ = 0.05;
  double follow_err_rad_ = 0.05;
};

}  // namespace kmpc
