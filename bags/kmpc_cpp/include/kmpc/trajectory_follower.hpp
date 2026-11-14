#pragma once
// 50 Hz certified-path follower for unified_multistage_v2.
// Monitors plan progress, stage stop/ACK, optional ProxQP δq offset.
// Never reselects waist/arm posture. 250 Hz samples the frozen package;
// this layer does not own the final send.
// Pitch outside the hard box uses RECOVERING: excess non-increasing,
// multi-tick, never widen cap or J1 58°. Impossible → PROTECT_STOP.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/types.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

enum class FollowerState {
  Hold = 0,
  Active = 1,
  Recovering = 2,
  Braking = 3,
  WaitAck = 4,
};

enum class FollowerAction {
  Continue = 0,
  Arrived = 1,       // time/geometry done locally; 250 Hz must still confirm
  NeedReplan = 2,
  Brake = 3,
  ProtectStop = 4,
  WaitExternalAck = 5,
};

// Physically motivated stage settle / ACK re-entry tolerances.
// Defaults are at least as tight as the old one-shot 0.03/0.05 rad gates and
// add TCP + measured velocity + multi-tick stability (grasp precision not
// loosened).
struct StageSettleTol {
  double q_abs_max_rad = 0.02;   // ~1.15° joint abs vs stage sample
  double tcp_abs_max_m = 0.01;   // 10 mm TCP vs stage FK
  double dq_abs_max = 0.05;      // rad/s infinity-norm "near stop"
  double meas_max_age_s = 0.06;  // stale feedback cannot claim settled
  int stable_ticks = 3;          // consecutive 50 Hz ticks
};

struct FollowerTick {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_plan = Vec18::Zero();   // certified nominal at exec_t
  Vec18 q_ref = Vec18::Zero();    // after optional δq (or recovery cmd)
  Vec18 q_follow = Vec18::Zero(); // monitor hint / offset target for 250 Hz
  Vec18 q_offset = Vec18::Zero(); // bounded correction about q_plan
  Vec18 freeze_q = Vec18::Zero(); // hold pose while timeline is frozen
  FollowerState state = FollowerState::Hold;
  FollowerAction action = FollowerAction::Continue;
  const char* reason = "hold";
  double s = 0.0;
  double t_s = 0.0;          // effective execution time (not wall clock)
  double exec_t_s = 0.0;     // alias of t_s for slot publishers
  bool timeline_frozen = false;
  // Next unreleased StageBoundary for 250 Hz independent clamp.
  bool barrier_pending = false;
  bool barrier_released = false;
  double barrier_t_s = 0.0;
  uint32_t barrier_stage_id = 0;
  bool barrier_stop_required = false;
  bool barrier_wait_ack = false;
  Vec18 barrier_q = Vec18::Zero();
  double pitch_excess = 0.0;
  double pitch_excess_ceiling = 0.0;
  bool qp_used = false;
  bool time_done = false;
  uint64_t generation = 0;
  uint32_t path_id = 0;
  uint32_t wait_stage_id = 0;
  int settle_stable_ticks = 0;
};

class TrajectoryFollower {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrajectoryFollower(const Kinematics& kin, const WholeBodyGuards* guards,
                     MotionConstraintPolicy policy);

  void reset();
  void hold_at(const Vec18& q);
  bool begin_pitch_recovery(const Vec18& q_meas);

  bool adopt(const TrajectoryPackage& package, const Vec18& q_meas,
             const Vec18& dq_meas, bool dq_known, double q_tol_rad = 1e-3,
             double dq_tol = 0.5, std::string* why = nullptr);

  // Request controlled brake along the time law (do not instant hold).
  void request_brake();

  // External stage ACK (generation + stage_id must match wait).
  // Success path re-checks settle with live feedback; drift / motion → replan.
  bool on_stage_ack(uint64_t generation, uint32_t stage_id, bool success,
                    const Vec18& q_meas, std::string* why = nullptr,
                    double meas_stamp_s =
                        std::numeric_limits<double>::quiet_NaN(),
                    double now_s = std::numeric_limits<double>::quiet_NaN());

  // Advance the *execution* timeline by dt (not wall-clock absolute time).
  // Stage stop / WaitAck freeze exec_t so 250 Hz cannot sample into the next
  // stage. ACK success resumes from the frozen stage time.
  // meas_stamp_s: feedback source time for age/velocity; NaN → estimate from
  // successive step() calls (tests). now_s used only for freshness.
  FollowerTick step(const Vec18& q_meas, double dt,
                    double meas_stamp_s =
                        std::numeric_limits<double>::quiet_NaN(),
                    double now_s = std::numeric_limits<double>::quiet_NaN());

  FollowerState state() const { return state_; }
  bool has_package() const { return has_package_; }
  const TrajectoryPackage& package() const { return package_; }
  const FollowerTick& last() const { return last_; }
  double pitch_excess_of(const Vec18& q) const;
  uint32_t waiting_stage_id() const { return wait_stage_id_; }
  bool waiting_ack() const { return state_ == FollowerState::WaitAck; }
  double exec_t() const { return exec_t_; }
  bool timeline_frozen() const { return timeline_frozen_; }
  const StageSettleTol& settle_tol() const { return settle_; }
  int settle_stable_ticks() const { return settle_stable_ticks_; }

  void set_tcp_tube_m(double m) { tcp_tube_m_ = std::max(m, 1e-4); }
  void set_track_err_m(double m) { track_err_m_ = std::max(m, 1e-4); }
  void set_enable_qp_correction(bool on) { enable_qp_ = on; }
  void set_ack_timeout_s(double s) { ack_timeout_s_ = std::max(s, 0.0); }
  void set_settle_tol(const StageSettleTol& tol) { settle_ = tol; }

 private:
  FollowerTick make_hold(const Vec18& q, const char* reason) const;
  FollowerTick recover_tick(const Vec18& q_meas, double dt);
  FollowerTick brake_tick(const Vec18& q_meas, double dt);
  bool try_qp_correction(const Vec18& q_nom, const Vec18& q_prev,
                         const Vec18& q_meas, double dt, Vec18* q_out,
                         const char** fail_tag) const;
  bool command_feasible(const Vec18& q_cmd, const Vec18& q_prev, double dt,
                        double pitch_excess_ceiling,
                        const char** fail_tag) const;
  Vec18 step_cap(double dt) const;
  const StageBoundary* stage_at_progress(double s) const;
  const StageBoundary* next_unreleased_barrier() const;
  void fill_barrier_fields(FollowerTick* tick) const;
  void note_feedback(const Vec18& q_meas, double meas_stamp_s, double dt);
  bool evaluate_stage_settle(const Vec18& q_stage, const Vec18& q_meas,
                             double now_s, const char** fail_tag,
                             bool* advance_stable) const;

  const Kinematics& kin_;
  const WholeBodyGuards* guards_;
  MotionConstraintPolicy policy_;
  TrajectoryPackage package_;
  bool has_package_ = false;
  FollowerState state_ = FollowerState::Hold;
  Vec18 q_cmd_ = Vec18::Zero();
  Vec18 q_prev_cmd_ = Vec18::Zero();
  bool has_cmd_ = false;
  double recovery_excess0_ = 0.0;
  int recovery_ticks_ = 0;
  int brake_ticks_ = 0;
  double tcp_tube_m_ = 0.02;
  double track_err_m_ = 0.03;
  bool enable_qp_ = true;
  uint32_t wait_stage_id_ = 0;
  double wait_ack_elapsed_s_ = 0.0;
  double ack_timeout_s_ = 5.0;
  std::size_t stage_cursor_ = 0;
  bool brake_requested_ = false;
  double exec_t_ = 0.0;
  bool timeline_frozen_ = false;
  double freeze_exec_t_ = 0.0;
  Vec18 freeze_q_ = Vec18::Zero();
  StageSettleTol settle_{};
  int settle_stable_ticks_ = 0;
  Vec18 q_meas_prev_ = Vec18::Zero();
  double q_meas_prev_stamp_s_ = std::numeric_limits<double>::quiet_NaN();
  bool have_meas_prev_ = false;
  Vec18 dq_meas_est_ = Vec18::Zero();
  bool dq_meas_valid_ = false;
  double last_meas_stamp_s_ = std::numeric_limits<double>::quiet_NaN();
  double synthetic_stamp_s_ = 0.0;
  FollowerTick last_;
};

}  // namespace kmpc
