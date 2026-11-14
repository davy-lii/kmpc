#pragma once
// 50 Hz certified-path follower for unified_multistage_v2.
// Monitors plan progress, stage stop/ACK, optional ProxQP δq offset.
// Never reselects waist/arm posture. 250 Hz samples the frozen package;
// this layer does not own the final send.
// Pitch outside the hard box uses RECOVERING: excess non-increasing,
// multi-tick, never widen cap or J1 58°. Impossible → PROTECT_STOP.

#include <algorithm>
#include <cstdint>
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

struct FollowerTick {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_plan = Vec18::Zero();   // certified nominal at t
  Vec18 q_ref = Vec18::Zero();    // after optional δq (or recovery cmd)
  Vec18 q_follow = Vec18::Zero(); // monitor hint / offset target for 250 Hz
  Vec18 q_offset = Vec18::Zero(); // bounded correction about q_plan
  FollowerState state = FollowerState::Hold;
  FollowerAction action = FollowerAction::Continue;
  const char* reason = "hold";
  double s = 0.0;
  double t_s = 0.0;
  double pitch_excess = 0.0;
  double pitch_excess_ceiling = 0.0;
  bool qp_used = false;
  bool time_done = false;
  uint64_t generation = 0;
  uint32_t path_id = 0;
  uint32_t wait_stage_id = 0;
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
  bool on_stage_ack(uint64_t generation, uint32_t stage_id, bool success,
                    const Vec18& q_meas, std::string* why = nullptr);

  FollowerTick step(const Vec18& q_meas, double t_s, double dt);

  FollowerState state() const { return state_; }
  bool has_package() const { return has_package_; }
  const TrajectoryPackage& package() const { return package_; }
  const FollowerTick& last() const { return last_; }
  double pitch_excess_of(const Vec18& q) const;
  uint32_t waiting_stage_id() const { return wait_stage_id_; }
  bool waiting_ack() const { return state_ == FollowerState::WaitAck; }

  void set_tcp_tube_m(double m) { tcp_tube_m_ = std::max(m, 1e-4); }
  void set_track_err_m(double m) { track_err_m_ = std::max(m, 1e-4); }
  void set_enable_qp_correction(bool on) { enable_qp_ = on; }
  void set_ack_timeout_s(double s) { ack_timeout_s_ = std::max(s, 0.0); }

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
  double wait_ack_t0_ = -1.0;
  double ack_timeout_s_ = 5.0;
  std::size_t stage_cursor_ = 0;
  bool brake_requested_ = false;
  FollowerTick last_;
};

}  // namespace kmpc
