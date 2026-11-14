#pragma once
// 50 Hz certified-path follower for unified_multistage_v2.
// Optional ProxQP δq correction only; never reselects waist/arm posture.
// Pitch outside the hard box uses RECOVERING (§7.4): excess non-increasing,
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
};

enum class FollowerAction {
  Continue = 0,
  Arrived = 1,
  NeedReplan = 2,
  Brake = 3,
  ProtectStop = 4,
};

struct FollowerTick {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_plan = Vec18::Zero();   // certified nominal at s(t)
  Vec18 q_ref = Vec18::Zero();    // after optional δq (or recovery cmd)
  Vec18 q_follow = Vec18::Zero(); // handed to 250 Hz (equals q_ref when accepted)
  FollowerState state = FollowerState::Hold;
  FollowerAction action = FollowerAction::Continue;
  const char* reason = "hold";
  double s = 0.0;
  double t_s = 0.0;
  double pitch_excess = 0.0;
  bool qp_used = false;
  uint64_t generation = 0;
  uint32_t path_id = 0;
};

class TrajectoryFollower {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrajectoryFollower(const Kinematics& kin, const WholeBodyGuards* guards,
                     MotionConstraintPolicy policy);

  void reset();

  // Drop any package and hold at q.
  void hold_at(const Vec18& q);

  // Begin §7.4 pitch recovery from measured q (no ordinary package).
  // Returns false only if q is non-finite.
  bool begin_pitch_recovery(const Vec18& q_meas);

  // Adopt an immutable certified package. Never rewrites samples.front().
  // Start must pass admit_package_start; on failure stays in HOLD.
  bool adopt(const TrajectoryPackage& package, const Vec18& q_meas,
             const Vec18& dq_meas, bool dq_known, double q_tol_rad = 1e-3,
             double dq_tol = 0.5, std::string* why = nullptr);

  // One 50 Hz tick. t_s is wall time along the package time law (0 at adopt).
  FollowerTick step(const Vec18& q_meas, double t_s, double dt);

  FollowerState state() const { return state_; }
  bool has_package() const { return has_package_; }
  const TrajectoryPackage& package() const { return package_; }
  const FollowerTick& last() const { return last_; }
  double pitch_excess_of(const Vec18& q) const;

  // TCP tube / tracking tolerances used by the optional δq QP.
  void set_tcp_tube_m(double m) { tcp_tube_m_ = std::max(m, 1e-4); }
  void set_track_err_m(double m) { track_err_m_ = std::max(m, 1e-4); }
  void set_enable_qp_correction(bool on) { enable_qp_ = on; }

 private:
  Vec18 interpolate_plan(double t_s, double* s_out) const;
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
  FollowerTick last_;
};

}  // namespace kmpc
