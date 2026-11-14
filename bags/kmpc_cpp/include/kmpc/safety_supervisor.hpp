#pragma once
// 250 Hz final gate before send for unified_multistage_v2.
// Hard box, pitch, step/accel, CoM, freshness, TCP tube, package generation,
// tracking error. No direction_envelope_accept / sign hard gate.
// Rejects unsafe cmds; never silent-clamps geometry; never fakes arrived.

#include <algorithm>
#include <cstdint>
#include <string>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/types.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct SafetySupervisorInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_cmd = Vec18::Zero();
  Vec18 q_prev = Vec18::Zero();
  Vec18 q_meas = Vec18::Zero();
  Vec18 q_follow = Vec18::Zero();  // 50 Hz layer
  Vec18 q_plan = Vec18::Zero();
  double dt = 0.004;               // 250 Hz
  double cmd_timestamp_s = 0.0;
  double meas_timestamp_s = 0.0;
  double now_s = 0.0;
  uint64_t expected_generation = 0;
  uint32_t expected_path_id = 0;
  bool claim_arrived = false;
  const TrajectoryPackage* package = nullptr;
};

struct SafetyDecision {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool accept = false;
  bool arrived = false;  // true only when claim is geometrically true
  const char* reason = "not_checked";
  Vec18 q_send = Vec18::Zero();  // last safe hold when rejected
};

class SafetySupervisor {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SafetySupervisor(const Kinematics& kin, const WholeBodyGuards* guards,
                   MotionConstraintPolicy policy);

  void seed(const Vec18& q_safe);
  void set_freshness_s(double s) { freshness_s_ = std::max(s, 1e-3); }
  void set_tcp_tube_m(double m) { tcp_tube_m_ = std::max(m, 1e-4); }
  void set_track_err_m(double m) { track_err_m_ = std::max(m, 1e-4); }
  void set_follow_err_rad(double r) { follow_err_rad_ = std::max(r, 1e-6); }

  // Updates last_safe_ only when accept=true. Reject never mutates geometry.
  SafetyDecision check(const SafetySupervisorInput& in);

  const Vec18& last_safe() const { return last_safe_; }

 private:
  const Kinematics& kin_;
  const WholeBodyGuards* guards_;
  MotionConstraintPolicy policy_;
  Vec18 last_safe_ = Vec18::Zero();
  Vec18 last_vel_ = Vec18::Zero();  // accepted command velocity (rad/s)
  bool has_safe_ = false;
  bool has_vel_ = false;
  double freshness_s_ = 0.05;
  double tcp_tube_m_ = 0.02;
  double track_err_m_ = 0.05;
  double follow_err_rad_ = 0.05;
};

}  // namespace kmpc
