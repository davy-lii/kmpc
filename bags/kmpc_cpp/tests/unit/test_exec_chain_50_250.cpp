// Offline hard gate A/D: simulate 50 Hz follower + 250 Hz supervisor with
// real measured timestamps (not command-as-meas).

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/safety_supervisor.hpp"
#include "kmpc/time_parameterizer.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/trajectory_validator.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::TrajectoryPackage make_hold_package(const kmpc::Vec18& q0,
                                          uint64_t gen) {
  kmpc::TrajectoryPackage pkg;
  pkg.generation = gen;
  pkg.path_id = 1;
  pkg.task_digest = 42;
  pkg.start_q = q0;
  pkg.left = true;
  pkg.samples = {q0, q0};
  pkg.time_law.s = {0.0, 1.0};
  pkg.time_law.t_s = {0.0, 0.2};
  pkg.time_law.ds_dt = {0.0, 0.0};
  pkg.time_law.duration_s = 0.2;
  pkg.time_law.software_limited = true;
  pkg.time_law.accel_source = "software_limited";
  pkg.time_law.jerk_source = "software_limited";
  pkg.certificate.passed = true;
  pkg.certificate.sampled_only = false;
  pkg.certificate.continuous_certified = true;
  pkg.planner = "unified_multistage_v2";
  return pkg;
}

kmpc::Vec18 home_q(const kmpc::Config& config) { return config.q_pref; }

}  // namespace

TEST(ExecChain50250, SampleAtProvidesFiniteDerivatives) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Vec18 q0 = home_q(config);
  kmpc::Vec18 q1 = q0;
  q1[4] += 0.05;
  kmpc::TrajectoryPackage pkg;
  pkg.generation = 1;
  pkg.start_q = q0;
  pkg.samples = {q0, q1, q1};
  pkg.time_law.s = {0.0, 0.5, 1.0};
  pkg.time_law.t_s = {0.0, 0.1, 0.2};
  pkg.time_law.duration_s = 0.2;
  pkg.certificate.passed = true;
  kmpc::TrajectoryTimeSample s;
  ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, 0.05, &s));
  EXPECT_TRUE(s.ok);
  EXPECT_TRUE(s.q.allFinite());
  EXPECT_TRUE(s.dq.allFinite());
  EXPECT_TRUE(s.ddq.allFinite());
}

TEST(ExecChain50250, SupervisorRejectsCommandAsMeasProxy) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor sup(kin, &guards, policy);
  const kmpc::Vec18 q = home_q(config);
  sup.seed(q);
  kmpc::SafetySupervisorInput in;
  in.q_cmd = q;
  in.q_meas = q;
  in.has_meas = false;  // pretend no real meas
  in.meas_timestamp_s = 0.0;
  in.now_s = 1.0;
  in.dt = 0.004;
  auto dec = sup.check(in);
  EXPECT_FALSE(dec.accept);
  EXPECT_STREQ(dec.reason, "safety_no_real_meas");
}

TEST(ExecChain50250, StepUsesInternalLastSendNotCallerPrev) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor sup(kin, &guards, policy);
  kmpc::Vec18 q = home_q(config);
  sup.seed(q);
  double t = 1.0;
  // First accept.
  {
    kmpc::SafetySupervisorInput in;
    in.q_cmd = q;
    in.q_meas = q;
    in.has_meas = true;
    in.meas_timestamp_s = t;
    in.now_s = t;
    in.dt = 0.004;
    in.q_prev = q;  // same
    ASSERT_TRUE(sup.check(in).accept);
  }
  // Caller lies about q_prev (far), but true last send is near — must use
  // internal last send and accept a small step.
  kmpc::Vec18 q2 = q;
  q2[4] += policy.dq_max[4] * 0.004 * 0.5;
  kmpc::Vec18 q_lie = q;
  q_lie[4] -= 1.0;  // would fail if trusted
  t += 0.004;
  kmpc::SafetySupervisorInput in;
  in.q_cmd = q2;
  in.q_prev = q_lie;
  in.q_meas = q2;
  in.has_meas = true;
  in.meas_timestamp_s = t;
  in.now_s = t;
  in.dt = 0.004;
  auto dec = sup.check(in);
  EXPECT_TRUE(dec.accept) << dec.reason;
}

TEST(PitchRecoveryThroughSupervisor, Excess254And613) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  const double limit = policy.torso_pitch_max_rad;
  ASSERT_GT(limit, 0.0);

  for (const double excess : {2.54e-4, 6.13e-4}) {
    kmpc::TrajectoryFollower follower(kin, &guards, policy);
    kmpc::SafetySupervisor sup(kin, &guards, policy);
    kmpc::Vec18 q = home_q(config);
    const double want = -limit - excess;
    const double d = want - (q[0] + q[1] + q[2]);
    q[0] += d / 3.0;
    q[1] += d / 3.0;
    q[2] += d / 3.0;
    // Keep inside joint hard box.
    for (int j = 0; j < 3; ++j) {
      q[j] = std::clamp(q[j], policy.q_min[j], policy.q_max[j]);
    }
    ASSERT_TRUE(follower.begin_pitch_recovery(q));
    sup.seed(q);
    double t = 10.0;
    bool inside = false;
    double prev_ex = follower.pitch_excess_of(q);
    for (int i = 0; i < 200; ++i) {
      const auto tick = follower.step(q, 0.0, 0.02);
      ASSERT_NE(tick.action, kmpc::FollowerAction::ProtectStop)
          << "excess=" << excess << " i=" << i << " " << tick.reason;
      EXPECT_LE(tick.pitch_excess, prev_ex + kmpc::kPitchRoundEpsRad + 1e-12);
      // 250 Hz gate with recovering contract.
      t += 0.004;
      for (int s = 0; s < 5; ++s) {
        kmpc::SafetySupervisorInput in;
        in.q_cmd = tick.q_follow;
        in.q_meas = q;
        in.has_meas = true;
        in.meas_timestamp_s = t;
        in.now_s = t;
        in.dt = 0.004;
        in.mode = kmpc::SafetyExecMode::Recovering;
        in.pitch_excess_ceiling = tick.pitch_excess_ceiling;
        in.q_follow = tick.q_follow;
        in.has_follow = true;
        const auto dec = sup.check(in);
        ASSERT_TRUE(dec.accept) << "supervisor rejected " << dec.reason
                                << " excess0=" << excess << " i=" << i;
        t += 0.004;
        q = dec.q_send;
      }
      prev_ex = tick.pitch_excess;
      if (std::abs(q[0] + q[1] + q[2]) <= limit + 1e-6) {
        inside = true;
        break;
      }
    }
    EXPECT_TRUE(inside) << "excess=" << excess;
  }
}

TEST(MotionTaskDigest, IncludesQuatAndStopAck) {
  kmpc::MotionTask a;
  a.generation = 1;
  a.left = true;
  a.frame = "ee";
  kmpc::TaskStage st;
  st.position = kmpc::Vec3(0.1, 0.2, 0.3);
  st.quat = kmpc::Vec4(0, 0, 0, 1);
  st.has_orientation = true;
  st.stop = kmpc::StageStop::StopRequired;
  st.wait_for_external_ack = true;
  st.stage_id = 7;
  a.stages.push_back(st);
  const uint64_t d0 = kmpc::motion_task_digest(a);
  a.stages[0].quat = kmpc::Vec4(0, 0, 0.1, 0.995);
  EXPECT_NE(d0, kmpc::motion_task_digest(a));
  a.stages[0].quat = kmpc::Vec4(0, 0, 0, 1);
  a.stages[0].wait_for_external_ack = false;
  EXPECT_NE(d0, kmpc::motion_task_digest(a));
}

TEST(StageAckGate, BlocksUntilMatchingAck) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q(config);
  kmpc::Vec18 q1 = q0;
  q1[4] += 0.02;
  kmpc::TrajectoryPackage pkg = make_hold_package(q0, 9);
  pkg.samples = {q0, q1, q1};
  pkg.time_law.s = {0.0, 0.5, 1.0};
  pkg.time_law.t_s = {0.0, 0.05, 0.1};
  pkg.time_law.duration_s = 0.1;
  kmpc::StageBoundary b;
  b.sample_index = 1;
  b.stage_id = 2;
  b.stop_required = true;
  b.wait_for_external_ack = true;
  pkg.stage_boundaries.push_back(b);

  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));
  // Advance to stage boundary time.
  auto tick = follower.step(q1, 0.05, 0.02);
  // May need a few ticks to enter WaitAck.
  for (int i = 0; i < 20 && tick.action != kmpc::FollowerAction::WaitExternalAck;
       ++i) {
    tick = follower.step(q1, 0.05 + 0.01 * i, 0.02);
  }
  ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck) << tick.reason;
  // Wrong gen rejected.
  EXPECT_FALSE(follower.on_stage_ack(8, 2, true, q1));
  EXPECT_TRUE(follower.waiting_ack());
  // Matching ACK releases.
  ASSERT_TRUE(follower.on_stage_ack(9, 2, true, q1));
  EXPECT_FALSE(follower.waiting_ack());
}
