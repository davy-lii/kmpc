// Offline hard gate A/D: simulate 50 Hz follower + 250 Hz supervisor with
// real measured timestamps (not command-as-meas).

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/safety_supervisor.hpp"
#include "kmpc/time_parameterizer.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/trajectory_validator.hpp"
#include "kmpc/whole_body_guards.hpp"
#include "kmpc/v2_execution_state.hpp"

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

kmpc::Vec18 home_q() { return kmpc::test_fixture_q0(); }

double tip_time_s(const kmpc::TrajectoryPackage& pkg) {
  return kmpc::time_at_sample_index(pkg,
                                    pkg.stage_boundaries.front().sample_index);
}

}  // namespace

TEST(ExecChain50250, SampleAtProvidesFiniteDerivatives) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Vec18 q0 = home_q();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor sup(kin, &guards, policy);
  const kmpc::Vec18 q = home_q();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor sup(kin, &guards, policy);
  kmpc::Vec18 q = home_q();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  const double limit = policy.torso_pitch_max_rad;
  ASSERT_GT(limit, 0.0);

  for (const double excess : {2.54e-4, 6.13e-4}) {
    kmpc::TrajectoryFollower follower(kin, &guards, policy);
    kmpc::SafetySupervisor sup(kin, &guards, policy);
    kmpc::Vec18 q = home_q();
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
      const auto tick = follower.step(q, 0.02);
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
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
  auto tick = follower.step(q1, 0.02);
  // May need a few ticks to enter WaitAck.
  for (int i = 0; i < 20 && tick.action != kmpc::FollowerAction::WaitExternalAck;
       ++i) {
    tick = follower.step(q1, 0.02);
  }
  ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck) << tick.reason;
  // Wrong gen rejected.
  EXPECT_FALSE(follower.on_stage_ack(8, 2, true, q1));
  EXPECT_TRUE(follower.waiting_ack());
  // Matching ACK releases.
  ASSERT_TRUE(follower.on_stage_ack(9, 2, true, q1));
  EXPECT_FALSE(follower.waiting_ack());
}

TEST(StageAckGate, TipWaitFiveSecondsBlocksLiftOn250Hz) {
  // Real 50/250 linkage: TIP stop → WaitAck → wall clock +5s → no LIFT samples.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q_tip = q0;
  q_tip[4] += 0.02;  // TIP approach
  kmpc::Vec18 q_lift = q_tip;
  q_lift[5] += 0.05;  // LIFT joint (must not move while waiting ACK)

  kmpc::TrajectoryPackage pkg = make_hold_package(q0, 42);
  pkg.samples = {q0, q_tip, q_lift};
  pkg.time_law.s = {0.0, 0.5, 1.0};
  pkg.time_law.t_s = {0.0, 0.08, 0.20};
  pkg.time_law.duration_s = 0.20;
  kmpc::StageBoundary tip;
  tip.sample_index = 1;
  tip.stage_id = 1;
  tip.stop_required = true;
  tip.wait_for_external_ack = true;
  pkg.stage_boundaries.push_back(tip);

  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  follower.set_enable_qp_correction(false);
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));

  kmpc::Vec18 q_meas = q0;
  kmpc::FollowerTick tick;
  for (int i = 0; i < 40; ++i) {
    // Approach TIP pose so stop condition can clear.
    q_meas = q_tip;
    tick = follower.step(q_meas, 0.02);
    if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
      break;
    }
  }
  ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck) << tick.reason;
  ASSERT_TRUE(tick.timeline_frozen);
  const double frozen_exec = tick.exec_t_s;
  const double tip_j5 = q_tip[5];

  // Simulate 5 s wall time at 250 Hz while 50 Hz stays in WaitAck.
  constexpr double kDt250 = 0.004;
  constexpr double kWaitS = 5.0;
  const int n250 = static_cast<int>(kWaitS / kDt250);
  int lift_cmds = 0;
  for (int i = 0; i < n250; ++i) {
    if (i % 5 == 0) {
      tick = follower.step(q_tip, 0.02);
      ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck)
          << "escaped WaitAck at i=" << i << " " << tick.reason;
      EXPECT_TRUE(tick.timeline_frozen);
      EXPECT_DOUBLE_EQ(tick.exec_t_s, frozen_exec);
    }
    // 250 Hz send uses frozen hold — never sample past TIP into LIFT.
    kmpc::Vec18 q_cmd =
        tick.freeze_q.allFinite() ? tick.freeze_q : tick.q_follow;
    if (std::abs(q_cmd[5] - tip_j5) > 1e-9) {
      ++lift_cmds;
    }
    kmpc::TrajectoryTimeSample samp;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, tick.exec_t_s, &samp));
    EXPECT_NEAR(samp.q[5], tip_j5, 1e-9) << "exec_t advanced into LIFT";
  }
  EXPECT_EQ(lift_cmds, 0);

  // ACK resumes from frozen stage time (not wall+5s).
  ASSERT_TRUE(follower.on_stage_ack(42, 1, true, q_tip));
  EXPECT_FALSE(follower.waiting_ack());
  EXPECT_NEAR(follower.exec_t(), frozen_exec, 1e-12);
  tick = follower.step(q_tip, 0.02);
  EXPECT_GE(tick.exec_t_s, frozen_exec);
  EXPECT_LT(tick.exec_t_s, frozen_exec + 0.05);
}

TEST(CertificatePolicy, NoContinuousFlagFromSamplingAndNo2xStep) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q1 = q0;
  q1[4] += 0.01;
  kmpc::MotionTask task;
  task.generation = 1;
  task.q = q0;
  task.left = true;
  task.frame = std::string(kmpc::kEeFrameLeft);
  kmpc::TaskStage st;
  st.geometry = kmpc::TaskGeometry::Lin;
  st.position = kin.fk_pos(q1, kmpc::kEeFrameLeft);
  st.stage_id = 1;
  task.stages.push_back(st);
  kmpc::TrajectoryPackage pkg;
  pkg.generation = 1;
  pkg.start_q = q0;
  pkg.samples = {q0, q1};
  pkg.time_law.s = {0.0, 1.0};
  pkg.time_law.t_s = {0.0, 0.5};
  pkg.time_law.duration_s = 0.5;
  pkg.time_law.software_limited = true;
  const auto cert =
      kmpc::validate_actual_q_of_t(pkg, task, policy, kin, &guards);
  EXPECT_FALSE(cert.continuous_certified);
  EXPECT_EQ(cert.coverage, kmpc::PathCoverage::AdaptiveSampledWithOnlineGuard);
  EXPECT_NE(cert.coverage_note.find("sampled+online_guard"), std::string::npos);
}

TEST(StageAckGate, AckFailErrorTimeoutAndPoseDrift) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q1 = q0;
  q1[4] += 0.02;
  kmpc::Vec18 q_lift = q1;
  q_lift[5] += 0.04;

  auto make_pkg = [&](uint64_t gen) {
    kmpc::TrajectoryPackage pkg = make_hold_package(q0, gen);
    pkg.samples = {q0, q1, q_lift};
    pkg.time_law.s = {0.0, 0.5, 1.0};
    pkg.time_law.t_s = {0.0, 0.06, 0.16};
    pkg.time_law.duration_s = 0.16;
    kmpc::StageBoundary b;
    b.sample_index = 1;
    b.stage_id = 3;
    b.stop_required = true;
    b.wait_for_external_ack = true;
    pkg.stage_boundaries.push_back(b);
    return pkg;
  };

  auto advance_to_wait = [&](kmpc::TrajectoryFollower& f) {
    kmpc::FollowerTick tick;
    for (int i = 0; i < 40; ++i) {
      tick = f.step(q1, 0.02);
      if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
        return tick;
      }
    }
    return tick;
  };

  // ACK success=false → NeedReplan, no LIFT.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(1), q0, kmpc::Vec18::Zero(), false));
    ASSERT_EQ(advance_to_wait(f).action, kmpc::FollowerAction::WaitExternalAck);
    std::string why;
    EXPECT_FALSE(f.on_stage_ack(1, 3, false, q1, &why));
    EXPECT_EQ(why, "ack_failed");
    EXPECT_EQ(f.state(), kmpc::FollowerState::Hold);
    EXPECT_FALSE(f.has_package());
  }

  // Wrong stage_id → mismatch, still waiting.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(2), q0, kmpc::Vec18::Zero(), false));
    ASSERT_EQ(advance_to_wait(f).action, kmpc::FollowerAction::WaitExternalAck);
    EXPECT_FALSE(f.on_stage_ack(2, 99, true, q1));
    EXPECT_TRUE(f.waiting_ack());
  }

  // Pose drift → reject, no advance.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(3), q0, kmpc::Vec18::Zero(), false));
    ASSERT_EQ(advance_to_wait(f).action, kmpc::FollowerAction::WaitExternalAck);
    kmpc::Vec18 drifted = q1;
    drifted[4] += 0.2;
    std::string why;
    EXPECT_FALSE(f.on_stage_ack(3, 3, true, drifted, &why));
    EXPECT_EQ(why, "ack_pose_drift");
    EXPECT_EQ(f.state(), kmpc::FollowerState::Hold);
  }

  // Timeout while waiting → NeedReplan; exec_t stays frozen until then.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    f.set_ack_timeout_s(0.1);
    ASSERT_TRUE(f.adopt(make_pkg(4), q0, kmpc::Vec18::Zero(), false));
    auto tick = advance_to_wait(f);
    ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck);
    const double frozen = tick.exec_t_s;
    for (int i = 0; i < 20; ++i) {
      tick = f.step(q1, 0.02);
      if (tick.action == kmpc::FollowerAction::NeedReplan) {
        break;
      }
      EXPECT_DOUBLE_EQ(tick.exec_t_s, frozen);
    }
    EXPECT_EQ(tick.action, kmpc::FollowerAction::NeedReplan);
    EXPECT_STREQ(tick.reason, "ack_timeout");
  }
}

TEST(ExecChain50250, SendSequenceStatsReport250Hz) {
  // Gate E: 250 Hz Hermite send sequence stats (dq/ddq/jerk/fold/pause).
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q1 = q0;
  q1[4] += 0.03;
  kmpc::TrajectoryPackage pkg = make_hold_package(q0, 7);
  pkg.samples = {q0, q1};
  pkg.time_law.s = {0.0, 1.0};
  pkg.time_law.t_s = {0.0, 0.4};
  pkg.time_law.duration_s = 0.4;
  pkg.time_law.software_limited = true;

  constexpr double kDt = 0.004;
  double max_dq = 0.0;
  double max_ddq = 0.0;
  double max_jerk = 0.0;
  int fold_events = 0;
  int pause_ticks = 0;
  kmpc::Vec18 prev_q = q0;
  kmpc::Vec18 prev_v = kmpc::Vec18::Zero();
  kmpc::Vec18 prev_a = kmpc::Vec18::Zero();
  kmpc::Vec18 prev_sign = kmpc::Vec18::Zero();
  bool have_prev = false;
  bool have_v = false;
  bool have_a = false;
  const int n = static_cast<int>(std::ceil(pkg.time_law.duration_s / kDt));
  for (int k = 0; k <= n; ++k) {
    const double t =
        std::min(pkg.time_law.duration_s, kDt * static_cast<double>(k));
    kmpc::TrajectoryTimeSample samp;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, t, &samp));
    if (!have_prev) {
      prev_q = samp.q;
      have_prev = true;
      continue;
    }
    const double dt = std::max(kDt, 1e-4);
    kmpc::Vec18 v = (samp.q - prev_q) / dt;
    for (int j = 0; j < kmpc::kNDof; ++j) {
      max_dq = std::max(max_dq, std::abs(v[j]));
      EXPECT_LE(std::abs(samp.q[j] - prev_q[j]), policy.dq_max[j] * dt + 1e-4);
      if (have_v) {
        const double a = (v[j] - prev_v[j]) / dt;
        max_ddq = std::max(max_ddq, std::abs(a));
        if (have_a) {
          max_jerk = std::max(max_jerk, std::abs((a - prev_a[j]) / dt));
        }
        prev_a[j] = a;
        have_a = true;
      }
      const double sgn = (std::abs(v[j]) < 1e-6) ? 0.0 : (v[j] > 0.0 ? 1.0 : -1.0);
      if (prev_sign[j] != 0.0 && sgn != 0.0 && sgn != prev_sign[j]) {
        ++fold_events;
      }
      if (sgn != 0.0) {
        prev_sign[j] = sgn;
      }
      if (std::abs(v[j]) < 1e-5) {
        ++pause_ticks;
      }
    }
    prev_v = v;
    have_v = true;
    prev_q = samp.q;
  }
  EXPECT_GE(max_dq, 0.0);
  EXPECT_GE(max_ddq, 0.0);
  // Soft assert: report values via gtest message for the delivery log.
  RecordProperty("max_dq", max_dq);
  RecordProperty("max_ddq", max_ddq);
  RecordProperty("max_jerk", max_jerk);
  RecordProperty("fold_events", fold_events);
  RecordProperty("pause_ticks", pause_ticks);
  std::printf(
      "E_250Hz_stats max_dq=%.6f max_ddq=%.6f max_jerk=%.6f folds=%d pauses=%d\n",
      max_dq, max_ddq, max_jerk, fold_events, pause_ticks);
}

TEST(GateA50250Matrix, DelayFirstTickReversalSwitchAndFeedbackLag) {
  // A: scheduling delay / first tick / reversal / task switch / feedback lag.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q1 = q0;
  q1[4] += 0.03;
  kmpc::Vec18 q2 = q0;
  q2[5] += 0.03;

  auto make_pkg = [&](uint64_t gen, const kmpc::Vec18& q_end) {
    kmpc::TrajectoryPackage pkg = make_hold_package(q0, gen);
    pkg.path_id = static_cast<uint32_t>(gen);
    pkg.task_digest = 1000 + gen;
    pkg.samples = {q0, q_end};
    pkg.time_law.s = {0.0, 1.0};
    pkg.time_law.t_s = {0.0, 0.2};
    pkg.time_law.duration_s = 0.2;
    return pkg;
  };

  // --- First tick: after adopt, exec_t starts at 0 and first sample is start.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(1, q1), q0, kmpc::Vec18::Zero(), false));
    EXPECT_DOUBLE_EQ(f.exec_t(), 0.0);
    const auto tick = f.step(q0, 0.02);
    EXPECT_NEAR(tick.exec_t_s, 0.02, 1e-12);
    EXPECT_LT((tick.q_plan - q0).cwiseAbs().maxCoeff(), 0.02);
    EXPECT_NE(tick.action, kmpc::FollowerAction::Arrived);
  }

  // --- Scheduling delay: 50 Hz advances exec_t; 250 Hz fill-in uses wall delta
  // from the published anchor without jumping past package duration.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(2, q1), q0, kmpc::Vec18::Zero(), false));
    const auto tick50 = f.step(q0, 0.02);
    const double anchor = tick50.exec_t_s;
    // Simulate 4× 250 Hz ticks using wall fill-in from the same anchor.
    for (int i = 1; i <= 4; ++i) {
      const double t_fill = anchor + 0.004 * i;
      kmpc::TrajectoryTimeSample samp;
      ASSERT_TRUE(kmpc::sample_trajectory_at(f.package(), t_fill, &samp));
      EXPECT_TRUE(samp.ok);
      EXPECT_LE(samp.t_s, f.package().time_law.duration_s + 1e-12);
    }
  }

  // --- Reversal: 250 Hz Supervisor accepts smooth sign change (shared with B/E).
  {
    kmpc::SafetySupervisor sup(kin, &guards, policy);
    sup.seed(q0);
    kmpc::Vec18 q = q0;
    const int j = 5;
    const double dt = 0.004;
    const double step = std::min(0.5 * policy.dq_max[j] * dt,
                                 2.0 * policy.ddq_max[j] * dt * dt);
    double t = 10.0;
    auto send = [&](double dq) {
      kmpc::SafetySupervisorInput in;
      in.q_cmd = q;
      in.q_cmd[j] = q[j] + dq;
      in.q_meas = q;
      in.has_meas = true;
      in.meas_timestamp_s = t;
      in.now_s = t;
      in.dt = dt;
      in.q_follow = in.q_cmd;
      in.has_follow = true;
      const auto dec = sup.check(in);
      ASSERT_TRUE(dec.accept) << dec.reason;
      q = dec.q_send;
      t += dt;
    };
    for (int k = 0; k < 8; ++k) {
      send(+step);
    }
    for (int k = 0; k < 16; ++k) {
      send(-step);
    }
  }

  // --- Task switch: new generation adopt replaces package; old gen cannot ACK.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(10, q1), q0, kmpc::Vec18::Zero(), false));
    EXPECT_EQ(f.package().generation, 10u);
    ASSERT_TRUE(f.adopt(make_pkg(11, q2), q0, kmpc::Vec18::Zero(), false));
    EXPECT_EQ(f.package().generation, 11u);
    EXPECT_EQ(f.package().path_id, 11u);
    // Stale ACK for old generation while not waiting is rejected.
    EXPECT_FALSE(f.on_stage_ack(10, 1, true, q0));
  }

  // --- Feedback lag: stale measurement timestamp rejected by Supervisor.
  {
    kmpc::SafetySupervisor sup(kin, &guards, policy);
    sup.seed(q0);
    kmpc::SafetySupervisorInput in;
    in.q_cmd = q0;
    in.q_meas = q0;
    in.has_meas = true;
    in.dt = 0.004;
    in.now_s = 100.0;
    in.meas_timestamp_s = 90.0;  // 10 s lag
    in.cmd_timestamp_s = 100.0;
    const auto dec = sup.check(in);
    EXPECT_FALSE(dec.accept);
    EXPECT_STREQ(dec.reason, "safety_stale_meas");
  }

  // --- Feedback lag (no real meas): also rejected.
  {
    kmpc::SafetySupervisor sup(kin, &guards, policy);
    sup.seed(q0);
    kmpc::SafetySupervisorInput in;
    in.q_cmd = q0;
    in.q_meas = q0;
    in.has_meas = false;
    in.dt = 0.004;
    in.now_s = 1.0;
    in.meas_timestamp_s = 1.0;
    const auto dec = sup.check(in);
    EXPECT_FALSE(dec.accept);
    EXPECT_STREQ(dec.reason, "safety_no_real_meas");
  }
}

TEST(P0Barrier250Hz, TipBetween50HzCyclesNeverEmitsLift) {
  // TIP barrier sits between two 50 Hz ticks; 250 Hz runs many samples in the
  // gap. Without independent clamp, wall extrapolation would enter LIFT.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q_tip = q0;
  q_tip[4] += 0.02;
  kmpc::Vec18 q_lift = q_tip;
  q_lift[5] += 0.08;

  kmpc::TrajectoryPackage pkg = make_hold_package(q0, 77);
  pkg.samples = {q0, q_tip, q_lift};
  // TIP at t=0.028 — between 50 Hz ticks at 0.02 and 0.04.
  pkg.time_law.s = {0.0, 0.5, 1.0};
  pkg.time_law.t_s = {0.0, 0.028, 0.12};
  pkg.time_law.duration_s = 0.12;
  kmpc::StageBoundary tip;
  tip.sample_index = 1;
  tip.stage_id = 1;
  tip.stop_required = true;
  tip.wait_for_external_ack = true;
  pkg.stage_boundaries.push_back(tip);

  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  follower.set_enable_qp_correction(false);
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));

  // One 50 Hz tick at t≈0.02 — still before TIP; barrier must already be
  // advertised so 250 Hz can clamp independently.
  auto tick = follower.step(q0, 0.02);
  ASSERT_TRUE(tick.barrier_pending);
  EXPECT_FALSE(tick.barrier_released);
  EXPECT_NEAR(tick.barrier_t_s, 0.028, 1e-9);
  EXPECT_FALSE(tick.timeline_frozen);
  EXPECT_LT(tick.exec_t_s, tip_time_s(pkg));
  const double tip_j5 = q_tip[5];
  const double barrier_t = tick.barrier_t_s;
  const double exec_anchor = tick.exec_t_s;
  const double wall0 = 1.0;

  // ~4.5× 50 Hz gap covered by 250 Hz ticks (0.004). No ACK → no LIFT stage.
  // Mirror stream_take_tick: clamp time, and at the barrier hold barrier_q.
  const double tip_s = 0.5;
  int past_tip = 0;
  int lift_hold_violations = 0;
  for (int i = 1; i <= 12; ++i) {
    const double wall = wall0 + 0.004 * i;
    const double t_raw = exec_anchor + (wall - wall0);
    const double t_eff = kmpc::clamp_exec_t_to_unreleased_barrier(
        t_raw, tick.barrier_pending, tick.barrier_released, barrier_t);
    EXPECT_LE(t_eff, barrier_t + 1e-12) << "i=" << i << " t_raw=" << t_raw;
    kmpc::TrajectoryTimeSample samp;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, t_eff, &samp));
    EXPECT_LE(samp.s, tip_s + 1e-9) << "progress past TIP stage at i=" << i;
    if (samp.s > tip_s + 1e-9) {
      ++past_tip;
    }
    kmpc::Vec18 q_cmd = samp.q;
    if (t_eff + 1e-12 >= barrier_t) {
      q_cmd = q_tip;  // node holds barrier_q once clamped at TIP
      EXPECT_NEAR(q_cmd[5], tip_j5, 1e-12);
      if (std::abs(q_cmd[5] - tip_j5) > 1e-12) {
        ++lift_hold_violations;
      }
    }
  }
  EXPECT_EQ(past_tip, 0);
  EXPECT_EQ(lift_hold_violations, 0);

  // Unclamped raw time past TIP would already enter the LIFT segment.
  {
    const double t_raw_past = 0.04;
    kmpc::TrajectoryTimeSample past;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, t_raw_past, &past));
    EXPECT_GT(past.s, tip_s + 1e-6);
    EXPECT_GT(std::abs(past.q[5] - tip_j5), 1e-4);
  }
}

TEST(P0StageSettle, EarlyAckWhileMovingAndTipDriftReject) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q_tip = q0;
  q_tip[4] += 0.02;
  kmpc::Vec18 q_lift = q_tip;
  q_lift[5] += 0.05;

  auto make_pkg = [&](uint64_t gen) {
    kmpc::TrajectoryPackage pkg = make_hold_package(q0, gen);
    pkg.samples = {q0, q_tip, q_lift};
    pkg.time_law.s = {0.0, 0.5, 1.0};
    pkg.time_law.t_s = {0.0, 0.06, 0.16};
    pkg.time_law.duration_s = 0.16;
    kmpc::StageBoundary b;
    b.sample_index = 1;
    b.stage_id = 3;
    b.stop_required = true;
    b.wait_for_external_ack = true;
    pkg.stage_boundaries.push_back(b);
    return pkg;
  };

  // Early success ACK before WaitAck (still approaching) must not release.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(1), q0, kmpc::Vec18::Zero(), false));
    auto tick = f.step(q0, 0.02);
    EXPECT_NE(tick.action, kmpc::FollowerAction::WaitExternalAck);
    std::string why;
    EXPECT_FALSE(f.on_stage_ack(1, 3, true, q0, &why));
    EXPECT_EQ(why, "not_waiting_ack");
    EXPECT_FALSE(f.waiting_ack());
  }

  // Reach WaitAck with settled tip, then ACK while feedback still moving.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(2), q0, kmpc::Vec18::Zero(), false));
    kmpc::FollowerTick tick;
    double t = 1.0;
    for (int i = 0; i < 60; ++i) {
      tick = f.step(q_tip, 0.02, t, t);
      t += 0.02;
      if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
        break;
      }
    }
    ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck) << tick.reason;
    // Hold tip with real stamps so dq≈0, then ACK with a short-dt jump.
    for (int i = 0; i < 3; ++i) {
      tick = f.step(q_tip, 0.02, t, t);
      t += 0.02;
    }
    kmpc::Vec18 moving = q_tip;
    moving[4] += 0.012;  // dts=0.02 → dq=0.6 ≫ 0.05 settle
    std::string why;
    EXPECT_FALSE(f.on_stage_ack(2, 3, true, moving, &why, t, t));
    EXPECT_EQ(why, "ack_feedback_moving") << why;
    EXPECT_EQ(f.state(), kmpc::FollowerState::Hold);
    EXPECT_FALSE(f.has_package());
  }

  // Small TIP drift after WaitAck → replan (tighter than legacy 0.05 rad).
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(3), q0, kmpc::Vec18::Zero(), false));
    kmpc::FollowerTick tick;
    for (int i = 0; i < 60; ++i) {
      tick = f.step(q_tip, 0.02);
      if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
        break;
      }
    }
    ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck);
    // Hold settled a few ticks so dq→0, then drift slightly over q tol.
    for (int i = 0; i < 5; ++i) {
      tick = f.step(q_tip, 0.02);
    }
    kmpc::Vec18 drifted = q_tip;
    drifted[4] += 0.025;  // > 0.02 rad settle tol, < old 0.05
    std::string why;
    EXPECT_FALSE(f.on_stage_ack(3, 3, true, drifted, &why));
    EXPECT_EQ(why, "ack_pose_drift");
    EXPECT_EQ(f.state(), kmpc::FollowerState::Hold);
  }

  // Multi-tick settle: one-shot position near tip is not enough.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    kmpc::StageSettleTol tol = f.settle_tol();
    tol.stable_ticks = 5;
    f.set_settle_tol(tol);
    ASSERT_TRUE(f.adopt(make_pkg(4), q0, kmpc::Vec18::Zero(), false));
    int first_near = -1;
    int wait_at = -1;
    for (int i = 0; i < 80; ++i) {
      auto tick = f.step(q_tip, 0.02);
      if (first_near < 0 &&
          (q_tip - tick.q_follow).cwiseAbs().maxCoeff() < 1e-9 &&
          tick.timeline_frozen) {
        first_near = i;
      }
      if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
        wait_at = i;
        break;
      }
    }
    ASSERT_GE(first_near, 0);
    ASSERT_GE(wait_at, 0);
    EXPECT_GE(wait_at - first_near, 4);
  }
}

// TIP settle must honor StageBoundary task TCP tol (3 mm), not a fixed 10 mm.
TEST(P0StageSettle, TipTaskTolRejectsEightMmAndAcceptsStable) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q();
  kmpc::Vec18 q_tip = q0;
  q_tip[4] += 0.02;
  kmpc::Vec18 q_lift = q_tip;
  q_lift[5] += 0.05;

  auto make_pkg = [&](uint64_t gen, double pos_tol_m) {
    kmpc::TrajectoryPackage pkg = make_hold_package(q0, gen);
    pkg.samples = {q0, q_tip, q_lift};
    pkg.time_law.s = {0.0, 0.5, 1.0};
    pkg.time_law.t_s = {0.0, 0.06, 0.16};
    pkg.time_law.duration_s = 0.16;
    kmpc::StageBoundary b;
    b.sample_index = 1;
    b.stage_id = 7;
    b.stop_required = true;
    b.wait_for_external_ack = true;
    b.position_tol_m = pos_tol_m;
    b.orientation_tol_rad = 0.05;
    b.has_orientation = false;
    pkg.stage_boundaries.push_back(b);
    return pkg;
  };

  const auto frame = kmpc::kEeFrameLeft;
  const kmpc::Vec3 p_tip = kin.fk_pos(q_tip, frame);

  // Build a live pose ~8 mm off tip TCP while keeping joints inside a loose
  // q gate so the reject is on the task TCP contract, not joint abs.
  kmpc::Vec18 q_8mm = q_tip;
  double tcp_err = 0.0;
  bool found = false;
  for (int j = kmpc::kSliceLeft.begin; j < kmpc::kSliceLeft.end && !found;
       ++j) {
    for (double dq = 0.002; dq <= 0.12; dq += 0.002) {
      q_8mm = q_tip;
      q_8mm[j] += dq;
      tcp_err = (kin.fk_pos(q_8mm, frame) - p_tip).norm();
      if (tcp_err >= 0.0075 && tcp_err <= 0.012) {
        found = true;
        break;
      }
    }
  }
  ASSERT_TRUE(found) << "tcp_err=" << tcp_err;
  ASSERT_GE(tcp_err, 0.0075) << "tcp_err=" << tcp_err;
  ASSERT_LE(tcp_err, 0.015) << "tcp_err=" << tcp_err;

  // 8 mm drift with 3 mm task tol → reject ACK (NeedReplan).
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    kmpc::StageSettleTol tol = f.settle_tol();
    tol.q_abs_max_rad = 0.20;  // isolate TCP contract
    f.set_settle_tol(tol);
    ASSERT_TRUE(f.adopt(make_pkg(11, 0.003), q0, kmpc::Vec18::Zero(), false));
    kmpc::FollowerTick tick;
    double t = 1.0;
    for (int i = 0; i < 80; ++i) {
      tick = f.step(q_tip, 0.02, t, t);
      t += 0.02;
      if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
        break;
      }
    }
    ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck) << tick.reason;
    for (int i = 0; i < 4; ++i) {
      tick = f.step(q_tip, 0.02, t, t);
      t += 0.02;
    }
    std::string why;
    EXPECT_FALSE(f.on_stage_ack(11, 7, true, q_8mm, &why, t, t));
    EXPECT_EQ(why, "ack_tcp_drift") << why;
    EXPECT_EQ(f.state(), kmpc::FollowerState::Hold);
    EXPECT_EQ(f.last().action, kmpc::FollowerAction::NeedReplan);
  }

  // Same tip, within 3 mm and multi-tick stable → ACK succeeds.
  {
    kmpc::TrajectoryFollower f(kin, &guards, policy);
    f.set_enable_qp_correction(false);
    ASSERT_TRUE(f.adopt(make_pkg(12, 0.003), q0, kmpc::Vec18::Zero(), false));
    kmpc::FollowerTick tick;
    double t = 2.0;
    for (int i = 0; i < 80; ++i) {
      tick = f.step(q_tip, 0.02, t, t);
      t += 0.02;
      if (tick.action == kmpc::FollowerAction::WaitExternalAck) {
        break;
      }
    }
    ASSERT_EQ(tick.action, kmpc::FollowerAction::WaitExternalAck) << tick.reason;
    for (int i = 0; i < 4; ++i) {
      tick = f.step(q_tip, 0.02, t, t);
      t += 0.02;
    }
    std::string why;
    ASSERT_TRUE(f.on_stage_ack(12, 7, true, q_tip, &why, t, t)) << why;
    EXPECT_EQ(why, "ok");
    EXPECT_EQ(f.state(), kmpc::FollowerState::Active);
    EXPECT_FALSE(f.timeline_frozen());
  }
}

// Regression for r53: the observer may jitter or disappear; only the sender
// advances q(t). These tests exercise the production clock and follower API.
TEST(R53Execution, ObserverJitterCannotReanchorSenderAndPauseNeverCatchesUp) {
  kmpc::V2ExecutionState exec;
  exec.adopt(7, 42, 9, 10.0);
  double wall = 10.0;
  for (int i = 1; i <= 100; ++i) {
    wall += 0.004;
    EXPECT_NEAR(exec.advance(wall, false, false, 0.0), 0.004 * i, 1e-10);
  }
  exec.pause(wall + 2.0);
  EXPECT_NEAR(exec.advance(wall + 2.004, false, false, 0.0), 0.404, 1e-10);
  EXPECT_NEAR(exec.advance(wall + 2.008, false, true, 0.406), 0.406, 1e-10);
  for (int i = 1; i <= 200; ++i)
    EXPECT_DOUBLE_EQ(exec.advance(wall + 2.008 + i * 0.004,
                                  true, true, 0.406), 0.406);
  EXPECT_NEAR(exec.advance(wall + 2.812, false, false, 0.0), 0.410, 1e-10);
  exec.latch_fault("safety_jerk");
  for (int i = 0; i < 200; ++i) {
    EXPECT_TRUE(exec.fault_matches(7, 42));
    EXPECT_NEAR(exec.advance(wall + 3.0 + i * 0.02,
                            false, false, 0.0), 0.410, 1e-10);
  }
  EXPECT_FALSE(exec.fault_matches(8, 42));
  exec.adopt(8, 43, 10, wall + 8.0);
  EXPECT_FALSE(exec.faulted());
  EXPECT_DOUBLE_EQ(exec.exec_s(), 0.0);
}

TEST(R53Execution, SenderTimedFollowerNeverAddsFeedbackOffset) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  const auto policy = kmpc::policy_from_config(config);
  const auto q0 = home_q();
  auto pkg = make_hold_package(q0, 7);
  pkg.samples.back()[kmpc::kSliceLeft.begin + 5] += 0.02;
  pkg.time_law.t_s = {0.0, 4.0};
  pkg.time_law.duration_s = 4.0;
  kmpc::TrajectoryFollower follower(kin, nullptr, policy);
  // Even an old caller opting into the legacy QP cannot alter production q(t).
  follower.set_enable_qp_correction(true);
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));
  kmpc::SafetySupervisor supervisor(kin, nullptr, policy);
  supervisor.seed(q0);
  kmpc::V2ExecutionState exec;
  exec.adopt(7, 42, 1, 10.0);
  double next_observer = 0.0;
  int observer_count = 0;
  double prev_observer_t = 0.0;
  for (int i = 0; i < 600; ++i) {
    const double wall = 10.0 + i * 0.004;
    const double t = exec.advance(wall, false, false, 0.0);
    kmpc::TrajectoryTimeSample sample;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, t, &sample));
    kmpc::Vec18 measured = sample.q;
    measured[kmpc::kSliceLeft.begin + 5] -= 0.002;
    if (t + 1e-9 >= next_observer) {
      const auto tick = follower.step(measured, std::max(0.02, t - prev_observer_t),
                                     wall, wall, t);
      ASSERT_EQ(tick.action, kmpc::FollowerAction::Continue) << tick.reason;
      EXPECT_FALSE(tick.qp_used);
      EXPECT_DOUBLE_EQ(tick.q_offset.norm(), 0.0);
      EXPECT_NEAR((tick.q_follow - sample.q).norm(), 0.0, 1e-12);
      EXPECT_DOUBLE_EQ(tick.exec_t_s, t);
      prev_observer_t = t;
      const double intervals[] = {0.012, 0.028, 0.016, 0.024, 0.06};
      next_observer = t + intervals[observer_count++ % 5];
    }
    kmpc::SafetySupervisorInput in;
    in.q_cmd = sample.q;
    in.q_plan = sample.q;
    in.q_meas = measured;
    in.has_meas = true;
    in.now_s = wall + 1000.0;
    in.meas_timestamp_s = in.now_s;
    in.send_timestamp_s = wall;
    const auto decision = supervisor.check(in, false);
    ASSERT_TRUE(decision.accept) << decision.reason << " tick=" << i;
    // Validation alone cannot change the last actual send.
    supervisor.note_sent(decision.q_send, wall);
  }
}

TEST(R53Execution, HoldsAreRecordedAndUnsentCandidatesNeverEnterHistory) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  const auto policy = kmpc::policy_from_config(config);
  const auto q0 = home_q();
  kmpc::SafetySupervisor supervisor(kin, nullptr, policy);
  supervisor.seed(q0);
  supervisor.note_sent(q0, 10.0);
  kmpc::SafetySupervisorInput in;
  in.q_cmd = q0;
  in.q_cmd[4] += policy.dq_max[4] * 0.004 * 0.1;
  in.q_meas = q0;
  in.has_meas = true;
  in.now_s = in.meas_timestamp_s = 100.0;
  in.send_timestamp_s = 10.004;
  ASSERT_TRUE(supervisor.check(in, false).accept);
  EXPECT_DOUBLE_EQ((supervisor.q_last_send() - q0).norm(), 0.0);
  supervisor.note_sent(in.q_cmd, 10.004);
  const auto hold = in.q_cmd;
  for (int i = 2; i <= 20; ++i) supervisor.note_sent(hold, 10.0 + i * 0.004);
  in.send_timestamp_s = 10.084;
  in.q_cmd = hold;
  in.q_cmd[4] += policy.dq_max[4] * 0.004 * 2.0;
  const auto rejected = supervisor.check(in, false);
  EXPECT_FALSE(rejected.accept);
  EXPECT_STREQ(rejected.reason, "safety_step");
  EXPECT_EQ(rejected.joint, 4);
  EXPECT_NEAR(rejected.dt_s, 0.004, 1e-12);
  EXPECT_GT(rejected.actual, rejected.limit);
  EXPECT_DOUBLE_EQ((supervisor.q_last_send() - hold).norm(), 0.0);
}

TEST(R53Execution, ShortFirstObserverAllowsAdmittedMeasurementDifference) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  const auto q0 = home_q();
  auto pkg = make_hold_package(q0, 7);
  kmpc::Vec18 measured = q0;
  measured[4] += 0.0005;
  kmpc::TrajectoryFollower follower(kin, nullptr, kmpc::policy_from_config(config));
  ASSERT_TRUE(follower.adopt(pkg, measured, kmpc::Vec18::Zero(), false));
  const auto tick = follower.step(measured, 0.0001, 10.0, 10.0, 0.0);
  EXPECT_EQ(tick.action, kmpc::FollowerAction::Continue) << tick.reason;
  EXPECT_NEAR((tick.q_follow - q0).norm(), 0.0, 1e-12);
}

TEST(R53Execution, PassThroughStageCannotHideLaterAckBarrier) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  const auto q0 = home_q();
  auto pkg = make_hold_package(q0, 7);
  pkg.samples = {q0, q0, q0, q0};
  pkg.time_law.s = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
  pkg.time_law.t_s = {0.0, 0.2, 0.4, 0.6};
  pkg.time_law.ds_dt = {0.0, 0.0, 0.0, 0.0};
  pkg.time_law.duration_s = 0.6;
  kmpc::StageBoundary pass, ack, end;
  pass.sample_index = 1; pass.stage_id = 1;
  ack.sample_index = 2; ack.stage_id = 2; ack.wait_for_external_ack = true;
  end.sample_index = 3; end.stage_id = 3; end.stop_required = true;
  pkg.stage_boundaries = {pass, ack, end};
  kmpc::TrajectoryFollower follower(kin, nullptr, kmpc::policy_from_config(config));
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));
  auto tick = follower.step(q0, 0.02, 10.2, 10.2, 0.2);
  EXPECT_TRUE(tick.barrier_pending);
  EXPECT_EQ(tick.barrier_stage_id, 2u);
  for (int i = 0; i < 15 && !follower.waiting_ack(); ++i)
    tick = follower.step(q0, 0.02, 10.4 + i * 0.02, 10.4 + i * 0.02, 0.4);
  ASSERT_TRUE(follower.waiting_ack()) << tick.reason;
  EXPECT_DOUBLE_EQ(tick.exec_t_s, 0.4);
  std::string why;
  EXPECT_FALSE(follower.on_stage_ack(6, 2, true, q0, &why));
  ASSERT_TRUE(follower.on_stage_ack(7, 2, true, q0, &why,
                                     10.7, 10.7)) << why;
  tick = follower.step(q0, 0.02, 10.72, 10.72, 0.404);
  EXPECT_NEAR(tick.exec_t_s, 0.404, 1e-12);
  EXPECT_EQ(tick.barrier_stage_id, 3u);
}

TEST(R53Execution, CertifiedPitchRecoveryUsesCurveNotMeasuredStaircase) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  const auto policy = kmpc::policy_from_config(config);
  auto q0 = home_q();
  q0[1] += -policy.torso_pitch_max_rad - 0.000254 - q0.head<3>().sum();
  auto pkg = make_hold_package(q0, 7);
  pkg.samples.back()[1] += 0.0005;
  pkg.time_law.t_s = {0.0, 2.0};
  pkg.time_law.duration_s = 2.0;
  kmpc::TrajectoryFollower follower(kin, nullptr, policy);
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));
  kmpc::SafetySupervisor supervisor(kin, nullptr, policy);
  supervisor.seed(q0);
  for (int i = 0; i <= 500; ++i) {
    const double t = i * 0.004;
    kmpc::TrajectoryTimeSample sample;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, t, &sample));
    if (i % 5 == 0) {
      const auto tick = follower.step(q0, 0.02, 10.0 + t, 10.0 + t, t);
      EXPECT_NEAR((tick.q_follow - sample.q).norm(), 0.0, 1e-12);
      EXPECT_NE(tick.state, kmpc::FollowerState::Recovering);
    }
    kmpc::SafetySupervisorInput in;
    in.q_cmd = sample.q; in.q_plan = sample.q;
    in.q_meas = q0; in.has_meas = true;
    in.now_s = in.meas_timestamp_s = 1000.0 + t;
    in.send_timestamp_s = 10.0 + t;
    in.mode = kmpc::SafetyExecMode::Recovering;
    in.pitch_excess_ceiling = follower.pitch_excess_of(supervisor.q_last_send());
    const auto d = supervisor.check(in, false);
    ASSERT_TRUE(d.accept) << d.reason << " i=" << i;
    supervisor.note_sent(d.q_send, in.send_timestamp_s);
  }
}

TEST(R53Execution, FieldLeftTwoStagesPassNominalSendingWithLaggedFeedback) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  // Measured q at t=35.6358 in 20261116T071131Z; radians, URDF 18-axis order.
  kmpc::Vec18 q0;
  q0 << 0.6981290376366417, -0.8726553049333495, -0.17453825152161312,
      -0.026161562968393396, -0.0017102986968056998, 0.5153259652775903,
      -0.13651973611635804, -2.126579257692943, -1.2752244744218761,
      -0.24291496577261937, 0.01394496394212196, -0.0017104484996170166,
      0.51532591534332, 0.1365200956431052, -2.126578658481698,
      1.2752249537908726, -0.2429148459303703, 0.013944844099872905;
  const auto policy = kmpc::policy_from_config(config, 0.3);
  auto weights = kmpc::load_trajectory_quality_weights(
      cfg("trajectory_quality.yaml"), policy.dq_max, nullptr);
  kmpc::MotionTask task;
  task.generation = 1794813127691249189ULL;
  task.q = q0; task.left = true; task.frame = std::string(kmpc::kEeFrameLeft);
  for (int i = 0; i < 2; ++i) {
    kmpc::TaskStage stage;
    stage.stage_id = i + 1;
    // Field grasp contract: PTP to PREP, LIN approach to TIP.
    stage.geometry =
        i == 0 ? kmpc::TaskGeometry::Ptp : kmpc::TaskGeometry::Lin;
    stage.position = kmpc::Vec3(0.456071795, 0.272798503, i == 0 ? 0.715 : 0.565);
    stage.has_orientation = true;
    stage.quat = kmpc::tool_to_flange_quat_xyzw(
        kmpc::Vec4(0.778376915, 0.229195584, -0.312912199, 0.493644324));
    stage.stop = i == 0 ? kmpc::StageStop::Continuous
                        : kmpc::StageStop::StopRequired;
    task.stages.push_back(stage);
  }
  kmpc::seal_motion_task_identity(&task, urdf, false, policy);
  kmpc::RobotState state; state.q = q0; state.fresh = true;
  const auto plan = kmpc::FastTrajectoryOptimizer(kin, &guards).plan(
      task, state, policy, weights, kmpc::kPlannerBudgetMs);
  ASSERT_TRUE(plan.ok) << plan.reason << " " << plan.report;
  const auto& pkg = plan.package;
  std::size_t stage_start = 0;
  for (const auto& b : pkg.stage_boundaries) {
    if (b.sample_index <= stage_start) continue;
    kmpc::JointCurve segment(pkg.samples.begin() + stage_start,
                             pkg.samples.begin() + b.sample_index + 1);
    const auto direction = kmpc::check_path_direction_strict(segment);
    EXPECT_TRUE(direction.ok) << "stage=" << b.stage_id << " " << direction.reason;
    stage_start = b.sample_index;
  }
  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false));
  kmpc::SafetySupervisor supervisor(kin, &guards, policy);
  supervisor.seed(q0);
  kmpc::Vec18 measured = q0, previous_send = q0;
  double max_waist = 0.0, max_arm = 0.0;
  bool arrived = false;
  const int ticks = static_cast<int>(std::ceil(pkg.time_law.duration_s / 0.004)) + 100;
  double exec_t = 0.0;
  for (int i = 0; i < ticks; ++i) {
    exec_t = std::min(i * 0.004, pkg.time_law.duration_s);
    kmpc::TrajectoryTimeSample sample;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, exec_t, &sample));
    if (i % 5 == 0) {
      const auto tick = follower.step(measured, 0.02, 100.0 + i * 0.004,
                                     100.0 + i * 0.004, exec_t);
      ASSERT_NE(tick.action, kmpc::FollowerAction::Brake) << tick.reason;
      ASSERT_NE(tick.action, kmpc::FollowerAction::NeedReplan) << tick.reason;
      EXPECT_FALSE(tick.qp_used);
      EXPECT_NEAR(tick.q_offset.norm(), 0.0, 1e-12);
      arrived = arrived || tick.action == kmpc::FollowerAction::Arrived;
    }
    kmpc::SafetySupervisorInput in;
    in.q_cmd = sample.q; in.q_plan = sample.q; in.q_meas = measured;
    in.has_meas = true; in.now_s = in.meas_timestamp_s = 100.0 + i * 0.004;
    in.send_timestamp_s = 10.0 + i * 0.004;
    in.package = &pkg; in.expected_generation = pkg.generation;
    in.expected_path_id = pkg.path_id; in.expected_task_digest = pkg.task_digest;
    const auto d = supervisor.check(in, false);
    ASSERT_TRUE(d.accept) << d.reason << " i=" << i << " joint=" << d.joint
                          << " actual=" << d.actual << " limit=" << d.limit;
    supervisor.note_sent(d.q_send, in.send_timestamp_s);
    max_waist = std::max(max_waist, (d.q_send - q0).head<3>().cwiseAbs().maxCoeff());
    max_arm = std::max(max_arm, (d.q_send - q0).segment<7>(kmpc::kSliceLeft.begin).cwiseAbs().maxCoeff());
    // One send-cycle plant delay; not a hardware replay or motor model.
    measured = previous_send;
    previous_send = d.q_send;
  }
  EXPECT_TRUE(arrived);
  EXPECT_GT(max_waist, 1e-4);
  EXPECT_GT(max_arm, 1e-4);
  const double tcp_error = (kin.fk_pos(previous_send, kmpc::kEeFrameLeft) -
                            task.stages.back().position).norm();
  EXPECT_LE(tcp_error, task.stages.back().position_tol_m);
  std::printf("R53_FIELD nominal_send=PASS duration_s=%.3f waist_exc_rad=%.6f "
              "arm_exc_rad=%.6f final_tcp_m=%.6f\n",
              pkg.time_law.duration_s, max_waist, max_arm, tcp_error);
}

TEST(R53Execution, NonAlignedAckStopAndRestartShareTheCertifiedSendGrid) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  const auto policy = kmpc::policy_from_config(config, 0.3);
  const auto q0 = home_q();
  auto pkg = make_hold_package(q0, 7);
  auto q1 = q0, q2 = q0;
  q1[11] += 0.03; q2[11] += 0.06;
  pkg.samples = {q0, q1, q2};
  kmpc::StageBoundary ack;
  ack.sample_index = 1; ack.stage_id = 1;
  ack.wait_for_external_ack = true;  // ACK alone must force an actual zero rate.
  pkg.stage_boundaries = {ack};
  pkg.time_law = kmpc::TimeParameterizer().compute(pkg.samples, policy,
                                                  pkg.stage_boundaries);
  // Deliberately place the barrier between 4 ms ticks, while slowing only.
  for (auto& t : pkg.time_law.t_s) t *= 1.003;
  pkg.time_law.duration_s *= 1.003;
  for (auto& v : pkg.time_law.ds_dt) v /= 1.003;
  EXPECT_DOUBLE_EQ(pkg.time_law.ds_dt[1], 0.0);
  const double barrier = kmpc::time_at_sample_index(pkg, 1);
  kmpc::TrajectoryTimeSample at_barrier;
  ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, barrier, &at_barrier));
  // The public diagnostic dq uses a coarse 1 ms finite difference. Probe
  // q(t) near the knot directly to test the actual zero derivative limit.
  kmpc::TrajectoryTimeSample before, after;
  constexpr double h = 1e-7;
  ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, barrier - h, &before));
  ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, barrier + h, &after));
  EXPECT_LT(((after.q - before.q) / (2.0 * h)).norm(), 1e-5);
  kmpc::SafetySupervisor supervisor(kin, nullptr, policy);
  supervisor.seed(q0);
  int held_at_barrier = 0;
  kmpc::Vec18 measured = q0;
  for (const auto& probe : kmpc::trajectory_send_probe_grid(pkg)) {
    kmpc::TrajectoryTimeSample sample;
    ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, probe.curve_s, &sample));
    if (std::abs(probe.curve_s - barrier) < 1e-12) ++held_at_barrier;
    kmpc::SafetySupervisorInput in;
    in.q_cmd = sample.q; in.q_plan = sample.q; in.q_meas = measured;
    in.has_meas = true; in.send_timestamp_s = 10.0 + probe.send_s;
    in.now_s = in.meas_timestamp_s = 100.0 + probe.send_s;
    const auto d = supervisor.check(in, false);
    ASSERT_TRUE(d.accept) << d.reason << " at curve=" << probe.curve_s;
    supervisor.note_sent(d.q_send, in.send_timestamp_s);
    measured = d.q_send;
  }
  EXPECT_GE(held_at_barrier, 3);
}

TEST(R53Execution, UnpublishedCommandCannotConsumePathTime) {
  kmpc::V2ExecutionState exec;
  exec.adopt(7, 42, 1, 10.0);
  const auto before = exec.exec_s();
  EXPECT_GT(exec.advance(10.004, false, false, 0.0), before);
  exec.discard_unsent_progress(before, 10.005);
  exec.latch_fault("safety_publish_failed");
  EXPECT_DOUBLE_EQ(exec.exec_s(), before);
  EXPECT_DOUBLE_EQ(exec.advance(11.0, false, false, 0.0), before);
}

TEST(R53Execution, SparseStopTimeInvertsTheExecutedCubic) {
  kmpc::TrajectoryPackage pkg;
  pkg.samples = {kmpc::Vec18::Zero(), kmpc::Vec18::Zero(), kmpc::Vec18::Zero()};
  pkg.time_law.s = {0.0, 1.0};
  pkg.time_law.t_s = {0.0, 2.0};
  pkg.time_law.ds_dt = {0.0, 0.5};
  pkg.time_law.duration_s = 2.0;
  const double at_middle_sample = kmpc::time_at_sample_index(pkg, 1);
  EXPECT_GT(at_middle_sample, 1.0);
  kmpc::TrajectoryTimeSample sample;
  ASSERT_TRUE(kmpc::sample_trajectory_at(pkg, at_middle_sample, &sample));
  EXPECT_NEAR(sample.s, 0.5, 1e-12);
}
