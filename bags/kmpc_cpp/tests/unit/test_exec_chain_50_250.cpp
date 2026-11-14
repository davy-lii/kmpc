// Offline hard gate A/D: simulate 50 Hz follower + 250 Hz supervisor with
// real measured timestamps (not command-as-meas).

#include <cmath>
#include <cstdio>
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q(config);
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q(config);
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q(config);
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = home_q(config);
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
