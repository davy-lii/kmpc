#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/safety_supervisor.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::SafetySupervisorInput base_in(const kmpc::Vec18& q) {
  kmpc::SafetySupervisorInput in;
  in.q_prev = q;
  in.q_meas = q;
  in.has_meas = true;
  in.meas_timestamp_s = 1.0;
  in.now_s = 1.0;
  in.q_follow = q;
  in.has_follow = true;
  in.q_plan = q;
  in.q_cmd = q;
  in.dt = 0.004;  // 250 Hz
  in.expected_generation = 0;
  return in;
}

}  // namespace

TEST(StreamV2, AllowsSmoothReversalAt250Hz) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor gate(kin, &guards, policy);
  gate.seed(kmpc::test_fixture_q0());

  // Smooth reversal at 250 Hz: no direction-sign hard gate.
  // SafetySupervisor accel proxy uses (q_prev-last_safe)/dt; after accept,
  // last_safe==q_prev so |v_cmd| must stay within 4*ddq_max*dt. Keep steps
  // inside that software band while changing sign.
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const int j = 5;  // RightArm J2
  const double dt = 0.004;
  const double step =
      std::min(0.5 * policy.dq_max[j] * dt, 2.0 * policy.ddq_max[j] * dt * dt);
  ASSERT_GT(step, 0.0);
  double t = 1.0;
  auto tick = [&](double dq, const char* tag, int k) {
    auto in = base_in(q);
    in.q_cmd = q;
    in.q_cmd[j] = q[j] + dq;
    in.q_follow = in.q_cmd;
    in.dt = dt;
    t += dt;
    in.now_s = t;
    in.meas_timestamp_s = t;
    const auto d = gate.check(in);
    ASSERT_TRUE(d.accept) << tag << " k=" << k << " " << d.reason;
    q = d.q_send;
  };
  for (int k = 0; k < 12; ++k) {
    tick(+step, "forward", k);
  }
  for (int k = 0; k < 24; ++k) {
    tick(-step, "reverse", k);  // sign change must remain acceptable
  }
}

TEST(StreamV2, RejectsHardBoxPitchAndStepViolations) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor gate(kin, &guards, policy);
  gate.seed(kmpc::test_fixture_q0());

  {
    auto in = base_in(kmpc::test_fixture_q0());
    in.q_cmd[0] = policy.q_max[0] + 0.02;
    const auto d = gate.check(in);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "safety_hard_box");
    EXPECT_LT((d.q_send - kmpc::test_fixture_q0()).cwiseAbs().maxCoeff(), 1e-12);
  }

  {
    kmpc::SafetySupervisor gate2(kin, &guards, policy);
    gate2.seed(kmpc::test_fixture_q0());
    auto in = base_in(kmpc::test_fixture_q0());
    const double hard = policy.torso_pitch_max_rad;
    ASSERT_GT(hard, 0.0);
    // Push pitch sum past hard box without necessarily tripping J1 alone.
    in.q_cmd[0] = policy.q_max[0] - 0.01;
    in.q_cmd[1] = policy.q_min[1] + 0.01;
    in.q_cmd[2] = -(hard + 0.05) - in.q_cmd[0] - in.q_cmd[1];
    in.q_cmd[2] = std::clamp(in.q_cmd[2], policy.q_min[2], policy.q_max[2]);
    // If clamp killed pitch excess, force via equal split outside hard.
    if (std::abs(in.q_cmd[0] + in.q_cmd[1] + in.q_cmd[2]) <= hard) {
      const double want = -(hard + 0.05);
      in.q_cmd[0] = std::clamp(want / 3.0, policy.q_min[0], policy.q_max[0]);
      in.q_cmd[1] = std::clamp(want / 3.0, policy.q_min[1], policy.q_max[1]);
      in.q_cmd[2] =
          std::clamp(want - in.q_cmd[0] - in.q_cmd[1], policy.q_min[2],
                     policy.q_max[2]);
    }
    // May hit hard_box first if a waist joint is outside; either reject is ok
    // as long as it is not accepted and not silent-clamped to a new pose.
    const auto d = gate2.check(in);
    EXPECT_FALSE(d.accept);
    EXPECT_TRUE(std::string(d.reason) == "safety_pitch" ||
                std::string(d.reason) == "safety_hard_box")
        << d.reason;
    EXPECT_LT((d.q_send - kmpc::test_fixture_q0()).cwiseAbs().maxCoeff(), 1e-12);
  }

  {
    kmpc::SafetySupervisor gate3(kin, &guards, policy);
    gate3.seed(kmpc::test_fixture_q0());
    auto in = base_in(kmpc::test_fixture_q0());
    const double dt = 0.004;
    in.dt = dt;
    // One joint step far beyond dq_max*dt.
    in.q_cmd[5] = kmpc::test_fixture_q0()[5] + 5.0 * policy.dq_max[5] * dt;
    const auto d = gate3.check(in);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "safety_step");
    EXPECT_LT((d.q_send - kmpc::test_fixture_q0()).cwiseAbs().maxCoeff(), 1e-12);
  }
}

// r55 field magnitude: waist J1 reverse jump Δ≈0.00348 rad in 4 ms
// (≈7× dq_max*dt). Hold/mode-switch must not publish that jump; last_send
// history stays put; a zero-step hold of last_send remains accepted.
TEST(StreamV2, R55HoldReverseJumpNeverEntersSendHistory) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor gate(kin, &guards, policy);

  // Stay inside pitch hard box: only perturb J4 (yaw), same Δ as field J1.
  constexpr double kFieldDelta = 0.00347760938;
  kmpc::Vec18 q_last = kmpc::test_fixture_q0();
  gate.seed(q_last);
  gate.note_sent(q_last, 50.812);

  kmpc::Vec18 q_hold_bad = q_last;
  q_hold_bad[kmpc::kIdxYaw] -= kFieldDelta;

  auto in = base_in(q_last);
  in.q_cmd = q_hold_bad;
  in.q_plan = q_hold_bad;
  in.q_follow = q_hold_bad;
  in.has_follow = false;  // hold path does not enforce follow hint
  in.dt = 0.004;
  in.send_timestamp_s = 50.816;
  in.now_s = 150.0;
  in.meas_timestamp_s = 150.0;
  const auto rejected = gate.check(in, false);
  EXPECT_FALSE(rejected.accept);
  EXPECT_STREQ(rejected.reason, "safety_step");
  EXPECT_EQ(rejected.joint, kmpc::kIdxYaw);
  EXPECT_GT(rejected.actual, rejected.limit);
  // Unsent candidate must not mutate history.
  EXPECT_NEAR((gate.q_last_send() - q_last).norm(), 0.0, 1e-15);

  // Correct hold: republish last successful send (zero step).
  in.q_cmd = q_last;
  in.q_plan = q_last;
  in.send_timestamp_s = 50.820;
  const auto hold_ok = gate.check(in, false);
  ASSERT_TRUE(hold_ok.accept) << hold_ok.reason;
  gate.note_sent(hold_ok.q_send, 50.820);
  EXPECT_NEAR((gate.q_last_send() - q_last).norm(), 0.0, 1e-15);

  // After a zero-vel hold, a full dq_max*dt step can trip safety_accel.
  // A bounded fraction of the limit must remain acceptable (mode-switch
  // continuous approach contract).
  const double lim = policy.dq_max[kmpc::kIdxYaw] * 0.004;
  ASSERT_GT(lim, 0.0);
  ASSERT_LT(lim, kFieldDelta);
  kmpc::Vec18 q_step = q_last;
  q_step[kmpc::kIdxYaw] = q_last[kmpc::kIdxYaw] - 0.25 * lim;
  in.q_cmd = q_step;
  in.q_plan = q_step;
  in.send_timestamp_s = 50.824;
  const auto step_ok = gate.check(in, false);
  ASSERT_TRUE(step_ok.accept) << step_ok.reason;
  gate.note_sent(step_ok.q_send, 50.824);
  EXPECT_NEAR(gate.q_last_send()[kmpc::kIdxYaw], q_step[kmpc::kIdxYaw], 1e-12);
}
