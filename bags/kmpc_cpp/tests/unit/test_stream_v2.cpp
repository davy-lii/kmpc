#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor gate(kin, &guards, policy);
  gate.seed(config.q_pref);

  // Smooth reversal at 250 Hz: no direction-sign hard gate.
  // SafetySupervisor accel proxy uses (q_prev-last_safe)/dt; after accept,
  // last_safe==q_prev so |v_cmd| must stay within 4*ddq_max*dt. Keep steps
  // inside that software band while changing sign.
  kmpc::Vec18 q = config.q_pref;
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor gate(kin, &guards, policy);
  gate.seed(config.q_pref);

  {
    auto in = base_in(config.q_pref);
    in.q_cmd[0] = policy.q_max[0] + 0.02;
    const auto d = gate.check(in);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "safety_hard_box");
    EXPECT_LT((d.q_send - config.q_pref).cwiseAbs().maxCoeff(), 1e-12);
  }

  {
    kmpc::SafetySupervisor gate2(kin, &guards, policy);
    gate2.seed(config.q_pref);
    auto in = base_in(config.q_pref);
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
    EXPECT_LT((d.q_send - config.q_pref).cwiseAbs().maxCoeff(), 1e-12);
  }

  {
    kmpc::SafetySupervisor gate3(kin, &guards, policy);
    gate3.seed(config.q_pref);
    auto in = base_in(config.q_pref);
    const double dt = 0.004;
    in.dt = dt;
    // One joint step far beyond dq_max*dt.
    in.q_cmd[5] = config.q_pref[5] + 5.0 * policy.dq_max[5] * dt;
    const auto d = gate3.check(in);
    EXPECT_FALSE(d.accept);
    EXPECT_STREQ(d.reason, "safety_step");
    EXPECT_LT((d.q_send - config.q_pref).cwiseAbs().maxCoeff(), 1e-12);
  }
}
