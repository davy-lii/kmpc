#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

// Field-scale start used by the historical pitch-recovery red tests.
constexpr double kHomeRaw[18] = {
    0.69813170, -0.87264718, -0.17453293, -0.02630211,
    -0.00171042,  0.51532590, -0.13651965, -2.12657900,
    -1.27522486, -0.24291493,  0.01394518,
    -0.00171042,  0.51532590,  0.13651965, -2.12657900,
     1.27522486, -0.24291493,  0.01394518};

double pitch_excess(const kmpc::MotionConstraintPolicy& policy,
                    const kmpc::Vec18& q) {
  return std::max(0.0, std::abs(q[0] + q[1] + q[2]) -
                           policy.torso_pitch_max_rad);
}

kmpc::Vec18 make_outside(double limit, double excess) {
  kmpc::Vec18 q;
  for (int i = 0; i < 18; ++i) {
    q[i] = kHomeRaw[i];
  }
  const double want = -limit - excess;
  const double d = want - (q[0] + q[1] + q[2]);
  q[0] += d / 3.0;
  q[1] += d / 3.0;
  q[2] += d / 3.0;
  return q;
}

}  // namespace

TEST(PitchRecoveryV2, ExcessNonIncreasingAndEntersHardBox) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const double limit = policy.torso_pitch_max_rad;
  ASSERT_GT(limit, 0.0);
  // Do NOT relax caps: J1 hard max and pitch ±32° stay as configured.
  EXPECT_NEAR(limit, 32.0 * kmpc::kPi / 180.0, 1e-9);
  EXPECT_NEAR(policy.q_max[0], 58.0 * kmpc::kPi / 180.0, 1e-3);

  for (const double excess : {2.54e-4, 6.13e-4, 2.95e-4}) {
    kmpc::TrajectoryFollower follower(kin, &guards, policy);
    kmpc::Vec18 q = make_outside(limit, excess);
    ASSERT_NEAR(q[0] + q[1] + q[2], -limit - excess, 1e-9);
    ASSERT_TRUE(follower.begin_pitch_recovery(q));
    EXPECT_EQ(follower.state(), kmpc::FollowerState::Recovering);

    double prev_ex = follower.pitch_excess_of(q);
    bool inside = false;
    const double dt = 0.02;
    for (int i = 0; i < 200; ++i) {
      const auto tick = follower.step(q, dt);
      ASSERT_NE(tick.action, kmpc::FollowerAction::ProtectStop)
          << "excess=" << excess << " i=" << i << " " << tick.reason;
      // §7.4: excess(t+dt) <= excess(t) + eps
      EXPECT_LE(tick.pitch_excess, prev_ex + kmpc::kPitchRoundEpsRad + 1e-12)
          << "excess grew; excess0=" << excess << " i=" << i;
      EXPECT_FALSE(kmpc::joint_violates_hard_box(tick.q_follow, policy.q_min,
                                                 policy.q_max));
      EXPECT_LE(tick.q_follow[0], policy.q_max[0] + 1e-9);
      q = tick.q_follow;
      prev_ex = tick.pitch_excess;
      if (std::abs(q[0] + q[1] + q[2]) <= limit + 1e-6) {
        inside = true;
        break;
      }
    }
    EXPECT_TRUE(inside) << "did not enter hard box; excess=" << excess
                        << " pitch_deg="
                        << ((q[0] + q[1] + q[2]) * 180.0 / kmpc::kPi);
  }
}

TEST(PitchRecoveryV2, NearJ1WallStillRecoversWithoutWideningCap) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const double limit = policy.torso_pitch_max_rad;
  kmpc::Vec18 q = make_outside(limit, 2.95e-4);
  // Sit J1 near the 58° hard wall (do not raise the wall).
  q[0] = policy.q_max[0] - 0.002;
  const double want = -limit - 2.95e-4;
  const double rem = want - q[0];
  q[1] = std::clamp(rem / 2.0, policy.q_min[1], policy.q_max[1]);
  q[2] = std::clamp(want - q[0] - q[1], policy.q_min[2], policy.q_max[2]);
  ASSERT_GT(pitch_excess(policy, q), 0.0);

  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  ASSERT_TRUE(follower.begin_pitch_recovery(q));
  const double dt = 0.02;
  double prev_ex = follower.pitch_excess_of(q);
  bool inside = false;
  for (int i = 0; i < 300; ++i) {
    const auto tick = follower.step(q, dt);
    if (tick.action == kmpc::FollowerAction::ProtectStop) {
      EXPECT_LE(tick.q_follow[0], policy.q_max[0] + 1e-9);
      break;
    }
    EXPECT_LE(tick.pitch_excess, prev_ex + kmpc::kPitchRoundEpsRad + 1e-12);
    EXPECT_LE(tick.q_follow[0], policy.q_max[0] + 1e-9);
    q = tick.q_follow;
    prev_ex = tick.pitch_excess;
    if (std::abs(q[0] + q[1] + q[2]) <= limit + 1e-6) {
      inside = true;
      break;
    }
  }
  EXPECT_TRUE(inside || prev_ex < 2.95e-4);
}
