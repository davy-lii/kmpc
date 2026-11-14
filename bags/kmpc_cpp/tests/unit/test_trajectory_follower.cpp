#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/safety_supervisor.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::TrajectoryPackage plan_short(const kmpc::Config& config,
                                   kmpc::Kinematics& kin,
                                   const kmpc::WholeBodyGuards& guards,
                                   const kmpc::Vec18& q0, bool left,
                                   const kmpc::Vec3& goal) {
  kmpc::MotionTask task;
  task.generation = 7;
  task.q = q0;
  task.dq_known = false;
  task.left = left;
  task.frame = left ? std::string(kmpc::kEeFrameLeft)
                    : std::string(kmpc::kEeFrameRight);
  kmpc::TaskStage st;
  st.geometry = kmpc::TaskGeometry::Lin;
  st.position = goal;
  st.position_tol_m = 0.003;
  st.stage_id = 1;
  task.stages.push_back(st);
  kmpc::RobotState state;
  state.q = q0;
  state.dq_known = false;
  state.fresh = true;
  kmpc::TrajectoryCostWeights w;
  w.calibrated = true;
  for (int j = 0; j < kmpc::kNDof; ++j) {
    w.motion_scale[j] = 1.0 / std::max(config.limits.dq_max[j], 1e-3);
  }
  kmpc::FastTrajectoryOptimizer opt(kin, &guards);
  const auto res =
      opt.plan(task, state, kmpc::policy_from_config(config), w, /*deadline*/0);
  EXPECT_TRUE(res.ok) << res.reason << " " << res.report;
  return res.package;
}

}  // namespace

TEST(TrajectoryFollower, FollowsCertifiedNominalAndRecordsLayers) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);
  auto package =
      plan_short(config, kin, guards, q0, false, p0 + kmpc::Vec3(0.05, 0, 0));
  ASSERT_GE(package.samples.size(), 2u);

  kmpc::TrajectoryFollower follower(kin, &guards,
                                    kmpc::policy_from_config(config));
  follower.set_enable_qp_correction(false);
  std::string why;
  ASSERT_TRUE(follower.adopt(package, q0, kmpc::Vec18::Zero(), false, 1e-3, 0.5,
                             &why))
      << why;
  EXPECT_EQ(follower.state(), kmpc::FollowerState::Active);

  kmpc::Vec18 q = q0;
  bool arrived = false;
  const double dt = 0.02;
  const double dur = std::max(package.time_law.duration_s, 1.0);
  for (int i = 0; i < static_cast<int>(dur / dt) + 50; ++i) {
    const double t = std::min(static_cast<double>(i) * dt, dur);
    const auto tick = follower.step(q, t, dt);
    EXPECT_TRUE(tick.q_plan.allFinite());
    EXPECT_TRUE(tick.q_ref.allFinite());
    EXPECT_TRUE(tick.q_follow.allFinite());
    EXPECT_EQ(tick.generation, package.generation);
    if (tick.action == kmpc::FollowerAction::Brake ||
        tick.action == kmpc::FollowerAction::ProtectStop) {
      FAIL() << "unexpected " << tick.reason << " at i=" << i;
    }
    q = tick.q_follow;
    if (tick.action == kmpc::FollowerAction::Arrived) {
      arrived = true;
      break;
    }
  }
  EXPECT_TRUE(arrived);
}

TEST(TrajectoryFollower, SafetySupervisorRejectsHardBoxWithoutClamp) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::SafetySupervisor gate(kin, &guards, policy);
  gate.seed(config.q_pref);

  kmpc::SafetySupervisorInput in;
  in.q_prev = config.q_pref;
  in.q_meas = config.q_pref;
  in.q_follow = config.q_pref;
  in.q_plan = config.q_pref;
  in.q_cmd = config.q_pref;
  in.q_cmd[0] = policy.q_max[0] + 0.05;  // past J1 58°
  in.dt = 0.004;
  in.expected_generation = 0;
  const auto bad = gate.check(in);
  EXPECT_FALSE(bad.accept);
  EXPECT_FALSE(bad.arrived);
  EXPECT_STREQ(bad.reason, "safety_hard_box");
  EXPECT_LT((bad.q_send - config.q_pref).cwiseAbs().maxCoeff(), 1e-12)
      << "must not silent-clamp to a new geometry";
}

TEST(TrajectoryFollower, SafetyNeverFakesArrived) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  kmpc::SafetySupervisor gate(kin, &guards, kmpc::policy_from_config(config));
  gate.seed(config.q_pref);

  kmpc::TrajectoryPackage pkg;
  pkg.generation = 3;
  pkg.certificate.passed = true;
  pkg.left = false;
  pkg.samples.push_back(config.q_pref);
  auto q_end = config.q_pref;
  q_end[5] += 0.2;
  pkg.samples.push_back(q_end);

  kmpc::SafetySupervisorInput in;
  in.q_prev = config.q_pref;
  in.q_cmd = config.q_pref;
  in.q_meas = config.q_pref;
  in.q_follow = config.q_pref;
  in.q_plan = config.q_pref;
  in.dt = 0.004;
  in.claim_arrived = true;
  in.package = &pkg;
  in.expected_generation = 3;
  const auto d = gate.check(in);
  EXPECT_FALSE(d.accept);
  EXPECT_FALSE(d.arrived);
  EXPECT_STREQ(d.reason, "safety_fake_arrived");
}
