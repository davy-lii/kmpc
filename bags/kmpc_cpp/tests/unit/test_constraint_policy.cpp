#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/trajectory_validator.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

std::string urdf() {
  return std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
}

kmpc::Config load_cfg() {
  return kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
}

kmpc::TrajectoryCostWeights scales_from(const kmpc::Config& config) {
  kmpc::TrajectoryCostWeights weights;
  for (int i = 0; i < kmpc::kNDof; ++i) {
    weights.motion_scale[i] =
        1.0 / std::max(config.limits.dq_max[i], 1e-6);
  }
  weights.calibrated = false;
  return weights;
}

kmpc::JointCurve hold(const kmpc::Vec18& q) {
  return kmpc::JointCurve(4, q);
}

}  // namespace

TEST(ConstraintPolicy, HardBoxMatchesFinalGateSlackNotSoftWall) {
  const kmpc::Config config = load_cfg();
  const auto policy = kmpc::policy_from_config(config);
  EXPECT_NEAR(policy.q_limit_margin_rad * 180.0 / kmpc::kPi, 0.3, 0.02);
  kmpc::Vec18 q = config.q_pref;
  q[0] = policy.q_max[0] - policy.q_limit_margin_rad + 1.53e-5;
  q[1] = -0.70;
  q[2] = -q[0] - q[1];
  ASSERT_LT(q[0], policy.q_max[0]);
  ASSERT_GT(q[0], policy.q_max[0] - policy.q_limit_margin_rad);
  EXPECT_FALSE(kmpc::joint_violates_hard_box(q, policy.q_min, policy.q_max));
  EXPECT_GT(kmpc::joint_hard_margin_rad(q, policy.q_min, policy.q_max), 1e-3);

  q[0] = policy.q_max[0] + 0.01;
  EXPECT_TRUE(kmpc::joint_violates_hard_box(q, policy.q_min, policy.q_max));
}

TEST(TrajectoryCost, SameCurveIgnoresHeightAndWallCostsMore) {
  const kmpc::Config config = load_cfg();
  const auto policy = kmpc::policy_from_config(config);
  const auto weights = scales_from(config);
  const auto home = evaluate_path_cost(
      hold(config.q_pref), config.q_pref, policy, weights);
  const auto home_again = evaluate_path_cost(
      hold(config.q_pref), config.q_pref, policy, weights);
  EXPECT_DOUBLE_EQ(home.total, home_again.total);
  EXPECT_FALSE(home.depends_on_target_height);
  EXPECT_FALSE(home.calibrated);
  EXPECT_FALSE(home.limit_outside_hard_box);

  kmpc::Vec18 near = config.q_pref;
  near[0] = policy.q_max[0] - 1e-4;
  near[1] = -0.70;
  near[2] = -near[0] - near[1];
  const auto wall = evaluate_path_cost(hold(near), config.q_pref, policy, weights);
  EXPECT_GT(wall.limit, home.limit);
  EXPECT_FALSE(wall.limit_outside_hard_box);
  EXPECT_FALSE(wall.depends_on_target_height);
}

TEST(TrajectoryPackage, ChangedStartInvalidatesDigestAndAdmitDoesNotRewrite) {
  kmpc::TrajectoryPackage package;
  package.generation = 7;
  package.start_q.setConstant(0.1);
  package.start_dq_known = false;
  package.samples = hold(package.start_q);
  const uint64_t digest = kmpc::package_digest(package);
  const kmpc::Vec18 front = package.samples.front();

  kmpc::Vec18 moved = package.start_q;
  moved[0] += 0.01;
  const auto rejected = kmpc::admit_package_start(
      package, moved, package.start_dq, false, 1e-8, 1e-6);
  EXPECT_FALSE(rejected.ok);
  EXPECT_STREQ(rejected.reason, "start_q_mismatch");
  EXPECT_NEAR((package.samples.front() - front).norm(), 0.0, 0.0);
  EXPECT_EQ(kmpc::package_digest(package), digest);

  package.start_q[0] += 1e-6;
  EXPECT_NE(kmpc::package_digest(package), digest);
}

TEST(TrajectoryPackage, RewrittenFirstSampleIsNotAdmitted) {
  kmpc::TrajectoryPackage package;
  package.start_q.setZero();
  package.samples = hold(package.start_q);
  package.samples.front()[3] = 0.2;
  const auto decision = kmpc::admit_package_start(
      package, package.start_q, package.start_dq, false, 1e-8, 1e-6);
  EXPECT_FALSE(decision.ok);
  EXPECT_STREQ(decision.reason, "curve_start_rewritten");
  EXPECT_NEAR(package.samples.front()[3], 0.2, 0.0);
}

TEST(TrajectoryValidator, DefaultLinTolIsNotTheLegacyHardTube) {
  kmpc::TaskStage stage;
  EXPECT_DOUBLE_EQ(stage.position_tol_m, 0.003);
  EXPECT_LT(stage.position_tol_m, 0.20);

  const kmpc::Config config = load_cfg();
  kmpc::Kinematics kin(urdf());
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const kmpc::Vec3 tcp = kin.fk_pos(config.q_pref, kmpc::kEeFrameLeft);

  kmpc::MotionTask task;
  task.generation = 1;
  task.q = config.q_pref;
  task.left = true;
  task.stages.resize(1);
  task.stages[0].geometry = kmpc::TaskGeometry::Lin;
  task.stages[0].position = tcp;
  const auto held = kmpc::validate_discrete_trajectory(
      task, hold(config.q_pref), policy, kin, &guards);
  EXPECT_FALSE(held.continuous_certified);
  EXPECT_EQ(held.coverage, kmpc::PathCoverage::DiscreteSamples);
  EXPECT_TRUE(held.hard_box_ok);
  EXPECT_TRUE(held.pitch_ok);
  EXPECT_TRUE(held.com_checked);
  EXPECT_STREQ(held.com_model, "xy_work_rectangle");
  EXPECT_TRUE(held.task_path_ok) << held.first_fault;
  EXPECT_TRUE(held.passed) << held.first_fault;

  task.stages[0].position = tcp + kmpc::Vec3(0.02, 0.0, 0.0);
  const auto off = kmpc::validate_discrete_trajectory(
      task, hold(config.q_pref), policy, kin, &guards);
  EXPECT_FALSE(off.task_path_ok);
  EXPECT_FALSE(off.passed);
  EXPECT_GT(off.worst_tcp_endpoint_m, 0.015);
  EXPECT_STREQ(off.first_fault, "tcp_endpoint");

  task.stages[0].position = tcp;
  task.stages[0].position_tol_m = 0.20;
  const auto legacy = kmpc::validate_discrete_trajectory(
      task, hold(config.q_pref), policy, kin, &guards);
  EXPECT_FALSE(legacy.passed);
  EXPECT_STREQ(legacy.first_fault, "lin_tol_is_legacy_hard_tube");

  task.stages[0].position_tol_m = 0.003;
  const auto missing = kmpc::validate_discrete_trajectory(
      task, hold(config.q_pref), policy, kin, nullptr);
  EXPECT_FALSE(missing.passed);
  EXPECT_STREQ(missing.first_fault, "com_not_checked");
  EXPECT_FALSE(missing.com_checked);
}

TEST(TrajectoryValidator, PastHardJointFailsInsideSoftWallDoesNot) {
  const kmpc::Config config = load_cfg();
  kmpc::Kinematics kin(urdf());
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);

  kmpc::Vec18 q = config.q_pref;
  q[0] = policy.q_max[0] - policy.q_limit_margin_rad + 1.53e-5;
  q[1] = -0.70;
  q[2] = -q[0] - q[1];
  kmpc::MotionTask task;
  task.left = true;
  task.stages.resize(1);
  task.stages[0].geometry = kmpc::TaskGeometry::Ptp;
  task.stages[0].position = kin.fk_pos(q, kmpc::kEeFrameLeft);
  task.stages[0].position_tol_m = 0.003;
  const auto inside = kmpc::validate_discrete_trajectory(
      task, hold(q), policy, kin, &guards);
  EXPECT_TRUE(inside.hard_box_ok);
  EXPECT_TRUE(inside.pitch_ok);
  EXPECT_GT(inside.min_joint_margin_rad, 1e-3);
  EXPECT_STRNE(inside.first_fault, "joint");

  q[0] = policy.q_max[0] + 0.01;
  task.stages[0].position = kin.fk_pos(config.q_pref, kmpc::kEeFrameLeft);
  const auto past = kmpc::validate_discrete_trajectory(
      task, hold(q), policy, kin, &guards);
  EXPECT_FALSE(past.hard_box_ok);
  EXPECT_FALSE(past.passed);
  EXPECT_STREQ(past.first_fault, "joint");
}
