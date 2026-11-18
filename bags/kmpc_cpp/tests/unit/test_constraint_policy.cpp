#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
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

TEST(ConstraintPolicy, YawTaskLimitFromPolicyConfig) {
  const kmpc::Config config = load_cfg();
  const auto policy = kmpc::policy_from_config(config);
  EXPECT_TRUE(policy.yaw_limited);
  EXPECT_NEAR(policy.yaw_abs_max_rad, 30.0 * kmpc::kPi / 180.0, 1e-6);
}

TEST(ConstraintPolicy, HardBoxMatchesFinalGateSlackNotSoftWall) {
  const kmpc::Config config = load_cfg();
  const auto policy = kmpc::policy_from_config(config);
  EXPECT_NEAR(policy.q_limit_margin_rad * 180.0 / kmpc::kPi, 0.3, 0.02);
  kmpc::Vec18 q = kmpc::test_fixture_q0();
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
      hold(kmpc::test_fixture_q0()), kmpc::test_fixture_q0(), policy, weights);
  const auto home_again = evaluate_path_cost(
      hold(kmpc::test_fixture_q0()), kmpc::test_fixture_q0(), policy, weights);
  EXPECT_DOUBLE_EQ(home.total, home_again.total);
  EXPECT_FALSE(home.depends_on_target_height);
  EXPECT_FALSE(home.calibrated);
  EXPECT_FALSE(home.limit_outside_hard_box);

  kmpc::Vec18 near = kmpc::test_fixture_q0();
  near[0] = policy.q_max[0] - 1e-4;
  near[1] = -0.70;
  near[2] = -near[0] - near[1];
  const auto wall = evaluate_path_cost(hold(near), kmpc::test_fixture_q0(), policy, weights);
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
  EXPECT_DOUBLE_EQ(stage.position_tol_m, kmpc::kDefaultTaskPositionTolM);
  EXPECT_LT(stage.position_tol_m, 0.20);

  const kmpc::Config config = load_cfg();
  kmpc::Kinematics kin(urdf());
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const kmpc::Vec3 tcp = kin.fk_pos(kmpc::test_fixture_q0(), kmpc::kEeFrameLeft);

  kmpc::MotionTask task;
  task.generation = 1;
  task.q = kmpc::test_fixture_q0();
  task.left = true;
  task.stages.resize(1);
  task.stages[0].geometry = kmpc::TaskGeometry::Lin;
  task.stages[0].position = tcp;
  const auto held = kmpc::validate_discrete_trajectory(
      task, hold(kmpc::test_fixture_q0()), policy, kin, &guards);
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
      task, hold(kmpc::test_fixture_q0()), policy, kin, &guards);
  EXPECT_FALSE(off.task_path_ok);
  EXPECT_FALSE(off.passed);
  EXPECT_GT(off.worst_tcp_endpoint_m, 0.015);
  EXPECT_STREQ(off.first_fault, "tcp_endpoint");

  task.stages[0].position = tcp;
  task.stages[0].position_tol_m = 0.20;
  const auto legacy = kmpc::validate_discrete_trajectory(
      task, hold(kmpc::test_fixture_q0()), policy, kin, &guards);
  EXPECT_FALSE(legacy.passed);
  EXPECT_STREQ(legacy.first_fault, "lin_tol_is_legacy_hard_tube");

  task.stages[0].position_tol_m = 0.003;
  const auto missing = kmpc::validate_discrete_trajectory(
      task, hold(kmpc::test_fixture_q0()), policy, kin, nullptr);
  EXPECT_FALSE(missing.passed);
  EXPECT_STREQ(missing.first_fault, "com_not_checked");
  EXPECT_FALSE(missing.com_checked);
}

TEST(WorkBox, EighteenAxisInsetAndNoGrowthRecovery) {
  const kmpc::Config config = load_cfg();
  const auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 work_lo;
  kmpc::Vec18 work_hi;
  kmpc::joint_work_limits(policy.q_min, policy.q_max, policy.q_limit_margin_rad,
                          work_lo, work_hi);
  for (int j = 0; j < kmpc::kNDof; ++j) {
    EXPECT_NEAR(work_lo[j], policy.q_min[j] + policy.q_limit_margin_rad, 1e-12)
        << "j=" << j;
    EXPECT_NEAR(work_hi[j], policy.q_max[j] - policy.q_limit_margin_rad, 1e-12)
        << "j=" << j;
  }
  EXPECT_NEAR(work_hi[0] * 180.0 / kmpc::kPi, 57.7, 0.05);

  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  q0[0] = policy.q_max[0] - 0.5 * policy.q_limit_margin_rad;
  kmpc::Vec18 rec_lo;
  kmpc::Vec18 rec_hi;
  kmpc::joint_recovery_limits(q0, policy.q_min, policy.q_max,
                              policy.q_limit_margin_rad, rec_lo, rec_hi);
  EXPECT_NEAR(rec_hi[0], q0[0], 1e-12);
  EXPECT_FALSE(kmpc::joint_violates_work_box(q0, q0, policy.q_min, policy.q_max,
                                             policy.q_limit_margin_rad));
  kmpc::Vec18 grown = q0;
  grown[0] = policy.q_max[0] - 1e-6;
  EXPECT_TRUE(kmpc::joint_violates_work_box(grown, q0, policy.q_min, policy.q_max,
                                            policy.q_limit_margin_rad));
  kmpc::Vec18 inward = q0;
  inward[0] = work_hi[0] - 0.01;
  EXPECT_FALSE(kmpc::joint_violates_work_box(
      inward, q0, policy.q_min, policy.q_max, policy.q_limit_margin_rad));
}

TEST(WorkBox, ChanGainOutwardCostlierThanInward) {
  const kmpc::Config config = load_cfg();
  const auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 work_lo;
  kmpc::Vec18 work_hi;
  kmpc::joint_work_limits(policy.q_min, policy.q_max, policy.q_limit_margin_rad,
                          work_lo, work_hi);
  const double q = work_hi[0] - 1e-4;
  const double g_out =
      kmpc::joint_chan_gain(q, work_lo[0], work_hi[0], +0.01);
  const double g_in =
      kmpc::joint_chan_gain(q, work_lo[0], work_hi[0], -0.01);
  EXPECT_GT(g_out, 5.0);
  EXPECT_NEAR(g_in, 1.0, 1e-12);
}

TEST(TrajectoryCost, FoldbackDensifyStableAndSeamNotIntraStage) {
  const kmpc::Config config = load_cfg();
  auto policy = kmpc::policy_from_config(config);
  auto weights = scales_from(config);
  weights.foldback = 1.0;
  weights.wrist_foldback = 1.0;
  weights.j4_twist = 0.0;
  weights.move = 0.0;
  weights.smooth = 0.0;
  weights.neutral = 0.0;
  weights.joint_limit = 0.0;
  weights.pitch_comfort = 0.0;
  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  kmpc::JointCurve fold;
  fold.push_back(q0);
  kmpc::Vec18 a = q0;
  a[3] += 0.20;
  fold.push_back(a);
  kmpc::Vec18 b = q0;
  b[3] -= 0.05;
  fold.push_back(b);
  const auto c_sparse = kmpc::evaluate_path_cost(fold, q0, policy, weights);

  kmpc::JointCurve dense = fold;
  dense.insert(dense.begin() + 1, 0.5 * (fold[0] + fold[1]));
  dense.insert(dense.begin() + 3, 0.5 * (fold[1] + fold[2]));
  const auto c_dense = kmpc::evaluate_path_cost(dense, q0, policy, weights);
  EXPECT_NEAR(c_sparse.foldback, c_dense.foldback, 1e-9);

  kmpc::Kinematics kin(urdf());
  kmpc::JointCurve seam;
  seam.push_back(q0);
  kmpc::Vec18 mid = q0;
  mid[3] += 0.15;
  seam.push_back(mid);
  kmpc::Vec18 end = q0;
  seam.push_back(end);
  std::vector<int> ends = {0, 1, 2};
  const auto c_seam = kmpc::evaluate_path_cost(
      seam, q0, policy, weights, kin, true, nullptr, &ends);
  const double seam_ruler =
      kmpc::joint_path_foldback_rad(seam, kmpc::kIdxYaw);
  EXPECT_GT(seam_ruler, 0.20);
  EXPECT_NEAR(c_seam.foldback, weights.foldback * seam_ruler * seam_ruler,
              1e-9);
}

TEST(TrajectoryCost, FoldbackGnMatchesJointPathRuler) {
  const kmpc::Config config = load_cfg();
  auto policy = kmpc::policy_from_config(config);
  auto weights = scales_from(config);
  weights.foldback = 1.0;
  weights.wrist_foldback = 1.0;
  weights.j4_twist = 0.0;
  weights.move = 0.0;
  weights.smooth = 0.0;
  weights.neutral = 0.0;
  weights.joint_limit = 0.0;
  weights.pitch_comfort = 0.0;
  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  kmpc::JointCurve curve;
  curve.push_back(q0);
  kmpc::Vec18 k1 = q0;
  k1[kmpc::kIdxYaw] += 0.30;
  curve.push_back(k1);
  kmpc::Vec18 k2 = q0;
  k2[kmpc::kIdxYaw] += 0.10;
  curve.push_back(k2);
  const double ruler = kmpc::joint_path_foldback_rad(curve, kmpc::kIdxYaw);
  EXPECT_NEAR(ruler, 0.40, 1e-12);
  const auto c =
      kmpc::evaluate_path_cost(curve, q0, policy, weights, kmpc::Kinematics(urdf()),
                               true, nullptr, nullptr);
  EXPECT_NEAR(std::sqrt(c.foldback / weights.foldback), ruler, 1e-9);
}

TEST(TrajectoryCost, FoldbackGradientMatchesFiniteDifference) {
  const kmpc::Config config = load_cfg();
  auto policy = kmpc::policy_from_config(config);
  auto weights = scales_from(config);
  weights.foldback = 1.0;
  weights.wrist_foldback = 1.0;
  weights.j4_twist = 0.0;
  weights.move = 0.0;
  weights.smooth = 0.0;
  weights.neutral = 0.0;
  weights.joint_limit = 0.0;
  weights.pitch_comfort = 0.0;
  weights.body_forward = 0.0;
  weights.com_margin = 0.0;
  weights.arm_manip = 0.0;
  weights.wrist_span = 0.0;
  weights.height_coop = 0.0;
  weights.pitch_near_barrier = 0.0;
  kmpc::Kinematics kin(urdf());
  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  kmpc::JointCurve knots(3, q0);
  knots[1][3] += 0.18;
  knots[2][3] -= 0.04;
  std::vector<int> active = {3};
  std::vector<int> ends = {0, 2};
  const auto lin = kmpc::value_and_linearize(
      knots, q0, q0, policy, weights, kin, true, nullptr, active, &ends);
  const double h = 1e-6;
  knots[2][3] += h;
  const auto plus = kmpc::evaluate_path_cost(
      knots, q0, policy, weights, kin, true, nullptr, &ends);
  knots[2][3] -= 2.0 * h;
  const auto minus = kmpc::evaluate_path_cost(
      knots, q0, policy, weights, kin, true, nullptr, &ends);
  const double fd = (plus.foldback - minus.foldback) / (2.0 * h);
  ASSERT_EQ(lin.grad.size(), 3u);
  const double fd_total = (plus.total - minus.total) / (2.0 * h);
  EXPECT_NEAR(lin.grad[2], fd, 2e-4);
  EXPECT_NEAR(lin.grad[2], fd_total, 2e-4);
}

TEST(TrajectoryValidator, PastHardJointFailsInsideSoftWallDoesNot) {
  const kmpc::Config config = load_cfg();
  kmpc::Kinematics kin(urdf());
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);

  kmpc::Vec18 q = kmpc::test_fixture_q0();
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
  EXPECT_TRUE(inside.work_box_ok);
  EXPECT_TRUE(inside.pitch_ok);
  EXPECT_GT(inside.min_joint_margin_rad, 1e-3);
  EXPECT_STRNE(inside.first_fault, "joint");
  EXPECT_STRNE(inside.first_fault, "work_box");

  q[0] = policy.q_max[0] + 0.01;
  task.stages[0].position = kin.fk_pos(kmpc::test_fixture_q0(), kmpc::kEeFrameLeft);
  const auto past = kmpc::validate_discrete_trajectory(
      task, hold(q), policy, kin, &guards);
  EXPECT_FALSE(past.hard_box_ok);
  EXPECT_FALSE(past.passed);
  EXPECT_STREQ(past.first_fault, "joint");
}

TEST(TrajectoryValidator, GrowingIntoMarginFailsWorkBoxKeepsHardOk) {
  const kmpc::Config config = load_cfg();
  kmpc::Kinematics kin(urdf());
  kmpc::WholeBodyGuards guards(config.params,
                               kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  kmpc::Vec18 q1 = q0;
  q1[0] = policy.q_max[0] - 0.4 * policy.q_limit_margin_rad;
  q1[2] -= (q1[0] - q0[0]);
  ASSERT_LT(q1[0], policy.q_max[0]);
  ASSERT_GT(q1[0], policy.q_max[0] - policy.q_limit_margin_rad);

  kmpc::MotionTask task;
  task.left = true;
  task.q = q0;
  task.stages.resize(1);
  task.stages[0].geometry = kmpc::TaskGeometry::Ptp;
  task.stages[0].position = kin.fk_pos(q1, kmpc::kEeFrameLeft);
  task.stages[0].position_tol_m = 1.0;
  kmpc::JointCurve curve;
  curve.push_back(q0);
  curve.push_back(q1);
  const auto grown = kmpc::validate_discrete_trajectory(
      task, curve, policy, kin, &guards);
  EXPECT_TRUE(grown.hard_box_ok);
  EXPECT_FALSE(grown.work_box_ok);
  EXPECT_FALSE(grown.passed);
  EXPECT_STREQ(grown.first_fault, "work_box");
}
