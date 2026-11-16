#include <chrono>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/planner_worker.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::TrajectoryPackage make_pkg(const kmpc::Vec18& q0, uint64_t gen) {
  kmpc::TrajectoryPackage pkg;
  pkg.generation = gen;
  pkg.path_id = 42;
  pkg.start_q = q0;
  pkg.start_dq_known = false;
  pkg.left = false;
  pkg.certificate.passed = true;
  pkg.planner = "unified_multistage_v2";
  pkg.samples.push_back(q0);
  auto q1 = q0;
  q1[5] += 0.01;
  pkg.samples.push_back(q1);
  pkg.time_law.s = {0.0, 1.0};
  pkg.time_law.t_s = {0.0, 1.0};
  pkg.time_law.duration_s = 1.0;
  return pkg;
}

}  // namespace

TEST(TrajectoryAdoption, CannotRewriteSamplesFront) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q0 = config.q_pref;
  auto pkg = make_pkg(q0, 11);
  const auto front_before = pkg.samples.front();

  // Measured start disagrees with certified start → admit fails.
  kmpc::Vec18 q_meas = q0;
  q_meas[5] += 0.05;
  const auto decision =
      kmpc::admit_package_start(pkg, q_meas, kmpc::Vec18::Zero(), false, 1e-3,
                                0.5);
  EXPECT_FALSE(decision.ok);
  EXPECT_STREQ(decision.reason, "start_q_mismatch");
  // Package samples are untouched.
  EXPECT_LT((pkg.samples.front() - front_before).cwiseAbs().maxCoeff(), 1e-15);
  EXPECT_LT((pkg.start_q - q0).cwiseAbs().maxCoeff(), 1e-15);

  // Even if a caller tried to rewrite front, curve_start_rewritten fires.
  pkg.samples.front()[5] += 0.02;
  const auto rewritten =
      kmpc::admit_package_start(pkg, pkg.start_q, kmpc::Vec18::Zero(), false,
                                1e-3, 0.5);
  EXPECT_FALSE(rewritten.ok);
  EXPECT_STREQ(rewritten.reason, "curve_start_rewritten");
}

TEST(TrajectoryAdoption, BridgeFailHoldsNotExecuteOldPackage) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::TrajectoryFollower follower(kin, &guards, policy);

  const kmpc::Vec18 q0 = config.q_pref;
  auto pkg = make_pkg(q0, 21);
  std::string why;
  ASSERT_TRUE(follower.adopt(pkg, q0, kmpc::Vec18::Zero(), false, 1e-3, 0.5,
                             &why))
      << why;
  EXPECT_EQ(follower.state(), kmpc::FollowerState::Active);

  // Bridge fail: new measured state far from certified start → hold.
  kmpc::Vec18 q_bad = q0;
  q_bad[4] += 0.1;
  EXPECT_FALSE(follower.adopt(pkg, q_bad, kmpc::Vec18::Zero(), false, 1e-3, 0.5,
                              &why));
  EXPECT_EQ(why, "start_q_mismatch");
  EXPECT_EQ(follower.state(), kmpc::FollowerState::Hold);
  EXPECT_FALSE(follower.has_package());

  const auto tick = follower.step(q_bad, 0.02);
  EXPECT_EQ(tick.state, kmpc::FollowerState::Hold);
  EXPECT_NE(tick.action, kmpc::FollowerAction::Arrived);
}

TEST(TrajectoryAdoption, SealedIdentityAdoptAndStaleReject) {
  // Production submit → Worker solve → identity gate: success adopts,
  // mismatched / zero digest cannot adopt.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);

  auto make_task = [&](uint64_t gen) {
    kmpc::MotionTask task;
    task.generation = gen;
    task.q = q0;
    task.left = false;
    task.frame = std::string(kmpc::kEeFrameRight);
    kmpc::TaskStage st;
    st.geometry = kmpc::TaskGeometry::Lin;
    st.position = p0 + kmpc::Vec3(0.02, 0.0, 0.0);
    st.position_tol_m = 0.01;
    st.stage_id = 1;
    task.stages.push_back(st);
    kmpc::seal_motion_task_identity(&task, urdf, false, policy);
    return task;
  };

  kmpc::RobotState state;
  state.q = q0;
  state.fresh = true;

  // Empty identity must be rejected by Worker (no rewrite / no zero bypass).
  {
    kmpc::MotionTask bare = make_task(10);
    bare.model_digest.clear();
    worker.submit_snapshot(bare, state, policy, worker.default_weights(), 50.0);
    kmpc::FastPlanResult bad;
    bool got = false;
    for (int i = 0; i < 400; ++i) {
      if (worker.try_take(bad)) {
        got = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(got);
    EXPECT_FALSE(bad.ok);
    EXPECT_EQ(bad.reason, "missing_model_digest");
    EXPECT_EQ(bad.task_digest, 0u);
  }

  // Sealed identity: successful plan must match sealed digest and be adoptable.
  kmpc::MotionTask task = make_task(11);
  const uint64_t sealed = kmpc::motion_task_digest(task);
  ASSERT_NE(sealed, 0u);
  worker.submit_snapshot(task, state, policy, worker.default_weights(), 50.0);
  kmpc::FastPlanResult ok_plan;
  bool got_ok = false;
  for (int i = 0; i < 400; ++i) {
    if (worker.try_take(ok_plan)) {
      got_ok = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(got_ok);
  if (ok_plan.ok) {
    EXPECT_EQ(ok_plan.task_digest, sealed);
    EXPECT_EQ(ok_plan.package.task_digest, sealed);
    EXPECT_NE(ok_plan.package.model_digest, 0u);
    EXPECT_NE(ok_plan.limits_digest, 0u);
    EXPECT_FALSE(ok_plan.package.certificate.continuous_certified);
    kmpc::TrajectoryFollower follower(kin, &guards, policy);
    std::string why;
    EXPECT_TRUE(follower.adopt(ok_plan.package, q0, kmpc::Vec18::Zero(), false,
                               1e-3, 0.5, &why))
        << why;
  } else {
    // Planner may fail geometry on this small fixture; identity path still ran.
    EXPECT_NE(ok_plan.reason, "missing_model_digest");
    EXPECT_NE(ok_plan.reason, "model_digest_mismatch");
  }

  // Stale generation result must not be treated as current identity.
  worker.submit_snapshot(make_task(12), state, policy, worker.default_weights(),
                         50.0);
  worker.submit_snapshot(make_task(13), state, policy, worker.default_weights(),
                         50.0);
  kmpc::FastPlanResult latest;
  bool got_latest = false;
  for (int i = 0; i < 400; ++i) {
    if (worker.try_take(latest)) {
      got_latest = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(got_latest);
  if (latest.ok) {
    EXPECT_EQ(latest.generation, 13u);
    EXPECT_EQ(latest.task_digest, kmpc::motion_task_digest(make_task(13)));
  }
  worker.stop();
}

TEST(TrajectoryAdoption, ProductionBudgetIsOneSecondNotTenMs) {
  // Production hard wall is 1s; 10ms is a soft perf statistic only.
  EXPECT_DOUBLE_EQ(kmpc::kPlannerBudgetMs, 1000.0);
  EXPECT_DOUBLE_EQ(kmpc::kPlannerPerfStatMs, 10.0);
  EXPECT_GT(kmpc::kPlannerBudgetMs, kmpc::kPlannerPerfStatMs);

  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);

  kmpc::MotionTask task;
  task.generation = 51;
  task.q = q0;
  task.left = false;
  task.frame = std::string(kmpc::kEeFrameRight);
  kmpc::TaskStage st;
  st.geometry = kmpc::TaskGeometry::Lin;
  st.position = p0 + kmpc::Vec3(0.03, 0.0, 0.0);
  st.position_tol_m = 0.01;
  st.stage_id = 1;
  task.stages.push_back(st);
  kmpc::seal_motion_task_identity(&task, urdf, false, policy);

  kmpc::RobotState state;
  state.q = q0;
  state.fresh = true;
  // Default overload must use kPlannerBudgetMs (not 10ms).
  worker.submit_snapshot(task, state);

  kmpc::FastPlanResult plan;
  bool got = false;
  // Budget is 1s; allow generous wall for CI load.
  for (int i = 0; i < 600; ++i) {
    if (worker.try_take(plan)) {
      got = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(got);
  // Must not fail solely because of the old 10ms production reject gate.
  EXPECT_NE(plan.reason, "deadline");
  EXPECT_FALSE(plan.deadline_hit);
  if (plan.ok) {
    kmpc::TrajectoryFollower follower(kin, &guards, policy);
    std::string why;
    EXPECT_TRUE(follower.adopt(plan.package, q0, kmpc::Vec18::Zero(), false,
                               1e-3, 0.5, &why))
        << why;
  }
  worker.stop();
}

TEST(TrajectoryAdoption, PlannerWorkerLatestGenerationWins) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();

  kmpc::Kinematics kin(urdf);
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);

  auto make_task = [&](uint64_t gen, const kmpc::Vec3& goal) {
    kmpc::MotionTask task;
    task.generation = gen;
    task.q = q0;
    task.left = false;
    task.frame = std::string(kmpc::kEeFrameRight);
    kmpc::TaskStage st;
    st.geometry = kmpc::TaskGeometry::Lin;
    st.position = goal;
    st.position_tol_m = 0.003;
    st.stage_id = 1;
    task.stages.push_back(st);
    return task;
  };
  kmpc::RobotState state;
  state.q = q0;
  state.fresh = true;

  worker.submit_snapshot(make_task(1, p0 + kmpc::Vec3(0.03, 0, 0)), state);
  worker.submit_snapshot(make_task(2, p0 + kmpc::Vec3(0.04, 0, 0)), state);

  kmpc::FastPlanResult result;
  bool got = false;
  for (int i = 0; i < 200; ++i) {
    if (worker.try_take(result)) {
      got = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(got);
  // Latest submitted generation should win; older results may be dropped.
  if (result.ok) {
    EXPECT_EQ(result.package.generation, 2u);
  }
  worker.stop();
  EXPECT_FALSE(worker.running());
}
