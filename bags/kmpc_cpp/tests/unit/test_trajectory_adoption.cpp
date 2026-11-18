#include <chrono>
#include <limits>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  kmpc::TrajectoryFollower follower(kin, &guards, policy);

  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
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
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();

  kmpc::Kinematics kin(urdf);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
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

namespace {

kmpc::FeedbackRing18 make_stable_ring(const kmpc::Vec18& q, double now_s) {
  kmpc::FeedbackRing18 ring;
  // Cover ~200 ms with 4 independent recv stamps (offline synthetic).
  // source left unknown — must not be filled with recv.
  ring.push(q, now_s - 0.18);
  ring.push(q, now_s - 0.12);
  ring.push(q, now_s - 0.06);
  ring.push(q, now_s - 0.01);
  return ring;
}

}  // namespace

TEST(TrajectoryAdoption, FastAdoptAllowsStableSubMilliDelta) {
  // Field reject was start_q_mismatch at 1e-3 while meas/cmd are different
  // signals. Stable ~0.06° plan-window change must fast-adopt from continuity.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  auto pkg = make_pkg(q0, 61);
  const double now_s = 100.0;
  kmpc::Vec18 q_meas = q0;
  q_meas[5] += 0.001;  // ~0.057° — above old 1e-3 vs start_q if compared
  // But plan-window delta vs meas_start is only 0.001 rad if start matched.
  kmpc::PlanStartSnapshot start;
  start.q_plan_start = q0;
  start.q_meas_start = q0;
  start.has_meas_start = true;
  start.q_cmd_start = q0;
  start.has_cmd_start = true;
  start.stamp_s = now_s - 0.8;

  // Plan-window meas change ~0.0015 rad (~0.086°) < 0.002, but still above
  // the old absolute start_q↔meas gate of 1e-3.
  q_meas = q0;
  q_meas[7] += 0.0015;
  auto ring = make_stable_ring(q_meas, now_s);
  kmpc::Vec18 q_noise = q_meas;
  q_noise[7] += 0.0003;
  ring.push(q_noise, now_s - 0.005);

  const auto dec = kmpc::evaluate_fast_adopt(
      pkg, start, q_meas, true, now_s - 0.005, q0, true, ring, now_s,
      /*track_err_max*/ 0.01, config.limits.dq_max, 0.004);
  EXPECT_TRUE(dec.ok) << dec.reason << " " << dec.detail;
  EXPECT_TRUE(dec.fast_path);
  EXPECT_FALSE(dec.request_replan);
  EXPECT_LT(dec.worst_plan_delta_rad, kmpc::kPlanMeasDeltaMaxRad + 1e-12);

  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  kmpc::TrajectoryFollower follower(kin, &guards,
                                    kmpc::policy_from_config(config));
  std::string why;
  // Continuity base is cmd/start — not raw meas — so strict admit still passes.
  ASSERT_TRUE(follower.adopt(pkg, q_meas, kmpc::Vec18::Zero(), false,
                             kmpc::kAdoptStartQTolRad, 0.5, &why))
      << why;
}

TEST(TrajectoryAdoption, FieldGapAdoptsFromMeasuredWithoutOldCmd) {
  // 20261117T083134Z: HOME vs measured max delta 1.27523 rad on j8/j15.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q_home = kmpc::test_fixture_q0();
  kmpc::Vec18 q_meas = q_home;
  q_meas[0] = 40.0 * kmpc::kPi / 180.0;
  q_meas[1] = -50.0 * kmpc::kPi / 180.0;
  q_meas[2] = -10.0 * kmpc::kPi / 180.0;
  q_meas[3] = -1.5 * kmpc::kPi / 180.0;
  q_meas[8] = 1.27523;
  q_meas[15] = -1.27523;
  auto pkg = make_pkg(q_meas, 701);
  const double now_s = 400.0;
  kmpc::PlanStartSnapshot start;
  start.q_plan_start = q_meas;
  start.q_meas_start = q_meas;
  start.has_meas_start = true;
  start.has_cmd_start = false;
  auto ring = make_stable_ring(q_meas, now_s);
  const auto dec = kmpc::evaluate_fast_adopt(
      pkg, start, q_meas, true, now_s - 0.01, q_home, true, ring, now_s, 0.01,
      config.limits.dq_max, 0.004);
  EXPECT_TRUE(dec.ok) << dec.reason << " " << dec.detail;
  EXPECT_NE(dec.reason, std::string("cmd_start_discontinuous"));
  EXPECT_NEAR(dec.meas_vs_plan_start_rad, 0.0, 1e-9);
}

TEST(TrajectoryAdoption, FastAdoptReplansWhenPlanMeasMoved) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  auto pkg = make_pkg(q0, 62);
  const double now_s = 200.0;
  kmpc::PlanStartSnapshot start;
  start.q_plan_start = q0;
  start.q_meas_start = q0;
  start.has_meas_start = true;
  start.q_cmd_start = q0;
  start.has_cmd_start = true;

  kmpc::Vec18 q_meas = q0;
  q_meas[4] += 0.005;  // > 0.002 rad during planning
  auto ring = make_stable_ring(q_meas, now_s);
  const auto dec = kmpc::evaluate_fast_adopt(
      pkg, start, q_meas, true, now_s - 0.01, q0, true, ring, now_s, 0.01,
      config.limits.dq_max, 0.004);
  EXPECT_FALSE(dec.ok);
  EXPECT_TRUE(dec.request_replan);
  EXPECT_STREQ(dec.reason, "plan_meas_moved");
  EXPECT_GT(dec.worst_plan_delta_rad, kmpc::kPlanMeasDeltaMaxRad);
}

TEST(TrajectoryAdoption, PlanAdoptDiagKeepsSourceAndRecvDistinct) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const double submit = 50.0;
  const double adopt = 50.85;
  kmpc::PlanStartSnapshot start;
  start.q_plan_start = q0;
  start.q_meas_start = q0;
  start.has_meas_start = true;
  start.q_cmd_start = q0;
  start.has_cmd_start = true;
  start.submit_ros_s = submit;
  start.meas_recv_s = submit - 0.01;
  start.has_meas_recv = true;
  // Collection intentionally unknown — must stay "unknown" in diag.
  start.has_meas_source = false;

  kmpc::FeedbackRing18 ring;
  for (int i = 0; i < 5; ++i) {
    kmpc::Vec18 q = q0;
    q[3] += 0.0002 * i;
    ring.push(q, submit - 0.18 + 0.04 * i, /*source*/ submit - 0.19 + 0.04 * i);
  }
  // During planning: one sample with larger delta on j4.
  kmpc::Vec18 q_mid = q0;
  q_mid[4] += 0.0012;
  ring.push(q_mid, submit + 0.4, submit + 0.39);

  const auto win =
      kmpc::summarize_feedback_windows(ring, submit, adopt, q0, true);
  EXPECT_TRUE(win.pre_ok) << win.pre_reason;
  EXPECT_TRUE(win.during_ok) << win.during_reason;
  EXPECT_EQ(win.during_worst_joint, 4);
  EXPECT_NEAR(win.during_worst_delta_rad, 0.0012, 1e-9);

  kmpc::Vec18 q_adopt = q0;
  q_adopt[4] += 0.0012;
  const std::string diag = kmpc::format_plan_adopt_diag(
      start, adopt, /*adopt_src*/ std::numeric_limits<double>::quiet_NaN(),
      false, adopt - 0.005, true, q_adopt, true, q0, true, win, "adopt_fast",
      "ok", adopt - submit);
  EXPECT_NE(diag.find("plan_meas_src=unknown"), std::string::npos) << diag;
  EXPECT_NE(diag.find("plan_meas_src_clock=unknown"), std::string::npos);
  EXPECT_NE(diag.find("adopt_meas_src=unknown"), std::string::npos);
  EXPECT_NE(diag.find("adopt_meas_recv="), std::string::npos);
  EXPECT_NE(diag.find("q_meas_plan=["), std::string::npos);
  EXPECT_NE(diag.find("d_meas_plan_to_adopt=["), std::string::npos);
  EXPECT_NE(diag.find("pre200_ok=1"), std::string::npos);
  EXPECT_NE(diag.find("during_worst_j=4"), std::string::npos);
  // Must not claim receive is collection.
  EXPECT_EQ(diag.find("plan_meas_src_clock=ros_time"), std::string::npos);

  // Empty cache → explicit insufficient, no blocking.
  kmpc::FeedbackRing18 empty;
  const auto thin =
      kmpc::summarize_feedback_windows(empty, submit, adopt, q0, true);
  EXPECT_FALSE(thin.pre_ok);
  EXPECT_STREQ(thin.pre_reason, "cache_empty");
}

TEST(TrajectoryAdoption, FastAdoptWaitsWhenFeedbackIncomplete) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  auto pkg = make_pkg(q0, 63);
  const double now_s = 300.0;
  kmpc::PlanStartSnapshot start;
  start.q_plan_start = q0;
  start.q_meas_start = q0;
  start.has_meas_start = true;
  start.q_cmd_start = q0;
  start.has_cmd_start = true;

  kmpc::FeedbackRing18 ring;  // empty / insufficient coverage
  ring.push(q0, now_s - 0.02);
  const auto dec = kmpc::evaluate_fast_adopt(
      pkg, start, q0, true, now_s - 0.02, q0, true, ring, now_s, 0.01,
      config.limits.dq_max, 0.004);
  EXPECT_FALSE(dec.ok);
  EXPECT_TRUE(dec.wait_for_stable);
  EXPECT_STREQ(dec.reason, "feedback_incomplete");
}
