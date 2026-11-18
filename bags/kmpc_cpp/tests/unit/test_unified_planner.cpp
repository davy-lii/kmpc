#include <chrono>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/unified_multistage_planner.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::UnifiedPlanRequest request_from(const kmpc::Config& config, const kmpc::Vec18& q0,
                                      bool left, const kmpc::Vec3& goal) {
  kmpc::UnifiedPlanRequest req;
  req.q0 = q0;
  req.q_neutral = kmpc::test_fixture_q0();
  req.dq_max = config.limits.dq_max;
  req.left = left;
  req.frame = left ? std::string(kmpc::kEeFrameLeft) : std::string(kmpc::kEeFrameRight);
  req.goal.xyz = goal;
  req.policy = kmpc::policy_from_config(config);
  req.pos_tol_m = 0.003;
  req.sample_max_m = 0.005;
  return req;
}

void expect_certified(const kmpc::Kinematics& kin, const kmpc::WholeBodyGuards& guards,
                      const kmpc::UnifiedPlanResult& plan, const kmpc::UnifiedPlanRequest& req) {
  ASSERT_TRUE(plan.ok) << plan.reason << " " << plan.report;
  EXPECT_NE(plan.report.find("planner=unified_multistage_v2"), std::string::npos)
      << plan.report;
  EXPECT_NE(plan.report.find("height_switch=0"), std::string::npos) << plan.report;
  EXPECT_GE(plan.segment.size(), 2u);
  EXPECT_LE(plan.tcp_worst_m, req.pos_tol_m + 1e-9);
  EXPECT_TRUE(plan.direction.ok) << plan.direction.reason;
  for (const auto& q : plan.segment) {
    EXPECT_FALSE(kmpc::joint_violates_hard_box(q, req.policy.q_min, req.policy.q_max));
    EXPECT_TRUE(guards.com_xy_inside(kin.com(q)));
    if (req.policy.torso_pitch_max_rad > 0.0) {
      const double cap = std::max(req.policy.torso_pitch_max_rad,
                                   std::abs(plan.segment.front().head<3>().sum()));
      EXPECT_LE(std::abs(q.head<3>().sum()), cap + 1e-6);
    }
  }
}

}  // namespace

TEST(UnifiedPlanner, HorizontalVerticalAndDiagonalStayInsideHardBox) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);
  const kmpc::Vec3 goals[] = {
      p0 + kmpc::Vec3(0.06, 0.0, 0.0),
      p0 + kmpc::Vec3(0.0, 0.0, -0.05),
      p0 + kmpc::Vec3(0.04, -0.03, 0.03),
  };
  for (const auto& goal : goals) {
    const auto req = request_from(config, q0, false, goal);
    const auto plan = kmpc::plan_unified_multistage(kin, guards, req);
    expect_certified(kin, guards, plan, req);
    const double end =
        (kin.fk_pos(plan.segment.back(), kmpc::kEeFrameRight) - goal).norm();
    EXPECT_LE(end, 0.003);
  }
}

TEST(UnifiedPlanner, KnownTipIsPlannedWithTheFirstGoal) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);
  auto req = request_from(config, q0, true, p0 + kmpc::Vec3(0.05, 0.0, 0.0));
  req.has_next = true;
  req.next.xyz = p0 + kmpc::Vec3(0.05, 0.0, 0.04);
  const auto plan = kmpc::plan_unified_multistage(kin, guards, req);
  expect_certified(kin, guards, plan, req);
  EXPECT_TRUE(plan.has_successor);
  EXPECT_GE(plan.successor.size(), 2u);
  EXPECT_LT((plan.successor.front() - plan.segment.back()).norm(), 1e-9);
  EXPECT_NE(plan.report.find("stages=2"), std::string::npos) << plan.report;
}

TEST(UnifiedPlanner, StartPastHardBoxIsNotAPath) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  q0[0] = config.limits.q_max[0] + 0.02;
  const kmpc::Vec3 p0 = kin.fk_pos(kmpc::test_fixture_q0(), kmpc::kEeFrameRight);
  const auto req = request_from(config, q0, false, p0 + kmpc::Vec3(0.02, 0.0, 0.0));
  const auto plan = kmpc::plan_unified_multistage(kin, guards, req);
  EXPECT_FALSE(plan.ok);
  EXPECT_NE(plan.reason.find("unified_hard_box"), std::string::npos) << plan.reason;
}

TEST(UnifiedPlanner, SyntheticNearJ1WallDoesNotCrossHardMax) {
  // Not an r40 18-vector replay. q_pref with J1 moved to 0.01 rad inside 58°.
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  q0[0] = config.limits.q_max[0] - 0.01;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);
  const auto req = request_from(config, q0, false, p0 + kmpc::Vec3(0.0, 0.0, 0.04));
  const auto plan = kmpc::plan_unified_multistage(kin, guards, req);
  expect_certified(kin, guards, plan, req);
  double j1_max = q0[0];
  for (const auto& q : plan.segment) {
    j1_max = std::max(j1_max, q[0]);
  }
  EXPECT_LE(j1_max, config.limits.q_max[0] + 1e-9);
  std::cout << "synthetic_j1_wall " << plan.report << " j1_max_deg="
            << (j1_max * 180.0 / kmpc::kPi) << "\n";
}

TEST(UnifiedPlanner, TrackerNamesUnifiedAndTimesTheBuild) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, config);
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  const auto t0 = std::chrono::steady_clock::now();
  tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
  const double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage_v2");
  EXPECT_NE(tracker.line_pose_report().find("planner=unified_multistage"), std::string::npos)
      << tracker.line_pose_report();
  EXPECT_EQ(tracker.line_pose_pref_decision(), "unified_only");
  EXPECT_EQ(tracker.reverse_line_decision(), "not_a_selector");
  std::cout << "x86_build_ms=" << tracker.line_build_ms()
            << " wall_ms=" << wall_ms
            << " report=" << tracker.line_pose_report() << "\n";
}
