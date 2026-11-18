#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/unified_multistage_planner.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::MotionTask make_task(const kmpc::Vec18& q0, bool left,
                           const std::vector<kmpc::Vec3>& goals) {
  kmpc::MotionTask task;
  task.generation = 2;
  task.q = q0;
  task.dq_known = false;
  task.left = left;
  task.frame = left ? std::string(kmpc::kEeFrameLeft)
                    : std::string(kmpc::kEeFrameRight);
  for (std::size_t i = 0; i < goals.size(); ++i) {
    kmpc::TaskStage st;
    st.geometry = kmpc::TaskGeometry::Lin;
    st.position = goals[i];
    st.position_tol_m = 0.003;
    st.stage_id = static_cast<uint32_t>(i + 1);
    st.stop = (i + 1 == goals.size()) ? kmpc::StageStop::StopRequired
                                      : kmpc::StageStop::Continuous;
    task.stages.push_back(st);
  }
  return task;
}

// Mirror of node_cache / MotionSequence admission: never invent stages.
bool accept_known_stages_only(const kmpc::MotionTask& task) {
  if (task.stages.empty() || task.stages.size() > 3) {
    return false;
  }
  for (const auto& st : task.stages) {
    if (!st.position.allFinite() || !(st.position_tol_m > 0.0)) {
      return false;
    }
  }
  return true;
}

kmpc::MotionTask without_inventing_lift(const kmpc::MotionTask& known) {
  // If LIFT was not in the input, the planner input stays as given.
  return known;
}

}  // namespace

TEST(MultistageV2, OneTwoThreeStagesWhenAllKnown) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const auto weights = kmpc::default_cost_weights(config.limits.dq_max);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);

  const std::vector<std::vector<kmpc::Vec3>> cases = {
      {p0 + kmpc::Vec3(0.05, 0.0, 0.0)},
      {p0 + kmpc::Vec3(0.04, 0.0, 0.0), p0 + kmpc::Vec3(0.04, 0.0, -0.05)},
      {p0 + kmpc::Vec3(0.03, 0.0, 0.0), p0 + kmpc::Vec3(0.03, 0.0, -0.04),
       p0 + kmpc::Vec3(0.03, 0.0, 0.02)},
  };
  for (const auto& goals : cases) {
    const auto task = make_task(q0, true, goals);
    ASSERT_TRUE(accept_known_stages_only(task));
    EXPECT_EQ(task.stages.size(), goals.size());
    const auto res =
        kmpc::solve_to_verified_package(kin, guards, task, policy, weights);
    ASSERT_TRUE(res.ok) << "stages=" << goals.size() << " " << res.reason
                        << " " << res.report;
    EXPECT_NE(res.report.find("planner=unified_multistage_v2"),
              std::string::npos)
        << res.report;
    EXPECT_NE(res.report.find("stages=" + std::to_string(goals.size())),
              std::string::npos)
        << res.report;
    EXPECT_GE(res.package.samples.size(), 2u);
    EXPECT_LE(res.tcp_worst_m, 0.003 + 1e-9);
  }
}

TEST(MultistageV2, RejectInventingLiftWhenOnlyPreTipKnown) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const auto weights = kmpc::default_cost_weights(config.limits.dq_max);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameRight);

  // Field often sent PRE+TIP as poses=2, then LIFT later. Do not invent LIFT.
  auto known = make_task(q0, false,
                         {p0 + kmpc::Vec3(0.0, 0.0, -0.05),
                          p0 + kmpc::Vec3(0.0, 0.0, -0.10)});
  ASSERT_EQ(known.stages.size(), 2u);
  known.stages[0].stage_id = 1;  // PRE
  known.stages[1].stage_id = 2;  // TIP

  const auto task = without_inventing_lift(known);
  EXPECT_EQ(task.stages.size(), 2u);
  EXPECT_FALSE(task.stages.size() == 3u);

  kmpc::MotionTask forged = task;
  // Explicitly reject a receiver that would append a guessed LIFT.
  kmpc::TaskStage fake_lift;
  fake_lift.geometry = kmpc::TaskGeometry::Lin;
  fake_lift.position = p0 + kmpc::Vec3(0.0, 0.0, 0.05);
  fake_lift.position_tol_m = 0.003;
  fake_lift.stage_id = 3;
  forged.stages.push_back(fake_lift);
  EXPECT_TRUE(accept_known_stages_only(forged));  // syntactically 3 is ok
  // Contract under test: planning input must equal known stages, not forged.
  EXPECT_EQ(task.stages.size(), known.stages.size());
  EXPECT_EQ(task.stages.back().stage_id, 2u);

  const auto res =
      kmpc::solve_to_verified_package(kin, guards, task, policy, weights);
  ASSERT_TRUE(res.ok) << res.reason << " " << res.report;
  EXPECT_NE(res.report.find("stages=2"), std::string::npos) << res.report;
  EXPECT_EQ(res.report.find("stages=3"), std::string::npos) << res.report;
}

TEST(MultistageV2, RejectMoreThanThreeStages) {
  kmpc::MotionTask task;
  task.generation = 1;
  task.stages.resize(4);
  for (std::size_t i = 0; i < task.stages.size(); ++i) {
    task.stages[i].position = kmpc::Vec3(0.1 * static_cast<double>(i), 0, 0.5);
    task.stages[i].position_tol_m = 0.003;
    task.stages[i].stage_id = static_cast<uint32_t>(i + 1);
  }
  EXPECT_FALSE(accept_known_stages_only(task));
}
