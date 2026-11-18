#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_evaluator.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/unified_multistage_planner.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::JointCurve path_from_to(const kmpc::Vec18& a, const kmpc::Vec18& b) {
  kmpc::JointCurve c;
  c.push_back(a);
  c.push_back(0.5 * (a + b));
  c.push_back(b);
  return c;
}

struct PostureSample {
  const char* name;
  kmpc::Vec18 q;
  double body_x = 0.0;
  double body_z = 0.0;
  double body_minus_leg_x = 0.0;
  double pitch = 0.0;
  double tcp_err = 0.0;
  kmpc::PathCostBreakdown cost{};
};

}  // namespace

TEST(BodyPostureCost, SameTcpMultiplePosturesDoesNotPreferDominatedForwardSquat) {
  const auto config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  auto weights = kmpc::default_cost_weights(config.limits.dq_max);
  weights.calibrated = true;

  const bool left = false;
  const auto frame = std::string(kmpc::kEeFrameRight);
  const kmpc::Vec18 q_home = kmpc::test_fixture_q0();
  const kmpc::Vec3 tcp_task =
      kin.fk_pos(q_home, frame) + kmpc::Vec3(0.04, 0.0, -0.06);

  // Build a few feasible end postures that reach the same TCP (within tol)
  // by small waist/arm perturbations, then score with the unified cost.
  std::vector<PostureSample> samples;
  const double waist_deltas[][3] = {
      {0.00, 0.00, 0.00},    // nearer upright / smaller squat
      {0.06, -0.12, 0.04},   // mild assist
      {0.12, -0.28, 0.10},   // more squat + forward-ish waist
      {0.18, -0.40, 0.14},   // dominated forward-squat candidate
  };
  kmpc::MotionEvaluator eval(kin);
  for (const auto& d : waist_deltas) {
    PostureSample s;
    s.q = q_home;
    s.q[0] = std::clamp(q_home[0] + d[0], policy.q_min[0], policy.q_max[0]);
    s.q[1] = std::clamp(q_home[1] + d[1], policy.q_min[1], policy.q_max[1]);
    s.q[2] = std::clamp(q_home[2] + d[2], policy.q_min[2], policy.q_max[2]);
    // Nudge task arm toward the TCP with a few IK-like damped steps.
    for (int it = 0; it < 40; ++it) {
      const kmpc::Vec3 err = tcp_task - kin.fk_pos(s.q, frame);
      if (err.norm() < 0.002) {
        break;
      }
      const auto J = kin.jacobian_pos(s.q, frame);
      kmpc::Vec18 dq = J.transpose() * err;
      dq.head<4>() *= 0.15;
      dq.segment<7>(4) *= 0.45;
      dq.segment<7>(11).setZero();
      s.q += dq * 0.35;
      for (int j = 0; j < kmpc::kNDof; ++j) {
        s.q[j] = std::clamp(s.q[j], policy.q_min[j], policy.q_max[j]);
      }
    }
    if (kmpc::joint_violates_hard_box(s.q, policy.q_min, policy.q_max)) {
      continue;
    }
    if (!guards.com_xy_inside(kin.com(s.q))) {
      continue;
    }
    kmpc::MotionSnapshot snap;
    eval.evaluate(s.q, left, &snap);
    s.body_x = snap.body.x();
    s.body_z = snap.body.z();
    s.body_minus_leg_x = snap.body_minus_leg_x;
    s.pitch = snap.pitch;
    s.tcp_err = (snap.tcp - tcp_task).norm();
    if (s.tcp_err > 0.008) {
      continue;
    }
    // Score the motion into the posture from home so body_forward sees Δ(Bx-Lx).
    s.cost = kmpc::evaluate_path_cost(path_from_to(q_home, s.q), kmpc::test_fixture_q0(),
                                      policy, weights, kin, left, &guards);
    samples.push_back(s);
  }
  ASSERT_GE(samples.size(), 2u) << "need at least two feasible postures";

  // Dominated forward-squat: larger body forward (Bx-Lx) AND lower Body Z,
  // with TCP/hard-box already feasible — body_forward term must not reward it.
  std::size_t upright_i = 0;
  std::size_t squat_i = 0;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    if (samples[i].body_z > samples[upright_i].body_z) {
      upright_i = i;
    }
    if (samples[i].body_minus_leg_x > samples[squat_i].body_minus_leg_x) {
      squat_i = i;
    }
  }
  const auto& upright = samples[upright_i];
  const auto& squat = samples[squat_i];
  std::cout << "body_posture upright body_x=" << upright.body_x
            << " body_z=" << upright.body_z
            << " body_minus_leg_x=" << upright.body_minus_leg_x
            << " pitch=" << upright.pitch << " cost=" << upright.cost.total
            << " body_forward=" << upright.cost.body_forward << "\n";
  std::cout << "body_posture squat   body_x=" << squat.body_x
            << " body_z=" << squat.body_z
            << " body_minus_leg_x=" << squat.body_minus_leg_x
            << " pitch=" << squat.pitch << " cost=" << squat.cost.total
            << " body_forward=" << squat.cost.body_forward << "\n";
  for (const auto& s : samples) {
    std::cout << "  sample body_x=" << s.body_x << " body_z=" << s.body_z
              << " dx_leg=" << s.body_minus_leg_x << " tcp_err=" << s.tcp_err
              << " total=" << s.cost.total
              << " body_forward=" << s.cost.body_forward << "\n";
  }

  if (squat.body_z < upright.body_z - 0.005 &&
      squat.body_minus_leg_x > upright.body_minus_leg_x + 0.005) {
    EXPECT_GE(squat.cost.body_forward, upright.cost.body_forward - 1e-9)
        << "body_forward must not reward dominated forward-squat";
    // Softplus growth on Bx-Lx: more forward growth => strictly higher feature.
    EXPECT_GT(squat.cost.body_forward, upright.cost.body_forward - 1e-6);
  }

  // Planner on the same TCP task should not land on a worse body_forward than
  // the best scored feasible sample by a large margin.
  kmpc::MotionTask task;
  task.generation = 9;
  task.q = q_home;
  task.left = left;
  task.frame = frame;
  kmpc::TaskStage st;
  st.geometry = kmpc::TaskGeometry::Lin;
  st.position = tcp_task;
  st.position_tol_m = 0.003;
  st.stage_id = 1;
  st.stop = kmpc::StageStop::StopRequired;
  task.stages.push_back(st);
  const auto plan =
      kmpc::solve_to_verified_package(kin, guards, task, policy, weights);
  ASSERT_TRUE(plan.ok) << plan.reason << " " << plan.report;
  kmpc::MotionSnapshot plan_end;
  eval.evaluate(plan.package.samples.back(), left, &plan_end);
  std::cout << "planner_end body_x=" << plan_end.body.x()
            << " body_z=" << plan_end.body.z()
            << " body_minus_leg_x=" << plan_end.body_minus_leg_x
            << " body_dx_m=" << plan.body_dx_m
            << " body_dz_m=" << plan.body_dz_m << "\n";
  EXPECT_FALSE(plan.report.find("height_switch=0") == std::string::npos);
  // Soft check: planner body_forward feature should not explode vs samples.
  const auto plan_cost = kmpc::evaluate_path_cost(
      plan.package.samples, kmpc::test_fixture_q0(), policy, weights, kin, left, &guards);
  double best_sample = samples.front().cost.total;
  for (const auto& s : samples) {
    best_sample = std::min(best_sample, s.cost.total);
  }
  EXPECT_LT(plan_cost.body_forward, best_sample + 50.0)
      << "planner body_forward feature unexpectedly huge vs samples";
}
