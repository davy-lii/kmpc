#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/types.hpp"
#include "kmpc/v2_submit_gate.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

kmpc::Vec18 zeros() { return kmpc::Vec18::Zero(); }

double finite_dq(const kmpc::JointCurve& samples, double p, int j) {
  constexpr double h = 1e-4;
  return (kmpc::hermite_joint_at(samples, p + h)[j] -
          kmpc::hermite_joint_at(samples, p - h)[j]) /
         (2.0 * h);
}

}  // namespace

TEST(HermitePchip, NearJ1NoOvershoot) {
  // Knots sit on / near the J1 hard wall (~58°). Classic central Hermite
  // overshoots; PCHIP must stay inside the segment hull.
  constexpr double j1_wall = 58.0 * kmpc::kPi / 180.0;
  kmpc::JointCurve samples(4);
  for (auto& q : samples) {
    q = zeros();
  }
  samples[0][0] = j1_wall - 0.02;
  samples[1][0] = j1_wall - 0.005;
  samples[2][0] = j1_wall;
  samples[3][0] = j1_wall;
  for (double p = 0.0; p <= 3.0 + 1e-9; p += 0.05) {
    const double q = kmpc::hermite_joint_at(samples, p)[0];
    const int i = std::min(2, static_cast<int>(std::floor(p)));
    const double lo = std::min(samples[static_cast<std::size_t>(i)][0],
                               samples[static_cast<std::size_t>(i + 1)][0]);
    const double hi = std::max(samples[static_cast<std::size_t>(i)][0],
                               samples[static_cast<std::size_t>(i + 1)][0]);
    EXPECT_GE(q, lo - 1e-12) << "p=" << p;
    EXPECT_LE(q, hi + 1e-12) << "p=" << p;
    EXPECT_LE(q, j1_wall + 1e-12) << "p=" << p;
  }
}

TEST(HermitePchip, MonotoneRunNoSpuriousStop) {
  kmpc::JointCurve samples(5);
  for (int i = 0; i < 5; ++i) {
    samples[static_cast<std::size_t>(i)] = zeros();
    samples[static_cast<std::size_t>(i)][4] = 0.05 * static_cast<double>(i);
  }
  double prev = samples[0][4];
  for (double p = 0.0; p <= 4.0 + 1e-9; p += 0.02) {
    const double q = kmpc::hermite_joint_at(samples, p)[4];
    EXPECT_GE(q + 1e-12, prev) << "p=" << p;
    prev = q;
  }
  // Interior tangents stay positive — no unnecessary dwell plateau mid-run.
  for (double p = 0.5; p <= 3.5; p += 0.5) {
    EXPECT_GT(finite_dq(samples, p, 4), 1e-4) << "p=" << p;
  }
}

TEST(HermitePchip, ReversalTangentsGoToZero) {
  kmpc::JointCurve samples(3);
  samples[0] = zeros();
  samples[1] = zeros();
  samples[2] = zeros();
  samples[0][4] = 0.0;
  samples[1][4] = 0.10;
  samples[2][4] = 0.0;
  // At the peak node the left/right slopes flip → PCHIP tangent is 0.
  const double v_l = finite_dq(samples, 1.0 - 1e-3, 4);
  const double v_r = finite_dq(samples, 1.0 + 1e-3, 4);
  EXPECT_NEAR(v_l, 0.0, 5e-2);
  EXPECT_NEAR(v_r, 0.0, 5e-2);
  EXPECT_NEAR(v_l, v_r, 5e-2);
}

TEST(HermitePchip, EqualKnotsNoMotion) {
  kmpc::JointCurve samples(3);
  for (auto& q : samples) {
    q = zeros();
    q[3] = 0.42;
  }
  for (double p = 0.0; p <= 2.0 + 1e-9; p += 0.1) {
    EXPECT_NEAR(kmpc::hermite_joint_at(samples, p)[3], 0.42, 1e-12);
  }
}

TEST(HermitePchip, FieldLikePlanGateFollower250Hz) {
  // Smoke: plan → V2SubmitGate identity → Follower adopt → 250 Hz Hermite
  // samples stay inside hard box / per-cycle step. Stage ACK path remains
  // reachable when the first stage requests wait_for_external_ack.
  const auto config = kmpc::Config::load(
      std::string(KMPC_CONFIG_DIR) + "/joint_limits.yaml",
      std::string(KMPC_CONFIG_DIR) + "/q_pref.yaml",
      std::string(KMPC_CONFIG_DIR) + "/kmpc_params.yaml");
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  auto policy = kmpc::policy_from_config(config);
  std::string weight_why;
  auto weights = kmpc::load_trajectory_quality_weights(
      std::string(KMPC_CONFIG_DIR) + "/trajectory_quality.yaml",
      config.limits.dq_max, &weight_why);
  for (int j = 0; j < kmpc::kNDof; ++j) {
    weights.motion_scale[j] = 1.0 / std::max(config.limits.dq_max[j], 1e-3);
  }

  kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);
  kmpc::MotionTask task;
  task.generation = 42;
  task.q = q0;
  task.left = true;
  task.frame = kmpc::kEeFrameLeft;
  for (int i = 0; i < 2; ++i) {
    kmpc::TaskStage st;
    st.geometry = kmpc::TaskGeometry::Lin;
    st.position = p0 + kmpc::Vec3(0.08 * (i + 1), 0.0, -0.02 * i);
    st.position_tol_m = 0.003;
    st.orientation_tol_rad = 0.05;
    st.stage_id = static_cast<uint32_t>(i + 1);
    st.stop = (i == 1) ? kmpc::StageStop::StopRequired
                       : kmpc::StageStop::Continuous;
    st.wait_for_external_ack = (i == 0);
    task.stages.push_back(st);
  }
  kmpc::seal_motion_task_identity(&task, urdf, false, policy);

  kmpc::FastTrajectoryOptimizer optimizer(kin, &guards);
  kmpc::RobotState state;
  state.q = q0;
  state.fresh = true;
  const auto plan =
      optimizer.plan(task, state, policy, weights, kmpc::kPlannerBudgetMs);
  ASSERT_TRUE(plan.ok) << plan.reason;
  ASSERT_TRUE(plan.package.certificate.passed);

  kmpc::V2TaskId tid;
  tid.generation = task.generation;
  tid.digest = kmpc::motion_task_digest(task);
  kmpc::V2SubmitGate gate;
  const auto dec = gate.on_tick(tid, /*has_follower_package=*/false,
                                /*explicit_request=*/false);
  EXPECT_TRUE(dec.submit);

  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  std::string why;
  ASSERT_TRUE(follower.adopt(plan.package, q0, kmpc::Vec18::Zero(), false, 1e-3,
                             0.5, &why))
      << why;

  kmpc::FollowerTick tick;
  for (int i = 0; i < 80; ++i) {
    tick = follower.step(q0, 0.02, 0.02 * i, 0.02 * i);
    if (tick.state == kmpc::FollowerState::WaitAck) {
      break;
    }
  }
  if (tick.state == kmpc::FollowerState::WaitAck) {
    EXPECT_TRUE(follower.on_stage_ack(plan.package.generation,
                                      tick.wait_stage_id, true, q0));
  }

  constexpr double dt = 0.004;
  kmpc::Vec18 q_prev = plan.package.samples.front();
  for (double t = 0.0; t <= plan.package.time_law.duration_s + 1e-12; t += dt) {
    kmpc::TrajectoryTimeSample samp;
    ASSERT_TRUE(kmpc::sample_trajectory_at(plan.package, t, &samp, false));
    ASSERT_TRUE(samp.ok);
    EXPECT_FALSE(kmpc::joint_violates_hard_box(samp.q, policy.q_min,
                                               policy.q_max));
    for (int j = 0; j < kmpc::kNDof; ++j) {
      const double step = samp.q[j] - q_prev[j];
      EXPECT_LE(std::abs(step), policy.dq_max[j] * dt + 1e-3)
          << "j=" << j << " t=" << t;
    }
    q_prev = samp.q;
  }
}
