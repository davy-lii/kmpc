#include <string>

#include <gtest/gtest.h>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_evaluator.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

std::string urdf() {
  return std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
}

}  // namespace

TEST(MotionEvaluator, ComMatchesInertialAndDoesNotRunCrba) {
  const kmpc::Config config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  const kmpc::Vec18 q = kmpc::test_fixture_q0();
  kmpc::Kinematics kin(urdf());
  kin.reset_kin_eval_stats();

  const kmpc::Vec3 com = kin.com(q);
  kmpc::KinEvalStats stats = kin.kin_eval_stats();
  EXPECT_EQ(stats.fk_updates, 1u);
  EXPECT_EQ(stats.crba_updates, 0u);
  const kmpc::Vec3 com_again = kin.com(q);
  stats = kin.kin_eval_stats();
  EXPECT_EQ(stats.fk_updates, 1u);
  EXPECT_EQ(stats.crba_updates, 0u);
  EXPECT_NEAR((com - com_again).norm(), 0.0, 1e-15);

  const auto full = kin.inertial_quantities(q);
  stats = kin.kin_eval_stats();
  EXPECT_EQ(stats.crba_updates, 1u);
  EXPECT_NEAR((com - full.p_com).norm(), 0.0, 1e-9);
  EXPECT_GT(full.mdiag.norm(), 1e-8);
  const auto full_again = kin.inertial_quantities(q);
  EXPECT_EQ(kin.kin_eval_stats().crba_updates, 1u);
  EXPECT_NEAR((full.J_com - full_again.J_com).norm(), 0.0, 1e-12);
  EXPECT_NEAR((full.mdiag - kin.mass_matrix_diag(q)).norm(), 0.0, 1e-12);
  EXPECT_EQ(kin.kin_eval_stats().crba_updates, 1u);

  kmpc::Kinematics kin_jac(urdf());
  const auto jac_before = kin_jac.jacobian_pos(q, kmpc::kEeFrameLeft);
  const kmpc::Vec18 mdiag = kin_jac.mass_matrix_diag(q);
  const auto jac_after = kin_jac.jacobian_pos(q, kmpc::kEeFrameLeft);
  EXPECT_GT(mdiag.norm(), 1e-8);
  EXPECT_NEAR((jac_before - jac_after).norm(), 0.0, 1e-12);
  EXPECT_NEAR(
      (kin_jac.jacobian_com(q) - full.J_com).norm(), 0.0, 1e-8);
  EXPECT_EQ(kin_jac.kin_eval_stats().crba_updates, 1u);
}

TEST(MotionEvaluator, BundleReadsBodyLegComWithoutCrba) {
  const kmpc::Config config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(urdf());
  kmpc::Kinematics reference(urdf());
  const kmpc::Vec18 q = kmpc::test_fixture_q0();
  const auto full = reference.inertial_quantities(q);

  kin.reset_kin_eval_stats();
  kmpc::MotionEvaluator evaluator(kin);
  kmpc::MotionSnapshot snap;
  evaluator.evaluate(q, true, &snap);
  const kmpc::KinEvalStats once = kin.kin_eval_stats();
  EXPECT_EQ(once.fk_updates, 1u);
  EXPECT_EQ(once.jac_updates, 1u);
  EXPECT_EQ(once.crba_updates, 0u);
  EXPECT_TRUE(snap.tcp.allFinite());
  EXPECT_TRUE(snap.body.allFinite());
  EXPECT_TRUE(snap.leg_y.allFinite());
  EXPECT_NEAR((snap.com - full.p_com).norm(), 0.0, 1e-9);
  EXPECT_NEAR((snap.j_com - full.J_com).norm(), 0.0, 1e-8);

  kmpc::MotionSnapshot again;
  evaluator.evaluate(q, true, &again);
  const kmpc::KinEvalStats twice = kin.kin_eval_stats();
  EXPECT_EQ(twice.fk_updates, 1u);
  EXPECT_EQ(twice.jac_updates, 1u);
  EXPECT_EQ(twice.crba_updates, 0u);
  EXPECT_NEAR((again.tcp - snap.tcp).norm(), 0.0, 1e-15);

  kmpc::Vec18 q2 = q;
  q2[5] += 0.05;
  evaluator.evaluate(q2, true, &again);
  EXPECT_EQ(kin.kin_eval_stats().crba_updates, 0u);
  EXPECT_GT(kin.kin_eval_stats().fk_updates, 1u);
  kmpc::Kinematics fresh(urdf());
  EXPECT_NEAR(
      (again.com - fresh.inertial_quantities(q2).p_com).norm(), 0.0, 1e-9);
}
