#include <gtest/gtest.h>
#include "node_cache.hpp"
using namespace kmpc;using namespace kmpc::ros2io;
TEST(LineSegment, AtomicGenerationTipAndArmReplacement) {
  TaskReceiver rx;CartesianTarget t;t.generation=100;t.xyz=Vec3(1,2,3);
  LineTarget tip;tip.xyz=Vec3(4,5,6);t.cascade=tip;
  ASSERT_TRUE(rx.on_segment(true,t));
  auto pair=rx.targets();ASSERT_TRUE(pair.first);EXPECT_FALSE(pair.second);
  EXPECT_EQ(pair.first->generation,100u);EXPECT_EQ(pair.first->cascade->xyz,tip.xyz);
  t.xyz[0]+=.01;t.cascade.reset();ASSERT_TRUE(rx.on_segment(true,t));
  EXPECT_TRUE(rx.target(true)->cascade); // drift cannot remove the validated next step
  t.generation=99;EXPECT_FALSE(rx.on_segment(true,t));
  t.generation=101;ASSERT_TRUE(rx.on_segment(false,t));
  pair=rx.targets();EXPECT_FALSE(pair.first);ASSERT_TRUE(pair.second);EXPECT_FALSE(pair.second->cascade);
  rx.clear_right();EXPECT_FALSE(rx.on_segment(false,t)); // old message cannot rearm after clear
}
TEST(LineSegment, LegacyUpdatesNeedClearForNewGeneration) {
  TaskReceiver rx;CartesianTarget t;rx.on_target_left(t);
  auto first=rx.target(true)->generation;t.xyz[0]=.15;rx.on_target_left(t);
  EXPECT_EQ(first,rx.target(true)->generation);
  rx.clear_left();rx.on_target_left(t);EXPECT_GT(rx.target(true)->generation,first);
}
TEST(LineSegment, InvalidCoordinatesNeverReplaceTarget) {
  TaskReceiver rx;CartesianTarget t;t.generation=1;t.frame_id="camera";
  EXPECT_FALSE(rx.on_segment(true,t));EXPECT_FALSE(rx.target(true));
  t.frame_id="base_link";t.xyz[0]=std::numeric_limits<double>::quiet_NaN();EXPECT_FALSE(rx.on_segment(true,t));
}

TEST(LineSegment, DelayedClearCannotEraseTheNextSegment) {
  TaskReceiver rx; CartesianTarget t;t.generation=100;
  ASSERT_TRUE(rx.on_segment(true,t));
  EXPECT_FALSE(rx.clear_segment(true,99));ASSERT_TRUE(rx.target(true));
  t.generation=101;ASSERT_TRUE(rx.on_segment(true,t));
  EXPECT_FALSE(rx.clear_segment(true,100));ASSERT_TRUE(rx.target(true));
  EXPECT_EQ(rx.target(true)->generation,101u);
  EXPECT_TRUE(rx.clear_segment(true,101));EXPECT_FALSE(rx.target(true));
  EXPECT_FALSE(rx.on_segment(true,t));
}

#include "line_stream.hpp"
TEST(LineStreamCommit, ReversalUsesReferenceAtCommitAndKeepsChord) {
  Vec18 snapshot=Vec18::Zero(),from=snapshot,planned=snapshot,cap=Vec18::Constant(.0132);
  from[5]=.000026;planned[5]=-.0132;planned[6]=.006;
  auto step=bound_line_stream_step(from,planned,cap);ASSERT_TRUE(step);
  EXPECT_LT(step->scale,1.0);EXPECT_GT(step->scale,.99);
  EXPECT_LE((step->target-from).cwiseAbs().maxCoeff(),.0132+1e-15);
  EXPECT_NEAR(step->target[6],.006*step->scale,1e-15);
  EXPECT_NEAR(step->target[5],from[5]+step->scale*(planned[5]-from[5]),1e-15);
}
TEST(LineStreamCommit, KeepsValidStepAndRejectsInvalidInput) {
  Vec18 from=Vec18::Zero(),to=Vec18::Constant(.001),cap=Vec18::Constant(.002);
  auto step=bound_line_stream_step(from,to,cap);ASSERT_TRUE(step);EXPECT_EQ(step->scale,1.0);
  EXPECT_EQ(step->target,to);
  cap[0]=0;step=bound_line_stream_step(from,to,cap);ASSERT_TRUE(step);EXPECT_EQ(step->scale,0.0);EXPECT_EQ(step->target,from);
  cap[0]=-.1;EXPECT_FALSE(bound_line_stream_step(from,to,cap));
  cap.setConstant(.002);to[0]=std::numeric_limits<double>::quiet_NaN();EXPECT_FALSE(bound_line_stream_step(from,to,cap));
}

#include "kmpc/path_progress.hpp"
TEST(PathProgress, GeometricBoundaryDoesNotForceVelocityDip) {
  Vec18 origin=Vec18::Zero(),cap=Vec18::Constant(.005);origin[0]=.008;
  auto at=[](double s){Vec18 q=Vec18::Zero();q[0]=.01*s;return q;};
  EXPECT_NEAR(kmpc::bounded_path_progress(.8,10,origin,cap,at),1.3,1e-8);
}
TEST(PathProgress, ChecksEveryBoundaryAndPreservesJointLimit) {
  Vec18 cap=Vec18::Constant(.005),origin=Vec18::Zero();
  auto at=[](double s){Vec18 q=Vec18::Zero();q[0]=s<=1?.02*s:.02*(2-s);return q;};
  EXPECT_NEAR(kmpc::bounded_path_progress(0,2,origin,cap,at),.25,1e-8);
}

TEST(PathProgress, RecoverResidualStillExposesNearestForBridge) {
  // r27 field: residual>15 mrad must not erase the nearest sample — callers
  // bridge from the projected command toward that sample.
  Vec18 accepted = Vec18::Zero();
  accepted[1] = 0.05;  // 50 mrad off the path at s≈1
  auto at = [](double s) {
    Vec18 q = Vec18::Zero();
    q[1] = 0.01 * s;
    return q;
  };
  const auto rec = kmpc::recover_path_progress(0.0, 3.0, accepted, at, 0.015);
  EXPECT_FALSE(rec.ok);
  EXPECT_GT(rec.residual_rad, 0.015);
  EXPECT_NEAR(rec.prog, 3.0, 0.05);  // nearest end of window
  EXPECT_EQ(kmpc::count_cap_substeps(accepted, at(rec.prog),
                                     Vec18::Constant(0.0024)),
            9);
}

TEST(PathProgress, FindProgressWithinCapReanchorsProjectedStep) {
  // r26-class case: projected `to` sits off the stale progress[0] by >cap, but a
  // nearby path sample still fits the velocity box.
  Vec18 cap = Vec18::Constant(0.0024);
  auto at = [](double s) {
    Vec18 q = Vec18::Zero();
    q[1] = 0.01 * s;
    return q;
  };
  Vec18 origin = Vec18::Zero();
  origin[1] = 0.010 + 0.0015;  // between s=1.0 and s=1.15
  const auto s =
      kmpc::find_progress_within_cap(0.0, 3.0, origin, cap, at);
  ASSERT_TRUE(s.has_value());
  EXPECT_NEAR(*s, 1.15, 0.05);
  EXPECT_TRUE(kmpc::within_velocity_cap(origin, at(*s), cap));
  // Far off-path: no sample within one tick (path max at s=3 is 0.03).
  origin[1] = 0.05;
  EXPECT_FALSE(kmpc::find_progress_within_cap(0.0, 3.0, origin, cap, at));
  EXPECT_EQ(kmpc::count_cap_substeps(Vec18::Zero(), at(1.0), cap), 5);
}

TEST(LineStreamCommit, MidChordRebaseWasFiveSixthsNowEndpointKeepsScaleOne) {
  // Field fingerprint: previous chord length == cap, commit at u=0.8 to a full
  // next step → |Δ|=1.2·cap → scale exactly 5/6. Anchoring at the accepted
  // endpoint keeps scale=1 (production queue arms only at u=1).
  const double cap_j = 0.0024;
  Vec18 cap = Vec18::Constant(cap_j);
  Vec18 endpoint = Vec18::Zero();
  endpoint[1] = cap_j;             // previous chord length == one tick
  Vec18 mid = 0.8 * endpoint;      // u=0.8
  Vec18 next = endpoint;
  next[1] += cap_j;                // + one full cruise step
  auto mid_step = bound_line_stream_step(mid, next, cap);
  ASSERT_TRUE(mid_step);
  EXPECT_NEAR(mid_step->scale, 5.0 / 6.0, 1e-9);
  auto end_step = bound_line_stream_step(endpoint, next, cap);
  ASSERT_TRUE(end_step);
  EXPECT_NEAR(end_step->scale, 1.0, 1e-12);
}

TEST(LineSmoother, PassthroughTracksRawAndSpeed) {
  // M09: FIR mean deleted; smoother is identity + speed_inf for settle checks.
  kmpc::ros2io::LineSmoother f;
  Vec18 q = Vec18::Zero();
  f.reset(q);
  for (int n = 0; n < 100; ++n) {
    q[0] += 0.002;
    const Vec18 out = f.push(q, 0.004);
    EXPECT_NEAR(out[0], q[0], 1e-12);
    EXPECT_NEAR(f.speed_inf(), 0.5, 1e-9);  // 0.002 / 0.004
  }
  const Vec18 held = f.push(q, 0.004);
  EXPECT_NEAR(held[0], 0.2, 1e-12);
  EXPECT_LT(f.speed_inf(), 1e-10);
}

#include "kmpc/path_layer_metrics.hpp"
TEST(PathCornerMetrics, DetectsInteriorTangentJump) {
  std::vector<Vec18, Eigen::aligned_allocator<Vec18>> samples(4, Vec18::Zero());
  samples[1][0] = 0.1;
  samples[2][0] = 0.1;
  samples[3][0] = 0.0;  // reverse after plateau → corner
  const auto m = kmpc::measure_path_corners(samples);
  EXPECT_GE(m.knots, 3);
  // Finite-diff tangent jump ≈ |(+0) - (−0.1)| / 1 ≈ 0.1 on this knot spacing.
  EXPECT_GT(m.max_tangent_jump_rad, 0.05);
  EXPECT_GE(m.worst_knot, 1);
  std::vector<Vec18, Eigen::aligned_allocator<Vec18>> straight(4, Vec18::Zero());
  for (int i = 0; i < 4; ++i) {
    straight[static_cast<std::size_t>(i)][0] = 0.05 * i;
  }
  const auto s = kmpc::measure_path_corners(straight);
  EXPECT_LT(s.max_tangent_jump_rad, m.max_tangent_jump_rad);
}

TEST(LineQpFollowMode, ParsesABC) {
  kmpc::LineQpFollowMode mode = kmpc::LineQpFollowMode::A;
  EXPECT_TRUE(kmpc::parse_line_qp_follow_mode("B", &mode));
  EXPECT_EQ(mode, kmpc::LineQpFollowMode::B);
  EXPECT_TRUE(kmpc::parse_line_qp_follow_mode("min_correction", &mode));
  EXPECT_EQ(mode, kmpc::LineQpFollowMode::C);
  EXPECT_FALSE(kmpc::parse_line_qp_follow_mode("Z", &mode));
}
