#include <gtest/gtest.h>
#include "node_cache.hpp"
using namespace kmpc;using namespace kmpc::ros2io;
TEST(LineSegment, AtomicGenerationTipAndArmReplacement) {
  TaskReceiver rx(Vec18::Zero());CartesianTarget t;t.generation=100;t.xyz=Vec3(1,2,3);
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
  TaskReceiver rx(Vec18::Zero());CartesianTarget t;rx.on_target_left(t);
  auto first=rx.target(true)->generation;t.xyz[0]=.15;rx.on_target_left(t);
  EXPECT_EQ(first,rx.target(true)->generation);
  rx.clear_left();rx.on_target_left(t);EXPECT_GT(rx.target(true)->generation,first);
}
TEST(LineSegment, InvalidCoordinatesNeverReplaceTarget) {
  TaskReceiver rx(Vec18::Zero());CartesianTarget t;t.generation=1;t.frame_id="camera";
  EXPECT_FALSE(rx.on_segment(true,t));EXPECT_FALSE(rx.target(true));
  t.frame_id="base_link";t.xyz[0]=std::numeric_limits<double>::quiet_NaN();EXPECT_FALSE(rx.on_segment(true,t));
}

TEST(LineSegment, DelayedClearCannotEraseTheNextSegment) {
  TaskReceiver rx(Vec18::Zero()); CartesianTarget t;t.generation=100;
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

TEST(LineSmoother, PreservesSpeedAndRemovesVelocityStep) {
  kmpc::ros2io::LineSmoother f;Vec18 q=Vec18::Zero(),last=q,prior_d=Vec18::Zero();f.reset(q);
  double peak_dd=0;
  for(int n=0;n<200;++n) {
    if(n<100)q[0]+=.002;
    const Vec18 out=f.push(q,.004),d=out-last;
    EXPECT_LE(d.cwiseAbs().maxCoeff(),.002+1e-12);
    peak_dd=std::max(peak_dd,(d-prior_d).cwiseAbs().maxCoeff());last=out;prior_d=d;
  }
  EXPECT_LT(peak_dd,.00011);EXPECT_NEAR(last[0],.2,1e-12);EXPECT_LT(f.speed_inf(),1e-10);
}
