#include <gtest/gtest.h>

#include "node_cache.hpp"

using namespace kmpc;
using namespace kmpc::ros2io;

namespace {

CartesianTarget make_left_approach(uint64_t generation) {
  CartesianTarget t;
  t.generation = generation;
  t.frame_id = std::string(kBaseFrame);
  t.xyz = Vec3(0.30, 0.20, 0.645);
  t.quat_xyzw = Vec4(0.0, 0.0, 0.0, 1.0);
  LineTarget tip;
  tip.xyz = Vec3(0.30, 0.20, 0.495);
  tip.quat_xyzw = Vec4(0.0, 0.0, 0.0, 1.0);
  tip.frame = kBaseFrame;
  t.cascade = tip;
  return t;
}

// Mirrors on_timer after alarm_clear_ready: drop stale, then nail only if empty.
bool would_nail_after_resume(const TaskReceiver& task) {
  const auto targets = task.targets();
  const bool use_cart = targets.first.has_value() || targets.second.has_value();
  return !use_cart && !task.has_q_target();
}

}  // namespace

// 20261109T014739Z: alarm cleared >2s, enable + segment same tick, first
// on_timer must keep the segment and must not take the empty-servo nail path.
TEST(AlarmResume, FreshEnableSegmentSurvivesResume) {
  TaskReceiver task;
  const uint64_t alarm_block_generation = task.generation();  // latched at alarm
  ASSERT_EQ(alarm_block_generation, 0u);

  // Alarm text empty already held >=2s; enable turns on; business sends one
  // /kmpc/segment_left before the first enabled on_timer.
  const uint64_t gen = 1794188890679472834ULL;
  ASSERT_TRUE(task.on_segment(true, make_left_approach(gen)));
  ASSERT_TRUE(task.target(true));

  task.drop_cartesian_up_to(alarm_block_generation);

  ASSERT_TRUE(task.target(true));
  EXPECT_EQ(task.target(true)->generation, gen);
  EXPECT_TRUE(task.target(true)->cascade.has_value());
  EXPECT_TRUE(task.saw_cartesian_target());
  EXPECT_FALSE(would_nail_after_resume(task));
}

TEST(AlarmResume, PreAlarmSegmentDroppedOnResume) {
  TaskReceiver task;
  ASSERT_TRUE(task.on_segment(true, make_left_approach(100)));
  const uint64_t alarm_block_generation = task.generation();
  EXPECT_EQ(alarm_block_generation, 100u);

  // No new enable-time segment. Resume must discard the pre-alarm goal.
  task.drop_cartesian_up_to(alarm_block_generation);
  EXPECT_FALSE(task.target(true));
  EXPECT_FALSE(task.saw_cartesian_target());
  EXPECT_TRUE(would_nail_after_resume(task));
  // Cleared stale id stays rejected; a newer id can still arm.
  EXPECT_FALSE(task.on_segment(true, make_left_approach(100)));
  ASSERT_TRUE(task.on_segment(true, make_left_approach(101)));
  EXPECT_EQ(task.target(true)->generation, 101u);
}

TEST(AlarmResume, ClearBeforeSegmentStillAllowsInstall) {
  TaskReceiver task;
  ASSERT_TRUE(task.on_segment(true, make_left_approach(50)));
  const uint64_t alarm_block_generation = task.generation();

  // Resume tick runs before the enable-time segment callback.
  task.drop_cartesian_up_to(alarm_block_generation);
  EXPECT_FALSE(task.target(true));

  const uint64_t gen = 1794188890679472834ULL;
  ASSERT_TRUE(task.on_segment(true, make_left_approach(gen)));
  EXPECT_EQ(task.target(true)->generation, gen);
  EXPECT_FALSE(would_nail_after_resume(task));
}
