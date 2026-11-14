#include <cmath>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/line_final_gate.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

// r40 rise: J1 already past the 0.3° soft wall by ~1.53e-5 rad, then the
// first chord grows another ~0.004°. Still short of the 58° hard stop.
kmpc::Vec18 rise_pose(const kmpc::Config& config, double extra_rad) {
  kmpc::Vec18 q = config.q_pref;
  const double soft =
      config.limits.q_max[0] - config.params.q_limit_margin;
  q[0] = soft + 1.53e-5 + extra_rad;
  q[1] = -0.70;
  q[2] = -q[0] - q[1];
  return q;
}

}  // namespace

TEST(R40RiseSoftWall, StreamAllowsHardBoxChordAndStillRejectsPastHard) {
  const kmpc::Config config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, config);
  tracker.set_line_pace(0.6, 0.0);
  tracker.set_line_direction_strict_enable(false);
  tracker.set_line_pitch_straighten_enable(false);
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q0, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat =
      kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.compute(q0, config.q_pref, &goal, nullptr, &quat, nullptr);
  ASSERT_TRUE(tracker.line_active()) << tracker.line_reason();

  const double extra = 0.004 * kmpc::kPi / 180.0;
  const kmpc::Vec18 from = rise_pose(config, 0.0);
  kmpc::Vec18 to = rise_pose(config, extra);
  ASSERT_GT(from[0], config.limits.q_max[0] - config.params.q_limit_margin);
  ASSERT_GT(to[0], from[0]);
  ASSERT_LT(to[0], config.limits.q_max[0]);
  ASSERT_LT(std::abs(from[0] + from[1] + from[2]), 1e-12);

  const auto chord = tracker.line_stream_segment_check(from, to);
  EXPECT_GT(chord.joint_margin_rad, 1e-3);
  EXPECT_STRNE(chord.reason, "joint");

  // Both ends past the hard stop. The checker returns on the first bad
  // sample; an off-line pose can trip the tube before a later sample reaches
  // a limit that only the chord endpoint violates.
  kmpc::Vec18 past_from = from;
  kmpc::Vec18 past_to = from;
  past_from[0] = config.limits.q_max[0] + 0.01;
  past_to[0] = config.limits.q_max[0] + 0.02;
  const auto past = tracker.line_stream_segment_check(past_from, past_to);
  EXPECT_FALSE(past.ok);
  EXPECT_STREQ(past.reason, "joint");
  EXPECT_LT(past.joint_margin_rad, 0.0);
}

TEST(R40RiseSoftWall, FinalGateUsesTheSameHardBox) {
  const kmpc::Config config = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto limits = kmpc::make_final_gate_limits(config);
  const kmpc::Vec18 inside = rise_pose(config, 0.004 * kmpc::kPi / 180.0);
  const char* inside_fault = kmpc::line_final_filtered_fault(
      inside, 0.0, limits, kin, guards, nullptr);
  EXPECT_TRUE(
      inside_fault == nullptr ||
      std::strcmp(inside_fault, "stream_filtered_joint_limit") != 0)
      << (inside_fault == nullptr ? "ok" : inside_fault);

  kmpc::Vec18 past = inside;
  past[0] = config.limits.q_max[0] + 0.01;
  EXPECT_STREQ(
      kmpc::line_final_filtered_fault(past, 0.0, limits, kin, guards, nullptr),
      "stream_filtered_joint_limit");
}
