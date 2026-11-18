#pragma once

// Shared ROS2 parameter declarations for kmpc_node.
// stage_ack / motion_sequence topics are fixed constants (see v2_submit_gate.hpp),
// not parameters — keep them out of this list so construct tests catch regressions.

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/v2_submit_gate.hpp"

namespace kmpc {
namespace ros2_params {

inline const std::vector<std::string>& string_params_read_by_str_or() {
  static const std::vector<std::string> kNames = {
      "robot_description",
      "joint_map_file",
      "joint_limits_file",
      "kmpc_params_file",
      "line_pace_mode",
      "line_brake_profile",
      "line_qp_follow_mode",
      "line_chain_profile",
      "planner",
      "layer_diag_path",
      "qp_fail_dump_dir",
      "config_snapshot_path",
      "stream_topic",
      "joint_states_topic",
      "enable_topic",
      "mode_topic",
      "q_target_topic",
      "arrived_topic",
      "target_left_topic",
      "target_right_topic",
      "clear_targets_topic",
      "waist_locked_topic",
      "tracking_state_topic",
      "q_calib_topic",
  };
  return kNames;
}

inline const std::vector<std::string>& forbidden_topic_params() {
  static const std::vector<std::string> kNames = {
      "stage_ack_topic",
      "motion_sequence_topic",
  };
  return kNames;
}

template <typename NodeT>
void declare_kmpc_node_params(NodeT& node) {
  node.template declare_parameter<std::string>("robot_description", "");
  node.template declare_parameter<std::string>("joint_map_file", "");
  node.template declare_parameter<std::string>("joint_limits_file", "");
  node.template declare_parameter<std::string>("kmpc_params_file", "");
  node.declare_parameter("auto_enable", false);
  node.declare_parameter("demo_wave_waist", false);
  node.declare_parameter("demo_wave_amp", 0.5);
  node.declare_parameter("demo_wave_period", 8.0);
  node.declare_parameter("state_timeout_sec", 0.1);
  node.declare_parameter("hold_deadband", 0.01);
  node.declare_parameter("waist_locked", false);
  node.declare_parameter("require_xoip", true);
  node.declare_parameter("dq_scale", 1.0);
  node.declare_parameter("line_dq_scale", 0.60);
  node.template declare_parameter<std::string>(
      "line_pace_mode",
      kmpc::CartesianTracker::line_pace_mode_name(
          kmpc::CartesianTracker::kDefaultLinePaceMode));
  node.declare_parameter("line_brake_distance_m", 0.08);
  node.declare_parameter("line_brake_floor", 0.20);
  node.template declare_parameter<std::string>(
      "line_brake_profile",
      kmpc::CartesianTracker::line_brake_profile_name(
          kmpc::CartesianTracker::kDefaultLineBrakeProfile));
  node.declare_parameter("line_qp_max_iter", 200);
  node.template declare_parameter<std::string>("line_qp_follow_mode", "A");
  node.template declare_parameter<std::string>("line_chain_profile",
                                               "new_unified");
  node.template declare_parameter<std::string>("planner",
                                               "unified_multistage_v2");
  // Legacy PoseArray/cascade two-point only. Typed MotionSequence is untouched.
  node.template declare_parameter<std::string>("legacy_prep_geometry", "ptp");
  node.template declare_parameter<std::string>("legacy_tip_geometry", "lin");
  node.declare_parameter("line_pose_pref_enable", false);
  node.declare_parameter("line_pitch_straighten_enable", true);
  node.declare_parameter("line_qp_pitch_monotonic_enable", true);
  node.declare_parameter("reverse_last_line_enable", true);
  node.declare_parameter("line_direction_strict_enable", true);
  node.declare_parameter("line_shadow_diag_enable", true);
  node.declare_parameter("line_diag_enable", false);
  node.template declare_parameter<std::string>("layer_diag_path", "");
  node.template declare_parameter<std::string>(
      "qp_fail_dump_dir", "/home/nvidia/workspaces/kmpc/qp_fail_dumps");
  node.template declare_parameter<std::string>("config_snapshot_path", "");
  node.declare_parameter("stream_hz", 250.0);
  node.declare_parameter("interp_catch_s", 0.02);
  node.declare_parameter("interp_accel_s", 0.50);
  node.declare_parameter("interp_jerk_s", 0.08);
  node.declare_parameter("stream_dedicated_thread", true);
  node.declare_parameter("stream_debug_q_hz", 50.0);
  node.template declare_parameter<std::string>(
      "stream_topic", "/xoip/robot/multi_joint_servo_cmd");
  node.template declare_parameter<std::string>("joint_states_topic",
                                               "/joint_states");
  node.template declare_parameter<std::string>("enable_topic", "/kmpc/enable");
  node.template declare_parameter<std::string>("mode_topic", "/kmpc/mode");
  node.template declare_parameter<std::string>("q_target_topic",
                                               "/kmpc/q_target");
  node.template declare_parameter<std::string>("arrived_topic",
                                               "/kmpc/arrived");
  node.template declare_parameter<std::string>("target_left_topic",
                                               "/kmpc/target_left");
  node.template declare_parameter<std::string>("target_right_topic",
                                               "/kmpc/target_right");
  node.template declare_parameter<std::string>("clear_targets_topic",
                                               "/kmpc/clear_targets");
  node.template declare_parameter<std::string>("waist_locked_topic",
                                               "/kmpc/waist_locked");
  node.template declare_parameter<std::string>("tracking_state_topic",
                                               "/kmpc/tracking_state");
  node.template declare_parameter<std::string>("q_calib_topic",
                                               "/kmpc/q_calib");
  // Intentionally NOT declaring stage_ack_topic / motion_sequence_topic:
  // subscriptions use kStageAckTopic / kMotionSequenceTopic constants.
  (void)kStageAckTopic;
  (void)kMotionSequenceTopic;
}

}  // namespace ros2_params
}  // namespace kmpc
