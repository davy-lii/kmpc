#include <future>
#include <geometry_msgs/msg/pose_array.hpp>
// 对应 Python kmpc/kmpc_node.py。只编排 Planner / Tracker / Safety；不改 Hessian。
// 本文件允许 rclcpp。src/core|solver|planner|tracker 禁止 include rclcpp。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#endif

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <array>
#include <kmpc_msgs/msg/motion_sequence.hpp>
#include <kmpc_msgs/msg/motion_stage.hpp>
#include <kmpc_msgs/msg/stage_ack.hpp>
#include "kmpc/trajectory_package.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <yaml-cpp/yaml.h>

#include "joint_space_tracker.hpp"
#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/line_async_handoff.hpp"
#include "kmpc/line_final_gate.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/planner_worker.hpp"
#include "kmpc/qp_fail_dump_worker.hpp"
#include "kmpc/safety.hpp"
#include "kmpc/safety_supervisor.hpp"
#include "kmpc/stream_interpolator.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/types.hpp"
#include "kmpc/upper_planner.hpp"
#include "kmpc/v2_submit_gate.hpp"
#include "kmpc/whole_body_guards.hpp"
#include "kmpc_declared_params.hpp"
#include "node_cache.hpp"
#include "layer_diag.hpp"
#include "line_stream.hpp"

#ifdef KMPC_HAVE_XOIP
#include <xoip_ros2/msg/joint_servo_target.hpp>
#include <xoip_ros2/msg/multi_joint_servo_command.hpp>
#include <xoip_ros2/msg/stream_status.hpp>
#include <xoip_ros2/srv/clear_route.hpp>
#endif

namespace {

std::string default_config(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

void check_joint_map(const std::string& path) {
  if (path.empty()) {
    return;
  }
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node robots = root["robots"];
  if (!robots) {
    throw kmpc::ConfigError("joint_map.yaml missing robots");
  }
  const auto expect = [&](int robot_id, kmpc::Slice sl) {
    YAML::Node spec = robots[robot_id];
    if (!spec) {
      spec = robots[std::to_string(robot_id)];
    }
    if (!spec || !spec["joints"] || !spec["joints"].IsSequence()) {
      throw kmpc::ConfigError("joint_map missing robot_id=" + std::to_string(robot_id));
    }
    if (static_cast<int>(spec["joints"].size()) != sl.size()) {
      throw kmpc::ConfigError("joint_map robot_id size mismatch");
    }
    for (int i = 0; i < sl.size(); ++i) {
      const std::string name = spec["joints"][i].as<std::string>();
      if (name != kmpc::kJointNames[static_cast<std::size_t>(sl.begin + i)]) {
        throw kmpc::ConfigError(
            "joint_map names != types.hpp: got " + name + " expected " +
            std::string(kmpc::kJointNames[static_cast<std::size_t>(sl.begin + i)]));
      }
    }
  };
  expect(kmpc::kRobotIdRight, kmpc::kSliceRight);
  expect(kmpc::kRobotIdLeft, kmpc::kSliceLeft);
  expect(kmpc::kRobotIdWaist, kmpc::kSliceWaist);
}

kmpc::ros2io::CartesianTarget from_pose(
    const geometry_msgs::msg::PoseStamped& msg) {
  kmpc::ros2io::CartesianTarget t;
  t.xyz = kmpc::Vec3(
      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
  t.quat_xyzw = kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(
      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
      msg.pose.orientation.w));
  t.frame_id = msg.header.frame_id.empty() ? std::string(kmpc::kBaseFrame)
                                          : msg.header.frame_id;
  return t;
}

kmpc::TaskGeometry geometry_from_msg(uint8_t g) {
  switch (g) {
    case kmpc_msgs::msg::MotionStage::PTP:
      return kmpc::TaskGeometry::Ptp;
    case kmpc_msgs::msg::MotionStage::FOLLOW_CURVE:
      return kmpc::TaskGeometry::FollowCurve;
    case kmpc_msgs::msg::MotionStage::LIN:
    default:
      return kmpc::TaskGeometry::Lin;
  }
}

// Builds MotionTask from typed MotionSequence. Does not start planning/follow.
std::optional<kmpc::MotionTask> motion_task_from_sequence(
    const kmpc_msgs::msg::MotionSequence& msg) {
  if (msg.stages.empty() ||
      msg.stages.size() > kmpc::ros2io::kMaxMotionStages ||
      msg.generation == 0) {
    return std::nullopt;
  }
  kmpc::MotionTask task;
  task.generation = msg.generation;
  task.left = msg.left;
  task.waist_locked = msg.waist_locked;
  task.frame = msg.header.frame_id.empty() ? std::string(kmpc::kBaseFrame)
                                           : msg.header.frame_id;
  task.stages.reserve(msg.stages.size());
  for (std::size_t i = 0; i < msg.stages.size(); ++i) {
    const auto& ms = msg.stages[i];
    kmpc::TaskStage st;
    st.geometry = geometry_from_msg(ms.geometry);
    st.position = kmpc::Vec3(ms.target.position.x, ms.target.position.y,
                             ms.target.position.z);
    st.quat = kmpc::Vec4(ms.target.orientation.x, ms.target.orientation.y,
                         ms.target.orientation.z, ms.target.orientation.w);
    st.has_orientation = ms.constrain_orientation;
    st.position_tol_m = ms.position_tol_m;
    st.orientation_tol_rad = ms.orientation_tol_rad;
    st.stop = ms.stop_at_end ? kmpc::StageStop::StopRequired
                             : kmpc::StageStop::Continuous;
    st.wait_for_external_ack = ms.wait_for_external_ack;
    st.stage_id = ms.stage_id != 0u ? ms.stage_id : static_cast<uint32_t>(i);
    task.stages.push_back(st);
  }
  return task;
}

std::optional<kmpc::Vec18> q_from_joint_state(
    const sensor_msgs::msg::JointState& msg, const kmpc::Vec18& fallback) {
  if (msg.position.empty()) {
    return std::nullopt;
  }
  if (!msg.name.empty()) {
    std::unordered_map<std::string, double> named;
    const std::size_t n = std::min(msg.name.size(), msg.position.size());
    for (std::size_t i = 0; i < n; ++i) {
      named[msg.name[i]] = msg.position[i];
    }
    return kmpc::ros2io::q_from_name_positions(named, fallback);
  }
  if (static_cast<int>(msg.position.size()) != kmpc::kNDof) {
    return std::nullopt;
  }
  kmpc::Vec18 q;
  for (int i = 0; i < kmpc::kNDof; ++i) {
    q[i] = msg.position[static_cast<std::size_t>(i)];
  }
  return q;
}

}  // namespace

class KmpcNode : public rclcpp::Node {
 public:
  KmpcNode() : Node("kmpc_node") {
    declare_params();
    const std::string limits_file = str_or("joint_limits_file", default_config("joint_limits.yaml"));
    const std::string q_pref_file = str_or("q_pref_file", default_config("q_pref.yaml"));
    const std::string params_file =
        str_or("kmpc_params_file", default_config("kmpc_params.yaml"));
    const std::string station = str_or("q_pref_station", "default");
    check_joint_map(str_or("joint_map_file", default_config("joint_map.yaml")));
    cfg_ = kmpc::Config::load(limits_file, q_pref_file, params_file, station);

    const bool require_xoip = bool_param("require_xoip", true);
#ifndef KMPC_HAVE_XOIP
    const char* xoip_msg =
        "xoip_ros2 MultiJointServoCommand unavailable. Source workspace overlay.";
    if (require_xoip) {
      throw kmpc::ConfigError(xoip_msg);
    }
    RCLCPP_ERROR(get_logger(), "%s", xoip_msg);
#else
    (void)require_xoip;
#endif

    std::string urdf_xml = str_or("robot_description", "");
    line_urdf_source_=urdf_xml.empty()?std::string(KMPC_URDF_DIR)+"/Light-duty wheeled robot.urdf":urdf_xml;
    line_urdf_xml_=!urdf_xml.empty();
    if (urdf_xml.empty()) {
      kin_ = std::make_unique<kmpc::Kinematics>(
          std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
    } else {
      kin_ = std::make_unique<kmpc::Kinematics>(
          urdf_xml, kmpc::Kinematics::UrdfInput::Xml);
    }
    stream_kin_=std::make_unique<kmpc::Kinematics>(line_urdf_source_,
        line_urdf_xml_?kmpc::Kinematics::UrdfInput::Xml:kmpc::Kinematics::UrdfInput::Path);
    // Legacy async prepare_line still owns a dedicated Pinocchio model when
    // planner!=unified_multistage_v2. V2 uses PlannerWorker instead.
    line_plan_kin_=std::make_unique<kmpc::Kinematics>(line_urdf_source_,
        line_urdf_xml_?kmpc::Kinematics::UrdfInput::Xml:kmpc::Kinematics::UrdfInput::Path);
    stream_guards_=std::make_unique<kmpc::WholeBodyGuards>(cfg_.params,cfg_.q_pref[kmpc::kIdxYaw]);
    tracker_ = std::make_unique<kmpc::CartesianTracker>(*kin_, cfg_);
    planner_ = std::make_unique<kmpc::UpperPlanner>(
        *kin_, cfg_.params.calib_tol_rad, cfg_.params.com_xmin,
        cfg_.params.com_xmax, cfg_.params.com_margin);
    safety_ = std::make_unique<kmpc::SafetyFilter>(
        cfg_.limits, cfg_.params.dt, cfg_.params.q_limit_margin);
    const auto motion_policy = kmpc::policy_from_config(cfg_);
    follower_ = std::make_unique<kmpc::TrajectoryFollower>(
        *kin_, &tracker_->guards(), motion_policy);
    safety_supervisor_ = std::make_unique<kmpc::SafetySupervisor>(
        *kin_, stream_guards_.get(), motion_policy);
    safety_supervisor_->seed(cfg_.q_pref);
    interpolator_.set_joint_box(
        cfg_.limits.q_min, cfg_.limits.q_max, cfg_.params.q_limit_margin);
    joint_tracker_ = std::make_unique<kmpc::ros2io::JointSpaceTracker>(
        *kin_, cfg_);
    q_des_star_ = cfg_.q_pref;

    task_ = std::make_unique<kmpc::ros2io::TaskReceiver>(cfg_.q_pref);
    planner_rx_ = std::make_unique<kmpc::ros2io::PlannerReceiver>(cfg_.q_pref);
    arbiter_ = std::make_unique<kmpc::ros2io::CommandArbiter>(
        bool_param("auto_enable", false));
    state_ = std::make_unique<kmpc::ros2io::StateAggregator>(
        double_param("state_timeout_sec", 0.1));
    if (bool_param("waist_locked", false)) {
      planner_->set_waist_locked(true);
      planner_rx_->on_waist_locked(true);
    }
    demo_wave_ = bool_param("demo_wave_waist", false);
    demo_amp_ = double_param("demo_wave_amp", 0.5);
    demo_period_ = double_param("demo_wave_period", 8.0);
    hold_deadband_ = double_param("hold_deadband", 0.01);
    dq_scale_ = std::clamp(double_param("dq_scale", 1.0), 0.0, 1.0);
    line_dq_scale_ = std::clamp(double_param("line_dq_scale", 0.60), 0.0, 1.0);
    {
      const std::string mode_text = str_or(
          "line_pace_mode",
          kmpc::CartesianTracker::line_pace_mode_name(
              kmpc::CartesianTracker::kDefaultLinePaceMode));
      kmpc::CartesianTracker::LinePaceMode mode =
          kmpc::CartesianTracker::kDefaultLinePaceMode;
      if (!kmpc::CartesianTracker::parse_line_pace_mode(mode_text, &mode)) {
        throw kmpc::ConfigError(
            "line_pace_mode must be r11_scurve or r10b_first_order");
      }
      line_pace_mode_ = mode;
      tracker_->set_line_pace_mode(mode);
    }
    line_brake_distance_m_ = double_param("line_brake_distance_m", 0.08);
    if (!tracker_->set_line_brake_distance_m(line_brake_distance_m_)) {
      throw kmpc::ConfigError(
          "line_brake_distance_m must be finite and > 0");
    }
    line_brake_distance_m_ = tracker_->line_brake_distance_m();
    line_brake_floor_ = double_param("line_brake_floor", 0.20);
    if (!tracker_->set_line_brake_floor(line_brake_floor_)) {
      throw kmpc::ConfigError(
          "line_brake_floor must be finite and in (0,1]");
    }
    line_brake_floor_ = tracker_->line_brake_floor();
    {
      const std::string profile_text =
          str_or("line_brake_profile",
                 kmpc::CartesianTracker::line_brake_profile_name(
                     kmpc::CartesianTracker::kDefaultLineBrakeProfile));
      kmpc::CartesianTracker::LineBrakeProfile profile =
          kmpc::CartesianTracker::kDefaultLineBrakeProfile;
      if (!kmpc::CartesianTracker::parse_line_brake_profile(profile_text,
                                                            &profile)) {
        throw kmpc::ConfigError(
            "line_brake_profile must be linear or sqrt");
      }
      line_brake_profile_ = profile;
      tracker_->set_line_brake_profile(profile);
    }
    line_qp_max_iter_ = int_param("line_qp_max_iter", 200);
    if (!tracker_->set_line_qp_max_iter(line_qp_max_iter_)) {
      throw kmpc::ConfigError(
          "line_qp_max_iter must be in [1,10000]");
    }
    line_qp_max_iter_ = tracker_->line_qp_max_iter();
    {
      const std::string mode_text = str_or("line_qp_follow_mode", "A");
      kmpc::LineQpFollowMode mode = kmpc::LineQpFollowMode::A;
      if (!kmpc::parse_line_qp_follow_mode(mode_text, &mode)) {
        throw kmpc::ConfigError(
            "line_qp_follow_mode must be A|B|C (current|direct|min_correction)");
      }
      line_qp_follow_mode_ = mode;
      tracker_->set_line_qp_follow_mode(mode);
    }
    {
      // P02: default NewUnified (offline suite covers it); FIR mean deleted.
      const std::string profile_text =
          str_or("line_chain_profile", "new_unified");
      kmpc::CartesianTracker::LineChainProfile profile =
          kmpc::CartesianTracker::LineChainProfile::Legacy;
      if (profile_text == "new_unified" || profile_text == "NewUnified") {
        profile = kmpc::CartesianTracker::LineChainProfile::NewUnified;
      } else if (profile_text != "legacy" && profile_text != "Legacy") {
        throw kmpc::ConfigError(
            "line_chain_profile must be legacy|new_unified");
      }
      line_chain_profile_ = profile;
      tracker_->set_line_chain_profile(profile);
      if (profile == kmpc::CartesianTracker::LineChainProfile::NewUnified) {
        line_qp_follow_mode_ = tracker_->line_qp_follow_mode();
        line_pace_mode_ = tracker_->line_pace_mode();
      }
    }
    final_gate_limits_ = kmpc::make_final_gate_limits(cfg_);
    planner_name_ = str_or("planner", "unified_multistage_v2");
    use_v2_chain_ = (planner_name_ == "unified_multistage_v2");
    line_pose_pref_enable_ = bool_param("line_pose_pref_enable", false);
    tracker_->set_line_pose_pref(line_pose_pref_enable_);
    tracker_->set_line_pose_pref_q(cfg_.q_pref);
    // P0 path-layer isolation switches (scheme defaults).
    line_pitch_straighten_enable_ =
        bool_param("line_pitch_straighten_enable", true);
    tracker_->set_line_pitch_straighten_enable(line_pitch_straighten_enable_);
    line_qp_pitch_monotonic_enable_ =
        bool_param("line_qp_pitch_monotonic_enable", true);
    tracker_->set_line_qp_pitch_monotonic_enable(
        line_qp_pitch_monotonic_enable_);
    reverse_last_line_enable_ = bool_param("reverse_last_line_enable", true);
    tracker_->set_reverse_last_line_enable(reverse_last_line_enable_);
    line_direction_strict_enable_ =
        bool_param("line_direction_strict_enable", true);
    tracker_->set_line_direction_strict_enable(line_direction_strict_enable_);
    line_shadow_diag_enable_ = bool_param("line_shadow_diag_enable", true);
    tracker_->set_line_shadow_diag_enable(line_shadow_diag_enable_);
    line_diag_enable_ = bool_param("line_diag_enable", false);
    tracker_->set_line_diag_enable(line_diag_enable_);
    if (use_v2_chain_) {
      // V2_CHAIN: these switches have no decision power on the production path.
      RCLCPP_INFO(get_logger(),
                  "planner=unified_multistage_v2: ignoring decision power of "
                  "line_pose_pref_enable / line_pitch_straighten_enable / "
                  "line_qp_pitch_monotonic_enable / reverse_last_line_enable / "
                  "line_direction_strict_enable");
      planner_worker_ = std::make_unique<kmpc::PlannerWorker>(
          line_urdf_source_, cfg_, line_urdf_xml_);
      planner_worker_->start();
    }
    // Node stream path must confirm/abort explicitly; never auto-confirm.
    tracker_->set_line_commit_requires_stream(true);
    run_id_ = kmpc::ros2io::layer_diag_mono_ns();
    tracker_->set_line_diag_run_id(run_id_);
    layer_diag_path_ = str_or("layer_diag_path", "");
    // Must be on a path the kmpc container can write (ljw is not bind-mounted).
    qp_fail_dump_dir_ = str_or(
        "qp_fail_dump_dir",
        "/home/nvidia/workspaces/kmpc/qp_fail_dumps");
    config_snapshot_path_ = str_or("config_snapshot_path", "");
    t0_ = now().seconds();
    configure_command_io();
    stream_topic_name_ = str_or("stream_topic", "/xoip/robot/multi_joint_servo_cmd");
    // Timer only schedules; ofstream runs on a detached writer thread.
    qp_fail_dump_timer_ = create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&KmpcNode::schedule_qp_fail_dump, this));

#ifdef KMPC_HAVE_XOIP
    servo_pub_ = create_publisher<xoip_ros2::msg::MultiJointServoCommand>(
        stream_topic_name_, rclcpp::SensorDataQoS());
    stream_status_sub_ = create_subscription<xoip_ros2::msg::StreamStatus>(
        "/xoip/robot/stream_status", 10,
        [this](xoip_ros2::msg::StreamStatus::ConstSharedPtr msg) {
          if (layer_diag_on_) {
            note_sdk_status(msg->sequence, msg->last_send_ok, msg->active);
          }
          if (msg->active && !msg->last_send_ok) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "stream batch rejected by controller seq=%llu mode=%s",
                static_cast<unsigned long long>(msg->sequence),
                msg->mode.c_str());
          }
        });
    clear_route_cli_ = create_client<xoip_ros2::srv::ClearRoute>(
        "/xoip/robot/clear_route");
#endif

    using std::placeholders::_1;
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        str_or("joint_states_topic", "/joint_states"), 10,
        std::bind(&KmpcNode::on_joint_states, this, _1));
    alarm_sub_ = create_subscription<std_msgs::msg::String>(
        "/xoip/system/alarm", 10,
        [this](std_msgs::msg::String::ConstSharedPtr msg) { on_alarm(msg->data); });
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
        str_or("enable_topic", "/kmpc/enable"), 10,
        [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
          const bool on = msg->data;
          const bool turning_off = arbiter_->enabled() && !on;
          arbiter_->set_enabled(on);
          std::lock_guard<std::mutex> lock(stream_mu_);
          if (!on) {
            task_->clear_cartesian();
            tracker_->reset();
            stream_clear_modes();
            publish_arrived(false);
          }
          if (turning_off) {
            release_servo_stream();
            release_waist_route();
          }
        });
    mode_sub_ = create_subscription<std_msgs::msg::String>(
        str_or("mode_topic", "/kmpc/mode"), 10,
        [this](std_msgs::msg::String::ConstSharedPtr msg) {
          if(!kmpc::is_valid_mode(msg->data)) return;
          if(task_->mode()!=msg->data) { task_->clear_cartesian(); tracker_->reset(); }
          task_->on_mode(msg->data);
        });
    q_target_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        str_or("q_target_topic", "/kmpc/q_target"), 10,
        [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
          const auto q = q_from_joint_state(*msg, cfg_.q_pref);
          if (q.has_value()) {
            task_->on_q_target(*q);
          }
        });
    arrived_pub_ = create_publisher<std_msgs::msg::Bool>(
        str_or("arrived_topic", "/kmpc/arrived"), 10);

    left_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        str_or("target_left_topic", "/kmpc/target_left"), 10,
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          task_->on_target_left(from_pose(*msg));
          RCLCPP_INFO(
              get_logger(), "kmpc target_left recv xyz[%.4f,%.4f,%.4f]",
              msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
          tracker_->clear_arrived();
          publish_arrived(false);
          if (use_v2_chain_) {
            v2_request_plan_ = true;
          }
        });
    right_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        str_or("target_right_topic", "/kmpc/target_right"), 10,
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          task_->on_target_right(from_pose(*msg));
          RCLCPP_INFO(
              get_logger(), "kmpc target_right recv xyz[%.4f,%.4f,%.4f]",
              msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
          tracker_->clear_arrived();
          publish_arrived(false);
          if (use_v2_chain_) {
            v2_request_plan_ = true;
          }
        });
    auto segment_callback=[this](bool left,geometry_msgs::msg::PoseArray::ConstSharedPtr msg) {
      if (msg->poses.size()>2 || msg->header.stamp.sec<0 ||
          msg->header.frame_id!=kmpc::kBaseFrame) {
        RCLCPP_ERROR(get_logger(),"invalid line segment/frame"); return;
      }
      const uint64_t generation=uint64_t(msg->header.stamp.sec)*1000000000ULL+msg->header.stamp.nanosec;
      if(generation==0 || msg->header.stamp.nanosec>=1000000000U) return;
      if(msg->poses.empty()) {
        if(task_->clear_segment(left,generation)) {
          tracker_->reset();
          publish_arrived(false);
          std::lock_guard<std::mutex> lock(stream_mu_);
          if (interpolator_.ready()) stream_enter_hold(interpolator_.q_ref());
          else stream_clear_modes();
        }
        return;
      }
      geometry_msgs::msg::PoseStamped pose;
      pose.header=msg->header; pose.pose=msg->poses[0];
      auto target=from_pose(pose);
      target.generation=generation;
      if(msg->poses.size()==2) {
        pose.pose=msg->poses[1]; const auto next=from_pose(pose);
        kmpc::LineTarget tip;
        tip.xyz=next.xyz; tip.quat_xyzw=next.quat_xyzw;
        tip.ori_on=!kmpc::quat_is_identity_xyzw(&tip.quat_xyzw); tip.frame=next.frame_id;
        target.cascade=tip;
      }
      if(!task_->on_segment(left,target)) {
        RCLCPP_WARN(get_logger(),"stale/invalid line generation rejected"); return;
      }
      tracker_->clear_arrived(); publish_arrived(false);
      if (use_v2_chain_) {
        v2_request_plan_ = true;
      }
    };
    left_segment_sub_=create_subscription<geometry_msgs::msg::PoseArray>("/kmpc/segment_left",10,
        [segment_callback](geometry_msgs::msg::PoseArray::ConstSharedPtr m){segment_callback(true,m);});
    right_segment_sub_=create_subscription<geometry_msgs::msg::PoseArray>("/kmpc/segment_right",10,
        [segment_callback](geometry_msgs::msg::PoseArray::ConstSharedPtr m){segment_callback(false,m);});
    // Typed multi-stage task input (1–3 stages). One topic; msg.left selects arm.
    stage_ack_sub_ = create_subscription<kmpc_msgs::msg::StageAck>(
        kmpc::kStageAckTopic, 10,
        [this](kmpc_msgs::msg::StageAck::ConstSharedPtr msg) {
          if (!follower_) {
            return;
          }
          const auto q_opt = state_->read();
          const kmpc::Vec18 q =
              q_opt && q_opt->allFinite() ? *q_opt : kmpc::Vec18::Zero();
          std::string why;
          const double meas_stamp =
              v2_meas_valid_ ? (v2_meas_source_stamp_s_ > 0.0
                                    ? v2_meas_source_stamp_s_
                                    : v2_meas_stamp_s_)
                             : std::numeric_limits<double>::quiet_NaN();
          if (!follower_->on_stage_ack(msg->generation, msg->stage_id,
                                       msg->success, q, &why, meas_stamp,
                                       now().seconds())) {
            RCLCPP_WARN(get_logger(),
                        "kmpc stage_ack rejected gen=%llu stage=%u why=%s",
                        static_cast<unsigned long long>(msg->generation),
                        msg->stage_id, why.c_str());
            if (!msg->success || why == "ack_failed" || why == "ack_pose_drift" ||
                why == "ack_tcp_drift" || why == "ack_feedback_moving" ||
                why == "ack_meas_stale" || why == "ack_meas_missing" ||
                why == "ack_dq_unknown" || why == "ack_timeout") {
              v2_request_plan_ = true;
            }
            return;
          }
          RCLCPP_INFO(get_logger(),
                      "kmpc stage_ack accepted gen=%llu stage=%u",
                      static_cast<unsigned long long>(msg->generation),
                      msg->stage_id);
        });
    motion_sequence_sub_ = create_subscription<kmpc_msgs::msg::MotionSequence>(
        kmpc::kMotionSequenceTopic, 10,
        [this](kmpc_msgs::msg::MotionSequence::ConstSharedPtr msg) {
          auto task = motion_task_from_sequence(*msg);
          if (!task) {
            RCLCPP_ERROR(get_logger(),
                         "invalid MotionSequence (empty/gen/stages>3)");
            return;
          }
          if (!task_->on_motion_sequence(*task)) {
            RCLCPP_WARN(get_logger(),
                        "stale/invalid MotionSequence generation rejected");
            return;
          }
          if (task_->waist_locked_hint()) {
            planner_rx_->on_waist_locked(true);
          }
          RCLCPP_INFO(get_logger(),
                      "kmpc motion_sequence recv gen=%llu left=%d stages=%zu",
                      static_cast<unsigned long long>(task->generation),
                      task->left ? 1 : 0, task->stages.size());
          tracker_->clear_arrived();
          publish_arrived(false);
          // V2_CHAIN: MotionTask is cached; on_timer submits to PlannerWorker.
          if (use_v2_chain_) {
            RCLCPP_INFO(
                get_logger(),
                "kmpc v2 motion_sequence queued gen=%llu digest=%llu stages=%zu",
                static_cast<unsigned long long>(task->generation),
                static_cast<unsigned long long>(
                    kmpc::motion_task_digest(*task)),
                task->stages.size());
            v2_request_plan_ = true;
          }
        });
    clear_sub_ = create_subscription<std_msgs::msg::Empty>(
        str_or("clear_targets_topic", "/kmpc/clear_targets"), 10,
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          task_->clear_cartesian();
          tracker_->reset();
          if (follower_) {
            follower_->reset();
          }
          if (planner_worker_) {
            planner_worker_->clear_pending_and_result();
          }
          if (planner_rx_) {
            planner_rx_->on_waist_locked(false);
          }
          v2_request_plan_ = false;
          v2_gate_.clear();
          publish_arrived(false);
          std::lock_guard<std::mutex> lock(stream_mu_);
          stream_clear_modes();
          publish_v2_slot_inactive();
        });
    clear_left_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/kmpc/clear_left", 10,
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          const bool active=task_->target(true).has_value();
          task_->clear_left();
          if(active) {
            tracker_->reset();
            if (planner_worker_) {
              planner_worker_->clear_pending_and_result();
            }
            if (follower_) {
              follower_->reset();
            }
            v2_request_plan_ = false;
            v2_gate_.clear();
          }
          publish_arrived(false);
        });
    clear_right_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/kmpc/clear_right", 10,
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          const bool active=task_->target(false).has_value();
          task_->clear_right();
          if(active) {
            tracker_->reset();
            if (planner_worker_) {
              planner_worker_->clear_pending_and_result();
            }
            if (follower_) {
              follower_->reset();
            }
            v2_request_plan_ = false;
            v2_gate_.clear();
          }
          publish_arrived(false);
        });

    waist_sub_ = create_subscription<std_msgs::msg::Bool>(
        str_or("waist_locked_topic", "/kmpc/waist_locked"), 10,
        [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
          planner_rx_->on_waist_locked(msg->data);
        });
    state_sub_ = create_subscription<std_msgs::msg::String>(
        str_or("tracking_state_topic", "/kmpc/tracking_state"), 10,
        [this](std_msgs::msg::String::ConstSharedPtr msg) {
          if(msg->data=="CALIBRATING" || msg->data=="calibrating") {
            task_->clear_cartesian(); tracker_->reset();
          }
          planner_rx_->on_tracking_state(msg->data);
        });
    q_calib_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        str_or("q_calib_topic", "/kmpc/q_calib"), 10,
        [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
          const auto q = q_from_joint_state(*msg, cfg_.q_pref);
          if (q.has_value()) {
            planner_rx_->on_q_calib(*q);
          }
        });

    dbg_q_ = create_publisher<sensor_msgs::msg::JointState>("/kmpc/debug/q", 10);
    dbg_dq_ =
        create_publisher<std_msgs::msg::Float64MultiArray>("/kmpc/debug/dq", 10);
    dbg_ee_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/kmpc/debug/ee_error", 10);
    dbg_status_ = create_publisher<std_msgs::msg::String>("/kmpc/debug/status", 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/kmpc/status", 10);
    dbg_com_ =
        create_publisher<visualization_msgs::msg::Marker>("/kmpc/debug/com_marker", 10);

    const double rate = std::max(cfg_.params.control_rate, 1.0);
    emit_effective_config();
    arm_layer_diag();
    timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / rate),
        std::bind(&KmpcNode::on_timer, this));
    if (!stream_dedicated_thread_) {
      stream_timer_ = create_wall_timer(
          std::chrono::duration<double>(interp_dt_),
          std::bind(&KmpcNode::on_stream_timer, this));
    }

    RCLCPP_INFO(
        get_logger(),
        "KMPC C++: dt=%.3fs rate=%.1fHz N=%d enable=%d dq_scale=%.3f line_dq_scale=%.3f r_lambda=%.4g",
        cfg_.params.dt, rate, cfg_.params.n_horizon,
        static_cast<int>(arbiter_->enabled()), dq_scale_, line_dq_scale_,
        tracker_->r_lambda());
    RCLCPP_INFO(
        get_logger(),
        "servo topic=%s publish=%.0fHz catch=%.3f accel_s=%.3f jerk_s=%.3f dq_scale=%.3f tcp=%s/%s",
        stream_topic_name_.c_str(), stream_hz_, interp_catch_s_, interp_accel_s_,
        interp_jerk_s_, dq_scale_,
        kmpc::kEeFrameLeft.data(), kmpc::kEeFrameRight.data());
    RCLCPP_INFO(
        get_logger(),
        "line_pace mode=%s brake_m=%.3f brake_floor=%.3f brake_profile=%s "
        "qp_max_iter=%d freeze_m=0.080 "
        "pitch_straighten=%d qp_pitch_mono=%d reverse_last_line=%d",
        kmpc::CartesianTracker::line_pace_mode_name(line_pace_mode_),
        line_brake_distance_m_, line_brake_floor_,
        kmpc::CartesianTracker::line_brake_profile_name(line_brake_profile_),
        line_qp_max_iter_,
        line_pitch_straighten_enable_ ? 1 : 0,
        line_qp_pitch_monotonic_enable_ ? 1 : 0,
        reverse_last_line_enable_ ? 1 : 0);
    RCLCPP_INFO(
        get_logger(),
        "stream_io dedicated_thread=%d debug_q_hz=%.0f",
        stream_dedicated_thread_ ? 1 : 0, stream_debug_q_hz_);
    if (stream_dedicated_thread_) {
      start_stream_thread();
    }
  }

  ~KmpcNode() override {
    stop_stream_thread();
    if (planner_worker_) {
      planner_worker_->stop();
    }
    if (layer_diag_on_) {
      drain_layer_diag();
    }
    // Join dump worker before destroying this / mutexes.
    qp_dump_worker_.stop_join();
  }

 private:
  void declare_params() {
    kmpc::ros2_params::declare_kmpc_node_params(*this);
  }

  std::string str_or(const char* name, const std::string& fallback) {
    const std::string v = get_parameter(name).as_string();
    return v.empty() ? fallback : v;
  }

  bool bool_param(const char* name, bool fallback) {
    if (!has_parameter(name)) {
      return fallback;
    }
    return get_parameter(name).as_bool();
  }

  double double_param(const char* name, double fallback) {
    if (!has_parameter(name)) {
      return fallback;
    }
    return get_parameter(name).as_double();
  }

  int int_param(const char* name, int fallback) {
    if (!has_parameter(name)) {
      return fallback;
    }
    return static_cast<int>(get_parameter(name).as_int());
  }

  void on_joint_states(sensor_msgs::msg::JointState::ConstSharedPtr msg) {
    std::unordered_map<std::string, double> named;
    const std::size_t n = std::min(msg->name.size(), msg->position.size());
    for (std::size_t i = 0; i < n; ++i) {
      named[msg->name[i]] = msg->position[i];
    }
    const kmpc::Vec18 q = kmpc::q_from_name_map(named, 0.0);
    std::vector<std::string> missing;
    for (int i = 0; i < kmpc::kNDof; ++i) {
      const std::string name(kmpc::kJointNames[static_cast<std::size_t>(i)]);
      if (named.find(name) == named.end()) {
        missing.push_back(name);
      }
    }
    const bool has_stamp =
        msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0;
    state_->on_q(q, missing, has_stamp, msg->header.stamp.sec,
                 msg->header.stamp.nanosec);
    {
      // Real measurement cache for 250 Hz SafetySupervisor (never use cmd).
      std::lock_guard<std::mutex> lock(stream_mu_);
      v2_meas_q_ = q;
      // Prefer local receive time for freshness; keep source stamp if present.
      v2_meas_stamp_s_ = now().seconds();
      if (has_stamp) {
        v2_meas_source_stamp_s_ =
            static_cast<double>(msg->header.stamp.sec) +
            1e-9 * static_cast<double>(msg->header.stamp.nanosec);
      }
      v2_meas_valid_ = missing.empty() && q.allFinite();
    }
    if (line_diag_enable_ && tracker_) {
      tracker_->line_diag_push_feedback(
          q, has_stamp ? (static_cast<double>(msg->header.stamp.sec) +
                          1e-9 * static_cast<double>(msg->header.stamp.nanosec))
                       : -1.0);
    }
    if (!missing.empty()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "/joint_states missing %zu MPC joints", missing.size());
    }
  }

  kmpc::Vec18 q_des_joint(const kmpc::Vec18& q0) {
    kmpc::Vec18 q_des = task_->q_des(q0);
    if (demo_wave_) {
      const double elapsed = now().seconds() - t0_;
      const double period = std::max(demo_period_, 1e-3);
      q_des[3] = demo_amp_ * std::sin(2.0 * kmpc::kPi * elapsed / period);
    }
    return q_des;
  }

  // stream_mu_ protects this model and its geometry; no Pinocchio data is
  // shared with the 50 Hz controller or the asynchronous geometry worker.
  const char* stream_filtered_fault(const kmpc::Vec18& q, double pitch_extra) {
    kmpc::LineFinalGateGeo geo;
    geo.start = stream_geo_start_;
    geo.target = stream_geo_target_;
    geo.quat0 = stream_geo_quat0_;
    geo.quat1 = stream_geo_quat1_;
    geo.frame = stream_geo_frame_;
    geo.ori_on = stream_geo_ori_;
    geo.previous_along = stream_geo_previous_;
    const char* fault = kmpc::line_final_filtered_fault(
        q, pitch_extra, final_gate_limits_, *stream_kin_, *stream_guards_,
        &geo);
    if (fault == nullptr) {
      stream_geo_previous_ = geo.previous_along;
    }
    return fault;
  }

  // Precondition: stream_mu_ held. Copy only; caller publishes unlocked.
  bool stream_take_tick(kmpc::Vec18* q_out) {
    if (!arbiter_->enabled()) {
      if (interpolator_.ready()) {
        interpolator_.reset(interpolator_.q_ref());
      }
      stream_clear_modes();
      return false;
    }
    // 球已清、没有新目标：内部姿势留着，不再往透传口发钉住包。
    if (stream_mute_) {
      return false;
    }
    if (!interpolator_.ready()) {
      return false;
    }
    if (alarm_block_ || feedback_block_) {
      return false;
    }
    if (stream_hold_) {
      *q_out = stream_q_hold_;
    } else if (stream_v2_) {
      // V2_CHAIN 250 Hz: copy complete slot, sample at send time; Supervisor
      // owns q_last_send. Never treat 50 Hz q_follow as the next 4 ms command.
      const V2ExecSlot slot =
          v2_slots_[static_cast<std::size_t>(
              v2_pub_idx_.load(std::memory_order_acquire))];
      if (!slot.active || slot.protect_stop) {
        stream_enter_hold(safety_supervisor_->last_safe().allFinite()
                              ? safety_supervisor_->last_safe()
                              : (interpolator_.ready() ? interpolator_.q_ref()
                                                       : stream_v2_q_follow_));
        stream_v2_ = false;
        *q_out = stream_q_hold_;
      } else if (!v2_meas_valid_) {
        stream_enter_hold(safety_supervisor_->last_safe().allFinite()
                              ? safety_supervisor_->last_safe()
                              : interpolator_.q_ref());
        stream_v2_ = false;
        *q_out = stream_q_hold_;
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                              "kmpc v2 safety reject reason=safety_no_real_meas");
      } else {
        const double now_s = now().seconds();
        kmpc::Vec18 q_cmd;
        kmpc::Vec18 q_plan;
        if (slot.timeline_frozen) {
          // Stage barrier / WaitAck: do not advance into the next stage.
          q_plan = slot.freeze_q.allFinite() ? slot.freeze_q : slot.q_follow_hint;
          q_cmd = q_plan + slot.q_offset;
        } else {
          // Extrapolate from last 50 Hz exec_t, then clamp to any unreleased
          // TIP/STOP barrier so 250 Hz cannot cross before ACK/settle.
          const double t_raw = std::max(
              0.0, slot.exec_t_s + (now_s - slot.exec_t_wall_s));
          const double t_s = kmpc::clamp_exec_t_to_unreleased_barrier(
              t_raw, slot.barrier_pending, slot.barrier_released,
              slot.barrier_t_s);
          kmpc::TrajectoryTimeSample samp;
          if (!kmpc::sample_trajectory_at(slot.package, t_s, &samp) ||
              !samp.ok) {
            stream_enter_hold(safety_supervisor_->last_safe());
            stream_v2_ = false;
            publish_v2_slot_inactive();
            *q_out = stream_q_hold_;
            return true;
          }
          q_plan = samp.q;
          q_cmd = samp.q + slot.q_offset;
          // At an unreleased STOP/ACK barrier time, prefer the certified
          // barrier sample (no wall-clock jump past the tip).
          if (slot.barrier_pending && !slot.barrier_released &&
              slot.barrier_q.allFinite() &&
              t_s + 1e-12 >= slot.barrier_t_s) {
            q_plan = slot.barrier_q;
            q_cmd = slot.barrier_q + slot.q_offset;
          }
        }
        if (slot.recovering) {
          q_cmd = slot.q_recovery_cmd.allFinite() ? slot.q_recovery_cmd : q_cmd;
        }
        kmpc::SafetySupervisorInput in;
        in.q_cmd = q_cmd;
        in.q_prev = q_cmd;  // ignored when supervisor has q_last_send
        in.q_meas = v2_meas_q_;
        in.has_meas = true;
        in.q_follow = slot.q_follow_hint;
        in.has_follow = slot.q_follow_hint.allFinite();
        in.q_plan = q_plan;
        in.dt = interp_dt_;
        in.cmd_timestamp_s = now_s;
        in.meas_timestamp_s = v2_meas_stamp_s_;
        in.now_s = now_s;
        in.expected_generation = slot.generation;
        in.expected_path_id = slot.path_id;
        in.expected_task_digest = slot.task_digest;
        in.claim_arrived = slot.claim_time_done;
        in.mode = slot.recovering ? kmpc::SafetyExecMode::Recovering
                                  : kmpc::SafetyExecMode::Normal;
        in.pitch_excess_ceiling = slot.pitch_excess_ceiling;
        in.package = &slot.package;
        const kmpc::SafetyDecision dec = safety_supervisor_->check(in);
        if (!dec.accept) {
          RCLCPP_ERROR_THROTTLE(
              get_logger(), *get_clock(), 1000,
              "kmpc v2 safety reject reason=%s gen=%llu path_id=%u",
              dec.reason,
              static_cast<unsigned long long>(slot.generation),
              slot.path_id);
          stream_enter_hold(dec.q_send.allFinite() ? dec.q_send
                                                   : interpolator_.q_ref());
          stream_v2_ = false;
          publish_v2_slot_inactive();
          *q_out = stream_q_hold_;
        } else {
          *q_out = dec.q_send;
          if (interpolator_.ready()) {
            interpolator_.set_q_ref(*q_out);
          }
          if (dec.arrived) {
            v2_safety_arrived_.generation.store(slot.generation,
                                                std::memory_order_relaxed);
            v2_safety_arrived_.path_id.store(slot.path_id,
                                             std::memory_order_relaxed);
            v2_safety_arrived_.flag.store(true, std::memory_order_release);
          }
        }
      }
    } else if (stream_line_) {
      const double qp_dt = std::max(cfg_.params.dt, 1e-3);
      // Arm a queued chord only after the active chord reaches its endpoint.
      // Mid-chord rebase was the source of the exact 5/6 commit_scale stutter.
      if (stream_chord_pending_ && stream_line_u_ >= 1.0 - 1e-15) {
        stream_line_from_ = stream_line_to_;
        stream_line_to_ = stream_chord_pending_to_;
        stream_line_u_ = 0.0;
        stream_raw_ref_ = stream_line_from_;
        stream_chord_pending_ = false;
        ++stream_chord_queue_arms_;
      }
      stream_line_u_ += interp_dt_ / qp_dt;
      if (stream_line_u_ > 1.0) {
        stream_line_u_ = 1.0;
      }
      const double u = stream_line_u_;
      stream_raw_ref_=(1.0-u)*stream_line_from_+u*stream_line_to_;
      double extra=0;
      const char* fault=nullptr;
      bool pitch_recovery_tick=false;
      if(stream_pitch_recovery_) {
        pitch_recovery_tick=true;
        const double limit=cfg_.params.torso_pitch_max_deg*kmpc::kPi/180;
        const double excess=std::max(0.0,stream_pitch_direction_*stream_raw_ref_.head<3>().sum()-limit);
        if(excess>stream_pitch_previous_+1e-7 || excess>stream_pitch_initial_+1e-7) fault="stream_pitch_recovery_regressed";
        stream_pitch_previous_=excess;
        kmpc::Vec18 budget=kmpc::Vec18::Zero(); budget[0]=excess;
        extra=std::max(0.0,stream_pitch_filter_.push(budget,interp_dt_)[0]);
        if(++stream_pitch_ticks_>250 && (excess>1e-6 || extra>1e-6)) fault="stream_pitch_recovery_timeout";
        if(excess<=1e-9 && extra<=1e-9) stream_pitch_recovery_=false;
      }
      const kmpc::Vec18 filtered=line_smoother_.push(stream_raw_ref_,interp_dt_);
      const uint32_t line_gen =
          stream_line_gen_ < 0 ? 0u : static_cast<uint32_t>(stream_line_gen_);
      if (layer_diag_on_) {
        push_stream_diag(kmpc::ros2io::LayerDiagLayer::StreamRawRef, stream_raw_ref_,
                         stream_goal_gen_, line_gen, "raw", 0, false);
        push_stream_diag(kmpc::ros2io::LayerDiagLayer::LineSmoother, filtered,
                         stream_goal_gen_, line_gen, "smoothed", 0, false);
      }
      if(!fault) fault=stream_filtered_fault(filtered,extra);
      // Final-send hard gate: never publish a waist/pitch reversal vs the
      // accepted direction contract, even if raw/QP were monotone.
      // Exception: active pitch recovery must be allowed to move frozen waist /
      // pitch toward the hard box (r32 hold: freeze gate canceled QP recovery).
      // Re-anchor on the clearing tick too — old extrema would false-fail.
      // V2_CHAIN: direction_envelope_accept is retired; SafetySupervisor is the
      // 250 Hz gate (stream_v2_ branch above). Keep this call behind
      // !use_v2_chain_ so a stray stream_line_ under v2 cannot revive it.
      if(!fault && !use_v2_chain_ && line_direction_strict_enable_ &&
         stream_dir_env_.active) {
        if(pitch_recovery_tick) {
          stream_dir_env_.last_q=filtered;
          stream_dir_env_.last_pitch=filtered.head<3>().sum();
          stream_dir_env_.extremum[0]=filtered[0];
          stream_dir_env_.extremum[1]=filtered[1];
          stream_dir_env_.extremum[2]=filtered[2];
          stream_dir_env_.extremum[3]=stream_dir_env_.last_pitch;
        } else {
          // Probe without mutating envelope until commit succeeds.
          kmpc::DirectionEnvelope probe_env=stream_dir_env_;
          const auto dir=kmpc::direction_envelope_accept(&probe_env,filtered);
          if(!dir.ok) {
            fault="final_send_direction_violation";
            stream_line_fault_=std::string(fault)+";"+dir.reason;
          } else {
            stream_dir_env_=probe_env;
          }
        }
      }
      if(fault) {
        RCLCPP_ERROR(get_logger(),"line stream fault gen=%llu reason=%s pitch=%.9fdeg limit=%.9fdeg start_excess=%.9frad allowed_extra=%.9frad recovery_ticks=%d",
          static_cast<unsigned long long>(stream_line_gen_),fault,filtered.head<3>().sum()*180/kmpc::kPi,
          cfg_.params.torso_pitch_max_deg,stream_pitch_initial_,extra,stream_pitch_ticks_);
        if (layer_diag_on_) {
          push_stream_diag(kmpc::ros2io::LayerDiagLayer::FinalSend, filtered,
                           stream_goal_gen_, line_gen, fault, 0, false);
        }
        const kmpc::Vec18 hold=interpolator_.q_ref();
        stream_enter_hold(hold);
        if(stream_line_fault_.empty()) stream_line_fault_=fault;
        *q_out=hold;
      } else {
        *q_out=filtered;interpolator_.set_q_ref(*q_out);
      }
    } else {
      *q_out = interpolator_.step(interp_dt_);
    }
    safety_->seed(*q_out);
    return true;
  }

  void start_stream_thread() {
    if (stream_thread_.joinable()) {
      return;
    }
    stream_thread_stop_.store(false, std::memory_order_relaxed);
    stream_thread_ = std::thread(&KmpcNode::stream_thread_main, this);
  }

  void stop_stream_thread() {
    stream_thread_stop_.store(true, std::memory_order_release);
    if (stream_thread_.joinable()) {
      stream_thread_.join();
    }
  }

  void stream_thread_main() {
#ifdef __linux__
    pthread_setname_np(pthread_self(), "kmpc_stream");
#endif
    using Clock = std::chrono::steady_clock;
    const auto period = std::chrono::nanoseconds(
        static_cast<long long>(std::llround(interp_dt_ * 1e9)));
    if (period <= Clock::duration::zero()) {
      RCLCPP_ERROR(get_logger(), "stream thread: interp_dt_ is not positive");
      return;
    }
    auto next = Clock::now() + period;
    while (rclcpp::ok() &&
           !stream_thread_stop_.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_until(next);
      if (stream_thread_stop_.load(std::memory_order_relaxed) || !rclcpp::ok()) {
        break;
      }
      kmpc::Vec18 q = kmpc::Vec18::Zero();
      bool send = false;
      uint64_t gen = 0;
      uint32_t line_gen = 0;
      uint64_t tick_snap = 0;
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        send = stream_take_tick(&q);
        gen = stream_goal_gen_;
        line_gen = stream_line_gen_ < 0 ? 0u : static_cast<uint32_t>(stream_line_gen_);
        // Snapshot tick with the accepted q under the same lock — do not reload
        // stream_tick_id_ after unlock (50 Hz may have advanced it).
        tick_snap = stream_segment_tick_;
      }
      if (send) {
        const uint64_t seq = publish_stream(q);
        if (layer_diag_on_) {
          push_stream_diag(kmpc::ros2io::LayerDiagLayer::FinalSend, q, gen, line_gen,
                           "sent", seq, true, 0.0, 1.0, 0, false, tick_snap,
                           /*use_tick=*/true);
        }
        maybe_publish_debug_q(q);
      }
      next += period;
      const auto now_st = Clock::now();
      // 方案写 now > next+2ms。若 next 已过期但仍在 +2ms 内，sleep_until(past)
      // 会立刻再进循环 = 补发连发。过期一律改期，本拍只 step 一次。
      if (now_st > next) {
        next = now_st + period;
      }
    }
  }

  void maybe_publish_debug_q(const kmpc::Vec18& q_rad) {
    if (!(stream_debug_q_hz_ > 0.0) || !std::isfinite(stream_debug_q_hz_)) {
      return;
    }
    const auto now_st = std::chrono::steady_clock::now();
    const auto min_dt =
        std::chrono::duration<double>(1.0 / stream_debug_q_hz_);
    if (have_last_debug_q_ && (now_st - last_debug_q_at_) < min_dt) {
      return;
    }
    last_debug_q_at_ = now_st;
    have_last_debug_q_ = true;
    publish_debug_q(q_rad);
  }

  void reset_interpolator_to(const kmpc::Vec18& q) {
    interpolator_.reset(q);
    stream_raw_ref_=q;line_smoother_.reset(q);
    interpolator_.set_catch_s(interp_catch_s_);
    interpolator_.set_jerk_s(interp_jerk_s_);
    stream_clear_modes();
  }

  void stream_clear_modes() {
    stream_hold_ = false;
    stream_fixed_ = false;
    stream_have_qp_ = false;
    stream_line_ = false;
    stream_v2_ = false;
    stream_v2_claim_arrived_ = false;
    stream_chord_pending_ = false;
    stream_pitch_recovery_ = false;
    kmpc::direction_envelope_reset(&stream_dir_env_);
    interpolator_.clear_q_star();
  }

  void stream_enter_hold(const kmpc::Vec18& q) {
    if (!q.allFinite()) {
      return;
    }
    stream_hold_ = true;
    stream_fixed_ = true;
    stream_line_ = false;
    stream_chord_pending_ = false;
    stream_q_hold_ = q;
    interpolator_.reset(q);
    stream_raw_ref_=q;line_smoother_.reset(q);
    interpolator_.set_stop_at_goal(true);
    interpolator_.set_goal(q);
    interpolator_.clear_segment();
    safety_->seed(q);
    // Keep envelope extrema across pause/hold so resume cannot reset fold budget.
    // Clear only when leaving the line contract entirely via stream_clear_modes.
  }

  // 直线段：250Hz 只在两个路径点之间线性插。不走各轴自己的加减速。
  void stream_on_line(const kmpc::Vec18& q_path, const kmpc::Vec18& planning_from) {
    if (!interpolator_.ready() || !q_path.allFinite()) {
      return;
    }
    const bool continuing=stream_line_;
    // Queue the next chord until the active one finishes. Binding from a
    // mid-chord sample made |new-mid|≈1.2·cap → exact scale 5/6 stutter.
    const bool queue_until_end =
        continuing && stream_line_u_ < 1.0 - 1e-9;
    stream_hold_ = false;
    stream_fixed_ = false;
    // Commit geometry is always anchored at the last accepted endpoint.
    const kmpc::Vec18 commit_from =
        continuing ? stream_line_to_ : interpolator_.q_ref();
    if(!continuing) {
      stream_line_from_ = commit_from;
      stream_raw_ref_=stream_line_from_;
      line_smoother_.reset(stream_line_from_);
      if(line_direction_strict_enable_) {
        kmpc::direction_envelope_seed(
            &stream_dir_env_, tracker_->line_direction_contract(),
            stream_line_from_);
      } else {
        kmpc::direction_envelope_reset(&stream_dir_env_);
      }
    }
    stream_line_gen_ = tracker_->line_gen();
    stream_goal_gen_ = tracker_->goal_generation();
    // Bind tick to the accepted propose (not the latest control tick). 250 Hz
    // still sending the previous segment must not inherit a new QP tick id.
    stream_segment_tick_ = tracker_->line_commit_tick_id();
    stream_tick_id_.store(stream_segment_tick_, std::memory_order_release);
    const uint32_t line_gen =
        stream_line_gen_ < 0 ? 0u : static_cast<uint32_t>(stream_line_gen_);
    const kmpc::Vec18 cap=tracker_->effective_dq_max()*cfg_.params.dt*line_dq_scale_;
    auto push_commit=[&](const char* note, const kmpc::Vec18& q, double scale) {
      push_stream_diag(kmpc::ros2io::LayerDiagLayer::StreamCommit, q,
                       stream_goal_gen_, line_gen, note, 0, false,
                       tracker_->line_progress(), scale,
                       kmpc::ros2io::kLayerDiagHasScale,
                       /*from_control_thread=*/true);
    };
    // A genuinely invalid solve still fails. A valid solve can be slightly
    // longer than one tick relative to the reference advanced during solving.
    if (((q_path-planning_from).cwiseAbs().array()>cap.array()+1e-6).any()) {
      for(int j=0;j<kmpc::kNDof;++j) if(std::abs(q_path[j]-commit_from[j])>cap[j]+1e-6) {
        RCLCPP_ERROR(get_logger(),"line step limit gen=%llu joint=%d delta=%.10g cap=%.10g planned_delta=%.10g reference_advance=%.10g",
          static_cast<unsigned long long>(tracker_->goal_generation()),j,q_path[j]-commit_from[j],cap[j],q_path[j]-planning_from[j],commit_from[j]-planning_from[j]);
      }
      push_commit("reject_first_step", q_path, 0.0);
      tracker_->abort_line_stream_commit();
      tracker_->refuse_line("stream_first_step_limit");
      stream_enter_hold(interpolator_.q_ref());
      return;
    }
    const auto bounded=kmpc::ros2io::bound_line_stream_step(commit_from,q_path,cap);
    if(!bounded) {
      push_commit("reject_nonfinite", q_path, 0.0);
      tracker_->abort_line_stream_commit();
      tracker_->refuse_line("stream_nonfinite_or_invalid_cap");
      stream_enter_hold(interpolator_.q_ref()); return;
    }
    line_commit_scale_=bounded->scale;
    line_commit_peak_ratio_=0.0;
    for(int j=0;j<kmpc::kNDof;++j) if(cap[j]>0.0)
      line_commit_peak_ratio_=std::max(line_commit_peak_ratio_,std::abs(bounded->target[j]-commit_from[j])/cap[j]);
    if(line_commit_scale_<1.0-1e-9) ++line_limited_steps_;
    // Validate the actual chord after limiting; velocity, line tube and
    // forward progress remain acceptance conditions.
    const auto check=tracker_->line_stream_segment_check(commit_from,bounded->target);
    if(!check.ok) {
      const auto vec_csv=[](const kmpc::Vec18& q) {
        std::ostringstream out;
        out.precision(12);
        for(int j=0;j<kmpc::kNDof;++j) {
          if(j) out << ',';
          out << q[j];
        }
        return out.str();
      };
      const kmpc::Vec18 sent=interpolator_.q_ref();
      RCLCPP_ERROR(get_logger(),
        "line stream reject gen=%llu check=%s sample=%d joint_margin_rad=%.9g com_margin_m=%.9g pitch_margin_rad=%.9g tube_margin_m=%.9g progress_margin_m=%.9g raw_to_sent_max_rad=%.9g from=[%s] to=[%s] sent=[%s]",
        static_cast<unsigned long long>(tracker_->goal_generation()),check.reason,
        check.sample_index,check.joint_margin_rad,check.com_margin_m,
        check.pitch_margin_rad,check.tube_margin_m,check.progress_margin_m,
        (commit_from-sent).cwiseAbs().maxCoeff(),
        vec_csv(commit_from).c_str(),vec_csv(bounded->target).c_str(),
        vec_csv(sent).c_str());
      char reject_note[48];
      std::snprintf(reject_note, sizeof(reject_note), "reject_%s", check.reason);
      push_commit(reject_note, bounded->target, line_commit_scale_);
      // Preserve the concrete gate (pitch/tube/progress/…) — never collapse to
      // the historical stream_tube_or_progress alias that hid pitch hard-box stops.
      tracker_->abort_line_stream_commit();
      tracker_->refuse_line(std::string("stream_") + check.reason);
      stream_enter_hold(interpolator_.q_ref());
      return;
    }
    interpolator_.clear_q_star();
    interpolator_.clear_segment();
    interpolator_.set_stop_at_goal(false);
    if (!tracker_->confirm_line_stream_commit(bounded->target,
                                              line_commit_scale_)) {
      // Progress mapping unknown — do not guess. Hold without arming stream.
      const std::string why =
          std::string("stream_progress_") + tracker_->last_progress_recover().reason;
      push_commit(why.c_str(), bounded->target, line_commit_scale_);
      tracker_->refuse_line(why);
      stream_enter_hold(interpolator_.q_ref());
      return;
    }
    tracker_->set_stream_commit_feedback_scale(line_commit_scale_);
    if (queue_until_end) {
      stream_chord_pending_ = true;
      stream_chord_pending_to_ = bounded->target;
      ++stream_chord_queue_waits_;
      push_commit(line_commit_scale_ < 1.0 - 1e-9 ? "scale_queued" : "full_queued",
                  bounded->target, line_commit_scale_);
      return;
    }
    if (continuing) {
      stream_line_from_ = stream_line_to_;
    }
    stream_line_ = true;
    stream_line_to_ = bounded->target;
    stream_line_u_ = 0.0;
    stream_chord_pending_ = false;
    push_commit(line_commit_scale_ < 1.0 - 1e-9 ? "scale" : "full",
                bounded->target, line_commit_scale_);
    // q_ref is the filtered command; raw advancement is maintained separately.
  }

  void stream_on_qp(
      const kmpc::Vec18& q_qp, const kmpc::Vec18& q_star, bool use_cart) {
    if (!interpolator_.ready()) {
      return;
    }
    stream_line_ = false;
    const kmpc::Vec18 v_bar = tracker_->effective_dq_max() * dq_scale_;
    const kmpc::Vec18 a_bar = v_bar / interp_accel_s_;
    interpolator_.set_jerk_s(interp_jerk_s_);
    interpolator_.set_limits(v_bar, a_bar);

    const kmpc::Vec18 q_ref = interpolator_.q_ref();
    const double dt = std::max(cfg_.params.dt, 1e-3);

    if (stream_hold_) {
      bool keep = false;
      if (use_cart) {
        const double e0 = tracker_->last_e0();
        keep = tracker_->last_arrived() ||
               (std::isfinite(e0) && e0 <= 0.012);
      } else {
        keep = (q_star - stream_q_hold_).cwiseAbs().maxCoeff() <= 0.03;
      }
      if (keep) {
        interpolator_.set_q_ref(stream_q_hold_);
        interpolator_.set_goal(stream_q_hold_);
        interpolator_.set_stop_at_goal(true);
        interpolator_.clear_segment();
        safety_->seed(stream_q_hold_);
        return;
      }
      stream_hold_ = false;
      stream_fixed_ = false;
      stream_have_qp_ = false;
    }

    if (use_cart) {
      interpolator_.clear_q_star();
      if (tracker_->last_arrived()) {
        stream_enter_hold(q_ref);
        return;
      }
      stream_fixed_ = false;
      interpolator_.set_stop_at_goal(false);
      interpolator_.set_catch_s(std::max(interp_catch_s_, dt));
      interpolator_.begin_qp_segment(q_ref, q_qp, dt);
      stream_qp_to_ = q_qp;
      stream_have_qp_ = true;
      safety_->seed(q_ref);
      return;
    }

    interpolator_.set_q_star(q_star);
    interpolator_.set_axis_e_near(0.05);
    interpolator_.set_axis_release_jump(0.03);
    interpolator_.set_axis_release_hyst(0.01);

    if (stream_fixed_ &&
        (q_star - stream_star_).cwiseAbs().maxCoeff() > 0.03) {
      stream_fixed_ = false;
    }

    const bool enter_brake = kmpc::stream_brake_enter(
        q_ref, interpolator_.v_ref(), interpolator_.a_ref(), q_star, a_bar,
        interp_jerk_s_);
    if (enter_brake || stream_fixed_) {
      stream_fixed_ = true;
      stream_star_ = q_star;
      interpolator_.set_stop_at_goal(true);
      interpolator_.set_goal(q_star);
      interpolator_.set_catch_s(interp_catch_s_);
      interpolator_.clear_segment();
      const double e_inf = (q_star - q_ref).cwiseAbs().maxCoeff();
      const double v_inf = interpolator_.v_ref().cwiseAbs().maxCoeff();
      if (e_inf <= 0.02 && v_inf <= 0.02) {
        stream_enter_hold(q_ref);
        return;
      }
      safety_->seed(q_ref);
      return;
    }

    interpolator_.set_stop_at_goal(false);
    interpolator_.set_catch_s(std::max(interp_catch_s_, dt));
    interpolator_.begin_qp_segment(interpolator_.q_goal(), q_qp, dt);
    stream_qp_to_ = q_qp;
    stream_have_qp_ = true;
    safety_->seed(q_ref);
  }

  static bool alarm_text_active(const std::string& text) {
    for (char c : text) {
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
        return true;
      }
    }
    return false;
  }

  void on_alarm(const std::string& text) {
    std::lock_guard<std::mutex> lock(stream_mu_);
    if (alarm_text_active(text)) {
      if (!alarm_block_) {
        RCLCPP_WARN(get_logger(), "alarm, stop stream: %s", text.c_str());
        // Only targets at-or-before this watermark are stale on resume.
        // A segment installed while blocked (enable-time approach) is kept.
        alarm_block_generation_ = task_->generation();
      }
      alarm_block_ = true;
      alarm_clear_since_valid_ = false;
      return;
    }
    if (alarm_block_ && !alarm_clear_since_valid_) {
      alarm_clear_since_ = std::chrono::steady_clock::now();
      alarm_clear_since_valid_ = true;
    }
  }

  // 调用方已持有 stream_mu_。空告警要连续 2 秒，单次清空不算能动。
  bool alarm_clear_ready() const {
    if (!alarm_block_ || !alarm_clear_since_valid_) {
      return false;
    }
    const double held = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - alarm_clear_since_).count();
    return held >= 2.0;
  }

  kmpc::Vec18 qp_q0(const kmpc::Vec18& q_meas) {
    // 停发期间每拍都把内部姿势改成实测的腰和两条臂。
    // 再开流的第一包从这里出发，不看上次指令，也不用 cmd_reset_rad。
    if (stream_mute_ && !line_future_.valid() && q_meas.allFinite()) {
      reset_interpolator_to(q_meas);
      safety_->seed(q_meas);
      return q_meas;
    }
    auto reset_to_meas = [&]() {
      reset_interpolator_to(q_meas);
      safety_->seed(q_meas);
      return q_meas;
    };
    if (!safety_->has_last() || !interpolator_.ready()) {
      return reset_to_meas();
    }
    // 不把实测一拍写成指令。关节还在跟：求解用发出去的角度。
    // 差过 1°（平时超前大约 0.25°）：这一轴指令停住，求解改用实测。
    kmpc::Vec18 q0 = interpolator_.q_ref();
    constexpr double kFollowEnter = 1.0 * kmpc::kPi / 180.0;
    constexpr double kFollowLeave = 0.5 * kmpc::kPi / 180.0;
    for (int i = 0; i < kmpc::kNDof; ++i) {
      const double gap = std::abs(q_meas[i] - interpolator_.q_ref()[i]);
      if (interpolator_.axis_held(i)) {
        if (gap < kFollowLeave) {
          interpolator_.release_axis(i);
        } else {
          interpolator_.hold_axis(i);
          q0[i] = q_meas[i];
        }
      } else if (gap > kFollowEnter) {
        interpolator_.hold_axis(i);
        q0[i] = q_meas[i];
      }
    }
    return q0;
  }

  void on_timer() {
    const auto t0 = std::chrono::steady_clock::now();
    if (!arbiter_->enabled()) {
      joint_tracker_->reset();
      tracker_->reset();
      return;
    }
    const auto q_meas = state_->read();
    if (!q_meas.has_value()) {
      std::lock_guard<std::mutex> lock(stream_mu_);
      if (!feedback_block_) {
        RCLCPP_WARN(
            get_logger(),
            "joint feedback stale, stop stream");
      }
      feedback_block_ = true;
      return;
    }
    kmpc::Vec18 q0;
    {
      std::lock_guard<std::mutex> lock(stream_mu_);
      if (alarm_block_) {
        if (!alarm_clear_ready() || !q_meas->allFinite()) {
          return;
        }
        reset_interpolator_to(*q_meas);
        safety_->seed(*q_meas);
        tracker_->reset();
        // Keep a line segment that arrived while blocked (same enable tick).
        // Still drop pre-alarm goals so an old reach is not chased again.
        task_->drop_cartesian_up_to(alarm_block_generation_);
        joint_tracker_->reset();
        alarm_block_ = false;
        feedback_block_ = false;
        alarm_clear_since_valid_ = false;
        RCLCPP_INFO(get_logger(), "alarm stayed clear, resume from measured joints");
      } else if (feedback_block_) {
        reset_interpolator_to(*q_meas);
        safety_->seed(*q_meas);
        tracker_->reset();
        task_->clear_cartesian();
        joint_tracker_->reset();
        feedback_block_ = false;
        RCLCPP_INFO(get_logger(), "joint feedback back, resume from measured joints");
      }
      q0 = qp_q0(*q_meas);
    }

    double line_scale = line_dq_scale_;
    if (tracker_->line_active() && q_meas->allFinite()) {
      kmpc::Vec18 q_ref = kmpc::Vec18::Zero();
      bool ready = false;
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        ready = interpolator_.ready();
        if (ready) {
          q_ref = interpolator_.q_ref();
        }
      }
      if (ready) {
        const double gap = (q_ref - *q_meas).cwiseAbs().maxCoeff();
        // 0.6 时正常跟随大约 1.2°。门槛停在 1° 会把整段速度打成 0，
        // 再爬 0.3 秒才回到巡航，手臂就一顿一顿。真卡住才会过 3°。
        constexpr double kLinePauseGap = 3.0 * kmpc::kPi / 180.0;
        if (gap > kLinePauseGap) {
          line_scale = 0.0;
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "line paused, |cmd-meas| %.2f deg", kmpc::rad_to_deg(gap));
        }
      }
    }
    tracker_->set_line_pace(line_scale, interp_accel_s_);
    tracker_->set_line_jerk_s(interp_jerk_s_);

    const auto targets=task_->targets();
    const auto& left=targets.first;
    const auto& right=targets.second;
    const kmpc::Vec3* xyz_l = left ? &left->xyz : nullptr;
    const kmpc::Vec3* xyz_r = right ? &right->xyz : nullptr;
    const kmpc::Vec4* quat_l = left ? &left->quat_xyzw : nullptr;
    const kmpc::Vec4* quat_r = right ? &right->quat_xyzw : nullptr;
    const bool invalid_frame=(left && left->frame_id!=kmpc::kBaseFrame) ||
        (right && right->frame_id!=kmpc::kBaseFrame);

    planner_rx_->apply(*planner_);
    const kmpc::PlannerStepResult result = planner_->step(q0);
    if (result.state_changed) {
      planner_rx_->note_auto_tracking();
    }
    tracker_->set_upper_commands(result.command);
    const bool calibrating =
        result.command.tracking_state == kmpc::TrackingState::Calibrating;
    kmpc::Vec18 q_pref_cart = cfg_.q_pref;
    if (calibrating && result.command.has_q_calib) {
      q_pref_cart = result.command.q_calib;
    }
    const kmpc::Vec3* use_l = calibrating ? nullptr : xyz_l;
    const kmpc::Vec3* use_r = calibrating ? nullptr : xyz_r;
    const kmpc::Vec4* use_ql = calibrating ? nullptr : quat_l;
    const kmpc::Vec4* use_qr = calibrating ? nullptr : quat_r;

    const bool use_cart = (use_l != nullptr || use_r != nullptr || calibrating);
    if (last_use_cart_ && !use_cart) {
      tracker_->reset();
      std::lock_guard<std::mutex> lock(stream_mu_);
      stream_clear_modes();
    }
    // 球清掉、又没有关节目标：钉住当前指令，不追 HOME。不往外发。
    const bool nail_pose =
        !use_cart && !demo_wave_ && !task_->has_q_target();
    bool exit_stream = false;
    if (nail_pose || (task_->in_hold_deadband(q0, hold_deadband_) && !use_cart)) {
      last_use_cart_ = false;
      kmpc::Vec18 q_cmd = kmpc::Vec18::Zero();
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        q_cmd = safety_->failsafe();
        const kmpc::Vec18 q_hold =
            interpolator_.ready() ? interpolator_.q_ref() : q_cmd;
        stream_enter_hold(q_hold);
        q_cmd = stream_q_hold_;
        // 只在球和关节目标都没了时停发。关节目标还在、只是进了死区，继续钉住。
        if (nail_pose && !stream_mute_) {
          exit_stream = true;
          RCLCPP_INFO(
              get_logger(),
              "kmpc nail: stop hold stream, enable stays on");
        }
        if (nail_pose) {
          stream_mute_ = true;
        }
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      // Request release before reporting that the segment has been cleared.
      if (exit_stream) release_servo_stream();
      publish_cmd(q_cmd, kmpc::Vec18::Zero(), ms, true);
      return;
    }

    if (stream_mute_ && !(bool(left)!=bool(right))) {
      // 上一拍停过发。求解前清掉旧够点状态，q0 已是本拍实测。
      tracker_->reset();
      tracker_->set_upper_commands(result.command);
      joint_tracker_->reset();
    }

    kmpc::Vec18 line_solve_from=q0;
    const bool single_line=!calibrating && bool(left)!=bool(right);

    // V2_CHAIN production: MotionTask → PlannerWorker → TrajectoryPackage →
    // TrajectoryFollower (50 Hz) → SafetySupervisor (250 Hz). No std::async
    // prepare_line, no line_path_qp posture reshape, no direction_envelope gate.
    if (single_line && use_v2_chain_ && planner_worker_) {
      const auto& target = left ? *left : *right;
      const bool is_left = bool(left);
      if (invalid_frame) {
        follower_->hold_at(q0);
        {
          std::lock_guard<std::mutex> lock(stream_mu_);
          stream_enter_hold(q0);
          stream_v2_ = false;
        }
        publish_cmd(q0, kmpc::Vec18::Zero(), 0.0, true);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        if (!interpolator_.ready() || stream_mute_) {
          reset_interpolator_to(*q_meas);
        }
        q0 = interpolator_.q_ref();
      }
      auto mt_opt = task_->motion_task();
      kmpc::MotionTask mt;
      if (mt_opt && mt_opt->generation == target.generation) {
        mt = *mt_opt;
      } else {
        mt.generation = target.generation;
        mt.left = is_left;
        kmpc::TaskStage st;
        st.geometry = kmpc::TaskGeometry::Lin;
        st.position = target.xyz;
        st.quat = target.quat_xyzw;
        st.has_orientation =
            !kmpc::quat_is_identity_xyzw(&target.quat_xyzw);
        st.stage_id = 0;
        mt.stages.push_back(st);
        if (target.cascade) {
          kmpc::TaskStage tip;
          tip.geometry = kmpc::TaskGeometry::Lin;
          tip.position = target.cascade->xyz;
          tip.quat = target.cascade->quat_xyzw;
          tip.has_orientation = target.cascade->ori_on;
          tip.stage_id = 1;
          mt.stages.push_back(tip);
        }
      }
      mt.waist_locked = result.command.waist_locked;
      mt.frame = is_left ? std::string(kmpc::kEeFrameLeft)
                         : std::string(kmpc::kEeFrameRight);
      mt.q = q0;
      // Seal immutable identity from real URDF bytes + limits before digest.
      const auto v2_policy = kmpc::policy_from_config(cfg_);
      const uint64_t digest = kmpc::seal_motion_task_identity(
          &mt, line_urdf_source_, line_urdf_xml_, v2_policy);
      const kmpc::V2TaskId req_id{mt.generation, digest};
      const bool explicit_req = v2_request_plan_;
      v2_request_plan_ = false;
      const auto gate_dec = v2_gate_.on_tick(req_id, follower_->has_package(),
                                             explicit_req);
      if (gate_dec.submit) {
        kmpc::RobotState st;
        st.q = q0;
        st.dq_known = false;
        st.fresh = true;
        st.timestamp_s = now().seconds();
        if (gate_dec.invalidate_old && follower_) {
          follower_->request_brake();
          planner_worker_->clear_pending_and_result();
        }
        planner_worker_->submit_snapshot(mt, st, v2_policy,
                                         planner_worker_->default_weights(),
                                         10.0);
        v2_gate_.note_submitted(req_id, planner_worker_->submitted_seq());
        ++line_plan_builds_;
        {
          std::lock_guard<std::mutex> lock(stream_mu_);
          stream_enter_hold(q0);
          stream_v2_ = false;
          publish_v2_slot_inactive();
        }
        RCLCPP_INFO(
            get_logger(),
            "kmpc v2 submit gen=%llu digest=%llu left=%d stages=%zu reason=%s "
            "submits=%llu",
            static_cast<unsigned long long>(mt.generation),
            static_cast<unsigned long long>(digest), is_left ? 1 : 0,
            mt.stages.size(), gate_dec.reason,
            static_cast<unsigned long long>(v2_gate_.counters().submits));
        publish_cmd(q0, kmpc::Vec18::Zero(), 0.0, true);
        return;
      }
      kmpc::FastPlanResult plan;
      if (gate_dec.poll_worker && planner_worker_->try_take(plan)) {
        const uint64_t plan_digest = plan.task_digest;
        const bool package_fields_ok =
            plan.package.task_digest == plan_digest &&
            plan.package.generation == mt.generation &&
            plan.package.model_digest != 0 && plan.limits_digest != 0 &&
            plan.package.samples.size() >= 2;
        const auto take = v2_gate_.on_worker_result(
            req_id, plan.generation, plan_digest, plan.submit_seq, plan.ok,
            package_fields_ok);
        if (!take.adopt) {
          RCLCPP_ERROR(
              get_logger(),
              "kmpc v2 plan reject gen=%llu reason=%s reject=%s seq=%llu "
              "rejects=%llu",
              static_cast<unsigned long long>(mt.generation),
              plan.ok ? "stale_or_mismatch" : plan.reason.c_str(),
              take.reject_reason,
              static_cast<unsigned long long>(plan.submit_seq),
              static_cast<unsigned long long>(v2_gate_.counters().rejects));
          v2_gate_.note_plan_failed(take.reject_reason);
          follower_->hold_at(q0);
          {
            std::lock_guard<std::mutex> lock(stream_mu_);
            stream_enter_hold(q0);
            stream_v2_ = false;
            publish_v2_slot_inactive();
          }
          // No per-tick auto replan; wait for new identity or explicit replan.
          publish_cmd(q0, kmpc::Vec18::Zero(), 0.0, true);
          return;
        }
        std::string why;
        const kmpc::Vec18 q_meas_adopt =
            (q_meas && q_meas->allFinite()) ? *q_meas : q0;
        if (!follower_->adopt(plan.package, q_meas_adopt, kmpc::Vec18::Zero(),
                              false, 1e-3, 0.5, &why)) {
          RCLCPP_WARN(get_logger(),
                      "kmpc v2 adopt rejected gen=%llu why=%s rejects=%llu",
                      static_cast<unsigned long long>(plan.package.generation),
                      why.c_str(),
                      static_cast<unsigned long long>(
                          v2_gate_.counters().rejects + 1));
          ++line_plan_discards_;
          v2_gate_.note_adopt_failed(why.c_str());
          {
            std::lock_guard<std::mutex> lock(stream_mu_);
            stream_enter_hold(q0);
            stream_v2_ = false;
            publish_v2_slot_inactive();
          }
          publish_cmd(q0, kmpc::Vec18::Zero(), 0.0, true);
          return;
        }
        v2_gate_.note_adopted(req_id, plan.submit_seq, plan.package.path_id);
        v2_follow_t0_ = now().seconds();
        v2_safety_arrived_.flag.store(false, std::memory_order_release);
        v2_safety_arrived_.generation.store(0, std::memory_order_relaxed);
        v2_safety_arrived_.path_id.store(0, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> lock(stream_mu_);
          reset_interpolator_to(q_meas_adopt);
          stream_v2_ = true;
          stream_v2_package_ = plan.package;
          stream_v2_q_follow_ = q_meas_adopt;
          stream_v2_q_plan_ = q_meas_adopt;
          stream_v2_q_ref_ = q_meas_adopt;
          stream_v2_claim_arrived_ = false;
          stream_goal_gen_ = plan.package.generation;
          stream_line_gen_ = static_cast<int>(plan.package.path_id);
          safety_supervisor_->seed(q_meas_adopt);
          V2ExecSlot slot;
          slot.active = true;
          slot.package = plan.package;
          slot.t0_s = v2_follow_t0_;
          slot.exec_t_s = 0.0;
          slot.exec_t_wall_s = v2_follow_t0_;
          slot.timeline_frozen = false;
          slot.freeze_q = q_meas_adopt;
          // Seed next unreleased barrier so 250 Hz clamps before first 50 Hz
          // tick observes the tip (TIP may sit between 50 Hz cycles).
          for (const auto& b : plan.package.stage_boundaries) {
            if (b.sample_index == 0) {
              continue;
            }
            if (!(b.stop_required || b.wait_for_external_ack)) {
              continue;
            }
            if (b.sample_index >= plan.package.samples.size()) {
              continue;
            }
            slot.barrier_pending = true;
            slot.barrier_released = false;
            slot.barrier_t_s =
                kmpc::time_at_sample_index(plan.package, b.sample_index);
            slot.barrier_stage_id = b.stage_id;
            slot.barrier_stop_required = b.stop_required;
            slot.barrier_wait_ack = b.wait_for_external_ack;
            slot.barrier_q = plan.package.samples[b.sample_index];
            break;
          }
          slot.generation = plan.package.generation;
          slot.path_id = plan.package.path_id;
          slot.task_digest = v2_gate_.active().digest;
          slot.q_follow_hint = q_meas_adopt;
          publish_v2_slot(slot);
        }
        RCLCPP_INFO(
            get_logger(),
            "kmpc v2 adopt planner=%s path_id=%u gen=%llu digest=%llu "
            "samples=%zu dur=%.3f seq=%llu quality=%s adopts=%llu",
            plan.package.planner.c_str(), plan.package.path_id,
            static_cast<unsigned long long>(plan.package.generation),
            static_cast<unsigned long long>(v2_gate_.active().digest),
            plan.package.samples.size(), plan.package.time_law.duration_s,
            static_cast<unsigned long long>(plan.submit_seq),
            planner_worker_->quality_load_note().c_str(),
            static_cast<unsigned long long>(v2_gate_.counters().adopts));
      }
      if (v2_gate_.phase() == kmpc::V2PlanPhase::Pending) {
        publish_cmd(q0, kmpc::Vec18::Zero(), 0.0, true);
        return;
      }
      if (!follower_->has_package()) {
        publish_cmd(q0, kmpc::Vec18::Zero(), 0.0, true);
        return;
      }
      const double dt = std::max(cfg_.params.dt, 1e-3);
      const kmpc::Vec18 q_meas_step =
          (q_meas && q_meas->allFinite()) ? *q_meas : q0;
      const double meas_stamp =
          v2_meas_valid_ ? (v2_meas_source_stamp_s_ > 0.0
                                ? v2_meas_source_stamp_s_
                                : v2_meas_stamp_s_)
                         : std::numeric_limits<double>::quiet_NaN();
      const kmpc::FollowerTick tick =
          follower_->step(q_meas_step, dt, meas_stamp, now().seconds());
      kmpc::Vec18 q_cmd = tick.q_follow;
      kmpc::Vec18 dq = tick.q_follow - q0;
      if (tick.action == kmpc::FollowerAction::ProtectStop) {
        RCLCPP_ERROR(get_logger(), "kmpc v2 follower protect_stop reason=%s",
                     tick.reason);
        follower_->hold_at(q_cmd);
        {
          std::lock_guard<std::mutex> lock(stream_mu_);
          stream_enter_hold(q_cmd);
          stream_v2_ = false;
          publish_v2_slot_inactive();
        }
        publish_arrived(false);
        publish_cmd(q_cmd, kmpc::Vec18::Zero(), 0.0, true);
        return;
      }
      if (tick.action == kmpc::FollowerAction::Brake) {
        // Controlled brake: keep 250 Hz sampling active; do not instant hold_at.
        RCLCPP_WARN(get_logger(), "kmpc v2 follower braking reason=%s",
                    tick.reason);
        follower_->request_brake();
      }
      if (tick.action == kmpc::FollowerAction::NeedReplan) {
        v2_request_plan_ = true;
        v2_gate_.request_replan();
        {
          std::lock_guard<std::mutex> lock(stream_mu_);
          // Decelerate / hold while waiting for a new certified package.
          stream_enter_hold(q_cmd);
          publish_v2_slot_inactive();
          stream_v2_ = false;
        }
        publish_cmd(q_cmd, kmpc::Vec18::Zero(), 0.0, true);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        stream_hold_ = false;
        stream_fixed_ = false;
        stream_line_ = false;
        stream_v2_ = true;
        stream_v2_q_follow_ = tick.q_follow;
        stream_v2_q_plan_ = tick.q_plan;
        stream_v2_q_ref_ = tick.q_ref;
        stream_v2_claim_arrived_ =
            (tick.action == kmpc::FollowerAction::Arrived);
        stream_goal_gen_ = tick.generation;
        stream_line_gen_ = static_cast<int>(tick.path_id);
        stream_mute_ = false;
        // Do NOT set interpolator from tick.q_follow — 250 Hz owns send truth.
        V2ExecSlot slot;
        slot.active = true;
        slot.package = follower_->package();
        slot.t0_s = v2_follow_t0_;
        slot.exec_t_s = tick.exec_t_s;
        slot.exec_t_wall_s = now().seconds();
        slot.timeline_frozen = tick.timeline_frozen;
        slot.barrier_pending = tick.barrier_pending;
        slot.barrier_released = tick.barrier_released;
        slot.barrier_t_s = tick.barrier_t_s;
        slot.barrier_stage_id = tick.barrier_stage_id;
        slot.barrier_stop_required = tick.barrier_stop_required;
        slot.barrier_wait_ack = tick.barrier_wait_ack;
        slot.barrier_q = tick.barrier_q;
        slot.freeze_q =
            tick.freeze_q.allFinite() ? tick.freeze_q : tick.q_follow;
        slot.generation = tick.generation;
        slot.path_id = tick.path_id;
        slot.task_digest = v2_gate_.active().digest;
        slot.q_offset = tick.q_offset;
        slot.q_follow_hint = tick.q_follow;
        slot.claim_time_done =
            (tick.action == kmpc::FollowerAction::Arrived);
        slot.recovering = (tick.state == kmpc::FollowerState::Recovering);
        slot.pitch_excess_ceiling = tick.pitch_excess_ceiling;
        slot.q_recovery_cmd = tick.q_follow;
        publish_v2_slot(slot);
      }
      // Arrived only after 250 Hz safety accepted the claim for this gen/path.
      if (v2_safety_arrived_.flag.load(std::memory_order_acquire)) {
        const uint64_t arrived_gen =
            v2_safety_arrived_.generation.load(std::memory_order_relaxed);
        const uint32_t arrived_path =
            v2_safety_arrived_.path_id.load(std::memory_order_relaxed);
        v2_safety_arrived_.flag.store(false, std::memory_order_release);
        if (arrived_gen == v2_gate_.active().generation &&
            arrived_path == v2_gate_.active_path_id()) {
          v2_gate_.note_arrived();
          publish_arrived(true);
          dq.setZero();
          {
            std::lock_guard<std::mutex> lock(stream_mu_);
            publish_v2_slot_inactive();
            stream_v2_ = false;
          }
        }
      } else if (tick.action == kmpc::FollowerAction::Arrived) {
        publish_arrived(false);
      }
      last_use_cart_ = true;
      last_ee_err_.setZero();
      publish_cmd(q_cmd, dq, 0.0, false);
      return;
    }

    if(single_line) {
      const auto& target=left?*left:*right;
      const bool is_left=bool(left);
      if(target.generation!=tracker_->goal_generation()) {
        tracker_->rearm_line(target.generation,target.cascade);
        line_limited_steps_=0;line_commit_scale_=1.0;line_plan_builds_=0;line_plan_discards_=0;
        {std::lock_guard<std::mutex> lock(stream_mu_);stream_line_fault_.clear();}
      }
      if(invalid_frame) tracker_->refuse_line("invalid_target_frame");
      const bool needs_build=tracker_->line_needs_build(is_left,target.xyz,&target.quat_xyzw);
      // Pause/snapshot/commit: the 250 Hz thread can continue publishing a fixed
      // reference while the worker builds a whole path on its own kinematics.
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        if(!interpolator_.ready() || (stream_mute_ && !line_future_.valid() && !tracker_->line_active()))
          reset_interpolator_to(*q_meas);
        q0=interpolator_.q_ref();
        if(!stream_line_fault_.empty()) tracker_->refuse_line(stream_line_fault_);
        // The next QP segment follows the last committed endpoint (including a
        // queued-but-not-yet-armed chord). Mid-segment samples alternate speed.
        if (stream_line_ && !needs_build && !line_future_.valid()) {
          line_solve_from = stream_chord_pending_ ? stream_chord_pending_to_
                                                  : stream_line_to_;
        } else {
          line_solve_from = q0;
        }
        if(needs_build || line_future_.valid()) stream_enter_hold(q0);
      }
      tracker_->set_line_reference(line_solve_from);
      if(line_future_.valid()) {
        if(line_future_.wait_for(std::chrono::seconds(0))!=std::future_status::ready) {
          publish_cmd(q0,kmpc::Vec18::Zero(),0.0,true); return;
        }
        kmpc::CartesianTracker::LineRun plan;
        try { plan=line_future_.get(); }
        catch(const std::exception&) { tracker_->refuse_line("line_worker_exception"); }
        kmpc::LineAsyncPendingSeed pending;
        pending.generation=line_pending_gen_;
        pending.left=line_pending_left_;
        pending.waist_locked=line_pending_waist_locked_;
        pending.xyz=line_pending_xyz_;
        pending.quat=line_pending_quat_;
        pending.q_seed=line_pending_q_;
        const bool current=kmpc::line_async_plan_matches_live(
            pending,target.generation,is_left,result.command.waist_locked,
            target.xyz,target.quat_xyzw,q0);
        bool adopt_plan = current && !tracker_->line_refused();
        if(adopt_plan) {
          // Snapping the certified start is a new curve. If the live joints
          // leave the certificate, drop this package and plan again.
          if(plan.on && !plan.samples.empty() && q_meas && q_meas->allFinite()) {
            if(!tracker_->recertify_line_start(plan, *q_meas)) {
              adopt_plan = false;
              ++line_plan_discards_;
            } else {
              plan.samples.front() = *q_meas;
              tracker_->set_line_reference(*q_meas);
              const auto start_pose=kin_->fk(*q_meas,plan.frame);
              std::lock_guard<std::mutex> lock(stream_mu_);
              reset_interpolator_to(*q_meas);
              stream_geo_start_=start_pose.col(3).head<3>();stream_geo_target_=plan.target;
              stream_geo_quat0_=kmpc::rot_to_quat_xyzw(start_pose.block<3,3>(0,0));
              stream_geo_quat1_=plan.quat;stream_geo_ori_=plan.ori_on;stream_geo_frame_=plan.frame;
              stream_geo_previous_=0.0;stream_line_fault_.clear();
              const double pitch=q_meas->head<3>().sum();
              const double limit=cfg_.params.torso_pitch_max_deg*kmpc::kPi/180;
              const double excess=std::max(0.0,std::abs(pitch)-limit);
              stream_pitch_recovery_=limit>0 && excess>1e-6;
              stream_pitch_direction_=pitch<0?-1.0:1.0;
              stream_pitch_initial_=stream_pitch_previous_=excess; stream_pitch_ticks_=0;
              kmpc::Vec18 budget=kmpc::Vec18::Zero();budget[0]=excess;
              stream_pitch_filter_.reset(budget);
            }
          }
          if(adopt_plan) {
            if(plan.candidate=="reverse_cache" || plan.candidate=="qa_seed" ||
               plan.reverse_decision.rfind("rejected_",0)==0) {
              tracker_->clear_last_line_cache();
            }
            tracker_->adopt_line(std::move(plan));
          }
        } else if(!current) {++line_plan_discards_;}
      }
      if(tracker_->line_needs_build(is_left,target.xyz,&target.quat_xyzw)) {
        // Seed once when launching, not again on each feedback update while
        // that worker is building. Encoder noise must not invalidate every plan.
        {
          std::lock_guard<std::mutex> lock(stream_mu_);
          // Always seed the plan from live joints — start is the robot now,
          // not a stale hold/q_ref (r32 fake descend; r38 QP s0 mismatch).
          if(!tracker_->line_active() && q_meas && q_meas->allFinite()) {
            reset_interpolator_to(*q_meas);
          } else if(stream_mute_ && !tracker_->line_active() &&
                    q_meas && q_meas->allFinite()) {
            reset_interpolator_to(*q_meas);
          }
          q0=interpolator_.q_ref();stream_enter_hold(q0);
        }
        line_solve_from=q0;tracker_->set_line_reference(q0);++line_plan_builds_;
        line_pending_gen_=target.generation; line_pending_left_=is_left;
        line_pending_xyz_=target.xyz; line_pending_quat_=target.quat_xyzw; line_pending_q_=q0;
        const auto config=cfg_;
        const auto upper=result.command;
        line_pending_waist_locked_=upper.waist_locked;
        const bool pose_pref=line_pose_pref_enable_;
        const bool pitch_straighten=line_pitch_straighten_enable_;
        const bool qp_pitch_mono=line_qp_pitch_monotonic_enable_;
        const bool reverse_last=reverse_last_line_enable_;
        const bool direction_strict=line_direction_strict_enable_;
        const bool shadow_diag=line_shadow_diag_enable_;
        const bool diag=line_diag_enable_;
        const auto pace_mode=line_pace_mode_;
        const double brake_m=line_brake_distance_m_;
        const double brake_floor=line_brake_floor_;
        const auto brake_profile=line_brake_profile_;
        const int qp_max_iter=line_qp_max_iter_;
        const auto qp_follow_mode=line_qp_follow_mode_;
        const auto chain_profile=line_chain_profile_;
        const kmpc::Vec18 q_pref=cfg_.q_pref;
        const auto line_cache=tracker_->last_line_cache();
        const auto successor_plan=tracker_->successor_plan();
        const uint64_t run_id=run_id_;
        kmpc::Kinematics* plan_kin=line_plan_kin_.get();
        line_future_=std::async(std::launch::async,
            [config,plan_kin,target,q0,is_left,upper,pose_pref,pitch_straighten,
             qp_pitch_mono,reverse_last,direction_strict,shadow_diag,diag,pace_mode,brake_m,brake_floor,
             brake_profile,qp_max_iter,qp_follow_mode,chain_profile,q_pref,line_cache,successor_plan,
             run_id]() {
          kmpc::CartesianTracker worker(*plan_kin,config);
          worker.set_upper_commands(upper);
          worker.set_line_pose_pref(pose_pref);
          worker.set_line_pose_pref_q(q_pref);
          worker.set_line_pitch_straighten_enable(pitch_straighten);
          worker.set_line_qp_pitch_monotonic_enable(qp_pitch_mono);
          worker.set_reverse_last_line_enable(reverse_last);
          worker.set_line_direction_strict_enable(direction_strict);
          worker.set_line_shadow_diag_enable(shadow_diag);
          // Profile first: NewUnified forces pace/QP; Legacy preserves later setters.
          worker.set_line_chain_profile(chain_profile);
          worker.set_line_qp_follow_mode(qp_follow_mode);
          worker.set_line_pace_mode(pace_mode);
          if (!worker.set_line_brake_distance_m(brake_m)) {
            throw kmpc::ConfigError("line_brake_distance_m must be finite and > 0");
          }
          if (!worker.set_line_brake_floor(brake_floor)) {
            throw kmpc::ConfigError("line_brake_floor must be finite and in (0,1]");
          }
          worker.set_line_brake_profile(brake_profile);
          if (!worker.set_line_qp_max_iter(qp_max_iter)) {
            throw kmpc::ConfigError("line_qp_max_iter must be in [1,10000]");
          }
          worker.import_last_line_cache(line_cache);
          worker.import_successor_plan(successor_plan);
          worker.set_line_diag_enable(diag);
          worker.set_line_diag_run_id(run_id);
          worker.rearm_line(target.generation,target.cascade);
          auto plan = worker.prepare_line(q0,is_left,target.xyz,&target.quat_xyzw);
          plan.diag_events = worker.line_diag_take();
          // prepare_line already copies successor + smooth/interp into plan.
          plan.successor = worker.successor_plan();
          return plan;
        });
        publish_cmd(q0,kmpc::Vec18::Zero(),0.0,true); return;
      }
    }

    kmpc::Vec18 dq = kmpc::Vec18::Zero();
    kmpc::Vec18 q_cmd = kmpc::Vec18::Zero();
    try {
      bool wait_chord_queue = false;
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        wait_chord_queue = use_cart && stream_line_ && stream_chord_pending_ &&
                           stream_line_u_ < 1.0 - 1e-9;
      }
      if (wait_chord_queue) {
        // Depth-1 chord queue is armed but not yet playing. Skip propose so
        // follow_line cannot abort/replace the confirmed pending endpoint.
        std::lock_guard<std::mutex> lock(stream_mu_);
        stream_goal_gen_ = tracker_->goal_generation();
        q_cmd = stream_chord_pending_to_;
        dq.setZero();
        if (interpolator_.ready()) {
          stream_mute_ = false;
        }
        publish_cmd(q_cmd, dq, 0.0, true);
        return;
      }
      if (use_cart) {
        const uint64_t tick_id =
            control_tick_id_.fetch_add(1, std::memory_order_relaxed) + 1;
        tracker_->set_control_tick_id(tick_id);
        // Do not bump stream_tick_id here: 250 Hz may still be emitting the
        // previous accepted segment. stream_on_line binds tick on confirm.
        const kmpc::TrackerStep step = tracker_->compute(
            q0, q_pref_cart, use_l, use_r, use_ql, use_qr,
            kmpc::kEeFrameLeft, kmpc::kEeFrameRight);
        last_ee_err_ = step.ee_error;
        dq = step.dq;
        if (tracker_->last_qp_status() != kmpc::QpStatus::Success ||
            !dq.allFinite()) {
          throw std::runtime_error("cartesian QP failed");
        }
      } else {
        const kmpc::Vec18 q_raw = q_des_joint(q0);
        kmpc::QDesFlags flags;
        kmpc::Vec18 star =
            joint_tracker_->guards().project_q_des(q_raw, flags);
        if (!joint_tracker_->guards().com_xy_inside(kin_->com(star))) {
          flags.com_reject = true;
          star = q_des_star_;
        } else {
          q_des_star_ = star;
        }
        last_qdes_flags_ = flags;
        if (flags.pitch_clip) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "pitch_clip theta %.1f -> %.1f deg",
              kmpc::rad_to_deg(kmpc::WholeBodyGuards::pitch(q_raw)),
              kmpc::rad_to_deg(kmpc::WholeBodyGuards::pitch(star)));
        }
        if (flags.elbow_clip) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "elbow_clip J4R %.1f->%.1f deg J4L %.1f->%.1f deg",
              kmpc::rad_to_deg(q_raw[kmpc::kIdxJ4R]),
              kmpc::rad_to_deg(star[kmpc::kIdxJ4R]),
              kmpc::rad_to_deg(q_raw[kmpc::kIdxJ4L]),
              kmpc::rad_to_deg(star[kmpc::kIdxJ4L]));
        }
        const kmpc::Vec18 r = tracker_->inertia_r(q0);
        dq = joint_tracker_->compute(q0, q_des_star_, &r);
        last_ee_err_.setZero();
        if (joint_tracker_->last_qp_status() != kmpc::QpStatus::Success ||
            !dq.allFinite()) {
          throw std::runtime_error("joint QP failed");
        }
      }
      last_use_cart_ = use_cart;
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        stream_goal_gen_ = tracker_->goal_generation();
        if (use_cart && tracker_->line_active()) {
          const kmpc::Vec18 q_path = tracker_->line_cmd();
          if (!q_path.allFinite()) {
            throw std::runtime_error("line command is not finite");
          }
          if (tracker_->line_holding() || tracker_->last_arrived() || line_scale<=1e-4) {
            // Drain the finite smoothing history before a normal arrival hold.
            // A feedback safety pause still holds immediately.
            if(stream_pitch_recovery_ && !stream_line_ && line_scale>1e-4) {
              tracker_->refuse_line("pitch_recovery_infeasible");
              stream_enter_hold(interpolator_.q_ref());
            } else if(line_scale<=1e-4 && stream_line_) {
              // Freeze the output and its pending raw/filter history together.
              // Dropping that history would rebase the QP backwards on resume.
              stream_hold_=true;stream_fixed_=false;
              stream_q_hold_=interpolator_.q_ref();
            } else if((!stream_line_ || line_smoother_.speed_inf()<0.002) && !stream_pitch_recovery_) {
              stream_enter_hold(interpolator_.ready()?interpolator_.q_ref():q_path);
            }
            q_cmd=interpolator_.q_ref();dq.setZero();
          } else if (!tracker_->line_commit_pending()) {
            // Tail-wait / already-confirmed tick has no new chord. Re-confirming
            // would reuse the previous "full_propose" reason and refuse a line
            // that has already reached the path end.
            q_cmd = stream_line_ ? stream_line_to_
                                 : (interpolator_.ready() ? interpolator_.q_ref()
                                                          : q_path);
            dq.setZero();
          } else {
            stream_on_line(q_path,line_solve_from);
            q_cmd = tracker_->line_refused()
                        ? stream_q_hold_
                        : (stream_chord_pending_ ? stream_chord_pending_to_
                                                 : stream_line_to_);
            if (tracker_->line_refused()) dq.setZero();
            else if (stream_chord_pending_) dq.setZero();
            else dq=q_cmd-stream_line_from_;
          }
        } else if (use_cart && tracker_->line_refused()) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "straight line refused, holding");
          const kmpc::Vec18 q_hold =
              interpolator_.ready() ? interpolator_.q_ref() : q0;
          stream_enter_hold(q_hold);
          q_cmd = stream_q_hold_;
          dq.setZero();
        } else {
          q_cmd = safety_->clamp(q0 + dq, q0, tracker_->effective_dq_max());
          const kmpc::Vec18 q_out =
              kmpc::ros2io::apply_dq_scale(q0, q_cmd, dq_scale_);
          const kmpc::Vec18 q_star = use_cart
              ? (calibrating ? q_pref_cart : q_out)
              : q_des_star_;
          stream_on_qp(q_out, q_star, use_cart);
          if (stream_hold_) {
            q_cmd = stream_q_hold_;
            dq.setZero();
          } else {
            q_cmd = q_out;
          }
          dq = q_cmd - q0;
        }
        // 段已经从实测接上，才允许流线程发出去。失败的拍继续停发。
        if (interpolator_.ready()) {
          if (stream_mute_) {
            RCLCPP_INFO(get_logger(), "kmpc stream open from measured joints");
          }
          stream_mute_ = false;
        }
      }
      fail_count_ = 0;
    } catch (const std::exception& exc) {
      ++fail_count_;
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 500, "QP/failsafe (%d): %s", fail_count_,
          exc.what());
      {
        std::lock_guard<std::mutex> lock(stream_mu_);
        q_cmd = safety_->failsafe();
        dq.setZero();
        if (interpolator_.ready()) {
          stream_enter_hold(interpolator_.q_ref());
          q_cmd = stream_q_hold_;
        } else {
          interpolator_.set_goal(q_cmd);
        }
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    bool held = false;
    {
      std::lock_guard<std::mutex> lock(stream_mu_);
      held = stream_hold_;
    }
    publish_cmd(q_cmd, dq, ms, held);
  }

  void configure_command_io() {
    stream_dedicated_thread_ = bool_param("stream_dedicated_thread", true);
    stream_debug_q_hz_ = double_param("stream_debug_q_hz", 50.0);
    stream_hz_ = double_param("stream_hz", 250.0);
    interp_catch_s_ = double_param("interp_catch_s", 0.02);
    interp_accel_s_ = double_param("interp_accel_s", 0.50);
    interp_jerk_s_ = double_param("interp_jerk_s", 0.08);
    if (!(stream_hz_ > 0.0) || !std::isfinite(stream_hz_)) {
      throw kmpc::ConfigError("stream_hz must be finite and > 0");
    }
    if (!(interp_catch_s_ > 0.0) || !std::isfinite(interp_catch_s_)) {
      throw kmpc::ConfigError("interp_catch_s must be finite and > 0");
    }
    if (!(interp_accel_s_ > 0.0) || !std::isfinite(interp_accel_s_)) {
      throw kmpc::ConfigError("interp_accel_s must be finite and > 0");
    }
    if (!(interp_jerk_s_ > 0.0) || !std::isfinite(interp_jerk_s_)) {
      throw kmpc::ConfigError("interp_jerk_s must be finite and > 0");
    }
    if (!std::isfinite(stream_debug_q_hz_) || stream_debug_q_hz_ < 0.0) {
      throw kmpc::ConfigError("stream_debug_q_hz must be finite and >= 0");
    }
    interp_dt_ = 1.0 / stream_hz_;
    interpolator_.set_catch_s(interp_catch_s_);
    interpolator_.set_jerk_s(interp_jerk_s_);
  }

  // from_control_thread: 50 Hz commit path → control_diag_q_ (single writer).
  // else: 250 Hz stream path → stream_diag_q_ (single writer). Never mix.
  void push_stream_diag(
      kmpc::ros2io::LayerDiagLayer layer, const kmpc::Vec18& q, uint64_t gen,
      uint32_t line_gen, const char* note, uint64_t seq_override, bool use_override,
      double prog = 0.0, double commit_scale = 1.0, uint8_t field_mask = 0,
      bool from_control_thread = false, uint64_t tick_override = 0,
      bool use_tick = false) {
    if (!layer_diag_on_) {
      return;
    }
    kmpc::ros2io::LayerDiagSample sample;
    sample.mono_ns = kmpc::ros2io::layer_diag_mono_ns();
    sample.recv_mono_ns = sample.mono_ns;
    sample.source_stamp_ns = -1;
    sample.run_id = run_id_;
    sample.goal_generation = gen;
    if (from_control_thread) {
      sample.seq = use_override ? seq_override : ++control_diag_seq_;
    } else {
      sample.seq = use_override ? seq_override : ++stream_diag_seq_;
    }
    sample.tick_id = use_tick
                         ? tick_override
                         : stream_tick_id_.load(std::memory_order_acquire);
    sample.line_gen = line_gen;
    sample.layer = layer;
    sample.next_layer = kmpc::ros2io::layer_diag_next(layer);
    sample.prog = prog;
    sample.commit_scale = commit_scale;
    sample.field_mask = field_mask;
    sample.qp_mode = static_cast<uint8_t>(line_qp_follow_mode_);
    kmpc::ros2io::layer_diag_copy_q(&sample, q);
    kmpc::ros2io::layer_diag_note(&sample, note);
    if (from_control_thread) {
      control_diag_q_.try_push(sample);
    } else {
      stream_diag_q_.try_push(sample);
    }
  }

  void note_sdk_status(uint64_t sequence, bool last_send_ok, bool active) {
    kmpc::ros2io::LayerDiagSample sample;
    sample.mono_ns = kmpc::ros2io::layer_diag_mono_ns();
    sample.recv_mono_ns = sample.mono_ns;
    sample.source_stamp_ns = -1;
    sample.run_id = run_id_;
    sample.goal_generation = 0;
    sample.seq = sequence;
    sample.layer = kmpc::ros2io::LayerDiagLayer::SdkStatus;
    sample.next_layer = sample.layer;
    const char* note = last_send_ok ? (active ? "active_ok" : "idle_ok") : "last_send_fail";
    kmpc::ros2io::layer_diag_note(&sample, note);
    if (sdk_notes_.size() >= 256) {
      sdk_notes_.pop_front();
      ++sdk_note_drops_;
    }
    sdk_notes_.push_back(sample);
  }

  void schedule_qp_fail_dump() {
    if (!tracker_ || qp_fail_dump_dir_.empty()) {
      return;
    }
    // Take is cheap; ofstream stays off every ROS/control callback thread.
    auto dump = std::make_unique<kmpc::CartesianTracker::LineQpFailDump>();
    if (!tracker_->take_line_qp_fail_dump(dump.get()) || !dump->valid) {
      return;
    }
    if (!qp_dump_worker_.running()) {
      const std::string dir = qp_fail_dump_dir_;
      qp_dump_worker_.start(
          [this, dir](
              std::unique_ptr<kmpc::CartesianTracker::LineQpFailDump>&& payload) {
            write_qp_fail_dump_file(std::move(payload), dir);
          });
    }
    if (!qp_dump_worker_.enqueue(std::move(dump))) {
      // Bounded queue full — counted, never spawn unbounded detach threads.
      return;
    }
  }

  void write_qp_fail_dump_file(
      std::unique_ptr<kmpc::CartesianTracker::LineQpFailDump> dump,
      const std::string& dir) {
    if (!dump || !dump->valid) {
      return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
      RCLCPP_ERROR(get_logger(), "qp fail dump mkdir %s failed: %s",
                   dir.c_str(), ec.message().c_str());
      return;
    }
    char name[256];
    std::snprintf(name, sizeof(name), "qp_fail_gen%llu_mono%llu.json",
                  static_cast<unsigned long long>(dump->goal_generation),
                  static_cast<unsigned long long>(dump->mono_ns));
    const fs::path path = fs::path(dir) / name;
    std::ofstream out(path);
    if (!out) {
      RCLCPP_ERROR(get_logger(), "qp fail dump open failed: %s", path.c_str());
      return;
    }
    auto csv18 = [](const kmpc::Vec18& q) {
      std::ostringstream oss;
      oss.setf(std::ios::scientific);
      oss.precision(17);
      for (int i = 0; i < 18; ++i) {
        if (i) oss << ',';
        oss << q[i];
      }
      return oss.str();
    };
    out << "{\n";
    out << "  \"schema\":\"kmpc_line_qp_fail_v1\",\n";
    out << "  \"note\":\"worst_c is the max violation index on the solver return; "
           "it does not prove a unique conflicting constraint\",\n";
    out << "  \"truncated\":" << (dump->truncated ? "true" : "false") << ",\n";
    out << "  \"mono_ns\":" << dump->mono_ns << ",\n";
    out << "  \"goal_generation\":" << dump->goal_generation << ",\n";
    out << "  \"line_gen\":" << dump->line_gen << ",\n";
    out << "  \"reason\":\"" << dump->reason << "\",\n";
    out << "  \"status\":" << dump->status << ",\n";
    out << "  \"status_name\":\"" << dump->status_name << "\",\n";
    out << "  \"iter\":" << dump->iter << ",\n";
    out << "  \"pri_res\":" << dump->pri_res << ",\n";
    out << "  \"dua_res\":" << dump->dua_res << ",\n";
    out << "  \"solve_ms\":" << dump->solve_ms << ",\n";
    out << "  \"worst_c\":" << dump->worst_c << ",\n";
    out << "  \"worst_tag\":\"" << dump->worst_tag << "\",\n";
    out << "  \"worst_violation\":" << dump->worst_violation << ",\n";
    out << "  \"prog\":" << dump->prog << ",\n";
    out << "  \"path_remain_m\":" << dump->path_remain_m << ",\n";
    out << "  \"pitch_deg\":" << dump->pitch_deg << ",\n";
    out << "  \"ee_err_m\":" << dump->ee_err_m << ",\n";
    out << "  \"n_horizon\":" << dump->n_horizon << ",\n";
    out << "  \"nx\":" << dump->nx << ",\n";
    out << "  \"nc\":" << dump->nc << ",\n";
    out << "  \"n_triplets\":" << dump->n_triplets << ",\n";
    out << "  \"waist_locked\":" << (dump->waist_locked ? "true" : "false") << ",\n";
    out << "  \"ori_on\":" << (dump->ori_on ? "true" : "false") << ",\n";
    out << "  \"pitch_mono_enable\":" << (dump->pitch_mono_enable ? "true" : "false")
        << ",\n";
    out << "  \"direction_strict\":" << (dump->direction_strict ? "true" : "false")
        << ",\n";
    out << "  \"candidate\":\"" << dump->candidate << "\",\n";
    out << "  \"frame\":\"" << dump->frame << "\",\n";
    out << "  \"torso_pitch_max\":" << dump->torso_pitch_max << ",\n";
    out << "  \"q_limit_margin\":" << dump->q_limit_margin << ",\n";
    out << "  \"com_box\":[" << dump->com_xmin << "," << dump->com_xmax << ","
        << dump->com_ymin << "," << dump->com_ymax << "],\n";
    out << "  \"com_margin\":" << dump->com_margin << ",\n";
    out << "  \"direction_sign\":[" << dump->direction_sign[0] << ","
        << dump->direction_sign[1] << "," << dump->direction_sign[2] << ","
        << dump->direction_sign[3] << "],\n";
    out << "  \"from\":[" << csv18(dump->from) << "],\n";
    out << "  \"cap\":[" << csv18(dump->cap) << "],\n";
    out << "  \"target\":[" << dump->target[0] << "," << dump->target[1] << ","
        << dump->target[2] << "],\n";
    out << "  \"quat_xyzw\":[" << dump->quat[0] << "," << dump->quat[1] << ","
        << dump->quat[2] << "," << dump->quat[3] << "],\n";
    out << "  \"progress\":[";
    for (int k = 0; k < dump->n_horizon; ++k) {
      if (k) out << ",";
      out << dump->progress[k];
    }
    out << "],\n  \"nominal\":[";
    for (int k = 0; k < dump->n_horizon; ++k) {
      if (k) out << ",";
      out << "[";
      for (int j = 0; j < 18; ++j) {
        if (j) out << ",";
        out << dump->nominal[k][j];
      }
      out << "]";
    }
    out << "],\n  \"g\":[";
    for (int i = 0; i < dump->nx; ++i) {
      if (i) out << ",";
      out << dump->g[i];
    }
    out << "],\n  \"lo\":[";
    for (int i = 0; i < dump->nc; ++i) {
      if (i) out << ",";
      out << dump->lo[i];
    }
    out << "],\n  \"hi\":[";
    for (int i = 0; i < dump->nc; ++i) {
      if (i) out << ",";
      out << dump->hi[i];
    }
    out << "],\n  \"tags\":[";
    for (int i = 0; i < dump->nc; ++i) {
      if (i) out << ",";
      out << "\"" << dump->tags[i] << "\"";
    }
    out << "],\n  \"C_triplets\":[";
    for (int i = 0; i < dump->n_triplets; ++i) {
      if (i) out << ",";
      out << "[" << dump->trip_r[i] << "," << dump->trip_c[i] << ","
          << dump->trip_v[i] << "]";
    }
    out << "]\n}\n";
    out.close();
    RCLCPP_ERROR(get_logger(),
                 "qp fail dump written path=%s worst_c=%d worst_tag=%s nc=%d "
                 "truncated=%d",
                 path.c_str(), dump->worst_c,
                 dump->worst_tag[0] ? dump->worst_tag : "-", dump->nc,
                 dump->truncated ? 1 : 0);
  }

  void drain_layer_diag() {
    if (!layer_diag_out_.is_open()) {
      return;
    }
    kmpc::ros2io::LayerDiagSample batch[64];
    auto drain_q = [&](kmpc::ros2io::LayerDiagQueue<8192>& q, uint64_t* exported) {
      for (;;) {
        const size_t n = q.try_pop(batch, 64);
        for (size_t i = 0; i < n; ++i) {
          layer_diag_out_ << kmpc::ros2io::layer_diag_jsonl(batch[i]) << '\n';
        }
        if (n < 64) {
          break;
        }
      }
      const uint64_t drops = q.drops();
      if (drops != *exported) {
        layer_diag_out_ << kmpc::ros2io::layer_diag_overflow_jsonl(run_id_, drops)
                        << '\n';
        *exported = drops;
      }
    };
    drain_q(control_diag_q_, &exported_control_drops_);
    drain_q(stream_diag_q_, &exported_stream_drops_);
    if (tracker_ && line_diag_enable_) {
      const uint64_t ov = tracker_->line_diag_overflows();
      const auto events = tracker_->line_diag_snapshot();
      size_t begin = 0;
      if (ov == exported_tracker_overflows_ && events.size() >= exported_ring_n_) {
        begin = exported_ring_n_;
      }
      for (size_t i = begin; i < events.size(); ++i) {
        const auto& ev = events[i];
        kmpc::ros2io::LayerDiagSample sample;
        sample.mono_ns = ev.mono_ns;
        sample.recv_mono_ns = ev.recv_mono_ns;
        sample.source_stamp_ns = ev.source_stamp_ns;
        sample.run_id = ev.run_id == 0 ? run_id_ : ev.run_id;
        sample.goal_generation = ev.goal_generation;
        sample.seq = ev.seq;
        sample.tick_id = ev.tick_id;
        sample.line_gen = ev.line_gen < 0 ? 0u : static_cast<uint32_t>(ev.line_gen);
        using L = kmpc::CartesianTracker::LineDiagLayer;
        sample.layer = ev.layer == L::FollowQp
                           ? kmpc::ros2io::LayerDiagLayer::FollowQp
                           : ev.layer == L::FollowCommit
                                 ? kmpc::ros2io::LayerDiagLayer::FollowCommit
                                 : ev.layer == L::Feedback
                                       ? kmpc::ros2io::LayerDiagLayer::Feedback
                                       : kmpc::ros2io::LayerDiagLayer::Build;
        sample.next_layer = kmpc::ros2io::layer_diag_next(sample.layer);
        sample.prog = ev.prog;
        sample.s_dot = ev.s_dot;
        sample.solve_us = ev.solve_us;
        sample.commit_scale = ev.commit_scale;
        sample.qp_dx_norm = ev.qp_dx_norm;
        sample.dq_proj_max = ev.dq_proj_max;
        sample.qp_mode = ev.qp_mode;
        sample.field_mask = ev.field_mask;
        kmpc::ros2io::layer_diag_copy_q(&sample, ev.q);
        if (ev.field_mask & kmpc::ros2io::kLayerDiagHasNominal) {
          kmpc::ros2io::layer_diag_copy_nominal(&sample, ev.q_nominal);
        }
        if (ev.field_mask & kmpc::ros2io::kLayerDiagHasPre) {
          kmpc::ros2io::layer_diag_copy_pre(&sample, ev.q_pre);
        }
        // Prefer q_aux as pre/from when FollowQp and q_pre empty.
        if (ev.layer == L::FollowQp &&
            !(sample.field_mask & kmpc::ros2io::kLayerDiagHasPre) &&
            ev.q_aux.allFinite()) {
          kmpc::ros2io::layer_diag_copy_pre(&sample, ev.q_aux);
        }
        kmpc::ros2io::layer_diag_note(
            &sample, ev.decision[0] != '\0' ? ev.decision : ev.candidate);
        layer_diag_out_ << kmpc::ros2io::layer_diag_jsonl(sample) << '\n';
      }
      exported_ring_n_ = events.size();
      if (ov != exported_tracker_overflows_) {
        layer_diag_out_ << kmpc::ros2io::layer_diag_overflow_jsonl(run_id_, ov) << '\n';
        exported_tracker_overflows_ = ov;
      }
    }
    if (sdk_note_drops_ != exported_sdk_drops_) {
      layer_diag_out_ << kmpc::ros2io::layer_diag_overflow_jsonl(run_id_, sdk_note_drops_)
                      << '\n';
      exported_sdk_drops_ = sdk_note_drops_;
    }
    while (!sdk_notes_.empty()) {
      layer_diag_out_ << kmpc::ros2io::layer_diag_jsonl(sdk_notes_.front()) << '\n';
      sdk_notes_.pop_front();
    }
    layer_diag_out_.flush();
  }

  void arm_layer_diag() {
    layer_diag_on_ = false;
    if (layer_diag_path_.empty()) {
      return;
    }
    if (!line_diag_enable_) {
      RCLCPP_WARN(get_logger(),
                  "layer_diag_path is set but line_diag_enable is false; export stays off");
      return;
    }
    layer_diag_out_.open(layer_diag_path_, std::ios::out | std::ios::trunc);
    if (!layer_diag_out_) {
      RCLCPP_ERROR(get_logger(), "cannot open layer diag %s; export stays off",
                   layer_diag_path_.c_str());
      return;
    }
    layer_diag_out_ << kmpc::ros2io::layer_diag_header_jsonl(run_id_) << '\n';
    layer_diag_out_.flush();
    layer_diag_on_ = true;
    layer_diag_timer_ = create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&KmpcNode::drain_layer_diag, this));
    RCLCPP_INFO(get_logger(), "layer diag export on path=%s run_id=%llu",
                layer_diag_path_.c_str(), static_cast<unsigned long long>(run_id_));
  }

  void emit_effective_config() {
    const std::string params_file =
        str_or("kmpc_params_file", default_config("kmpc_params.yaml"));
    const std::string limits_file =
        str_or("joint_limits_file", default_config("joint_limits.yaml"));
    std::vector<kmpc::ros2io::ConfigSnapshotRow> rows;
    auto add = [&](const std::string& name, const std::string& value, const char* unit,
                   const std::string& source, const char* range, bool ok) {
      rows.push_back({name, value, unit, source, range, ok});
    };
    auto num = [&](const char* name, double value, const char* unit,
                   const std::string& source, const char* range, bool ok) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.9g", value);
      add(name, buf, unit, source, range, ok && std::isfinite(value));
    };
    auto ros_num = [&](const char* name, double fallback, double effective, const char* unit,
                       const char* range) {
      const double raw = double_param(name, fallback);
      num(name, effective, unit, raw == fallback ? "node_default" : "ros_override", range,
          true);
    };
    auto ros_bool = [&](const char* name, bool fallback) {
      const bool value = bool_param(name, fallback);
      add(name, value ? "true" : "false", "bool",
          value == fallback ? "node_default" : "ros_override", "true|false", true);
    };
    const auto& p = cfg_.params;
    num("dt", p.dt, "s", params_file, ">0", p.dt > 0.0);
    num("control_rate", p.control_rate, "Hz", params_file, ">=1", p.control_rate >= 1.0);
    num("n_horizon", static_cast<double>(p.n_horizon), "steps", params_file, ">=1",
        p.n_horizon >= 1);
    num("q_limit_margin", p.q_limit_margin, "rad", params_file, "finite", true);
    num("arrived_err", p.arrived_err, "m", params_file, "finite", true);
    num("arrived_ori_rad", p.arrived_ori_rad, "rad", params_file, "finite", true);
    num("com_xmin", p.com_xmin, "m", params_file, "finite", true);
    num("com_xmax", p.com_xmax, "m", params_file, "finite", true);
    num("com_ymin", p.com_ymin, "m", params_file, "finite", true);
    num("com_ymax", p.com_ymax, "m", params_file, "finite", true);
    num("com_margin", p.com_margin, "m", params_file, "finite", true);
    num("torso_pitch_max_deg", p.torso_pitch_max_deg, "deg", params_file, "finite", true);
    num("calib_tol_rad", p.calib_tol_rad, "rad", params_file, "finite", true);
    for (int i = 0; i < kmpc::kNDof; ++i) {
      const std::string joint(kmpc::kJointNames[static_cast<std::size_t>(i)]);
      num(("q_min." + joint).c_str(), cfg_.limits.q_min[i], "rad", limits_file, "finite",
          true);
      num(("q_max." + joint).c_str(), cfg_.limits.q_max[i], "rad", limits_file, "finite",
          true);
      num(("dq_max." + joint).c_str(), cfg_.limits.dq_max[i], "rad/s", limits_file, ">=0",
          cfg_.limits.dq_max[i] >= 0.0);
    }
    ros_num("dq_scale", 1.0, dq_scale_, "1", "[0,1]");
    ros_num("line_dq_scale", 0.60, line_dq_scale_, "1", "[0,1]");
    add("line_pace_mode",
        kmpc::CartesianTracker::line_pace_mode_name(line_pace_mode_), "enum",
        "ros_param", "r11_scurve|r10b_first_order", true);
    ros_num("line_brake_distance_m", 0.08, line_brake_distance_m_, "m", ">0");
    ros_num("line_brake_floor", 0.20, line_brake_floor_, "1", "(0,1]");
    add("line_brake_profile",
        kmpc::CartesianTracker::line_brake_profile_name(line_brake_profile_),
        "enum", "ros_param", "linear|sqrt", true);
    {
      const int raw = int_param("line_qp_max_iter", 200);
      num("line_qp_max_iter", static_cast<double>(line_qp_max_iter_), "iters",
          raw == 200 ? "node_default" : "ros_override", "[1,10000]", true);
    }
    {
      const std::string raw = str_or("line_qp_follow_mode", "A");
      add("line_qp_follow_mode",
          kmpc::line_qp_follow_mode_name(line_qp_follow_mode_), "enum",
          raw == "A" ? "node_default" : "ros_override", "A|B|C", true);
    }
    add("line_freeze_distance_m", "0.08", "m", "compile_const", "=0.08", true);
    ros_bool("line_pitch_straighten_enable", true);
    ros_bool("line_qp_pitch_monotonic_enable", true);
    ros_bool("line_shadow_diag_enable", true);
    ros_bool("reverse_last_line_enable", true);
    ros_bool("line_direction_strict_enable", true);
    ros_num("stream_hz", 250.0, stream_hz_, "Hz", ">0");
    ros_num("interp_catch_s", 0.02, interp_catch_s_, "s", ">0");
    ros_num("interp_accel_s", 0.50, interp_accel_s_, "s", ">0");
    ros_num("interp_jerk_s", 0.08, interp_jerk_s_, "s", ">0");
    ros_num("stream_debug_q_hz", 50.0, stream_debug_q_hz_, "Hz", ">=0");
    ros_num("state_timeout_sec", 0.1, double_param("state_timeout_sec", 0.1), "s", "finite");
    ros_num("hold_deadband", 0.01, hold_deadband_, "rad", "finite");
    ros_bool("line_diag_enable", false);
    add("planner", planner_name_, "enum", "ros_param",
        "unified_multistage_v2|legacy", true);
    ros_bool("line_pose_pref_enable", false);
    ros_bool("stream_dedicated_thread", true);
    ros_bool("auto_enable", false);
    ros_bool("waist_locked", false);
    ros_bool("require_xoip", true);
    add("stream_topic", stream_topic_name_, "topic", "ros_param", "nonempty",
        !stream_topic_name_.empty());
    add("joint_states_topic", str_or("joint_states_topic", "/joint_states"), "topic",
        "ros_param", "nonempty", true);
    add("layer_diag_path", layer_diag_path_, "path",
        layer_diag_path_.empty() ? "node_default" : "ros_override", "empty_or_writable",
        true);
    add("qp_fail_dump_dir", qp_fail_dump_dir_, "path",
        qp_fail_dump_dir_.empty() ? "disabled" : "ros_or_default", "empty_or_writable",
        true);
    add("config_snapshot_path", config_snapshot_path_, "path",
        config_snapshot_path_.empty() ? "node_default" : "ros_override", "empty_or_writable",
        true);
    add("q_pref_station", str_or("q_pref_station", "default"), "name",
        str_or("q_pref_file", default_config("q_pref.yaml")), "nonempty", true);
    add("sdk_servo_msg",
#ifdef KMPC_HAVE_XOIP
        "xoip_ros2/MultiJointServoCommand",
#else
        "unavailable",
#endif
        "msg", "compile", "KMPC_HAVE_XOIP", 
#ifdef KMPC_HAVE_XOIP
        true
#else
        false
#endif
    );
    const std::string json = kmpc::ros2io::config_snapshot_json(run_id_, "unrecorded", rows);
    bool bad = false;
    for (const auto& row : rows) {
      if (!row.ok) {
        bad = true;
      }
    }
    if (!config_snapshot_path_.empty()) {
      std::ofstream out(config_snapshot_path_, std::ios::out | std::ios::trunc);
      if (!out) {
        RCLCPP_ERROR(get_logger(), "cannot write config snapshot %s",
                     config_snapshot_path_.c_str());
      } else {
        out << json << '\n';
        RCLCPP_INFO(get_logger(), "config snapshot wrote %s rows=%zu run_id=%llu",
                    config_snapshot_path_.c_str(), rows.size(),
                    static_cast<unsigned long long>(run_id_));
      }
    }
    RCLCPP_INFO(get_logger(), "effective_config %s", json.c_str());
    if (bad) {
      RCLCPP_ERROR(get_logger(), "effective config has a row marked ok=false");
    }
  }



  void release_waist_route() {
#ifdef KMPC_HAVE_XOIP
    if (!clear_route_cli_) {
      return;
    }
    auto req = std::make_shared<xoip_ros2::srv::ClearRoute::Request>();
    req->robot_id = kmpc::kRobotIdWaist;
    req->emergency_stop = false;
    clear_route_cli_->async_send_request(
        req,
        [this](rclcpp::Client<xoip_ros2::srv::ClearRoute>::SharedFuture fut) {
          const auto resp = fut.get();
          if (!resp) {
            RCLCPP_WARN(get_logger(), "clear_route waist: empty response");
            return;
          }
          if (resp->success) {
            RCLCPP_INFO(get_logger(), "clear_route waist ok: %s", resp->message.c_str());
          } else {
            RCLCPP_WARN(
                get_logger(), "clear_route waist failed: %s", resp->message.c_str());
          }
        });
    RCLCPP_INFO(get_logger(), "kmpc off: clear_route robot_id=%d", kmpc::kRobotIdWaist);
#endif
  }

  // 空 targets 退出透传。20260921 控制器按 stream_control_rate 取最新一包，
  // 调用 PluseMultiToServo。没有每包加减速。
  void release_servo_stream() {
#ifdef KMPC_HAVE_XOIP
    if (!servo_pub_) {
      return;
    }
    xoip_ros2::msg::MultiJointServoCommand msg;
    msg.header.stamp = now();
    msg.sequence = ++servo_seq_;
    servo_pub_->publish(msg);
    RCLCPP_INFO(
        get_logger(), "kmpc off: empty servo batch seq=%llu",
        static_cast<unsigned long long>(msg.sequence));
#endif
  }

  uint64_t publish_stream(const kmpc::Vec18& q_rad) {
#ifdef KMPC_HAVE_XOIP
    if (!servo_pub_) {
      return 0;
    }
    const auto packed = kmpc::pack_stream_deg(q_rad);
    xoip_ros2::msg::MultiJointServoCommand msg;
    msg.header.stamp = now();
    msg.sequence = ++servo_seq_;
    auto add = [&](int robot_id, const auto& joints) {
      xoip_ros2::msg::JointServoTarget target;
      target.robot_id = robot_id;
      target.joints.assign(joints.begin(), joints.end());
      msg.targets.push_back(std::move(target));
    };
    add(kmpc::kRobotIdWaist, packed.waist_deg);
    add(kmpc::kRobotIdRight, packed.right_deg);
    add(kmpc::kRobotIdLeft, packed.left_deg);
    servo_pub_->publish(msg);
    return msg.sequence;
#else
    (void)q_rad;
    return 0;
#endif
  }


  void publish_cmd(
      const kmpc::Vec18& q_cmd, const kmpc::Vec18& dq, double solve_ms,
      bool held) {
    // 透传由 250 Hz 线程发。控制拍只发调试，不把 50 Hz 台阶打到伺服口。
    publish_debug(q_cmd, dq, solve_ms, held, false);
  }



  void on_stream_timer() {
    kmpc::Vec18 q = kmpc::Vec18::Zero();
    bool send = false;
    uint64_t gen = 0;
    uint32_t line_gen = 0;
    {
      std::lock_guard<std::mutex> lock(stream_mu_);
      send = stream_take_tick(&q);
      gen = stream_goal_gen_;
      line_gen = stream_line_gen_ < 0 ? 0u : static_cast<uint32_t>(stream_line_gen_);
    }
    if (send) {
      const uint64_t seq = publish_stream(q);
      if (layer_diag_on_) {
        push_stream_diag(kmpc::ros2io::LayerDiagLayer::FinalSend, q, gen, line_gen,
                         "sent", seq, true);
      }
      maybe_publish_debug_q(q);
    }
  }

  void publish_debug_q(const kmpc::Vec18& q_rad) {
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    for (int i = 0; i < kmpc::kNDof; ++i) {
      js.name.emplace_back(kmpc::kJointNames[static_cast<std::size_t>(i)]);
      js.position.push_back(q_rad[i]);
    }
    dbg_q_->publish(js);
  }

  void publish_debug(
      const kmpc::Vec18& q_cmd, const kmpc::Vec18& dq, double solve_ms,
      bool held, bool publish_q) {
    if (publish_q) {
      publish_debug_q(q_cmd);
    }

    std_msgs::msg::Float64MultiArray dq_msg;
    std_msgs::msg::Float64MultiArray ee_msg;
    for (int i = 0; i < kmpc::kNDof; ++i) {
      dq_msg.data.push_back(dq[i]);
    }
    for (int i = 0; i < 6; ++i) {
      ee_msg.data.push_back(last_ee_err_[i]);
    }
    dbg_dq_->publish(dq_msg);
    dbg_ee_->publish(ee_msg);

    const double e_l = last_ee_err_.head<3>().norm();
    const double e_r = last_ee_err_.tail<3>().norm();
    const double e0 = tracker_->last_e0();
    const auto measured=state_->read();
    const bool line_done=tracker_->line_holding();
    bool settled=true, recovery_complete=true;
    {std::lock_guard<std::mutex> lock(stream_mu_);recovery_complete=!stream_pitch_recovery_;settled=(!stream_line_ || line_smoother_.speed_inf()<0.002) && recovery_complete;}
    const bool arrived=tracker_->last_arrived() &&
        (!tracker_->line_active() || recovery_complete) &&
        (!line_done || (settled && measured && tracker_->line_measured_arrived(*measured)));
    if(measured && line_done) {
      tracker_->maybe_commit_last_line_cache(*measured);
    }
    if(tracker_->line_refused() && (last_reported_line_stop_gen_!=tracker_->goal_generation() ||
        last_reported_line_stop_reason_!=tracker_->line_reason())) {
      last_reported_line_stop_gen_=tracker_->goal_generation();
      last_reported_line_stop_reason_=tracker_->line_reason();
      const auto& fail = tracker_->last_line_qp_fail();
      if (fail.valid) {
        RCLCPP_ERROR(
            get_logger(),
            "line stopped generation=%llu reason=%s status=%s(%d) iter=%d "
            "pri_res=%.3e dua_res=%.3e solve_ms=%.3f worst_c=%d worst_tag=%s "
            "worst_viol=%.3e path_remain_m=%.4f prog=%.4f pitch_deg=%.3f "
            "ee_err_m=%.4f",
            static_cast<unsigned long long>(tracker_->goal_generation()),
            tracker_->line_reason().c_str(), fail.status_name, fail.status,
            fail.iter, fail.pri_res, fail.dua_res, fail.solve_ms, fail.worst_c,
            fail.worst_tag[0] ? fail.worst_tag : "-",
            fail.worst_violation, fail.path_remain_m, fail.prog, fail.pitch_deg,
            fail.ee_err_m);
      } else {
        RCLCPP_ERROR(get_logger(),"line stopped generation=%llu reason=%s",
            static_cast<unsigned long long>(tracker_->goal_generation()),
            tracker_->line_reason().c_str());
      }
      // Drain promptly: empty poses trigger reset()/line_clear within ~10ms,
      // and the 100ms wall timer alone often loses that race.
      schedule_qp_fail_dump();
    }
    publish_arrived(arbiter_->enabled() && last_use_cart_ && arrived);
    const kmpc::Vec3 com = last_use_cart_ ? tracker_->last_com()
                                         : joint_tracker_->last_com();
    bool sfix = false;
    int nbrk = 0;
    {
      std::lock_guard<std::mutex> lock(stream_mu_);
      sfix = stream_fixed_;
      nbrk = stream_hold_ ? kmpc::kNDof : interpolator_.n_axis_brake();
    }
    const double pitch_deg = kmpc::rad_to_deg(
        last_use_cart_ ? tracker_->last_pitch() : joint_tracker_->last_pitch());
    const double e_l_ori = tracker_->last_e_ori().head<3>().norm();
    const double e_r_ori = tracker_->last_e_ori().tail<3>().norm();
    publish_com_marker(com);

    const double work_lo = last_use_cart_
        ? tracker_->last_com_work_xmin()
        : joint_tracker_->guards().com_work_xmin();
    const double work_hi = last_use_cart_
        ? tracker_->last_com_work_xmax()
        : joint_tracker_->guards().com_work_xmax();
    const double slack_x = last_use_cart_
        ? tracker_->last_com_slack_x()
        : (work_hi - com[0]);
    const bool work_wall = last_use_cart_
        ? tracker_->last_com_work_wall()
        : joint_tracker_->guards().com_at_work_wall(com);

    std_msgs::msg::String status;
    status.data =
        "enable=" + std::string(arbiter_->enabled() ? "True" : "False") +
        " mode=" + task_->mode() +
        " line=" +
        std::string(
            tracker_->line_holding()
                ? "hold"
                : (tracker_->line_active()
                       ? "on"
                       : (tracker_->line_refused() ? "stop" : "off"))) +
        " goal_generation=" + std::to_string(tracker_->goal_generation()) +
        " segment_active=" + std::to_string(task_->saw_cartesian_target()) +
        " frozen=" + std::to_string(tracker_->line_frozen()) +
        " abort=" + std::to_string(tracker_->line_abort_latched()) +
        " reason=" + (tracker_->line_reason().empty()?"none":tracker_->line_reason()) +
        " cascade=" + (tracker_->line_cascade_checked()?(tracker_->line_cascade_passed()?"passed":"failed"):"not_provided") +
        " forced_waist=" + std::to_string(tracker_->line_cascade_forced_waist()) +
        " cascade_full_body=" + std::to_string(tracker_->line_cascade_full_body()) +
        " planner=" +
        std::string(use_v2_chain_
                        ? "unified_multistage_v2"
                        : (tracker_->line_candidate() == "unified_multistage_v2"
                               ? "unified_multistage_v2"
                               : "none")) +
        " path_id=" + std::to_string(v2_gate_.active_path_id()) +
        " task_digest=" + std::to_string(v2_gate_.active().digest) +
        " v2_submits=" + std::to_string(v2_gate_.counters().submits) +
        " v2_adopts=" + std::to_string(v2_gate_.counters().adopts) +
        " v2_rejects=" + std::to_string(v2_gate_.counters().rejects) +
        " line_candidate=" + (tracker_->line_candidate().empty()?"none":tracker_->line_candidate()) +
        " pose_report=" + (tracker_->line_pose_report().empty()?"none":tracker_->line_pose_report()) +
        " line_waist_locked=" + std::string(tracker_->line_waist_locked() ? "True" : "False") +
        " pose_pref=" + (tracker_->line_pose_pref_decision().empty()?"disabled":tracker_->line_pose_pref_decision()) +
        " reverse_line=" + (tracker_->reverse_line_decision().empty()?"disabled":tracker_->reverse_line_decision()) +
        " max_arm_fold_deg=" + std::to_string(kmpc::rad_to_deg(tracker_->line_max_arm_fold())) +
        " line_build_ms=" + std::to_string(tracker_->line_build_ms()) +
        " line_samples=" + std::to_string(tracker_->line_sample_count()) +
        " line_length_m=" + std::to_string(tracker_->line_length()) +
        " line_prog=" + std::to_string(tracker_->line_progress()) +
        " path_remain_m=" + std::to_string(tracker_->last_line_path_remain_m()) +
        " line_brake=" + std::to_string(tracker_->last_line_brake()) +
        " line_want=" + std::to_string(tracker_->last_line_want()) +
        " line_rate=" + std::to_string(tracker_->line_rate()) +
        " line_rate_acc=" +
        std::string(
            tracker_->line_pace_mode() ==
                    kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder
                ? "na"
                : std::to_string(tracker_->line_rate_acc())) +
        " line_pace_mode=" +
        kmpc::CartesianTracker::line_pace_mode_name(tracker_->line_pace_mode()) +
        " line_brake_m=" + std::to_string(tracker_->line_brake_distance_m()) +
        " line_brake_floor=" + std::to_string(tracker_->line_brake_floor()) +
        " line_brake_profile=" +
        kmpc::CartesianTracker::line_brake_profile_name(
            tracker_->line_brake_profile()) +
        " line_qp_max_iter=" + std::to_string(tracker_->line_qp_max_iter()) +
        " pitch_straighten=" +
        std::string(tracker_->line_pitch_straighten_enabled() ? "1" : "0") +
        " pitch_straighten_hits=" +
        std::to_string(tracker_->line_pitch_straighten_hits()) +
        " qp_pitch_mono=" +
        std::string(tracker_->line_qp_pitch_monotonic_enabled() ? "1" : "0") +
        " qp_pitch_mono_hits=" +
        std::to_string(tracker_->line_qp_pitch_monotonic_hits()) +
        " reverse_last_line=" +
        std::string(tracker_->reverse_last_line_enabled() ? "1" : "0") +
        " tail_brake_dt_s=" +
        std::to_string(
            (tracker_->line_tail_enter_brake_s() < 0.0 ||
             tracker_->line_seg_start_s() < 0.0)
                ? -1.0
                : (tracker_->line_tail_enter_brake_s() -
                   tracker_->line_seg_start_s())) +
        " tail_2cm_dt_s=" +
        std::to_string(
            (tracker_->line_tail_enter_2cm_s() < 0.0 ||
             tracker_->line_seg_start_s() < 0.0)
                ? -1.0
                : (tracker_->line_tail_enter_2cm_s() -
                   tracker_->line_seg_start_s())) +
        " tail_hold_dt_s=" +
        std::to_string(
            (tracker_->line_tail_hold_s() < 0.0 ||
             tracker_->line_seg_start_s() < 0.0)
                ? -1.0
                : (tracker_->line_tail_hold_s() - tracker_->line_seg_start_s())) +
        " tail_2cm_to_hold_s=" +
        std::to_string(
            (tracker_->line_tail_hold_s() < 0.0 ||
             tracker_->line_tail_enter_2cm_s() < 0.0)
                ? -1.0
                : (tracker_->line_tail_hold_s() -
                   tracker_->line_tail_enter_2cm_s())) +
        " line_planning=" + std::to_string(line_future_.valid()) +
        " line_plan_builds=" + std::to_string(line_plan_builds_) +
        " line_plan_discards=" + std::to_string(line_plan_discards_) +
        " stream_commit_scale=" + std::to_string(line_commit_scale_) +
        " stream_step_ratio=" + std::to_string(line_commit_peak_ratio_) +
        " stream_limited_steps=" + std::to_string(line_limited_steps_) +
        " stream_limited=" + std::to_string(line_limited_steps_ > 0 ? 1 : 0) +
        " tracking_state=" + std::string(kmpc::to_cstring(planner_->tracking_state())) +
        " waist_locked=" + std::string(planner_->waist_locked() ? "True" : "False") +
        " com_reject=" + std::string(planner_rx_->last_com_reject() ? "True" : "False") +
        " held=" + std::string(held ? "True" : "False") +
        " sfix=" + std::string(sfix ? "1" : "0") +
        " nbrk=" +
        std::to_string(nbrk) +
        " solve_ms=" + std::to_string(solve_ms) +
        " fail=" + std::to_string(fail_count_) +
        " line_dq=" + std::to_string(line_dq_scale_) +
        " dq_scale=" + std::to_string(dq_scale_) +
        " js_source_stamp_s=" + std::to_string(state_->source_stamp_s()) +
        " js_receive_age_s=" + std::to_string(state_->receive_age_s()) +
        " line_diag_n=" + std::to_string(tracker_->line_diag_size()) +
        " line_diag_overflow=" + std::to_string(tracker_->line_diag_overflows()) +
        " N=" + std::to_string(cfg_.params.n_horizon) +
        " qp_nodes=2 arrived=" + std::string(arrived ? "True" : "False") +
        " e0=" + std::to_string(e0) +
        " eL=" + std::to_string(e_l) + "m eR=" + std::to_string(e_r) +
        " com_xy=(" + std::to_string(com[0]) + "," + std::to_string(com[1]) + ")" +
        " com_work_x=(" + std::to_string(work_lo) + "," + std::to_string(work_hi) +
        ")" +
        " com_slack_x=" + std::to_string(slack_x) +
        " com_work_wall=" + std::string(work_wall ? "True" : "False") +
        " pitch=" + std::to_string(pitch_deg) + "deg" +
        " eLori=" + std::to_string(kmpc::rad_to_deg(e_l_ori)) + "deg" +
        " eRori=" + std::to_string(kmpc::rad_to_deg(e_r_ori)) + "deg" +
        " pitch_clip=" + std::string(last_qdes_flags_.pitch_clip ? "True" : "False") +
        " elbow_clip=" + std::string(last_qdes_flags_.elbow_clip ? "True" : "False") +
        " q_des_projected=" +
        std::string(last_qdes_flags_.projected ? "True" : "False") +
        " q_des_com_reject=" +
        std::string(last_qdes_flags_.com_reject ? "True" : "False");
    if (last_use_cart_) {
      status.data +=
          " theta_des=" +
          std::to_string(kmpc::rad_to_deg(tracker_->last_theta_des())) + "deg" +
          " c_des_x=" + std::to_string(tracker_->last_c_des_x()) +
          " com_hold=" +
          std::string(tracker_->last_com_hold() ? "True" : "False") +
          " rise_hold=" +
          std::string(tracker_->last_rise_hold() ? "True" : "False");
    } else {
      status.data +=
          " theta_star=" +
          std::to_string(kmpc::rad_to_deg(joint_tracker_->last_theta_star())) +
          "deg" +
          " pitch_seg=(" +
          std::to_string(kmpc::rad_to_deg(joint_tracker_->last_pitch_seg_lo())) +
          "," +
          std::to_string(kmpc::rad_to_deg(joint_tracker_->last_pitch_seg_hi())) +
          ")";
    }
    if (tracker_->last_line_qp_fail().valid) {
      const auto& fail = tracker_->last_line_qp_fail();
      status.data +=
          " qp_fail_status=" + std::string(fail.status_name) +
          "(" + std::to_string(fail.status) + ")" +
          " qp_fail_iter=" + std::to_string(fail.iter) +
          " qp_fail_pri_res=" + std::to_string(fail.pri_res) +
          " qp_fail_dua_res=" + std::to_string(fail.dua_res) +
          " qp_fail_solve_ms=" + std::to_string(fail.solve_ms) +
          " qp_fail_worst_c=" + std::to_string(fail.worst_c) +
          " qp_fail_worst_tag=" +
          std::string(fail.worst_tag[0] ? fail.worst_tag : "-") +
          " qp_fail_worst_viol=" + std::to_string(fail.worst_violation) +
          " qp_fail_path_remain_m=" + std::to_string(fail.path_remain_m) +
          " qp_fail_pitch_deg=" + std::to_string(fail.pitch_deg) +
          " qp_fail_ee_err_m=" + std::to_string(fail.ee_err_m);
    }
    dbg_status_->publish(status);
    status_pub_->publish(status);
  }

  void publish_arrived(bool arrived) {
    if (!arrived_pub_) {
      return;
    }
    std_msgs::msg::Bool msg;
    msg.data = arrived;
    arrived_pub_->publish(msg);
  }

  struct V2ExecSlot {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool active = false;
    bool protect_stop = false;
    bool recovering = false;
    bool claim_time_done = false;
    bool timeline_frozen = false;
    kmpc::TrajectoryPackage package{};
    double t0_s = 0.0;
    // Effective execution time at publish, plus wall stamp for 250 Hz fill-in
    // between 50 Hz ticks. When frozen, 250 Hz must hold freeze_q.
    double exec_t_s = 0.0;
    double exec_t_wall_s = 0.0;
    // Next unreleased StageBoundary — 250 Hz clamps before sampling so a TIP
    // between two 50 Hz ticks cannot produce LIFT commands.
    bool barrier_pending = false;
    bool barrier_released = false;
    double barrier_t_s = 0.0;
    uint32_t barrier_stage_id = 0;
    bool barrier_stop_required = false;
    bool barrier_wait_ack = false;
    kmpc::Vec18 barrier_q = kmpc::Vec18::Zero();
    uint64_t generation = 0;
    uint32_t path_id = 0;
    uint64_t task_digest = 0;
    kmpc::Vec18 q_offset = kmpc::Vec18::Zero();
    kmpc::Vec18 q_follow_hint = kmpc::Vec18::Zero();
    kmpc::Vec18 q_recovery_cmd = kmpc::Vec18::Zero();
    kmpc::Vec18 freeze_q = kmpc::Vec18::Zero();
    double pitch_excess_ceiling = 0.0;
  };

  struct V2SafetyArrived {
    std::atomic<bool> flag{false};
    std::atomic<uint64_t> generation{0};
    std::atomic<uint32_t> path_id{0};
  };

  // Precondition: stream_mu_ held. Swap a complete slot (no half-update).
  void publish_v2_slot(const V2ExecSlot& slot) {
    const int cur = v2_pub_idx_.load(std::memory_order_relaxed);
    const int next = 1 - cur;
    v2_slots_[static_cast<std::size_t>(next)] = slot;
    v2_pub_idx_.store(next, std::memory_order_release);
  }

  void publish_v2_slot_inactive() {
    V2ExecSlot empty;
    empty.active = false;
    empty.protect_stop = true;
    publish_v2_slot(empty);
  }

  void publish_com_marker(const kmpc::Vec3& p_com) {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = std::string(kmpc::kBaseFrame);
    marker.ns = "kmpc_com";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = p_com[0];
    marker.pose.position.y = p_com[1];
    marker.pose.position.z = p_com[2];
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.06;
    marker.scale.y = 0.06;
    marker.scale.z = 0.06;
    marker.color.r = 1.0f;
    marker.color.g = 0.75f;
    marker.color.b = 0.1f;
    marker.color.a = 0.9f;
    marker.frame_locked = true;
    dbg_com_->publish(marker);
  }

  std::string line_urdf_source_;
  bool line_urdf_xml_=false;
  kmpc::ros2io::LineSmoother line_smoother_;
  kmpc::ros2io::LineSmoother stream_pitch_filter_;
  bool stream_pitch_recovery_=false;
  double stream_pitch_direction_=1.0,stream_pitch_previous_=0,stream_pitch_initial_=0;
  int stream_pitch_ticks_=0;
  std::unique_ptr<kmpc::Kinematics> stream_kin_;
  std::unique_ptr<kmpc::Kinematics> line_plan_kin_;
  std::unique_ptr<kmpc::WholeBodyGuards> stream_guards_;
  kmpc::Vec18 stream_raw_ref_=kmpc::Vec18::Zero();
  kmpc::Vec3 stream_geo_start_=kmpc::Vec3::Zero(),stream_geo_target_=kmpc::Vec3::Zero();
  kmpc::Vec4 stream_geo_quat0_=kmpc::Vec4(0,0,0,1),stream_geo_quat1_=kmpc::Vec4(0,0,0,1);
  std::string stream_geo_frame_,stream_line_fault_;
  double stream_geo_previous_=0;
  bool stream_geo_ori_=false;
  uint64_t line_plan_builds_=0,line_plan_discards_=0;
  uint64_t line_limited_steps_=0;
  double line_commit_scale_=1.0;
  double line_commit_peak_ratio_=0.0;
  uint64_t last_reported_line_stop_gen_=0;
  std::string last_reported_line_stop_reason_;
  std::future<kmpc::CartesianTracker::LineRun> line_future_;
  uint64_t line_pending_gen_=0;
  bool line_pending_left_=false;
  bool line_pending_waist_locked_=false;
  kmpc::Vec3 line_pending_xyz_=kmpc::Vec3::Zero();
  kmpc::Vec4 line_pending_quat_=kmpc::Vec4::Zero();
  kmpc::Vec18 line_pending_q_=kmpc::Vec18::Zero();
  // V2_CHAIN resident planner + 50/250 Hz layers.
  std::unique_ptr<kmpc::PlannerWorker> planner_worker_;
  std::unique_ptr<kmpc::TrajectoryFollower> follower_;
  std::unique_ptr<kmpc::SafetySupervisor> safety_supervisor_;
  std::string planner_name_ = "unified_multistage_v2";
  bool use_v2_chain_ = true;
  bool v2_request_plan_ = false;
  kmpc::V2SubmitGate v2_gate_{};
  double v2_follow_t0_ = 0.0;
  bool stream_v2_ = false;
  bool stream_v2_claim_arrived_ = false;
  V2SafetyArrived v2_safety_arrived_{};
  kmpc::TrajectoryPackage stream_v2_package_{};
  kmpc::Vec18 stream_v2_q_follow_ = kmpc::Vec18::Zero();
  kmpc::Vec18 stream_v2_q_plan_ = kmpc::Vec18::Zero();
  kmpc::Vec18 stream_v2_q_ref_ = kmpc::Vec18::Zero();
  std::array<V2ExecSlot, 2> v2_slots_{};
  std::atomic<int> v2_pub_idx_{0};
  kmpc::Vec18 v2_meas_q_ = kmpc::Vec18::Zero();
  double v2_meas_stamp_s_ = 0.0;
  double v2_meas_source_stamp_s_ = 0.0;
  bool v2_meas_valid_ = false;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr left_segment_sub_,right_segment_sub_;
  rclcpp::Subscription<kmpc_msgs::msg::MotionSequence>::SharedPtr motion_sequence_sub_;
  rclcpp::Subscription<kmpc_msgs::msg::StageAck>::SharedPtr stage_ack_sub_;
  kmpc::Config cfg_;
  std::unique_ptr<kmpc::Kinematics> kin_;
  std::unique_ptr<kmpc::CartesianTracker> tracker_;
  std::unique_ptr<kmpc::UpperPlanner> planner_;
  std::unique_ptr<kmpc::SafetyFilter> safety_;
  std::unique_ptr<kmpc::ros2io::JointSpaceTracker> joint_tracker_;
  std::unique_ptr<kmpc::ros2io::TaskReceiver> task_;
  std::unique_ptr<kmpc::ros2io::PlannerReceiver> planner_rx_;
  std::unique_ptr<kmpc::ros2io::CommandArbiter> arbiter_;
  std::unique_ptr<kmpc::ros2io::StateAggregator> state_;

  kmpc::Vec6 last_ee_err_ = kmpc::Vec6::Zero();
  int fail_count_ = 0;
  bool demo_wave_ = false;
  double demo_amp_ = 0.5;
  double demo_period_ = 8.0;
  double hold_deadband_ = 0.01;
  double dq_scale_ = 1.0;
  double line_dq_scale_ = 0.60;
  kmpc::CartesianTracker::LinePaceMode line_pace_mode_ =
      kmpc::CartesianTracker::kDefaultLinePaceMode;
  double line_brake_distance_m_ = 0.08;
  double line_brake_floor_ = 0.20;
  kmpc::CartesianTracker::LineBrakeProfile line_brake_profile_ =
      kmpc::CartesianTracker::kDefaultLineBrakeProfile;
  int line_qp_max_iter_ = 200;
  kmpc::LineQpFollowMode line_qp_follow_mode_ = kmpc::LineQpFollowMode::A;
  kmpc::CartesianTracker::LineChainProfile line_chain_profile_ =
      kmpc::CartesianTracker::LineChainProfile::Legacy;
  kmpc::LineFinalGateLimits final_gate_limits_{};
  bool line_pose_pref_enable_ = false;
  bool line_pitch_straighten_enable_ = true;
  bool line_qp_pitch_monotonic_enable_ = true;
  bool reverse_last_line_enable_ = true;
  bool line_direction_strict_enable_ = true;
  bool line_shadow_diag_enable_ = true;
  kmpc::DirectionEnvelope stream_dir_env_{};
  bool line_diag_enable_ = false;
  bool layer_diag_on_ = false;
  uint64_t run_id_ = 0;
  std::atomic<uint64_t> control_tick_id_{0};
  std::atomic<uint64_t> stream_tick_id_{0};
  uint64_t stream_segment_tick_ = 0;  // tick of last accepted segment
  uint64_t stream_goal_gen_ = 0;
  uint64_t stream_diag_seq_ = 0;
  uint64_t control_diag_seq_ = 0;
  uint64_t exported_stream_drops_ = 0;
  uint64_t exported_control_drops_ = 0;
  uint64_t exported_tracker_overflows_ = 0;
  size_t exported_ring_n_ = 0;
  uint64_t sdk_note_drops_ = 0;
  uint64_t exported_sdk_drops_ = 0;
  std::string layer_diag_path_;
  std::string qp_fail_dump_dir_;
  std::string config_snapshot_path_;
  std::ofstream layer_diag_out_;
  std::deque<kmpc::ros2io::LayerDiagSample> sdk_notes_;
  // Separate SPSC queues: 50 Hz control writer vs 250 Hz stream writer.
  kmpc::ros2io::LayerDiagQueue<8192> control_diag_q_;
  kmpc::ros2io::LayerDiagQueue<8192> stream_diag_q_;
  kmpc::QpFailDumpWorker<
      std::unique_ptr<kmpc::CartesianTracker::LineQpFailDump>>
      qp_dump_worker_{4};
  bool stream_dedicated_thread_ = true;
  double stream_debug_q_hz_ = 50.0;
  double stream_hz_ = 250.0;
  double interp_catch_s_ = 0.02;
  double interp_accel_s_ = 0.50;
  double interp_jerk_s_ = 0.08;
  double interp_dt_ = 1.0 / 250.0;
  mutable std::mutex stream_mu_;
  std::atomic<bool> stream_thread_stop_{false};
  std::thread stream_thread_;
  std::chrono::steady_clock::time_point last_debug_q_at_{};
  bool have_last_debug_q_ = false;
  kmpc::StreamInterpolator interpolator_;
  bool stream_hold_ = false;
  // 直线段不进各轴插补。from/to 都在路径上，u 在一个控制周期里从 0 走到 1。
  bool stream_line_ = false;
  int stream_line_gen_ = -1;
  double stream_line_u_ = 1.0;
  kmpc::Vec18 stream_line_from_ = kmpc::Vec18::Zero();
  kmpc::Vec18 stream_line_to_ = kmpc::Vec18::Zero();
  // Depth-1 queue: confirm immediately, arm only when active chord reaches u=1.
  bool stream_chord_pending_ = false;
  kmpc::Vec18 stream_chord_pending_to_ = kmpc::Vec18::Zero();
  uint64_t stream_chord_queue_waits_ = 0;
  uint64_t stream_chord_queue_arms_ = 0;
  // 无目标钉住：不往伺服口重复发同一包。有球或有关节目标时为 false。
  bool stream_mute_ = false;
  bool alarm_block_ = false;
  bool feedback_block_ = false;
  bool alarm_clear_since_valid_ = false;
  std::chrono::steady_clock::time_point alarm_clear_since_{};
  uint64_t alarm_block_generation_ = 0;
  bool stream_fixed_ = false;
  bool stream_have_qp_ = false;
  kmpc::Vec18 stream_qp_to_ = kmpc::Vec18::Zero();
  kmpc::Vec18 stream_star_ = kmpc::Vec18::Zero();
  kmpc::Vec18 stream_q_hold_ = kmpc::Vec18::Zero();
  double t0_ = 0.0;
  bool cart_frame_warned_ = false;
  bool last_use_cart_ = false;
  kmpc::Vec18 q_des_star_ = kmpc::Vec18::Zero();
  kmpc::QDesFlags last_qdes_flags_{};

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr stream_timer_;
  rclcpp::TimerBase::SharedPtr layer_diag_timer_;
  rclcpp::TimerBase::SharedPtr qp_fail_dump_timer_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr alarm_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr q_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr left_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr right_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr clear_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr clear_left_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr clear_right_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr arrived_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr waist_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr q_calib_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr dbg_q_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr dbg_dq_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr dbg_ee_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr dbg_status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr dbg_com_;
  std::string stream_topic_name_;
  uint64_t servo_seq_ = 0;
#ifdef KMPC_HAVE_XOIP
  rclcpp::Publisher<xoip_ros2::msg::MultiJointServoCommand>::SharedPtr servo_pub_;
  rclcpp::Subscription<xoip_ros2::msg::StreamStatus>::SharedPtr stream_status_sub_;
  rclcpp::Client<xoip_ros2::srv::ClearRoute>::SharedPtr clear_route_cli_;
#endif
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<KmpcNode>());
  } catch (const std::exception& exc) {
    fprintf(stderr, "kmpc_node FAILED: %s\n", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
