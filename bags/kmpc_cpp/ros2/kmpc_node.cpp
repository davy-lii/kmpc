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
#include "kmpc/kinematics.hpp"
#include "kmpc/safety.hpp"
#include "kmpc/stream_interpolator.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"
#include "kmpc/upper_planner.hpp"
#include "kmpc/whole_body_guards.hpp"
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
    stream_guards_=std::make_unique<kmpc::WholeBodyGuards>(cfg_.params,cfg_.q_pref[kmpc::kIdxYaw]);
    tracker_ = std::make_unique<kmpc::CartesianTracker>(*kin_, cfg_);
    planner_ = std::make_unique<kmpc::UpperPlanner>(
        *kin_, cfg_.params.calib_tol_rad, cfg_.params.com_xmin,
        cfg_.params.com_xmax, cfg_.params.com_margin);
    safety_ = std::make_unique<kmpc::SafetyFilter>(
        cfg_.limits, cfg_.params.dt, cfg_.params.q_limit_margin);
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
    line_diag_enable_ = bool_param("line_diag_enable", false);
    tracker_->set_line_diag_enable(line_diag_enable_);
    run_id_ = kmpc::ros2io::layer_diag_mono_ns();
    tracker_->set_line_diag_run_id(run_id_);
    layer_diag_path_ = str_or("layer_diag_path", "");
    qp_fail_dump_dir_ = str_or(
        "qp_fail_dump_dir", "/home/nvidia/ljw/issues/kmpc/qp_fail_dumps");
    config_snapshot_path_ = str_or("config_snapshot_path", "");
    t0_ = now().seconds();
    configure_command_io();
    stream_topic_name_ = str_or("stream_topic", "/xoip/robot/multi_joint_servo_cmd");

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
    };
    left_segment_sub_=create_subscription<geometry_msgs::msg::PoseArray>("/kmpc/segment_left",10,
        [segment_callback](geometry_msgs::msg::PoseArray::ConstSharedPtr m){segment_callback(true,m);});
    right_segment_sub_=create_subscription<geometry_msgs::msg::PoseArray>("/kmpc/segment_right",10,
        [segment_callback](geometry_msgs::msg::PoseArray::ConstSharedPtr m){segment_callback(false,m);});
    clear_sub_ = create_subscription<std_msgs::msg::Empty>(
        str_or("clear_targets_topic", "/kmpc/clear_targets"), 10,
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          task_->clear_cartesian();
          tracker_->reset();
          publish_arrived(false);
          std::lock_guard<std::mutex> lock(stream_mu_);
          stream_clear_modes();
        });
    clear_left_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/kmpc/clear_left", 10,
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          const bool active=task_->target(true).has_value();
          task_->clear_left();
          if(active) tracker_->reset();
          publish_arrived(false);
        });
    clear_right_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/kmpc/clear_right", 10,
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          const bool active=task_->target(false).has_value();
          task_->clear_right();
          if(active) tracker_->reset();
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
    if (layer_diag_on_) {
      drain_layer_diag();
    }
  }

 private:
  void declare_params() {
    declare_parameter<std::string>("robot_description", "");
    declare_parameter<std::string>("joint_map_file", "");
    declare_parameter<std::string>("joint_limits_file", "");
    declare_parameter<std::string>("q_pref_file", "");
    declare_parameter<std::string>("kmpc_params_file", "");
    declare_parameter<std::string>("q_pref_station", "default");
    declare_parameter("auto_enable", false);
    declare_parameter("demo_wave_waist", false);
    declare_parameter("demo_wave_amp", 0.5);
    declare_parameter("demo_wave_period", 8.0);
    declare_parameter("state_timeout_sec", 0.1);
    declare_parameter("hold_deadband", 0.01);
    declare_parameter("waist_locked", false);
    declare_parameter("require_xoip", true);
    declare_parameter("dq_scale", 1.0);
    declare_parameter("line_dq_scale", 0.60);
    declare_parameter<std::string>(
        "line_pace_mode",
        kmpc::CartesianTracker::line_pace_mode_name(
            kmpc::CartesianTracker::kDefaultLinePaceMode));
    declare_parameter("line_brake_distance_m", 0.08);
    declare_parameter("line_brake_floor", 0.20);
    declare_parameter<std::string>(
        "line_brake_profile",
        kmpc::CartesianTracker::line_brake_profile_name(
            kmpc::CartesianTracker::kDefaultLineBrakeProfile));
    declare_parameter("line_qp_max_iter", 200);
    declare_parameter("line_pose_pref_enable", false);
    declare_parameter("line_pitch_straighten_enable", true);
    declare_parameter("line_qp_pitch_monotonic_enable", true);
    declare_parameter("reverse_last_line_enable", true);
    declare_parameter("line_direction_strict_enable", true);
    declare_parameter("line_diag_enable", false);
    declare_parameter<std::string>("layer_diag_path", "");
    declare_parameter<std::string>(
        "qp_fail_dump_dir", "/home/nvidia/ljw/issues/kmpc/qp_fail_dumps");
    declare_parameter<std::string>("config_snapshot_path", "");
    declare_parameter("stream_hz", 250.0);
    declare_parameter("interp_catch_s", 0.02);
    declare_parameter("interp_accel_s", 0.50);
    declare_parameter("interp_jerk_s", 0.08);
    declare_parameter("stream_dedicated_thread", true);
    declare_parameter("stream_debug_q_hz", 50.0);
    declare_parameter<std::string>("stream_topic", "/xoip/robot/multi_joint_servo_cmd");
    declare_parameter<std::string>("joint_states_topic", "/joint_states");
    declare_parameter<std::string>("enable_topic", "/kmpc/enable");
    declare_parameter<std::string>("mode_topic", "/kmpc/mode");
    declare_parameter<std::string>("q_target_topic", "/kmpc/q_target");
    declare_parameter<std::string>("arrived_topic", "/kmpc/arrived");
    declare_parameter<std::string>("target_left_topic", "/kmpc/target_left");
    declare_parameter<std::string>("target_right_topic", "/kmpc/target_right");
    declare_parameter<std::string>("clear_targets_topic", "/kmpc/clear_targets");
    declare_parameter<std::string>("waist_locked_topic", "/kmpc/waist_locked");
    declare_parameter<std::string>(
        "tracking_state_topic", "/kmpc/tracking_state");
    declare_parameter<std::string>("q_calib_topic", "/kmpc/q_calib");
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
    if(!q.allFinite()) return "stream_filtered_nonfinite";
    if((q.array()<cfg_.limits.q_min.array()+cfg_.params.q_limit_margin-1e-7).any() ||
       (q.array()>cfg_.limits.q_max.array()-cfg_.params.q_limit_margin+1e-7).any()) return "stream_filtered_joint_limit";
    const double limit=cfg_.params.torso_pitch_max_deg*kmpc::kPi/180;
    // Ordinary final gate: hard + recovery extra + round eps only.
    // Never add kStartPitchAdmitRad here — that admit is for start-outside
    // recovery / validate residue, not a permanent 250 Hz widen.
    if(limit>0 &&
       std::abs(q.head<3>().sum())>
           limit+pitch_extra+kmpc::kPitchRoundEpsRad) {
      return "stream_filtered_pitch";
    }
    if(!stream_guards_->com_xy_inside(stream_kin_->com(q))) return "stream_filtered_com";
    const auto pose=stream_kin_->fk(q,stream_geo_frame_);
    const kmpc::Vec3 ab=stream_geo_target_-stream_geo_start_;
    const double along=(pose.col(3).head<3>()-stream_geo_start_).dot(ab);
    const double t=std::clamp(along/std::max(ab.squaredNorm(),1e-12),0.0,1.0);
    if((pose.col(3).head<3>()-stream_geo_start_-t*ab).norm()>.003) return "stream_filtered_path";
    if(along<stream_geo_previous_-1e-6*std::max(ab.norm(),1e-6)) return "stream_filtered_progress";
    if(stream_geo_ori_ && kmpc::ori_error_xyzw(kmpc::quat_slerp_xyzw(stream_geo_quat0_,stream_geo_quat1_,t),
       kmpc::rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm()>cfg_.params.arrived_ori_rad+1e-6) return "stream_filtered_orientation";
    stream_geo_previous_=along;return nullptr;
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
    } else if (stream_line_) {
      const double qp_dt = std::max(cfg_.params.dt, 1e-3);
      stream_line_u_ += interp_dt_ / qp_dt;
      if (stream_line_u_ > 1.0) {
        stream_line_u_ = 1.0;
      }
      const double u = stream_line_u_;
      stream_raw_ref_=(1.0-u)*stream_line_from_+u*stream_line_to_;
      double extra=0;
      const char* fault=nullptr;
      if(stream_pitch_recovery_) {
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
      if (layer_diag_on_) {
        const uint32_t line_gen = stream_line_gen_ < 0 ? 0u : static_cast<uint32_t>(stream_line_gen_);
        push_stream_diag(kmpc::ros2io::LayerDiagLayer::StreamRawRef, stream_raw_ref_,
                         stream_goal_gen_, line_gen, "raw", 0, false);
        push_stream_diag(kmpc::ros2io::LayerDiagLayer::LineSmoother, filtered,
                         stream_goal_gen_, line_gen, "smoothed", 0, false);
      }
      if(!fault) fault=stream_filtered_fault(filtered,extra);
      // Final-send hard gate: never publish a waist/pitch reversal vs the
      // accepted direction contract, even if raw/QP were monotone.
      if(!fault && line_direction_strict_enable_ && stream_dir_env_.active) {
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
      if(fault) {
        RCLCPP_ERROR(get_logger(),"line stream fault gen=%llu reason=%s pitch=%.9fdeg limit=%.9fdeg start_excess=%.9frad allowed_extra=%.9frad recovery_ticks=%d",
          static_cast<unsigned long long>(stream_line_gen_),fault,filtered.head<3>().sum()*180/kmpc::kPi,
          cfg_.params.torso_pitch_max_deg,stream_pitch_initial_,extra,stream_pitch_ticks_);
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
    stream_hold_ = false;
    stream_fixed_ = false;
    stream_line_from_ = continuing?stream_raw_ref_:interpolator_.q_ref();
    if(!continuing) {
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
    const kmpc::Vec18 cap=tracker_->effective_dq_max()*cfg_.params.dt*line_dq_scale_;
    // A genuinely invalid solve still fails. A valid solve can be slightly
    // longer than one tick relative to the reference advanced during solving.
    if (((q_path-planning_from).cwiseAbs().array()>cap.array()+1e-6).any()) {
      for(int j=0;j<kmpc::kNDof;++j) if(std::abs(q_path[j]-stream_line_from_[j])>cap[j]+1e-6) {
        RCLCPP_ERROR(get_logger(),"line step limit gen=%llu joint=%d delta=%.10g cap=%.10g planned_delta=%.10g reference_advance=%.10g",
          static_cast<unsigned long long>(tracker_->goal_generation()),j,q_path[j]-stream_line_from_[j],cap[j],q_path[j]-planning_from[j],stream_line_from_[j]-planning_from[j]);
      }
      tracker_->refuse_line("stream_first_step_limit");
      stream_enter_hold(interpolator_.q_ref());
      return;
    }
    const auto bounded=kmpc::ros2io::bound_line_stream_step(stream_line_from_,q_path,cap);
    if(!bounded) {
      tracker_->refuse_line("stream_nonfinite_or_invalid_cap");
      stream_enter_hold(interpolator_.q_ref()); return;
    }
    line_commit_scale_=bounded->scale;
    line_commit_peak_ratio_=0.0;
    for(int j=0;j<kmpc::kNDof;++j) if(cap[j]>0.0)
      line_commit_peak_ratio_=std::max(line_commit_peak_ratio_,std::abs(bounded->target[j]-stream_line_from_[j])/cap[j]);
    if(line_commit_scale_<1.0-1e-9) ++line_limited_steps_;
    // Validate the actual chord after limiting; velocity, line tube and
    // forward progress remain acceptance conditions.
    const auto check=tracker_->line_stream_segment_check(stream_line_from_,bounded->target);
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
        (stream_line_from_-sent).cwiseAbs().maxCoeff(),
        vec_csv(stream_line_from_).c_str(),vec_csv(bounded->target).c_str(),
        vec_csv(sent).c_str());
      // Preserve the concrete gate (pitch/tube/progress/…) — never collapse to
      // the historical stream_tube_or_progress alias that hid pitch hard-box stops.
      tracker_->refuse_line(std::string("stream_") + check.reason);
      stream_enter_hold(interpolator_.q_ref());
      return;
    }
    stream_line_ = true;
    stream_line_to_ = bounded->target;
    stream_line_u_ = 0.0;
    interpolator_.clear_q_star();
    interpolator_.clear_segment();
    interpolator_.set_stop_at_goal(false);
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
        // The next QP segment follows the last committed endpoint. Anchoring
        // it to a moving mid-segment sample creates an alternating speed.
        line_solve_from=(stream_line_ && !needs_build && !line_future_.valid())?stream_line_to_:q0;
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
        const bool current=line_pending_gen_==target.generation && line_pending_left_==is_left &&
            (line_pending_xyz_-target.xyz).norm()<0.008 &&
            kmpc::ori_error_xyzw(line_pending_quat_,target.quat_xyzw).norm()<kmpc::kTargetJumpOriRad &&
            line_pending_waist_locked_==result.command.waist_locked &&
            (q0-line_pending_q_).cwiseAbs().maxCoeff()<1e-9;
        if(current && !tracker_->line_refused()) {
          const double moved=(*q_meas-line_pending_q_).cwiseAbs().maxCoeff();
          if(moved>0.1*kmpc::kPi/180.0 && stream_mute_)
            tracker_->refuse_line("line_start_moved_during_plan");
          else {
            if(plan.on && !plan.samples.empty()) {
              const auto start_pose=kin_->fk(plan.samples.front(),plan.frame);
              std::lock_guard<std::mutex> lock(stream_mu_);
              stream_geo_start_=start_pose.col(3).head<3>();stream_geo_target_=plan.target;
              stream_geo_quat0_=kmpc::rot_to_quat_xyzw(start_pose.block<3,3>(0,0));
              stream_geo_quat1_=plan.quat;stream_geo_ori_=plan.ori_on;stream_geo_frame_=plan.frame;
              stream_geo_previous_=0.0;stream_line_fault_.clear();
              const double pitch=plan.samples.front().head<3>().sum();
              const double limit=cfg_.params.torso_pitch_max_deg*kmpc::kPi/180;
              const double excess=std::max(0.0,std::abs(pitch)-limit);
              stream_pitch_recovery_=limit>0 && excess>1e-6;
              stream_pitch_direction_=pitch<0?-1.0:1.0;
              stream_pitch_initial_=stream_pitch_previous_=excess; stream_pitch_ticks_=0;
              kmpc::Vec18 budget=kmpc::Vec18::Zero();budget[0]=excess;
              stream_pitch_filter_.reset(budget);
            }
            // Worker cache mutations stay local; mirror consume/reject on main.
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
          if(stream_mute_ && !tracker_->line_active()) reset_interpolator_to(*q_meas);
          q0=interpolator_.q_ref();stream_enter_hold(q0);
        }
        line_solve_from=q0;tracker_->set_line_reference(q0);++line_plan_builds_;
        line_pending_gen_=target.generation; line_pending_left_=is_left;
        line_pending_xyz_=target.xyz; line_pending_quat_=target.quat_xyzw; line_pending_q_=q0;
        const auto config=cfg_; const auto source=line_urdf_source_; const bool xml=line_urdf_xml_;
        const auto upper=result.command;
        line_pending_waist_locked_=upper.waist_locked;
        const bool pose_pref=line_pose_pref_enable_;
        const bool pitch_straighten=line_pitch_straighten_enable_;
        const bool qp_pitch_mono=line_qp_pitch_monotonic_enable_;
        const bool reverse_last=reverse_last_line_enable_;
        const bool direction_strict=line_direction_strict_enable_;
        const bool diag=line_diag_enable_;
        const auto pace_mode=line_pace_mode_;
        const double brake_m=line_brake_distance_m_;
        const double brake_floor=line_brake_floor_;
        const auto brake_profile=line_brake_profile_;
        const int qp_max_iter=line_qp_max_iter_;
        const kmpc::Vec18 q_pref=cfg_.q_pref;
        const auto line_cache=tracker_->last_line_cache();
        const auto successor_plan=tracker_->successor_plan();
        const uint64_t run_id=run_id_;
        line_future_=std::async(std::launch::async,
            [config,source,xml,target,q0,is_left,upper,pose_pref,pitch_straighten,
             qp_pitch_mono,reverse_last,direction_strict,diag,pace_mode,brake_m,brake_floor,
             brake_profile,qp_max_iter,q_pref,line_cache,successor_plan,run_id]() {
          kmpc::Kinematics model(source,xml?kmpc::Kinematics::UrdfInput::Xml:kmpc::Kinematics::UrdfInput::Path);
          kmpc::CartesianTracker worker(model,config);
          worker.set_upper_commands(upper);
          worker.set_line_pose_pref(pose_pref);
          worker.set_line_pose_pref_q(q_pref);
          worker.set_line_pitch_straighten_enable(pitch_straighten);
          worker.set_line_qp_pitch_monotonic_enable(qp_pitch_mono);
          worker.set_reverse_last_line_enable(reverse_last);
          worker.set_line_direction_strict_enable(direction_strict);
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
          // prepare_line already copies successor into plan; keep explicit.
          plan.successor = worker.successor_plan();
          return plan;
        });
        publish_cmd(q0,kmpc::Vec18::Zero(),0.0,true); return;
      }
    }

    kmpc::Vec18 dq = kmpc::Vec18::Zero();
    kmpc::Vec18 q_cmd = kmpc::Vec18::Zero();
    try {
      if (use_cart) {
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
          } else {
            stream_on_line(q_path,line_solve_from);
            q_cmd = tracker_->line_refused() ? stream_q_hold_ : stream_line_to_;
            if (tracker_->line_refused()) dq.setZero();
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

  void push_stream_diag(
      kmpc::ros2io::LayerDiagLayer layer, const kmpc::Vec18& q, uint64_t gen,
      uint32_t line_gen, const char* note, uint64_t seq_override, bool use_override) {
    if (!layer_diag_on_) {
      return;
    }
    kmpc::ros2io::LayerDiagSample sample;
    sample.mono_ns = kmpc::ros2io::layer_diag_mono_ns();
    sample.recv_mono_ns = sample.mono_ns;
    sample.source_stamp_ns = -1;
    sample.run_id = run_id_;
    sample.goal_generation = gen;
    sample.seq = use_override ? seq_override : ++stream_diag_seq_;
    sample.line_gen = line_gen;
    sample.layer = layer;
    sample.next_layer = kmpc::ros2io::layer_diag_next(layer);
    kmpc::ros2io::layer_diag_copy_q(&sample, q);
    kmpc::ros2io::layer_diag_note(&sample, note);
    stream_diag_q_.try_push(sample);
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

  void drain_qp_fail_dump() {
    if (!tracker_ || qp_fail_dump_dir_.empty()) {
      return;
    }
    // Dump is ~0.6MB; keep it off the stack.
    auto dump = std::make_unique<kmpc::CartesianTracker::LineQpFailDump>();
    if (!tracker_->take_line_qp_fail_dump(dump.get()) || !dump->valid) {
      return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(qp_fail_dump_dir_, ec);
    if (ec) {
      RCLCPP_ERROR(get_logger(), "qp fail dump mkdir %s failed: %s",
                   qp_fail_dump_dir_.c_str(), ec.message().c_str());
      return;
    }
    char name[256];
    std::snprintf(name, sizeof(name), "qp_fail_gen%llu_mono%llu.json",
                  static_cast<unsigned long long>(dump->goal_generation),
                  static_cast<unsigned long long>(dump->mono_ns));
    const fs::path path = fs::path(qp_fail_dump_dir_) / name;
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
    for (;;) {
      const size_t n = stream_diag_q_.try_pop(batch, 64);
      for (size_t i = 0; i < n; ++i) {
        layer_diag_out_ << kmpc::ros2io::layer_diag_jsonl(batch[i]) << '\n';
      }
      if (n < 64) {
        break;
      }
    }
    const uint64_t drops = stream_diag_q_.drops();
    if (drops != exported_stream_drops_) {
      layer_diag_out_ << kmpc::ros2io::layer_diag_overflow_jsonl(run_id_, drops) << '\n';
      exported_stream_drops_ = drops;
    }
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
        kmpc::ros2io::layer_diag_copy_q(&sample, ev.q);
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
    add("line_freeze_distance_m", "0.08", "m", "compile_const", "=0.08", true);
    ros_bool("line_pitch_straighten_enable", true);
    ros_bool("line_qp_pitch_monotonic_enable", true);
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
      // Off the 50/250 Hz hot path: drain a pre-filled QP input dump to disk.
      drain_qp_fail_dump();
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
        " line_candidate=" + (tracker_->line_candidate().empty()?"none":tracker_->line_candidate()) +
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
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr left_segment_sub_,right_segment_sub_;
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
  bool line_pose_pref_enable_ = false;
  bool line_pitch_straighten_enable_ = true;
  bool line_qp_pitch_monotonic_enable_ = true;
  bool reverse_last_line_enable_ = true;
  bool line_direction_strict_enable_ = true;
  kmpc::DirectionEnvelope stream_dir_env_{};
  bool line_diag_enable_ = false;
  bool layer_diag_on_ = false;
  uint64_t run_id_ = 0;
  uint64_t stream_goal_gen_ = 0;
  uint64_t stream_diag_seq_ = 0;
  uint64_t exported_stream_drops_ = 0;
  uint64_t exported_tracker_overflows_ = 0;
  size_t exported_ring_n_ = 0;
  uint64_t sdk_note_drops_ = 0;
  uint64_t exported_sdk_drops_ = 0;
  std::string layer_diag_path_;
  std::string qp_fail_dump_dir_;
  std::string config_snapshot_path_;
  std::ofstream layer_diag_out_;
  std::deque<kmpc::ros2io::LayerDiagSample> sdk_notes_;
  kmpc::ros2io::LayerDiagQueue<8192> stream_diag_q_;
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
