#pragma once
// 对应 Python task_receiver / planner_receiver / command_arbiter / state_aggregator。
// 非阻塞写、控制拍读。无 rclcpp，消息转换留在 kmpc_node.cpp。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmpc/types.hpp"
#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/upper_planner.hpp"

namespace kmpc {
namespace ros2io {

struct CartesianTarget {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 xyz = Vec3::Zero();
  Vec4 quat_xyzw = Vec4(0.0, 0.0, 0.0, 1.0);
  std::string frame_id = std::string(kBaseFrame);
  uint64_t generation = 0;
  std::optional<LineTarget> cascade;

};

// MotionSequence stage payload without ROS types (conversion stays in kmpc_node).
using StageTarget = TaskStage;
static constexpr std::size_t kMaxMotionStages = 3;

class CommandArbiter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit CommandArbiter(bool enabled = false) : enabled_(enabled) {}

  void set_enabled(bool value) {
    std::lock_guard<std::mutex> lock(mu_);
    enabled_ = value;
  }

  bool enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return enabled_;
  }

 private:
  mutable std::mutex mu_;
  bool enabled_ = false;
};

class TaskReceiver {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TaskReceiver() = default;

  void on_mode(std::string_view mode) {
    if (!is_valid_mode(mode)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    mode_ = std::string(mode);
  }

  void on_q_target(const Vec18& q) {
    std::lock_guard<std::mutex> lock(mu_);
    q_target_ = q;
  }

  void on_target_left(const CartesianTarget& t) {
    std::lock_guard<std::mutex> lock(mu_);
    CartesianTarget next=t;
    next.generation=left_?left_->generation:++generation_;
    next.cascade=left_?left_->cascade:std::nullopt;
    left_=next;
    saw_cartesian_=true;
  }

  void on_target_right(const CartesianTarget& t) {
    std::lock_guard<std::mutex> lock(mu_);
    CartesianTarget next=t;
    next.generation=right_?right_->generation:++generation_;
    next.cascade=right_?right_->cascade:std::nullopt;
    right_=next;
    saw_cartesian_=true;
  }

  // Single message installs current pose, real next pose and generation atomically.
  // PoseArray adapter also fills motion_ with 1 (or 2 if cascade) StageTarget entries.
  bool on_segment(bool left, const CartesianTarget& t) {
    if (!t.generation || t.frame_id!=kBaseFrame || !t.xyz.allFinite() ||
        !t.quat_xyzw.allFinite() || t.quat_xyzw.norm()<1e-6) return false;
    if (t.cascade && (t.cascade->frame!=kBaseFrame || !t.cascade->xyz.allFinite() ||
        !t.cascade->quat_xyzw.allFinite() || t.cascade->quat_xyzw.norm()<1e-6)) return false;
    std::lock_guard<std::mutex> lock(mu_);
    auto& current=left?left_:right_;
    if (current && current->generation==t.generation) {
      current->xyz=t.xyz; current->quat_xyzw=t.quat_xyzw;
      if (motion_ && motion_->generation == t.generation && !motion_->stages.empty()) {
        motion_->stages[0].position = t.xyz;
        motion_->stages[0].quat = t.quat_xyzw;
      }
      return true; // Same segment drift cannot replace its cascade or rearm.
    }
    if (t.generation<=generation_) return false;
    generation_=t.generation;
    mode_=std::string(kModeSingleEe);
    if(left) { left_=t; right_.reset(); } else { right_=t; left_.reset(); }
    saw_cartesian_=true;
    motion_ = motion_task_from_segment(left, t);
    return true;
  }

  // Atomic generation replace for typed MotionSequence (1..3 stages). Rejects >3.
  bool on_motion_sequence(const MotionTask& task) {
    if (!task.generation || task.stages.empty() ||
        task.stages.size() > kMaxMotionStages) {
      return false;
    }
    const std::string frame =
        task.frame.empty() ? std::string(kBaseFrame) : task.frame;
    if (frame != kBaseFrame) {
      return false;
    }
    for (const auto& st : task.stages) {
      if (!st.position.allFinite() || !st.quat.allFinite() ||
          st.quat.norm() < 1e-6 || !(st.position_tol_m > 0.0) ||
          !(st.orientation_tol_rad > 0.0)) {
        return false;
      }
      if (st.geometry != TaskGeometry::Lin && st.geometry != TaskGeometry::Ptp &&
          st.geometry != TaskGeometry::FollowCurve) {
        return false;
      }
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (task.generation <= generation_) {
      return false;
    }
    generation_ = task.generation;
    mode_ = std::string(kModeSingleEe);
    MotionTask next = task;
    next.frame = frame;
    motion_ = next;
    // Mirror first 1–2 stages into legacy CartesianTarget so PoseArray path stays usable.
    CartesianTarget cart = cartesian_from_stages(next);
    if (task.left) {
      left_ = cart;
      right_.reset();
    } else {
      right_ = cart;
      left_.reset();
    }
    saw_cartesian_ = true;
    waist_locked_hint_ = task.waist_locked;
    return true;
  }

  std::optional<MotionTask> motion_task() const {
    std::lock_guard<std::mutex> lock(mu_);
    return motion_;
  }

  bool waist_locked_hint() const {
    std::lock_guard<std::mutex> lock(mu_);
    return waist_locked_hint_;
  }
  std::pair<std::optional<CartesianTarget>,std::optional<CartesianTarget>> targets() const {
    std::lock_guard<std::mutex> lock(mu_);
    return {left_,right_};
  }

  bool clear_segment(bool left, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& target=left?left_:right_;
    if(!target || target->generation!=generation) return false;
    target.reset();
    if (motion_ && motion_->generation == generation && motion_->left == left) {
      motion_.reset();
    }
    saw_cartesian_=left_.has_value() || right_.has_value();
    return true;
  }

  void clear_cartesian() {
    std::lock_guard<std::mutex> lock(mu_);
    left_.reset();
    right_.reset();
    motion_.reset();
    saw_cartesian_ = false;
  }

  // Accepted generation watermark. Alarm resume uses this to drop only
  // targets that were already known when the alarm latched.
  uint64_t generation() const {
    std::lock_guard<std::mutex> lock(mu_);
    return generation_;
  }

  // Drop cartesian targets with generation <= gen. If none remain, rewind
  // generation_ to gen so a later segment with a newer id can still install
  // (alarm resume must not permanently reject the enable-time goal).
  void drop_cartesian_up_to(uint64_t gen) {
    std::lock_guard<std::mutex> lock(mu_);
    if (left_ && left_->generation <= gen) {
      left_.reset();
    }
    if (right_ && right_->generation <= gen) {
      right_.reset();
    }
    if (motion_ && motion_->generation <= gen) {
      motion_.reset();
    }
    saw_cartesian_ = left_.has_value() || right_.has_value();
    if (!saw_cartesian_) {
      generation_ = gen;
    }
  }

  void clear_left() {
    std::lock_guard<std::mutex> lock(mu_);
    left_.reset();
    if (motion_ && motion_->left) {
      motion_.reset();
    }
    if (!right_) {
      saw_cartesian_ = false;
    }
  }

  void clear_right() {
    std::lock_guard<std::mutex> lock(mu_);
    right_.reset();
    if (motion_ && !motion_->left) {
      motion_.reset();
    }
    if (!left_) {
      saw_cartesian_ = false;
    }
  }

  bool saw_cartesian_target() const {
    std::lock_guard<std::mutex> lock(mu_);
    return saw_cartesian_;
  }

  std::string mode() const {
    std::lock_guard<std::mutex> lock(mu_);
    return mode_;
  }

  bool has_q_target() const {
    std::lock_guard<std::mutex> lock(mu_);
    return q_target_.has_value();
  }

  // 没发过 /kmpc/q_target 时钉在 q0（原地）；软偏置由本任务 begin_task 的实测参考提供。
  Vec18 q_des(const Vec18& q0) const {
    std::lock_guard<std::mutex> lock(mu_);
    return q_target_.has_value() ? *q_target_ : q0;
  }

  std::optional<CartesianTarget> target(bool left) const {
    std::lock_guard<std::mutex> lock(mu_);
    return left ? left_ : right_;
  }

  bool in_hold_deadband(const Vec18& q0, double deadband) const {
    if (mode() != kModeHold) {
      return false;
    }
    return (q_des(q0) - q0).norm() < deadband;
  }

 private:
  static MotionTask motion_task_from_segment(bool left, const CartesianTarget& t) {
    MotionTask mt;
    mt.generation = t.generation;
    mt.left = left;
    mt.frame = t.frame_id;
    StageTarget s0;
    // Match kmpc_node legacy two-point defaults: prep=PTP, tip=LIN.
    // Single-point stays LIN (unknown insert must not free-path).
    s0.geometry = t.cascade ? TaskGeometry::Ptp : TaskGeometry::Lin;
    s0.position = t.xyz;
    s0.quat = t.quat_xyzw;
    // Legacy PoseArray: identity quat means unconstrained orientation.
    s0.has_orientation = !quat_is_identity_xyzw(&t.quat_xyzw);
    s0.stage_id = 0;
    mt.stages.push_back(s0);
    if (t.cascade) {
      StageTarget s1;
      s1.geometry = TaskGeometry::Lin;
      s1.position = t.cascade->xyz;
      s1.quat = t.cascade->quat_xyzw;
      s1.has_orientation = t.cascade->ori_on;
      s1.stage_id = 1;
      mt.stages.push_back(s1);
    }
    return mt;
  }

  static CartesianTarget cartesian_from_stages(const MotionTask& task) {
    CartesianTarget cart;
    cart.generation = task.generation;
    cart.frame_id = task.frame.empty() ? std::string(kBaseFrame) : task.frame;
    cart.xyz = task.stages[0].position;
    cart.quat_xyzw = task.stages[0].quat;
    if (task.stages.size() >= 2) {
      LineTarget tip;
      tip.xyz = task.stages[1].position;
      tip.quat_xyzw = task.stages[1].quat;
      tip.ori_on = task.stages[1].has_orientation;
      tip.frame = cart.frame_id;
      cart.cascade = tip;
    }
    return cart;
  }

  mutable std::mutex mu_;
  uint64_t generation_ = 0;
  std::optional<Vec18> q_target_;
  std::string mode_ = std::string(kModeSingleEe);
  std::optional<CartesianTarget> left_;
  std::optional<CartesianTarget> right_;
  std::optional<MotionTask> motion_;
  bool saw_cartesian_ = false;
  bool waist_locked_hint_ = false;
};

class PlannerReceiver {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PlannerReceiver() = default;

  bool last_com_reject() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_com_reject_;
  }

  void on_waist_locked(bool locked) {
    std::lock_guard<std::mutex> lock(mu_);
    waist_locked_ = locked;
  }

  void on_tracking_state(std::string_view state) {
    std::string s(state);
    for (char& c : s) {
      if (c >= 'a' && c <= 'z') {
        c = static_cast<char>(c - 'a' + 'A');
      }
    }
    if (s != "CALIBRATING" && s != "TRACKING") {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    tracking_state_ = s;
  }

  void on_q_calib(const Vec18& q) {
    std::lock_guard<std::mutex> lock(mu_);
    q_calib_ = q;
  }

  void apply(UpperPlanner& planner) {
    bool waist_locked = false;
    std::optional<std::string> state;
    std::optional<Vec18> q_calib;
    {
      std::lock_guard<std::mutex> lock(mu_);
      waist_locked = waist_locked_;
      state = tracking_state_;
      q_calib = q_calib_;
    }
    planner.set_waist_locked(waist_locked);
    if (!state.has_value()) {
      return;
    }
    if (state != last_applied_state_) {
      if (*state == "CALIBRATING") {
        if (!q_calib.has_value()) {
          last_com_reject_ = true;
          return;
        }
        const bool ok = planner.request_calibrate(*q_calib);
        last_com_reject_ = !ok;
        if (ok) {
          last_applied_q_calib_ = *q_calib;
          last_applied_state_ = "CALIBRATING";
        }
      } else {
        planner.request_tracking();
        last_applied_state_ = "TRACKING";
        last_applied_q_calib_.reset();
        last_com_reject_ = false;
      }
      return;
    }
    if (*state == "CALIBRATING" && q_calib.has_value()) {
      const bool same = last_applied_q_calib_.has_value() &&
                        (*q_calib - *last_applied_q_calib_).cwiseAbs().maxCoeff() <
                            1e-9;
      if (!same) {
        const bool ok = planner.request_calibrate(*q_calib);
        last_com_reject_ = !ok;
        if (ok) {
          last_applied_q_calib_ = *q_calib;
        }
      }
    }
  }

  void note_auto_tracking() {
    std::lock_guard<std::mutex> lock(mu_);
    if (tracking_state_ == "CALIBRATING") {
      tracking_state_ = "TRACKING";
    }
    last_applied_state_ = "TRACKING";
    last_applied_q_calib_.reset();
    last_com_reject_ = false;
  }

 private:
  mutable std::mutex mu_;
  bool waist_locked_ = false;
  std::optional<std::string> tracking_state_;
  std::optional<Vec18> q_calib_;
  std::optional<std::string> last_applied_state_;
  std::optional<Vec18> last_applied_q_calib_;
  bool last_com_reject_ = false;
};

class StateAggregator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit StateAggregator(double timeout_sec = 0.5)
      : timeout_sec_(timeout_sec) {}

  void on_q(const Vec18& q, const std::vector<std::string>& missing,
            bool has_source_stamp = false, int32_t source_sec = 0,
            uint32_t source_nsec = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    q_ = q;
    stamp_ = std::chrono::steady_clock::now();
    has_ = true;
    last_missing_ = missing;
    has_source_stamp_ = has_source_stamp;
    source_sec_ = source_sec;
    source_nsec_ = source_nsec;
  }

  std::optional<Vec18> read() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!has_) {
      return std::nullopt;
    }
    const auto age = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - stamp_)
                         .count();
    if (age > timeout_sec_) {
      return std::nullopt;
    }
    if (!last_missing_.empty()) {
      return std::nullopt;
    }
    if (!q_.allFinite()) {
      return std::nullopt;
    }
    return q_;
  }

  // Local receive age of the last /joint_states sample (steady clock).
  double receive_age_s() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!has_) return -1.0;
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - stamp_)
        .count();
  }

  bool has_source_stamp() const {
    std::lock_guard<std::mutex> lock(mu_);
    return has_ && has_source_stamp_;
  }

  // ROS header stamp from JointState when present; seconds since epoch.
  double source_stamp_s() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!has_ || !has_source_stamp_) return -1.0;
    return static_cast<double>(source_sec_) +
           1e-9 * static_cast<double>(source_nsec_);
  }

  std::vector<std::string> last_missing() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_missing_;
  }

 private:
  mutable std::mutex mu_;
  double timeout_sec_ = 0.5;
  Vec18 q_ = Vec18::Zero();
  std::chrono::steady_clock::time_point stamp_{};
  bool has_ = false;
  bool has_source_stamp_ = false;
  int32_t source_sec_ = 0;
  uint32_t source_nsec_ = 0;
  std::vector<std::string> last_missing_;
};

inline std::optional<Vec18> q_from_name_positions(
    const std::unordered_map<std::string, double>& named) {
  Vec18 q = Vec18::Zero();
  for (int i = 0; i < kNDof; ++i) {
    const auto it =
        named.find(std::string(kJointNames[static_cast<std::size_t>(i)]));
    if (it == named.end() || !std::isfinite(it->second)) {
      return std::nullopt;
    }
    q[i] = it->second;
  }
  return q;
}

// C10：只乘 Safety 之后的下发步，不改 Hessian / yaml 限位。s 钉在 [0,1]。
inline Vec18 apply_dq_scale(const Vec18& q0, const Vec18& q_safe, double scale) {
  if (!std::isfinite(scale)) {
    return q0;
  }
  const double s = std::clamp(scale, 0.0, 1.0);
  return q0 + s * (q_safe - q0);
}

}  // namespace ros2io
}  // namespace kmpc
