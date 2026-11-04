#pragma once
// 对应 Python task_receiver / planner_receiver / command_arbiter / state_aggregator。
// 非阻塞写、控制拍读。无 rclcpp，消息转换留在 kmpc_node.cpp。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmpc/types.hpp"
#include "kmpc/cartesian_tracker.hpp"
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
  explicit TaskReceiver(const Vec18& q_pref) : q_pref_(q_pref) {}

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
  bool on_segment(bool left, const CartesianTarget& t) {
    if (!t.generation || t.frame_id!=kBaseFrame || !t.xyz.allFinite() ||
        !t.quat_xyzw.allFinite() || t.quat_xyzw.norm()<1e-6) return false;
    if (t.cascade && (t.cascade->frame!=kBaseFrame || !t.cascade->xyz.allFinite() ||
        !t.cascade->quat_xyzw.allFinite() || t.cascade->quat_xyzw.norm()<1e-6)) return false;
    std::lock_guard<std::mutex> lock(mu_);
    auto& current=left?left_:right_;
    if (current && current->generation==t.generation) {
      current->xyz=t.xyz; current->quat_xyzw=t.quat_xyzw;
      return true; // Same segment drift cannot replace its cascade or rearm.
    }
    if (t.generation<=generation_) return false;
    generation_=t.generation;
    mode_=std::string(kModeSingleEe);
    if(left) { left_=t; right_.reset(); } else { right_=t; left_.reset(); }
    saw_cartesian_=true;
    return true;
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
    saw_cartesian_=left_.has_value() || right_.has_value();
    return true;
  }

  void clear_cartesian() {
    std::lock_guard<std::mutex> lock(mu_);
    left_.reset();
    right_.reset();
    saw_cartesian_ = false;
  }

  void clear_left() {
    std::lock_guard<std::mutex> lock(mu_);
    left_.reset();
    if (!right_) {
      saw_cartesian_ = false;
    }
  }

  void clear_right() {
    std::lock_guard<std::mutex> lock(mu_);
    right_.reset();
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

  // 没发过 /kmpc/q_target 时钉在 q0（原地），不再默认追 q_pref/HOME。
  // 要回家就明确发 HOME 的 q_target。q_pref 仍给空闲臂偏置 / 标定用。
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
  mutable std::mutex mu_;
  uint64_t generation_ = 0;
  Vec18 q_pref_ = Vec18::Zero();
  std::optional<Vec18> q_target_;
  std::string mode_ = std::string(kModeSingleEe);
  std::optional<CartesianTarget> left_;
  std::optional<CartesianTarget> right_;
  bool saw_cartesian_ = false;
};

class PlannerReceiver {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit PlannerReceiver(const Vec18& q_pref) : q_pref_(q_pref) {}

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
    const Vec18 q_target = q_calib.has_value() ? *q_calib : q_pref_;
    if (state != last_applied_state_) {
      if (*state == "CALIBRATING") {
        const bool ok = planner.request_calibrate(q_target);
        last_com_reject_ = !ok;
        if (ok) {
          last_applied_q_calib_ = q_target;
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
  Vec18 q_pref_ = Vec18::Zero();
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

  void on_q(const Vec18& q, const std::vector<std::string>& missing) {
    std::lock_guard<std::mutex> lock(mu_);
    q_ = q;
    stamp_ = std::chrono::steady_clock::now();
    has_ = true;
    last_missing_ = missing;
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
    return q_;
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
  std::vector<std::string> last_missing_;
};

inline Vec18 q_from_name_positions(
    const std::unordered_map<std::string, double>& named, const Vec18& fallback) {
  Vec18 q = fallback;
  for (int i = 0; i < kNDof; ++i) {
    const auto it = named.find(std::string(kJointNames[static_cast<std::size_t>(i)]));
    if (it != named.end()) {
      q[i] = it->second;
    }
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
