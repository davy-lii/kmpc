#pragma once
// 核内弧度，无 ROS。QP 50 Hz 给稀疏路点；任务终点是 q_star。
// S 曲线：加速度连续（|Δa| ≤ j_bar Δt）。s_stop 主导刹车；catch_s 只远场饱和。
// 禁止用 pin 当刹车：只有 |e| 与 |v| 都已经很小才钉住。
// 禁止用降低 v_bar 当刹车：新上限低于当前 v 时按 a_bar 往下收。
// stop_at_goal=false：跟踪移动路点，限速限加，不按静止梯形刹停。
// 按轴 FSM（set_q_star）：track / brake / pin。进 brake = |e*|≤e_near 或 s_stop≥|e*|。

#include <cmath>

#include "kmpc/types.hpp"

namespace kmpc {

// 单轴刹车距。v=0 → 0，不会一 enable 就刹。
inline double stream_s_stop(double v, double a_now, double a_bar, double jerk_s) {
  if (!(a_bar > 1e-9) || !std::isfinite(a_bar) || !std::isfinite(v)) {
    return 0.0;
  }
  double s = (v * v) / (2.0 * a_bar);
  if (jerk_s > 1e-12 && std::isfinite(jerk_s) && std::abs(v) > 0.0) {
    const double j_bar = a_bar / jerk_s;
    if (j_bar > 1e-12) {
      const double a_brake = (v > 0.0) ? -a_bar : a_bar;
      const double a = std::isfinite(a_now) ? a_now : 0.0;
      s += std::abs(v) * std::abs(a - a_brake) / j_bar;
    }
  }
  return s;
}

// 按轴进 brake：|e|≤e_near（几何尾）或 s_stop≥|e|。纯 s_stop 在 e<1° 进不去。
inline bool stream_axis_brake_enter(
    double e, double v, double a_now, double a_bar, double jerk_s,
    double e_near = 0.05) {
  const double ae = std::abs(e);
  if (!(ae > 1e-12)) {
    return true;
  }
  if (e_near > 0.0 && std::isfinite(e_near) && ae <= e_near) {
    return true;
  }
  return stream_s_stop(v, a_now, a_bar, jerk_s) >= ae;
}

// 整包 FixedGoal：最远轴用同一两项或。短轴刹车走 per-axis，不靠这一票。
inline bool stream_brake_enter(
    const Vec18& q_ref, const Vec18& v_ref, const Vec18& a_ref,
    const Vec18& q_star, const Vec18& a_bar, double jerk_s,
    double e_force = 0.05) {
  int imax = 0;
  double emax = 0.0;
  for (int i = 0; i < kNDof; ++i) {
    const double e = std::abs(q_star[i] - q_ref[i]);
    if (e > emax) {
      emax = e;
      imax = i;
    }
  }
  if (!(emax > 1e-12)) {
    return true;
  }
  return stream_axis_brake_enter(
      q_star[imax] - q_ref[imax], v_ref[imax], a_ref[imax], a_bar[imax], jerk_s,
      e_force);
}

enum class AxisMode : unsigned char { Track = 0, Brake = 1, Pin = 2 };

class StreamInterpolator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void reset(const Vec18& q) {
    if (!q.allFinite()) {
      return;
    }
    q_ref_ = q;
    v_ref_.setZero();
    a_ref_.setZero();
    q_goal_ = q;
    q_seg_from_ = q;
    q_seg_to_ = q;
    q_star_.setZero();
    q_star_locked_.setZero();
    seg_period_ = 0.0;
    seg_elapsed_ = 0.0;
    blending_ = false;
    axis_fsm_ = false;
    for (int i = 0; i < kNDof; ++i) {
      axis_mode_[i] = AxisMode::Track;
      axis_hold_[i] = false;
    }
    ready_ = true;
    clamp_inside();
    q_goal_ = q_ref_;
    q_seg_from_ = q_ref_;
    q_seg_to_ = q_ref_;
  }

  // 硬限位内侧 inner_rad。未设置时不刹车。
  void set_joint_box(const Vec18& q_min, const Vec18& q_max, double inner_rad) {
    if (!q_min.allFinite() || !q_max.allFinite() || !std::isfinite(inner_rad) ||
        inner_rad < 0.0) {
      return;
    }
    q_min_ = q_min;
    q_max_ = q_max;
    inner_rad_ = inner_rad;
    joint_box_ = true;
  }

  void hold_axis(int i) {
    if (i < 0 || i >= kNDof || !ready_) {
      return;
    }
    axis_hold_[i] = true;
    v_ref_[i] = 0.0;
    a_ref_[i] = 0.0;
    q_goal_[i] = q_ref_[i];
    q_seg_from_[i] = q_ref_[i];
    q_seg_to_[i] = q_ref_[i];
  }

  void release_axis(int i) {
    if (i < 0 || i >= kNDof) {
      return;
    }
    axis_hold_[i] = false;
  }

  bool axis_held(int i) const {
    return i >= 0 && i < kNDof && axis_hold_[i];
  }

  void set_goal(const Vec18& q_goal) {
    if (!q_goal.allFinite()) {
      return;
    }
    q_goal_ = q_goal;
    blending_ = false;
  }

  // lerp 从当前 q_goal 接到 q_to，禁止弹回 q_from（那会让 e≈0 而 v 仍巡航）。
  void begin_qp_segment(const Vec18& q_from, const Vec18& q_to, double period) {
    (void)q_from;
    if (!q_to.allFinite() || !(period > 0.0) || !std::isfinite(period)) {
      return;
    }
    q_seg_from_ = q_goal_;
    q_seg_to_ = q_to;
    for (int i = 0; i < kNDof; ++i) {
      if (!axis_hold_[i]) {
        continue;
      }
      q_seg_from_[i] = q_ref_[i];
      q_seg_to_[i] = q_ref_[i];
      q_goal_[i] = q_ref_[i];
    }
    seg_period_ = period;
    seg_elapsed_ = 0.0;
    blending_ = true;
  }

  void clear_segment() { blending_ = false; }

  void set_limits(const Vec18& v_bar, const Vec18& a_bar) {
    if (!v_bar.allFinite() || !a_bar.allFinite()) {
      return;
    }
    v_bar_ = v_bar;
    a_bar_ = a_bar;
  }

  void set_catch_s(double t) {
    if (!std::isfinite(t) || t <= 0.0) {
      return;
    }
    catch_s_ = t;
  }

  void set_jerk_s(double t) {
    if (!std::isfinite(t) || t <= 0.0) {
      return;
    }
    jerk_s_ = t;
  }

  void set_stop_at_goal(bool on) {
    stop_at_goal_ = on;
    if (on && axis_fsm_) {
      for (int i = 0; i < kNDof; ++i) {
        if (axis_mode_[i] == AxisMode::Track) {
          axis_mode_[i] = AxisMode::Brake;
          q_star_locked_[i] = q_star_[i];
        }
      }
    }
  }

  void set_q_star(const Vec18& q_star) {
    if (!q_star.allFinite()) {
      return;
    }
    q_star_ = q_star;
    axis_fsm_ = true;
  }

  void clear_q_star() {
    axis_fsm_ = false;
    for (int i = 0; i < kNDof; ++i) {
      axis_mode_[i] = AxisMode::Track;
    }
  }

  void set_axis_e_near(double e) {
    if (std::isfinite(e) && e > 0.0) {
      e_near_ = e;
    }
  }

  void set_axis_release_jump(double e) {
    if (std::isfinite(e) && e > 0.0) {
      release_jump_ = e;
    }
  }

  void set_axis_release_hyst(double e) {
    if (std::isfinite(e) && e >= 0.0) {
      release_hyst_ = e;
    }
  }

  // 只改位置，禁止把 v,a 清零（那是一拍撞停）。
  void set_q_ref(const Vec18& q) {
    if (!q.allFinite()) {
      return;
    }
    q_ref_ = q;
    clamp_inside();
  }

  const Vec18& step(double dt) {
    if (!ready_ || !std::isfinite(dt) || dt <= 0.0) {
      return q_ref_;
    }
    update_axis_modes();
    if (blending_ && !stop_at_goal_ && seg_period_ > 0.0) {
      seg_elapsed_ += dt;
      double a = seg_elapsed_ / seg_period_;
      if (a < 0.0) {
        a = 0.0;
      }
      if (a > 1.0) {
        a = 1.0;
      }
      const Vec18 q_lerp = (1.0 - a) * q_seg_from_ + a * q_seg_to_;
      for (int i = 0; i < kNDof; ++i) {
        if (axis_hold_[i]) {
          q_goal_[i] = q_ref_[i];
        } else if (axis_sticky(i)) {
          q_goal_[i] = q_star_locked_[i];
        } else {
          q_goal_[i] = q_lerp[i];
        }
      }
    } else if (axis_fsm_) {
      for (int i = 0; i < kNDof; ++i) {
        if (axis_sticky(i)) {
          q_goal_[i] = q_star_locked_[i];
        }
      }
    }
    for (int i = 0; i < kNDof; ++i) {
      step_axis(i, dt);
    }
    clamp_inside();
    return q_ref_;
  }

  bool ready() const { return ready_; }
  const Vec18& q_ref() const { return q_ref_; }
  const Vec18& v_ref() const { return v_ref_; }
  const Vec18& a_ref() const { return a_ref_; }
  const Vec18& q_goal() const { return q_goal_; }
  bool axis_fsm() const { return axis_fsm_; }
  AxisMode axis_mode(int i) const {
    if (i < 0 || i >= kNDof) {
      return AxisMode::Track;
    }
    return axis_mode_[i];
  }
  int n_axis_brake() const {
    if (!axis_fsm_) {
      return 0;
    }
    int n = 0;
    for (int i = 0; i < kNDof; ++i) {
      if (axis_mode_[i] != AxisMode::Track) {
        ++n;
      }
    }
    return n;
  }

 private:
  static constexpr double kPinAbs = 1e-8;
  static constexpr double kPinVel = 1e-4;

  bool axis_sticky(int i) const {
    return stop_at_goal_ ||
           (axis_fsm_ && axis_mode_[i] != AxisMode::Track);
  }

  void update_axis_modes() {
    if (!axis_fsm_) {
      return;
    }
    for (int i = 0; i < kNDof; ++i) {
      const double e_live = q_star_[i] - q_ref_[i];
      if (axis_mode_[i] == AxisMode::Track) {
        if (stream_axis_brake_enter(
                e_live, v_ref_[i], a_ref_[i], a_bar_[i], jerk_s_, e_near_)) {
          axis_mode_[i] = AxisMode::Brake;
          q_star_locked_[i] = q_star_[i];
        }
      } else if (
          std::abs(q_star_[i] - q_star_locked_[i]) > release_jump_ ||
          std::abs(e_live) > e_near_ + release_hyst_) {
        axis_mode_[i] = AxisMode::Track;
      }
    }
  }

  static double clip(double x, double lim) {
    if (!std::isfinite(x) || !(lim > 0.0) || !std::isfinite(lim)) {
      return 0.0;
    }
    if (x > lim) {
      return lim;
    }
    if (x < -lim) {
      return -lim;
    }
    return x;
  }

  static double sgn(double x) {
    if (x > 0.0) {
      return 1.0;
    }
    if (x < 0.0) {
      return -1.0;
    }
    return 0.0;
  }

  void pin_axis(int i, double g) {
    q_ref_[i] = g;
    v_ref_[i] = 0.0;
    a_ref_[i] = 0.0;
    if (axis_fsm_) {
      axis_mode_[i] = AxisMode::Pin;
    }
  }

  static double velocity_cap(double v, double v_lim, double a_bar, double dt) {
    const double speed = std::abs(v);
    if (!(a_bar > 0.0) || !(dt > 0.0) || v_lim >= speed) {
      return v_lim;
    }
    return std::max(v_lim, speed - a_bar * dt);
  }

  void integrate(int i, double q, double v, double a, double a_des, double a_bar,
                 double j_bar, double v_lim, double dt) {
    const double j_cmd = clip((a_des - a) / dt, j_bar);
    const double a_new = clip(a + j_cmd * dt, a_bar);
    const double v_cap = velocity_cap(v, v_lim, a_bar, dt);
    const double v_new = clip(v + a_new * dt, v_cap);
    q_ref_[i] = q + v_new * dt;
    v_ref_[i] = v_new;
    a_ref_[i] = a_new;
  }

  void pull_goal_inside(int i, double& g) {
    if (!joint_box_) {
      return;
    }
    const double lo = q_min_[i] + inner_rad_;
    const double hi = q_max_[i] - inner_rad_;
    if (!(hi > lo)) {
      return;
    }
    if (g > hi) {
      g = hi;
    }
    if (g < lo) {
      g = lo;
    }
    q_goal_[i] = g;
  }

  // 只刹朝限位的那一侧。反方向返回 false。
  bool toward_limit_brake(
      int i, double q, double v, double a, double a_bar) const {
    if (!joint_box_ || !(a_bar > 0.0) || !(std::abs(v) > 0.0)) {
      return false;
    }
    const double lo = q_min_[i] + inner_rad_;
    const double hi = q_max_[i] - inner_rad_;
    if (!(hi > lo)) {
      return false;
    }
    const double d = (v > 0.0) ? (hi - q) : (q - lo);
    if (!(d > 0.0)) {
      return true;
    }
    return stream_s_stop(v, a, a_bar, jerk_s_) >= d;
  }

  void clamp_inside() {
    if (!joint_box_) {
      return;
    }
    for (int i = 0; i < kNDof; ++i) {
      const double lo = q_min_[i] + inner_rad_;
      const double hi = q_max_[i] - inner_rad_;
      if (!(hi > lo)) {
        continue;
      }
      if (q_ref_[i] > hi) {
        q_ref_[i] = hi;
        if (v_ref_[i] > 0.0) {
          v_ref_[i] = 0.0;
          a_ref_[i] = 0.0;
        }
      } else if (q_ref_[i] < lo) {
        q_ref_[i] = lo;
        if (v_ref_[i] < 0.0) {
          v_ref_[i] = 0.0;
          a_ref_[i] = 0.0;
        }
      }
    }
  }

  void step_axis(int i, double dt) {
    if (axis_hold_[i]) {
      q_goal_[i] = q_ref_[i];
      v_ref_[i] = 0.0;
      a_ref_[i] = 0.0;
      return;
    }
    const double q = q_ref_[i];
    const double v = v_ref_[i];
    const double a = a_ref_[i];
    double g = q_goal_[i];
    const double v_bar = v_bar_[i];
    const double a_bar = a_bar_[i];
    if (!std::isfinite(q) || !std::isfinite(v) || !std::isfinite(a) ||
        !std::isfinite(g) || !std::isfinite(v_bar) || !std::isfinite(a_bar) ||
        !(a_bar > 0.0) || !std::isfinite(catch_s_) || catch_s_ <= 0.0 ||
        !std::isfinite(jerk_s_) || jerk_s_ <= 0.0) {
      return;
    }
    pull_goal_inside(i, g);
    const double e = g - q;
    const double j_bar = a_bar / jerk_s_;
    const double v_lim = v_bar > 0.0 ? v_bar : 0.0;
    const bool limit_brake = toward_limit_brake(i, q, v, a, a_bar);
    if (!axis_sticky(i)) {
      double v_ff = 0.0;
      if (blending_ && seg_period_ > 0.0) {
        v_ff = (q_seg_to_[i] - q_seg_from_[i]) / seg_period_;
      }
      const double v_des = clip(
          v_ff + ((std::abs(e) <= kPinAbs) ? 0.0 : e / catch_s_), v_lim);
      const double a_track = clip((v_des - v) / dt, a_bar);
      const double a_des = limit_brake ? -sgn(v) * a_bar : a_track;
      integrate(i, q, v, a, a_des, a_bar, j_bar, v_lim, dt);
      return;
    }
    if (std::abs(e) <= kPinAbs) {
      if (std::abs(v) <= kPinVel) {
        pin_axis(i, g);
        return;
      }
      integrate(i, q, v, a, -sgn(v) * a_bar, a_bar, j_bar, v_lim, dt);
      q_ref_[i] = g;
      if (v * v_ref_[i] <= 0.0 || std::abs(v_ref_[i]) <= kPinVel) {
        pin_axis(i, g);
      }
      return;
    }
    const double v_des = clip(e / catch_s_, v_lim);
    const double s_stop = stream_s_stop(v, a, a_bar, jerk_s_);
    const bool need_brake =
        (e * v > 0.0 && s_stop >= std::abs(e)) || (std::abs(v) > v_lim);
    const double a_des = (need_brake || limit_brake)
        ? -sgn(v) * a_bar
        : clip((v_des - v) / dt, a_bar);
    integrate(i, q, v, a, a_des, a_bar, j_bar, v_lim, dt);
    const double v_new = v_ref_[i];
    const double q_try = q_ref_[i];
    const double e_try = g - q_try;
    if (std::abs(e_try) <= kPinAbs && std::abs(v_new) <= kPinVel) {
      pin_axis(i, g);
      return;
    }
    if (e * e_try <= 0.0) {
      q_ref_[i] = g;
      if (std::abs(v_new) <= kPinVel) {
        v_ref_[i] = 0.0;
        a_ref_[i] = 0.0;
        if (axis_fsm_) {
          axis_mode_[i] = AxisMode::Pin;
        }
      }
    }
  }

  Vec18 q_ref_ = Vec18::Zero();
  Vec18 v_ref_ = Vec18::Zero();
  Vec18 a_ref_ = Vec18::Zero();
  Vec18 q_goal_ = Vec18::Zero();
  Vec18 q_seg_from_ = Vec18::Zero();
  Vec18 q_seg_to_ = Vec18::Zero();
  Vec18 q_star_ = Vec18::Zero();
  Vec18 q_star_locked_ = Vec18::Zero();
  Vec18 v_bar_ = Vec18::Zero();
  Vec18 a_bar_ = Vec18::Zero();
  Vec18 q_min_ = Vec18::Zero();
  Vec18 q_max_ = Vec18::Zero();
  AxisMode axis_mode_[kNDof] = {};
  bool axis_hold_[kNDof] = {};
  double inner_rad_ = 0.0;
  bool joint_box_ = false;
  double catch_s_ = 0.02;
  double jerk_s_ = 0.01;
  double e_near_ = 0.05;
  double release_jump_ = 0.03;
  double release_hyst_ = 0.01;
  double seg_period_ = 0.0;
  double seg_elapsed_ = 0.0;
  bool blending_ = false;
  bool stop_at_goal_ = true;
  bool axis_fsm_ = false;
  bool ready_ = false;
};

}  // namespace kmpc
