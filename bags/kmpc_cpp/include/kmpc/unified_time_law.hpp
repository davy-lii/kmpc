#pragma once
// Whole-body scalar time law s(t) for a verified geometric path q(s).
// Geometry C1 and time-domain joint continuity are reported separately.
// Non-C1 corners: do NOT multiply a linear-to-zero corner factor into want
// (that deadlocks with follow_line's line_rate>1e-4 gate; r31 field).
// Crossing uses sqrt stop-limited look-ahead + axis v/a caps instead.
// Axis velocity/accel use q_s*s_dot and q_ss*s_dot^2 + q_s*s_ddot.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kmpc/path_layer_metrics.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct TimeLawLimits {
  Vec18 step_cap = Vec18::Ones();  // rad per control tick
  Vec18 dq_max = Vec18::Ones();    // rad/s
  double speed_scale = 1.0;
  double path_remain_m = 0.0;
  double brake_distance_m = 0.08;
  double brake_floor = 0.20;  // end brake floor; corners use look-ahead only
  double corner_remain_s = 1e9;
  double end_remain_s = 1e9;  // path cells remaining to end
  double feedback_scale = 1.0;
  double rate = 0.0;
  double rate_acc = 0.0;
  double accel_s = 0.3;
  double jerk_s = 0.08;
  double dt = 0.02;
  bool geometry_c1 = false;
};

struct TimeLawSample {
  double want = 0.0;
  double rate = 0.0;
  double rate_acc = 0.0;
  double brake = 1.0;
  double q_s_max = 0.0;
  double q_ss_max = 0.0;
  bool geometry_c1 = false;
  bool time_joint_continuous = false;
  bool axis_limit_ok = true;
  bool look_ahead_brake = false;
  const char* note = "ok";
};

template <class At>
inline Vec18 path_qs(const At& at, double s, double eps = 1e-4) {
  return (at(s + eps) - at(s - eps)) / (2.0 * eps);
}

template <class At>
inline Vec18 path_qss(const At& at, double s, double eps = 1e-4) {
  return (at(s + eps) - 2.0 * at(s) + at(s - eps)) / (eps * eps);
}

inline double rate_from_axis_velocity(const Vec18& qs, const Vec18& dq_max,
                                      double speed_scale) {
  double max_rate = speed_scale;
  for (int j = 0; j < kNDof; ++j) {
    const double qsj = std::abs(qs[j]);
    if (qsj > 1e-9 && dq_max[j] > 1e-12) {
      max_rate = std::min(max_rate, dq_max[j] / qsj);
    }
  }
  return std::clamp(max_rate, 0.0, speed_scale);
}

inline double rate_from_axis_accel(const Vec18& qs, const Vec18& qss,
                                   const Vec18& step_cap, double dt,
                                   double speed_scale, double rate_guess) {
  // Discrete accel bound via step_cap/dt^2 as stand-in when device a_max is
  // not separately configured (report as unverified device accel).
  if (!(dt > 1e-6)) {
    return rate_guess;
  }
  double max_rate = rate_guess;
  const Vec18 a_cap = step_cap / (dt * dt);
  for (int j = 0; j < kNDof; ++j) {
    const double qsj = std::abs(qs[j]);
    const double qssj = std::abs(qss[j]);
    const double aj = a_cap[j];
    if (!(aj > 1e-9)) {
      continue;
    }
    // |q_ss|*s_dot^2 + |q_s|*|s_ddot| ≲ a_j; |s_ddot|≲s_dot/dt
    if (qsj > 1e-9) {
      const double disc = aj / (qssj + qsj / dt + 1e-12);
      if (disc > 0.0) {
        max_rate = std::min(max_rate, std::sqrt(disc));
      }
    } else if (qssj > 1e-9) {
      max_rate = std::min(max_rate, std::sqrt(aj / qssj));
    }
  }
  return std::clamp(max_rate, 0.0, speed_scale);
}

// Stop-by-remain: s_dot^2 <= 2 * a_bar * remain (cells/s).
inline double stop_limited_rate(double remain_s, double a_bar) {
  if (!(remain_s > 1e-12) || !(a_bar > 1e-12)) {
    return 0.0;
  }
  return std::sqrt(2.0 * a_bar * remain_s);
}

inline double end_brake_factor(double path_remain_m, double brake_distance_m,
                              double brake_floor) {
  if (!(brake_distance_m > 1e-9) || !(path_remain_m >= 0.0)) {
    return 1.0;
  }
  if (path_remain_m >= brake_distance_m) {
    return 1.0;
  }
  const double x = path_remain_m / brake_distance_m;
  return std::max(brake_floor, std::sqrt(std::max(0.0, x)));
}

template <class At>
inline TimeLawSample unified_time_law_step(const At& at, double s,
                                           const TimeLawLimits& lim) {
  TimeLawSample out;
  out.geometry_c1 = lim.geometry_c1;
  // Non-C1 joints stay discontinuous in time; look-ahead only limits rate.
  out.time_joint_continuous = lim.geometry_c1;
  const Vec18 qs = path_qs(at, s);
  const Vec18 qss = path_qss(at, s);
  out.q_s_max = qs.cwiseAbs().maxCoeff();
  out.q_ss_max = qss.cwiseAbs().maxCoeff();
  double rate_cap =
      rate_from_axis_velocity(qs, lim.dq_max, lim.speed_scale);
  rate_cap = rate_from_axis_accel(qs, qss, lim.step_cap, lim.dt,
                                  lim.speed_scale, rate_cap);
  out.brake = end_brake_factor(lim.path_remain_m, lim.brake_distance_m,
                             lim.brake_floor);
  const double fb = std::clamp(lim.feedback_scale, 0.0, 1.0);
  out.want = rate_cap * out.brake * fb;

  // Velocity look-ahead: cannot exceed stoppable rate before corner / end.
  const double a_bar =
      (lim.accel_s > 1e-3) ? (1.0 / lim.accel_s) : (1.0 / lim.dt);
  if (!lim.geometry_c1 && lim.corner_remain_s < 1e8) {
    const double stop_r = stop_limited_rate(lim.corner_remain_s, a_bar);
    if (out.want > stop_r) {
      out.want = stop_r;
      out.look_ahead_brake = true;
      out.note = "corner_look_ahead";
    }
  }
  if (lim.end_remain_s < 1e8) {
    const double stop_r = stop_limited_rate(lim.end_remain_s, a_bar);
    if (out.want > stop_r) {
      out.want = stop_r;
      out.look_ahead_brake = true;
      out.note = "end_look_ahead";
    }
  }

  if (out.brake < 1.0 - 1e-9 && !out.look_ahead_brake) {
    out.note = "end_brake";
  } else if (fb < 1.0 - 1e-9 && !out.look_ahead_brake) {
    out.note = "feedback_scale";
  }

  double rate = lim.rate;
  double acc = lim.rate_acc;
  const double dt = std::max(lim.dt, 1e-4);
  if (!(lim.accel_s > 1e-3)) {
    rate = out.want;
    acc = 0.0;
  } else {
    const double a_lim = 1.0 / lim.accel_s;
    const double j_bar = a_lim / std::max(lim.jerk_s, 1e-3);
    const auto clip = [](double x, double limv) {
      return std::clamp(x, -limv, limv);
    };
    const double a_des = clip((out.want - rate) / dt, a_lim);
    const double j_cmd = clip((a_des - acc) / dt, j_bar);
    acc = clip(acc + j_cmd * dt, a_lim);
    double v = rate + acc * dt;
    if (v < 0.0) {
      v = 0.0;
      acc = 0.0;
    }
    if (v > lim.speed_scale) {
      v = lim.speed_scale;
      if (acc > 0.0) {
        acc = 0.0;
      }
    }
    rate = v;
  }

  // Re-check look-ahead against the chosen rate (not just want).
  if (!lim.geometry_c1 && lim.corner_remain_s < 1e8) {
    const double stop_r = stop_limited_rate(lim.corner_remain_s, a_bar);
    if (rate > stop_r) {
      rate = stop_r;
      if (acc > 0.0) {
        acc = 0.0;
      }
      out.look_ahead_brake = true;
      out.note = "corner_look_ahead_rate";
    }
  }
  if (lim.end_remain_s < 1e8) {
    const double stop_r = stop_limited_rate(lim.end_remain_s, a_bar);
    if (rate > stop_r) {
      rate = stop_r;
      if (acc > 0.0) {
        acc = 0.0;
      }
      out.look_ahead_brake = true;
      out.note = "end_look_ahead_rate";
    }
  }

  // Discrete axis velocity and accel vs caps with chosen rate/acc.
  const Vec18 a_cap = lim.step_cap / (dt * dt);
  for (int j = 0; j < kNDof; ++j) {
    const double vj = std::abs(qs[j]) * rate;
    if (lim.dq_max[j] > 1e-12 && vj > lim.dq_max[j] * 1.001) {
      out.axis_limit_ok = false;
      rate = std::min(rate, lim.dq_max[j] / std::max(std::abs(qs[j]), 1e-12));
      out.note = "axis_v_clip";
    }
    const double aj =
        std::abs(qss[j]) * rate * rate + std::abs(qs[j]) * std::abs(acc);
    if (a_cap[j] > 1e-9 && aj > a_cap[j] * 1.001) {
      out.axis_limit_ok = false;
      // Reduce rate to satisfy |q_ss| r^2 + |q_s| |acc| <= a_cap.
      const double qssj = std::abs(qss[j]);
      const double qsj = std::abs(qs[j]);
      double r_ok = rate;
      if (qssj > 1e-12) {
        const double disc =
            std::max(0.0, a_cap[j] - qsj * std::abs(acc)) / qssj;
        r_ok = std::sqrt(disc);
      } else if (qsj > 1e-12 && std::abs(acc) > 1e-12) {
        r_ok = rate;  // accel term dominates; clip acc below
        acc = std::copysign(std::min(std::abs(acc), a_cap[j] / qsj), acc);
      }
      rate = std::min(rate, r_ok);
      out.note = "axis_a_clip";
    }
  }
  out.rate = rate;
  out.rate_acc = acc;
  return out;
}

// Next corner remain among *all* interior knots above jump threshold.
inline double next_corner_remain_all(double s, int knots,
                                     double jump_thresh_rad,
                                     const std::vector<double>& jumps) {
  if (knots < 3 || jumps.empty()) {
    return 1e9;
  }
  double best = 1e9;
  for (int k = 1; k < knots - 1 && k < static_cast<int>(jumps.size()); ++k) {
    if (jumps[static_cast<std::size_t>(k)] < jump_thresh_rad) {
      continue;
    }
    const double knot = static_cast<double>(k);
    if (s < knot - 1e-12) {
      best = std::min(best, knot - s);
    }
  }
  return best;
}

inline double next_corner_remain(double s, const PathCornerMetrics& corners,
                                 double jump_thresh_rad) {
  if (!corners.tangent_jumps.empty()) {
    return next_corner_remain_all(s, corners.knots, jump_thresh_rad,
                                  corners.tangent_jumps);
  }
  if (corners.worst_knot < 0 || corners.knots < 3) {
    return 1e9;
  }
  if (corners.max_tangent_jump_rad < jump_thresh_rad) {
    return 1e9;
  }
  const double knot = static_cast<double>(corners.worst_knot);
  if (s >= knot) {
    return 1e9;
  }
  return knot - s;
}

}  // namespace kmpc
