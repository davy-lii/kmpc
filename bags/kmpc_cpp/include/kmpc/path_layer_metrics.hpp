#pragma once
// Read-only path geometry metrics for planner/QP/stream attribution.
// Does not change live line_at selection.

#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct PathCornerMetrics {
  double max_joint_turn_rad = 0.0;
  double max_waist_turn_rad = 0.0;
  double max_pitch_turn_rad = 0.0;
  double max_tangent_jump_rad = 0.0;
  int worst_knot = -1;
  int knots = 0;
  // Per-knot tangent jump (index = knot). Empty unless measure filled it.
  std::vector<double> tangent_jumps;
};

// Interior knot tangent jump of piecewise-linear joint path q(s).
// Uses ±eps finite difference of the live linear interpolant.
template <class Samples>
inline PathCornerMetrics measure_path_corners(
    const Samples& samples, double eps = 1e-4) {
  PathCornerMetrics out;
  const int n = static_cast<int>(samples.size());
  out.knots = n;
  if (n < 3) {
    return out;
  }
  auto at = [&](double s) -> Vec18 {
    // Local linear sample matching live line_at semantics without tracker.
    const double hi = static_cast<double>(n - 1);
    double p = s;
    if (p < 0.0) {
      p = 0.0;
    } else if (p > hi) {
      p = hi;
    }
    const int i = static_cast<int>(std::floor(p));
    if (i >= n - 1) {
      return samples[static_cast<std::size_t>(n - 1)];
    }
    const double u = p - static_cast<double>(i);
    return (1.0 - u) * samples[static_cast<std::size_t>(i)] +
           u * samples[static_cast<std::size_t>(i + 1)];
  };
  out.tangent_jumps.assign(static_cast<std::size_t>(n), 0.0);
  for (int k = 1; k < n - 1; ++k) {
    const double s = static_cast<double>(k);
    const Vec18 t0 = (at(s) - at(s - eps)) / eps;
    const Vec18 t1 = (at(s + eps) - at(s)) / eps;
    const Vec18 jump = t1 - t0;
    const double joint_jump = jump.cwiseAbs().maxCoeff();
    out.tangent_jumps[static_cast<std::size_t>(k)] = joint_jump;
    double waist_jump = 0.0;
    for (int a = 0; a < 3; ++a) {
      waist_jump = std::max(waist_jump, std::abs(jump[a]));
    }
    const double pitch0 = t0[0] + t0[1] + t0[2];
    const double pitch1 = t1[0] + t1[1] + t1[2];
    const double pitch_jump = std::abs(pitch1 - pitch0);
    if (joint_jump >= out.max_tangent_jump_rad) {
      out.max_tangent_jump_rad = joint_jump;
      out.worst_knot = k;
    }
    out.max_joint_turn_rad = std::max(out.max_joint_turn_rad, joint_jump);
    out.max_waist_turn_rad = std::max(out.max_waist_turn_rad, waist_jump);
    out.max_pitch_turn_rad = std::max(out.max_pitch_turn_rad, pitch_jump);
  }
  out.max_pitch_turn_rad = std::max(
      out.max_pitch_turn_rad,
      chord_turnback(n, [&](int i) {
        return samples[static_cast<std::size_t>(i)][0] +
               samples[static_cast<std::size_t>(i)][1] +
               samples[static_cast<std::size_t>(i)][2];
      }));
  return out;
}

enum class LineQpFollowMode : uint8_t {
  A = 0,  // current Cartesian-weighted QP
  B = 1,  // nominal path + full hard checks / projection (no ProxQP reshape)
  C = 2,  // min-correction QP (keep hard rows; drop large Cartesian reshape)
};

inline const char* line_qp_follow_mode_name(LineQpFollowMode mode) {
  switch (mode) {
    case LineQpFollowMode::A: return "A";
    case LineQpFollowMode::B: return "B";
    case LineQpFollowMode::C: return "C";
  }
  return "A";
}

inline bool parse_line_qp_follow_mode(std::string_view text, LineQpFollowMode* out) {
  if (out == nullptr) {
    return false;
  }
  if (text == "A" || text == "a" || text == "current") {
    *out = LineQpFollowMode::A;
    return true;
  }
  if (text == "B" || text == "b" || text == "direct") {
    *out = LineQpFollowMode::B;
    return true;
  }
  if (text == "C" || text == "c" || text == "min_correction") {
    *out = LineQpFollowMode::C;
    return true;
  }
  return false;
}

}  // namespace kmpc
