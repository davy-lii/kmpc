#pragma once
// Hermite may become the live interpolant only after the *actual* Hermite
// curve (not a densified polyline through Hermite samples) revalidates.
//
// Validation is adaptive sampling with an explicit joint/FK residual bound.
// It is NOT a mathematical proof that the continuous curve is safe everywhere:
// on success we only claim that every refined sample (mid + quarter points of
// each accepted interval) passed check_q within the configured depth/eps.
// Failure keeps the linear path — never relax protections to force Hermite.

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/StdVector>

#include "kmpc/path_layer_metrics.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct PathSmoothReport {
  bool applied = false;
  bool geometry_c1 = false;
  bool hermite_candidate = false;
  bool hermite_verified = false;
  int fail_knot = -1;
  double fail_value = 0.0;
  char reason[128]{"none"};
  PathCornerMetrics before{};
  PathCornerMetrics after{};
};

using JointPath =
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>>;

// Validate the Hermite curve on [0, n-1] by adaptive sampling.
//
// Do NOT accept an interval solely because endpoints (or even the midpoint)
// are close to the chord. Antisymmetric bulges can place mid≈0 while
// quarter-points deviate (e.g. knots [-0.2,0,0,0.2] → quarters ≈ ±0.009375).
// Always evaluate mid + both quarters; only stop when end_span and all three
// chord residuals are ≤ joint_eps. Hitting max_depth → fail closed.
template <class CheckQ>
inline bool validate_hermite_curve(const JointPath& src, CheckQ&& check_q,
                                   std::string* why, double joint_eps = 5e-4,
                                   int max_depth = 8) {
  const int n = static_cast<int>(src.size());
  if (n < 2) {
    if (why) {
      *why = "too_few_knots";
    }
    return false;
  }
  auto chord_at = [](const Vec18& q0, const Vec18& q1, double u) {
    return (1.0 - u) * q0 + u * q1;
  };
  std::function<bool(double, double, const Vec18&, const Vec18&, int)> refine;
  refine = [&](double s0, double s1, const Vec18& q0, const Vec18& q1,
               int depth) -> bool {
    if (!check_q(q0, why) || !check_q(q1, why)) {
      return false;
    }
    const double sm = 0.5 * (s0 + s1);
    const double sq1 = 0.75 * s0 + 0.25 * s1;
    const double sq3 = 0.25 * s0 + 0.75 * s1;
    const Vec18 qm = hermite_joint_at(src, sm);
    const Vec18 qq1 = hermite_joint_at(src, sq1);
    const Vec18 qq3 = hermite_joint_at(src, sq3);
    if (!check_q(qm, why) || !check_q(qq1, why) || !check_q(qq3, why)) {
      return false;
    }
    const double end_span = (q1 - q0).cwiseAbs().maxCoeff();
    const double mid_bulge =
        (qm - chord_at(q0, q1, 0.5)).cwiseAbs().maxCoeff();
    const double q1_bulge =
        (qq1 - chord_at(q0, q1, 0.25)).cwiseAbs().maxCoeff();
    const double q3_bulge =
        (qq3 - chord_at(q0, q1, 0.75)).cwiseAbs().maxCoeff();
    if (end_span <= joint_eps && mid_bulge <= joint_eps &&
        q1_bulge <= joint_eps && q3_bulge <= joint_eps) {
      return true;
    }
    if (depth >= max_depth) {
      if (why) {
        *why = "hermite_sample_bound";
      }
      return false;
    }
    return refine(s0, sm, q0, qm, depth + 1) &&
           refine(sm, s1, qm, q1, depth + 1);
  };
  for (int i = 0; i < n - 1; ++i) {
    const double s0 = static_cast<double>(i);
    const double s1 = static_cast<double>(i + 1);
    const Vec18 q0 = hermite_joint_at(src, s0);
    const Vec18 q1 = hermite_joint_at(src, s1);
    if (!refine(s0, s1, q0, q1, 0)) {
      return false;
    }
  }
  return true;
}

// Attempt verified Hermite. On success, caller may switch live interpolant to
// Hermite on the *original* knots (progress domain unchanged).
template <class CheckQ>
inline PathSmoothReport try_verified_hermite_smooth(const JointPath& src,
                                                    CheckQ&& check_q) {
  PathSmoothReport rep;
  rep.before = measure_path_corners(src);
  if (src.size() < 3) {
    std::snprintf(rep.reason, sizeof(rep.reason), "too_few_knots");
    return rep;
  }
  if (rep.before.max_tangent_jump_rad < 1e-4) {
    // Already C0-smooth in tangent jump; Hermite not required. Do NOT claim
    // live Hermite was verified — keep linear.
    std::snprintf(rep.reason, sizeof(rep.reason), "already_smooth_keep_linear");
    rep.geometry_c1 = false;
    return rep;
  }
  rep.hermite_candidate = true;
  std::string why;
  if (!validate_hermite_curve(src, check_q, &why)) {
    rep.fail_knot = rep.before.worst_knot;
    rep.fail_value = rep.before.max_tangent_jump_rad;
    std::snprintf(rep.reason, sizeof(rep.reason), "hermite_curve_fail:%s",
                  why.empty() ? "unknown" : why.c_str());
    return rep;
  }
  // Measure Hermite tangent jumps at original knots.
  PathCornerMetrics herm;
  herm.knots = static_cast<int>(src.size());
  herm.tangent_jumps.assign(src.size(), 0.0);
  constexpr double eps = 1e-4;
  for (int k = 1; k < herm.knots - 1; ++k) {
    const double s = static_cast<double>(k);
    const Vec18 t0 =
        (hermite_joint_at(src, s) - hermite_joint_at(src, s - eps)) / eps;
    const Vec18 t1 =
        (hermite_joint_at(src, s + eps) - hermite_joint_at(src, s)) / eps;
    const double jump = (t1 - t0).cwiseAbs().maxCoeff();
    herm.tangent_jumps[static_cast<std::size_t>(k)] = jump;
    if (jump >= herm.max_tangent_jump_rad) {
      herm.max_tangent_jump_rad = jump;
      herm.worst_knot = k;
    }
    herm.max_joint_turn_rad = herm.max_tangent_jump_rad;
  }
  rep.after = herm;
  rep.hermite_verified = true;
  rep.applied = true;
  rep.geometry_c1 = herm.max_tangent_jump_rad <= 5e-3;
  std::snprintf(rep.reason, sizeof(rep.reason),
                rep.geometry_c1 ? "hermite_curve_verified_sampled_c1"
                                : "hermite_curve_verified_sampled");
  return rep;
}

}  // namespace kmpc
