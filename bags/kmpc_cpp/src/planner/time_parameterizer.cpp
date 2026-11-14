#include "kmpc/time_parameterizer.hpp"

#include <algorithm>
#include <cmath>

namespace kmpc {
namespace {

// Raw parameter u is the sample index domain [0, N-1]; du per segment is 1.
// Rate v = du/dt. Caps are derived per segment from the joint deltas:
//   v_max = min_j dq_max_j / |dq_j|,  a_max = min_j ddq_max_j / |dq_j|,
//   j_max = min_j dddq_max_j / |dq_j|  (jerk smooths the accel profile).
constexpr double kNominalDt = 0.02;
constexpr double kRateFloor = 1e-6;

}  // namespace

TimeLawTable TimeParameterizer::compute(
    const JointCurve& samples, const MotionConstraintPolicy& policy,
    const std::vector<StageBoundary>& stage_boundaries) const {
  TimeLawTable out;
  out.software_limited =
      !(policy.a_max_calibrated && policy.j_max_calibrated);
  out.accel_source = policy.accel_source;
  out.jerk_source = policy.jerk_source;
  const std::size_t n = samples.size();
  if (n == 0) {
    return out;
  }
  if (n == 1) {
    out.s.push_back(0.0);
    out.t_s.push_back(0.0);
    out.duration_s = 0.0;
    return out;
  }

  // Per-segment rate / accel / jerk caps in index domain.
  std::vector<double> v_max(n - 1, 1e9);
  std::vector<double> a_max(n - 1, 1e9);
  std::vector<double> j_max(n - 1, 1e9);
  bool any_motion = false;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const Vec18 dq = samples[i + 1] - samples[i];
    for (int j = 0; j < kNDof; ++j) {
      const double d = std::abs(dq[j]);
      if (d < 1e-12) {
        continue;
      }
      any_motion = true;
      if (policy.dq_max[j] > 1e-12) {
        v_max[i] = std::min(v_max[i], policy.dq_max[j] / d);
      }
      if (policy.ddq_max[j] > 1e-12) {
        a_max[i] = std::min(a_max[i], policy.ddq_max[j] / d);
      }
      if (policy.dddq_max[j] > 1e-12) {
        j_max[i] = std::min(j_max[i], policy.dddq_max[j] / d);
      }
    }
  }

  // Degenerate curve (no joint motion): uniform nominal timing.
  if (!any_motion) {
    out.s.resize(n);
    out.t_s.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.s[i] = static_cast<double>(i) / static_cast<double>(n - 1);
      out.t_s[i] = static_cast<double>(i) * kNominalDt;
    }
    out.duration_s = out.t_s.back();
    return out;
  }

  // Node rates; endpoints stop, stop_required stage boundaries stop.
  std::vector<double> v(n, 0.0);
  for (const StageBoundary& b : stage_boundaries) {
    if (b.stop_required && b.sample_index < n) {
      v[b.sample_index] = 0.0;
    }
  }
  std::vector<bool> pinned(n, false);
  pinned[0] = true;
  pinned[n - 1] = true;
  for (const StageBoundary& b : stage_boundaries) {
    if (b.stop_required && b.sample_index < n) {
      pinned[b.sample_index] = true;
    }
  }
  // Rate cap per node from adjacent segments.
  for (std::size_t i = 0; i < n; ++i) {
    double cap = 1e9;
    if (i > 0) {
      cap = std::min(cap, v_max[i - 1]);
    }
    if (i + 1 < n) {
      cap = std::min(cap, v_max[i]);
    }
    if (!pinned[i]) {
      v[i] = cap;
    }
  }
  // Forward accel pass: v[i+1] <= sqrt(v[i]^2 + 2 a du), du = 1.
  // Pinned nodes are already at 0, so the same min applies everywhere.
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const double a = std::min(a_max[i], 1e9);
    const double reach = std::sqrt(v[i] * v[i] + 2.0 * a);
    v[i + 1] = std::min(v[i + 1], reach);
    v[i + 1] = std::min(v[i + 1], v_max[i]);
  }
  // Backward brake pass.
  for (std::size_t i = n - 1; i-- > 0;) {
    const double a = std::min(a_max[i], 1e9);
    const double reach = std::sqrt(v[i + 1] * v[i + 1] + 2.0 * a);
    v[i] = std::min(v[i], reach);
    v[i] = std::min(v[i], v_max[i]);
  }
  // Jerk smoothing on the accel profile (one forward pass).
  double a_prev = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    double a = 0.5 * (v[i + 1] * v[i + 1] - v[i] * v[i]);
    const double dt_est =
        2.0 / std::max(v[i] + v[i + 1], kRateFloor);
    const double da = std::min(j_max[i], 1e9) * dt_est;
    a = std::clamp(a, a_prev - da, a_prev + da);
    a = std::clamp(a, -a_max[i], a_max[i]);
    const double v2 = v[i] * v[i] + 2.0 * a;
    if (!pinned[i + 1]) {
      v[i + 1] = std::min(v[i + 1], std::sqrt(std::max(v2, 0.0)));
    }
    a_prev = a;
  }
  // Brake again after jerk smoothing so stops stay stops.
  for (std::size_t i = n - 1; i-- > 0;) {
    const double a = std::min(a_max[i], 1e9);
    const double reach = std::sqrt(v[i + 1] * v[i + 1] + 2.0 * a);
    v[i] = std::min(v[i], reach);
  }

  // Integrate: du = 1 per segment, dt = 2 du / (v_i + v_{i+1}).
  out.s.resize(n);
  out.t_s.resize(n);
  out.s[0] = 0.0;
  out.t_s[0] = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    out.s[i + 1] = static_cast<double>(i + 1) / static_cast<double>(n - 1);
    const double rates = v[i] + v[i + 1];
    double dt = kNominalDt;
    if (rates > kRateFloor) {
      dt = 2.0 / rates;
    } else {
      // Both endpoints ~stopped: fall back to the capped-rate estimate.
      const double vref = std::max(v_max[i], kRateFloor);
      dt = vref > 1e8 ? kNominalDt : std::max(2.0 / vref, kNominalDt);
    }
    out.t_s[i + 1] = out.t_s[i] + dt;
  }
  out.duration_s = out.t_s.back();
  return out;
}

}  // namespace kmpc
