#include "kmpc/time_parameterizer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

// Raw parameter u is the sample index domain [0, N-1]; du per segment is 1.
// Rate v = du/dt. Caps are derived per segment from the joint deltas:
//   v_max = min_j dq_max_j / |dq_j|,  a_max = min_j ddq_max_j / |dq_j|,
//   j_max = min_j dddq_max_j / |dq_j|  (jerk smooths the accel profile).
constexpr double kNominalDt = 0.02;
constexpr double kRateFloor = 1e-6;
constexpr double kSendDt = 0.004;

// Chord-based rate planning underestimates Hermite mid-segment accel/jerk.
// Scale wall time until the *executed* 250 Hz Hermite send sequence fits 1×.
bool hermite_send_sequence_ok(const JointCurve& samples,
                              const TimeLawTable& law,
                              const MotionConstraintPolicy& policy) {
  if (samples.size() < 2 || law.t_s.size() < 2 || !(law.duration_s > 0.0)) {
    return true;
  }
  TrajectoryPackage pkg;
  pkg.samples = samples;
  pkg.time_law = law;
  const int n =
      std::max(1, static_cast<int>(std::ceil(law.duration_s / kSendDt)));
  std::vector<Vec18, Eigen::aligned_allocator<Vec18>> qs;
  std::vector<double> ts;
  qs.reserve(static_cast<std::size_t>(n) + 2);
  ts.reserve(static_cast<std::size_t>(n) + 2);
  auto push = [&](double t) -> bool {
    if (!ts.empty() && t <= ts.back() + 1e-12) {
      return true;
    }
    TrajectoryTimeSample samp;
    if (!sample_trajectory_at(pkg, t, &samp) || !samp.ok) {
      return false;
    }
    qs.push_back(samp.q);
    ts.push_back(samp.t_s);
    return true;
  };
  for (int k = 0; k < n; ++k) {
    const double t = kSendDt * static_cast<double>(k);
    if (t >= law.duration_s - 1e-12) {
      break;
    }
    if (!push(t)) {
      return false;
    }
  }
  if (!push(law.duration_s)) {
    return false;
  }
  for (std::size_t i = 1; i < qs.size(); ++i) {
    const double dt = std::max(ts[i] - ts[i - 1], 1e-4);
    for (int j = 0; j < kNDof; ++j) {
      const double dq_max = std::max(policy.dq_max[j], 1e-6);
      if (std::abs(qs[i][j] - qs[i - 1][j]) > dq_max * dt + 1e-4) {
        return false;
      }
    }
  }
  for (std::size_t i = 2; i < qs.size(); ++i) {
    const double dt1 = std::max(ts[i] - ts[i - 1], 1e-4);
    const double dt0 = std::max(ts[i - 1] - ts[i - 2], 1e-4);
    for (int j = 0; j < kNDof; ++j) {
      const double dq_max = std::max(policy.dq_max[j], 1e-6);
      const double ddq_pol = std::max(policy.ddq_max[j], 1e-6);
      const double ddq_soft =
          policy.a_max_calibrated ? ddq_pol : std::max(ddq_pol, dq_max / 0.02);
      const double v1 = (qs[i][j] - qs[i - 1][j]) / dt1;
      const double v0 = (qs[i - 1][j] - qs[i - 2][j]) / dt0;
      if (std::abs(v1 - v0) > 4.0 * ddq_soft * dt1 + 1e-6) {
        return false;
      }
    }
  }
  for (std::size_t i = 3; i < qs.size(); ++i) {
    const double dt2 = std::max(ts[i] - ts[i - 1], 1e-4);
    const double dt1 = std::max(ts[i - 1] - ts[i - 2], 1e-4);
    const double dt0 = std::max(ts[i - 2] - ts[i - 3], 1e-4);
    for (int j = 0; j < kNDof; ++j) {
      const double dq_max = std::max(policy.dq_max[j], 1e-6);
      const double ddq_pol = std::max(policy.ddq_max[j], 1e-6);
      const double ddq_soft =
          policy.a_max_calibrated ? ddq_pol : std::max(ddq_pol, dq_max / 0.02);
      const double j_pol = std::max(policy.dddq_max[j], 1e-6);
      const double j_soft =
          policy.j_max_calibrated ? j_pol : std::max(j_pol, ddq_soft / 0.02);
      const double v2 = (qs[i][j] - qs[i - 1][j]) / dt2;
      const double v1 = (qs[i - 1][j] - qs[i - 2][j]) / dt1;
      const double v0 = (qs[i - 2][j] - qs[i - 3][j]) / dt0;
      const double a1 = (v2 - v1) / dt2;
      const double a0 = (v1 - v0) / dt1;
      if (std::abs(a1 - a0) > 16.0 * j_soft * dt2 + 1e-3) {
        return false;
      }
    }
  }
  return true;
}

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
    out.ds_dt.push_back(0.0);
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
    out.ds_dt.assign(n, 0.0);
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
  // v[i] is du/dt in sample-index domain; s = u/(n-1) so ds/dt = v/(n-1).
  out.s.resize(n);
  out.t_s.resize(n);
  out.ds_dt.resize(n);
  out.s[0] = 0.0;
  out.t_s[0] = 0.0;
  const double s_scale = 1.0 / static_cast<double>(n - 1);
  out.ds_dt[0] = v[0] * s_scale;
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
    out.ds_dt[i + 1] = v[i + 1] * s_scale;
  }
  out.duration_s = out.t_s.back();

  // Stretch time (keep s knots) until Hermite@250Hz send sequence meets 1×
  // dq/ddq/jerk. Never widen policy caps — only slow the motion.
  for (int attempt = 0; attempt < 12; ++attempt) {
    if (hermite_send_sequence_ok(samples, out, policy)) {
      break;
    }
    constexpr double kStretch = 1.25;
    for (double& t : out.t_s) {
      t *= kStretch;
    }
    out.duration_s *= kStretch;
    for (double& r : out.ds_dt) {
      r /= kStretch;
    }
  }
  return out;
}

}  // namespace kmpc
