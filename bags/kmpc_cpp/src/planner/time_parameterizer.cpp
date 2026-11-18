#include "kmpc/time_parameterizer.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
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

struct HermiteLimitReport {
  bool ok = true;
  double max_vel_ratio = 0.0;
  double max_acc_ratio = 0.0;
  double max_jerk_ratio = 0.0;
  int worst_joint = -1;
  std::string limit_source = "none";
};

// Measure Hermite@250Hz send-sequence over-limit ratios.
// Accel uses 4× and jerk 16× soft multipliers when not device-calibrated —
// these are software estimates, not strict hardware certification.
HermiteLimitReport hermite_send_limit_report(
    const JointCurve& samples, const TimeLawTable& law,
    const MotionConstraintPolicy& policy,
    const std::vector<StageBoundary>& boundaries) {
  HermiteLimitReport rep;
  if (samples.size() < 2 || law.t_s.size() < 2 || !(law.duration_s > 0.0)) {
    return rep;
  }
  TrajectoryPackage pkg;
  pkg.samples = samples;
  pkg.time_law = law;
  pkg.stage_boundaries = boundaries;
  const int n =
      std::max(1, static_cast<int>(std::ceil(law.duration_s / kSendDt)));
  std::vector<Vec18, Eigen::aligned_allocator<Vec18>> qs;
  std::vector<double> ts;
  qs.reserve(static_cast<std::size_t>(n) + 2);
  ts.reserve(static_cast<std::size_t>(n) + 2);
  auto push = [&](double t, double send_t) -> bool {
    if (!ts.empty() && send_t <= ts.back() + 1e-12) {
      return true;
    }
    TrajectoryTimeSample samp;
    if (!sample_trajectory_at(pkg, t, &samp) || !samp.ok) {
      return false;
    }
    qs.push_back(samp.q);
    ts.push_back(send_t);
    return true;
  };
  for (const auto& probe : trajectory_send_probe_grid(pkg, kSendDt)) {
    if (!push(probe.curve_s, probe.send_s)) {
      rep.ok = false;
      rep.limit_source = "sample_failed";
      return rep;
    }
  }
  for (std::size_t i = 1; i < qs.size(); ++i) {
    const double dt = std::max(ts[i] - ts[i - 1], 1e-4);
    for (int j = 0; j < kNDof; ++j) {
      const double dq_max = std::max(policy.dq_max[j], 1e-6);
      const double r =
          std::abs(qs[i][j] - qs[i - 1][j]) / (dq_max * dt + 1e-12);
      if (r > rep.max_vel_ratio) {
        rep.max_vel_ratio = r;
        rep.worst_joint = j;
        rep.limit_source = "vel";
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
      // 4× soft when uncalibrated: software estimate, not device cert.
      const double acc_cap =
          (policy.a_max_calibrated ? 1.0 : 4.0) * ddq_soft * dt1;
      const double v1 = (qs[i][j] - qs[i - 1][j]) / dt1;
      const double v0 = (qs[i - 1][j] - qs[i - 2][j]) / dt0;
      const double r = std::abs(v1 - v0) / (acc_cap + 1e-12);
      if (r > rep.max_acc_ratio) {
        rep.max_acc_ratio = r;
        rep.worst_joint = j;
        rep.limit_source = policy.a_max_calibrated ? "acc" : "acc_soft_x4";
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
      // 16× soft when uncalibrated: software estimate, not device cert.
      const double jerk_cap =
          (policy.j_max_calibrated ? 1.0 : 16.0) * j_soft * dt2;
      const double v2 = (qs[i][j] - qs[i - 1][j]) / dt2;
      const double v1 = (qs[i - 1][j] - qs[i - 2][j]) / dt1;
      const double v0 = (qs[i - 2][j] - qs[i - 3][j]) / dt0;
      const double a1 = (v2 - v1) / dt2;
      const double a0 = (v1 - v0) / dt1;
      const double r = std::abs(a1 - a0) / (jerk_cap + 1e-12);
      if (r > rep.max_jerk_ratio) {
        rep.max_jerk_ratio = r;
        rep.worst_joint = j;
        rep.limit_source = policy.j_max_calibrated ? "jerk" : "jerk_soft_x16";
      }
    }
  }
  rep.ok = (rep.max_vel_ratio <= 1.0 + 1e-9) &&
           (rep.max_acc_ratio <= 1.0 + 1e-6) &&
           (rep.max_jerk_ratio <= 1.0 + 1e-3);
  return rep;
}

void scale_time_law(TimeLawTable* law, double stretch) {
  if (!(stretch > 1.0) || law == nullptr) {
    return;
  }
  for (double& t : law->t_s) {
    t *= stretch;
  }
  law->duration_s *= stretch;
  for (double& r : law->ds_dt) {
    r /= stretch;
  }
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
    out.ok = false;
    out.reason = "empty_samples";
    return out;
  }
  if (n == 1) {
    out.s.push_back(0.0);
    out.t_s.push_back(0.0);
    out.ds_dt.push_back(0.0);
    out.duration_s = 0.0;
    out.ok = true;
    out.reason = "ok";
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
    out.duration_before_s = out.duration_s;
    out.ok = true;
    out.reason = "ok_zero_motion";
    return out;
  }

  // Node rates; endpoints stop, stop_required stage boundaries stop.
  std::vector<double> v(n, 0.0);
  for (const StageBoundary& b : stage_boundaries) {
    if ((b.stop_required || b.wait_for_external_ack) && b.sample_index < n) {
      v[b.sample_index] = 0.0;
    }
  }
  std::vector<bool> pinned(n, false);
  pinned[0] = true;
  pinned[n - 1] = true;
  for (const StageBoundary& b : stage_boundaries) {
    if ((b.stop_required || b.wait_for_external_ack) && b.sample_index < n) {
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
    const double dt_est = 2.0 / std::max(v[i] + v[i + 1], kRateFloor);
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
      const double vref = std::max(v_max[i], kRateFloor);
      dt = vref > 1e8 ? kNominalDt : std::max(2.0 / vref, kNominalDt);
    }
    out.t_s[i + 1] = out.t_s[i] + dt;
    out.ds_dt[i + 1] = v[i + 1] * s_scale;
  }
  out.duration_s = out.t_s.back();
  out.duration_before_s = out.duration_s;

  // Replace blind 1.25××12: measure ratios, scale by
  // max(r_v, sqrt(r_a), cbrt(r_j)) with a small margin; recheck.
  constexpr int kMaxScaleIters = 6;
  constexpr double kMargin = 1.02;
  for (int attempt = 0; attempt < kMaxScaleIters; ++attempt) {
    const HermiteLimitReport rep =
        hermite_send_limit_report(samples, out, policy, stage_boundaries);
    out.max_vel_ratio = rep.max_vel_ratio;
    out.max_acc_ratio = rep.max_acc_ratio;
    out.max_jerk_ratio = rep.max_jerk_ratio;
    out.worst_joint = rep.worst_joint;
    out.limit_source = rep.limit_source;
    out.scale_iters = attempt;
    if (rep.ok) {
      out.ok = true;
      out.reason = "ok";
      return out;
    }
    const double stretch = std::max(
        {1.0, rep.max_vel_ratio,
         (rep.max_acc_ratio > 1e-12) ? std::sqrt(rep.max_acc_ratio) : 1.0,
         (rep.max_jerk_ratio > 1e-12) ? std::cbrt(rep.max_jerk_ratio) : 1.0});
    if (!(stretch > 1.0 + 1e-6)) {
      break;
    }
    scale_time_law(&out, stretch * kMargin);
  }

  const HermiteLimitReport final_rep =
      hermite_send_limit_report(samples, out, policy, stage_boundaries);
  out.max_vel_ratio = final_rep.max_vel_ratio;
  out.max_acc_ratio = final_rep.max_acc_ratio;
  out.max_jerk_ratio = final_rep.max_jerk_ratio;
  out.worst_joint = final_rep.worst_joint;
  out.limit_source = final_rep.limit_source;
  out.scale_iters = kMaxScaleIters;
  if (final_rep.ok) {
    out.ok = true;
    out.reason = "ok";
    return out;
  }
  out.ok = false;
  std::ostringstream os;
  os << "time_law_over_limit:src=" << final_rep.limit_source
     << ":joint=" << final_rep.worst_joint << ":rv=" << final_rep.max_vel_ratio
     << ":ra=" << final_rep.max_acc_ratio << ":rj=" << final_rep.max_jerk_ratio
     << ":dur=" << out.duration_s;
  out.reason = os.str();
  return out;
}

}  // namespace kmpc
