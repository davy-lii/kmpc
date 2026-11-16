#include "kmpc/trajectory_package.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "kmpc/motion_task.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
// v2: digest also covers path_id, stage boundaries, and the time law.
constexpr uint64_t kDigestVersion = 2;

uint64_t mix(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

uint64_t mix_double(uint64_t hash, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double must be 8 bytes");
  std::memcpy(&bits, &value, sizeof(bits));
  return mix(hash, bits);
}

uint64_t mix_vec(uint64_t hash, const Vec18& q) {
  for (int i = 0; i < kNDof; ++i) {
    hash = mix_double(hash, q[i]);
  }
  return hash;
}

uint64_t mix_str(uint64_t hash, const std::string& s) {
  hash = mix(hash, static_cast<uint64_t>(s.size()));
  for (char c : s) {
    hash = mix(hash, static_cast<uint64_t>(static_cast<unsigned char>(c)));
  }
  return hash;
}

}  // namespace

uint64_t package_digest(const TrajectoryPackage& package) {
  uint64_t hash = kFnvOffset;
  hash = mix(hash, kDigestVersion);
  hash = mix(hash, package.generation);
  hash = mix(hash, package.task_digest);
  hash = mix(hash, package.model_digest);
  hash = mix(hash, package.limits_digest);
  hash = mix(hash, package.path_id);
  hash = mix(hash, package.left ? 1u : 0u);
  hash = mix(hash, package.start_dq_known ? 1u : 0u);
  hash = mix_vec(hash, package.start_q);
  hash = mix_vec(hash, package.start_dq);
  hash = mix(hash, static_cast<uint64_t>(package.samples.size()));
  for (const Vec18& q : package.samples) {
    hash = mix_vec(hash, q);
  }
  hash = mix(hash, static_cast<uint64_t>(package.stage_boundaries.size()));
  for (const StageBoundary& b : package.stage_boundaries) {
    hash = mix(hash, static_cast<uint64_t>(b.sample_index));
    hash = mix(hash, b.stage_id);
    hash = mix(hash, b.stop_required ? 1u : 0u);
    hash = mix(hash, b.wait_for_external_ack ? 1u : 0u);
    hash = mix_double(hash, b.position_tol_m);
    hash = mix_double(hash, b.orientation_tol_rad);
    hash = mix(hash, b.has_orientation ? 1u : 0u);
  }
  const TimeLawTable& law = package.time_law;
  hash = mix(hash, static_cast<uint64_t>(law.s.size()));
  for (double v : law.s) {
    hash = mix_double(hash, v);
  }
  hash = mix(hash, static_cast<uint64_t>(law.t_s.size()));
  for (double v : law.t_s) {
    hash = mix_double(hash, v);
  }
  hash = mix_double(hash, law.duration_s);
  hash = mix(hash, law.software_limited ? 1u : 0u);
  hash = mix_str(hash, law.accel_source);
  hash = mix_str(hash, law.jerk_source);
  hash = mix_str(hash, package.planner);
  return hash;
}

uint64_t limits_digest_of(const MotionConstraintPolicy& policy) {
  uint64_t hash = kFnvOffset;
  hash = mix(hash, kDigestVersion);
  hash = mix_vec(hash, policy.q_min);
  hash = mix_vec(hash, policy.q_max);
  hash = mix_vec(hash, policy.dq_max);
  hash = mix_vec(hash, policy.ddq_max);
  hash = mix_vec(hash, policy.dddq_max);
  hash = mix_double(hash, policy.torso_pitch_max_rad);
  hash = mix_double(hash, policy.q_limit_margin_rad);
  hash = mix_double(hash, policy.yaw_abs_max_rad);
  hash = mix(hash, policy.yaw_limited ? 1u : 0u);
  hash = mix(hash, policy.a_max_calibrated ? 1u : 0u);
  hash = mix(hash, policy.j_max_calibrated ? 1u : 0u);
  return hash;
}

namespace {

uint64_t fnv1a_bytes(const std::string& bytes) {
  uint64_t hash = kFnvOffset;
  for (unsigned char c : bytes) {
    hash = mix(hash, static_cast<uint64_t>(c));
  }
  return hash;
}

std::string load_urdf_bytes(const std::string& urdf, bool is_xml) {
  if (is_xml) {
    return urdf;
  }
  std::ifstream in(urdf, std::ios::binary);
  if (!in) {
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string hex_u64(uint64_t v) {
  std::ostringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(16) << v;
  return ss.str();
}

}  // namespace

std::string model_identity_string(const std::string& urdf, bool is_xml,
                                  uint64_t limits_digest) {
  const std::string bytes = load_urdf_bytes(urdf, is_xml);
  const uint64_t content = fnv1a_bytes(bytes);
  return "urdf:" + hex_u64(content) + ":limits:" + hex_u64(limits_digest);
}

uint64_t seal_motion_task_identity(MotionTask* task, const std::string& urdf,
                                   bool is_xml,
                                   const MotionConstraintPolicy& policy) {
  if (task == nullptr) {
    return 0;
  }
  const uint64_t lim = limits_digest_of(policy);
  task->model_digest = model_identity_string(urdf, is_xml, lim);
  return motion_task_digest(*task);
}

double time_at_sample_index(const TrajectoryPackage& package,
                            std::size_t sample_index) {
  const auto& law = package.time_law;
  if (law.t_s.empty()) {
    return 0.0;
  }
  if (law.t_s.size() == package.samples.size() &&
      sample_index < law.t_s.size()) {
    return law.t_s[sample_index];
  }
  if (package.samples.size() < 2 || law.s.size() != law.t_s.size()) {
    return std::min(law.duration_s, law.t_s.back());
  }
  const double s_tgt = std::clamp(
      static_cast<double>(sample_index) /
          static_cast<double>(package.samples.size() - 1),
      0.0, 1.0);
  for (std::size_t i = 1; i < law.s.size(); ++i) {
    if (law.s[i] + 1e-12 >= s_tgt) {
      const double s0 = law.s[i - 1];
      const double s1 = law.s[i];
      const double t0 = law.t_s[i - 1];
      const double t1 = law.t_s[i];
      if (s1 - s0 <= 1e-12) {
        return t1;
      }
      if (law.ds_dt.size() == law.s.size()) {
        // Sparse time tables need the inverse of the executed cubic s(t),
        // not a linear inverse that would stop at the wrong joint sample.
        double lo = t0, hi = t1;
        for (int k = 0; k < 48; ++k) {
          const double mid = 0.5 * (lo + hi);
          TrajectoryTimeSample sample;
          if (!sample_trajectory_at(package, mid, &sample, false) || !sample.ok)
            return t1;
          if (sample.s < s_tgt) lo = mid;
          else hi = mid;
        }
        return 0.5 * (lo + hi);
      }
      const double a = (s_tgt - s0) / (s1 - s0);
      return t0 + a * (t1 - t0);
    }
  }
  return law.t_s.back();
}

std::vector<TrajectorySendProbe> trajectory_send_probe_grid(
    const TrajectoryPackage& package, double send_dt) {
  std::vector<TrajectorySendProbe> out;
  const double duration = package.time_law.duration_s;
  if (!(send_dt > 0.0) || !std::isfinite(duration) || duration < 0.0) return out;
  std::vector<double> stops;
  for (const auto& b : package.stage_boundaries) {
    if ((b.stop_required || b.wait_for_external_ack) &&
        b.sample_index > 0 && b.sample_index < package.samples.size()) {
      stops.push_back(time_at_sample_index(package, b.sample_index));
    }
  }
  stops.push_back(duration);
  std::sort(stops.begin(), stops.end());
  stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
  double sending = -3.0 * send_dt;
  auto push = [&](double curve) {
    sending += send_dt;
    out.push_back({sending, curve});
  };
  push(0.0); push(0.0); push(0.0);
  double start = 0.0;
  for (double end : stops) {
    end = std::clamp(end, start, duration);
    const int ticks = static_cast<int>(std::ceil((end - start) / send_dt));
    for (int i = 1; i <= ticks; ++i)
      push(std::min(end, start + i * send_dt));
    push(end); push(end);
    start = end;
  }
  return out;
}

AdmitDecision admit_package_start(
    const TrajectoryPackage& package, const Vec18& q, const Vec18& dq,
    bool dq_known, double q_tol_rad, double dq_tol) {
  AdmitDecision decision;
  if (!q.allFinite() || (dq_known && !dq.allFinite())) {
    decision.reason = "start_nonfinite";
    return decision;
  }
  if ((package.start_q - q).cwiseAbs().maxCoeff() > q_tol_rad) {
    decision.reason = "start_q_mismatch";
    return decision;
  }
  if (!package.samples.empty() &&
      (package.samples.front() - package.start_q).cwiseAbs().maxCoeff() >
          q_tol_rad) {
    decision.reason = "curve_start_rewritten";
    return decision;
  }
  if (package.start_dq_known != dq_known) {
    decision.reason = "start_dq_unknown";
    return decision;
  }
  if (package.start_dq_known &&
      (package.start_dq - dq).cwiseAbs().maxCoeff() > dq_tol) {
    decision.reason = "start_dq_mismatch";
    return decision;
  }
  decision.ok = true;
  decision.reason = "ok";
  return decision;
}

bool sample_trajectory_at(const TrajectoryPackage& package, double t_s,
                          TrajectoryTimeSample* out, bool need_derivatives) {
  if (out == nullptr) {
    return false;
  }
  *out = TrajectoryTimeSample{};
  const JointCurve& samples = package.samples;
  const TimeLawTable& law = package.time_law;
  if (samples.empty()) {
    out->reason = "empty_samples";
    return false;
  }
  if (samples.size() == 1) {
    out->q = samples.front();
    out->dq.setZero();
    out->ddq.setZero();
    out->s = 0.0;
    out->t_s = 0.0;
    out->ok = true;
    out->reason = "ok";
    return true;
  }
  if (law.t_s.size() < 2 || law.s.size() != law.t_s.size()) {
    out->reason = "bad_time_law";
    return false;
  }
  const double t_end = std::max(law.duration_s, law.t_s.back());
  const double t = std::clamp(t_s, 0.0, t_end);
  out->t_s = t;

  auto s_at_t = [&](double tt) -> double {
    std::size_t i = 0;
    while (i + 1 < law.t_s.size() && law.t_s[i + 1] < tt - 1e-15) {
      ++i;
    }
    const std::size_t i1 = std::min(i + 1, law.t_s.size() - 1);
    const double t0 = law.t_s[i];
    const double t1 = law.t_s[i1];
    const double s0 = law.s[i];
    const double s1 = law.s[i1];
    if (t1 - t0 <= 1e-12) {
      return s1;
    }
    const double h = t1 - t0;
    const double a = std::clamp((tt - t0) / h, 0.0, 1.0);
    if (law.ds_dt.size() <= i1) return s0 + a * (s1 - s0);
    // Honour planned endpoint/STOP/ACK rates. Linear interpolation ignored
    // ds_dt=0 and turned every stop into an abrupt velocity discontinuity.
    const double secant = (s1 - s0) / h;
    if (!(secant > 0.0)) return s0;
    double m0 = std::clamp(law.ds_dt[i], 0.0, 3.0 * secant);
    double m1 = std::clamp(law.ds_dt[i1], 0.0, 3.0 * secant);
    const double norm = std::hypot(m0 / secant, m1 / secant);
    if (norm > 3.0) { m0 *= 3.0 / norm; m1 *= 3.0 / norm; }
    const double a2 = a * a, a3 = a2 * a;
    return std::clamp((2*a3 - 3*a2 + 1)*s0 + (a3 - 2*a2 + a)*h*m0 +
                      (-2*a3 + 3*a2)*s1 + (a3 - a2)*h*m1, s0, s1);
  };

  const double s = s_at_t(t);
  out->s = s;
  const double prog = s * static_cast<double>(samples.size() - 1);
  out->q = hermite_joint_at(samples, prog);

  if (!need_derivatives) {
    out->dq.setZero();
    out->ddq.setZero();
    if (!out->q.allFinite()) {
      out->reason = "nonfinite_sample";
      out->ok = false;
      return false;
    }
    out->ok = true;
    out->reason = "ok";
    return true;
  }

  // Time derivatives via central differences on the executed Hermite curve.
  const double h = 1e-3;
  const double s_m = s_at_t(std::max(0.0, t - h));
  const double s_p = s_at_t(std::min(t_end, t + h));
  const Vec18 q_m =
      hermite_joint_at(samples, s_m * static_cast<double>(samples.size() - 1));
  const Vec18 q_p =
      hermite_joint_at(samples, s_p * static_cast<double>(samples.size() - 1));
  const double denom = std::max(2.0 * h, 1e-9);
  out->dq = (q_p - q_m) / denom;
  // Second derivative: (q_p - 2 q + q_m) / h^2
  out->ddq = (q_p - 2.0 * out->q + q_m) / (h * h);
  if (!out->q.allFinite() || !out->dq.allFinite() || !out->ddq.allFinite()) {
    out->reason = "nonfinite_sample";
    out->ok = false;
    return false;
  }
  out->ok = true;
  out->reason = "ok";
  return true;
}

}  // namespace kmpc
