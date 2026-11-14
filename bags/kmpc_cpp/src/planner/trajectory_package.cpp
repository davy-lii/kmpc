#include "kmpc/trajectory_package.hpp"

#include <cstring>

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

}  // namespace kmpc
