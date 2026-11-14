#include "kmpc/trajectory_package.hpp"

#include <cstring>

namespace kmpc {
namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr uint64_t kDigestVersion = 1;

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

}  // namespace

uint64_t package_digest(const TrajectoryPackage& package) {
  uint64_t hash = kFnvOffset;
  hash = mix(hash, kDigestVersion);
  hash = mix(hash, package.generation);
  hash = mix(hash, package.start_dq_known ? 1u : 0u);
  hash = mix_vec(hash, package.start_q);
  hash = mix_vec(hash, package.start_dq);
  hash = mix(hash, static_cast<uint64_t>(package.samples.size()));
  for (const Vec18& q : package.samples) {
    hash = mix_vec(hash, q);
  }
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
