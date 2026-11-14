#pragma once
// Immutable adoption check. This function does not rewrite the curve.

#include <cstdint>
#include <vector>

#include "kmpc/types.hpp"

namespace kmpc {

using JointCurve = std::vector<Vec18, Eigen::aligned_allocator<Vec18>>;

struct TrajectoryPackage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t generation = 0;
  Vec18 start_q = Vec18::Zero();
  Vec18 start_dq = Vec18::Zero();
  bool start_dq_known = false;
  JointCurve samples;
};

struct AdmitDecision {
  bool ok = false;
  const char* reason = "not_checked";
};

// FNV-1a over the fields adopt is allowed to trust. Not a cryptographic hash.
uint64_t package_digest(const TrajectoryPackage& package);

// Rejects a measured start that disagrees with the certified start.
// Does not write package.samples.front(). A bridge segment is not built here.
AdmitDecision admit_package_start(
    const TrajectoryPackage& package, const Vec18& q, const Vec18& dq,
    bool dq_known, double q_tol_rad, double dq_tol);

}  // namespace kmpc
