#pragma once
// Shared async line-build handoff gate used by the production ROS node and
// offline tests. Keep this the single source of truth for "may adopt worker
// plan onto main tracker?".

#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"

namespace kmpc {

struct LineAsyncPendingSeed {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t generation = 0;
  bool left = true;
  bool waist_locked = false;
  Vec3 xyz = Vec3::Zero();
  Vec4 quat = Vec4(0.0, 0.0, 0.0, 1.0);
  Vec18 q_seed = Vec18::Zero();
};

// True only when the finished worker plan still matches the live target and
// the seed pose used when the worker was launched. Covers the case where a
// newer generation/target became active while the old future was building.
inline bool line_async_plan_matches_live(
    const LineAsyncPendingSeed& pending, uint64_t live_generation,
    bool live_left, bool live_waist_locked, const Vec3& live_xyz,
    const Vec4& live_quat, const Vec18& live_q0) {
  return pending.generation == live_generation && pending.left == live_left &&
         pending.waist_locked == live_waist_locked &&
         (pending.xyz - live_xyz).norm() < 0.008 &&
         ori_error_xyzw(pending.quat, live_quat).norm() < kTargetJumpOriRad &&
         (pending.q_seed - live_q0).cwiseAbs().maxCoeff() < 1e-9;
}

}  // namespace kmpc
