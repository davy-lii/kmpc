#include "kmpc/trajectory_cost.hpp"

#include <algorithm>
#include <cmath>

namespace kmpc {
namespace {

constexpr double kLimitEpsRad = 1e-3;

double scaled_step(const Vec18& dq, const Vec18& scale) {
  double sum = 0.0;
  for (int i = 0; i < kNDof; ++i) {
    const double scaled = scale[i] * dq[i];
    sum += scaled * scaled;
  }
  return sum;
}

}  // namespace

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights) {
  PathCostBreakdown out;
  out.calibrated = weights.calibrated;
  out.depends_on_target_height = false;
  if (samples.size() < 2 || !q_neutral.allFinite()) {
    out.total = 0.0;
    return out;
  }
  const double steps = static_cast<double>(samples.size() - 1);
  for (std::size_t i = 1; i < samples.size(); ++i) {
    out.motion += scaled_step(samples[i] - samples[i - 1], weights.motion_scale);
  }
  out.motion /= steps;
  if (samples.size() >= 3) {
    const double accel_steps = static_cast<double>(samples.size() - 2);
    for (std::size_t i = 1; i + 1 < samples.size(); ++i) {
      const Vec18 accel =
          samples[i + 1] - 2.0 * samples[i] + samples[i - 1];
      out.smooth += scaled_step(accel, weights.motion_scale);
    }
    out.smooth /= accel_steps;
  }
  for (const Vec18& q : samples) {
    out.neutral += scaled_step(q - q_neutral, Vec18::Ones());
    for (int j = 0; j < kNDof; ++j) {
      const double margin = std::min(
          q[j] - policy.q_min[j], policy.q_max[j] - q[j]);
      if (!std::isfinite(q[j]) || margin < -kHardBoxSlackRad) {
        out.limit_outside_hard_box = true;
        out.limit += 1.0e6;
        continue;
      }
      const double clearance = std::max(margin, 0.0);
      const double ratio = kLimitEpsRad / (clearance + kLimitEpsRad);
      out.limit += ratio * ratio;
    }
  }
  out.neutral /= static_cast<double>(samples.size());
  out.limit /= static_cast<double>(samples.size());
  out.neutral *= weights.neutral;
  out.limit *= weights.limit;
  out.total = out.motion + out.smooth + out.neutral + out.limit;
  return out;
}

}  // namespace kmpc
