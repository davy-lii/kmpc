#include "kmpc/trajectory_cost.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <yaml-cpp/yaml.h>

#include "kmpc/motion_evaluator.hpp"

namespace kmpc {
namespace {

constexpr double kLimitEpsRad = 1e-3;
constexpr double kComMarginRefM = 0.025;
// softplus input beyond |30| is linear / zero to machine precision.
constexpr double kSoftplusClamp = 30.0;

double softplus(double z) {
  if (z > kSoftplusClamp) {
    return z;
  }
  if (z < -kSoftplusClamp) {
    return std::exp(z);
  }
  return std::log1p(std::exp(z));
}

double sigmoid(double z) {
  if (z >= 0.0) {
    const double e = std::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

double scaled_sq(const Vec18& dq, const Vec18& scale) {
  double sum = 0.0;
  for (int i = 0; i < kNDof; ++i) {
    const double v = scale[i] * dq[i];
    sum += v * v;
  }
  return sum;
}

// Gradient / diagonal-Gauss-Newton accumulator over knots x active joints.
// Layout is knot-major: index k * na + i addresses knot k, active slot i.
// Every add already carries the term weight, so grad/hess match value.total.
struct LinAccum {
  const std::vector<int>* active = nullptr;
  std::size_t knots = 0;
  std::vector<double> grad;
  std::vector<double> hess;

  void reset(std::size_t k, const std::vector<int>* act) {
    knots = k;
    active = act;
    if (active != nullptr) {
      grad.assign(knots * active->size(), 0.0);
      hess.assign(knots * active->size(), 0.0);
    } else {
      grad.clear();
      hess.clear();
    }
  }

  bool on() const { return active != nullptr; }

  void add(std::size_t k, int joint, double g, double h) {
    if (!on()) {
      return;
    }
    const int na = static_cast<int>(active->size());
    for (int i = 0; i < na; ++i) {
      if ((*active)[static_cast<std::size_t>(i)] == joint) {
        const std::size_t idx =
            k * static_cast<std::size_t>(na) + static_cast<std::size_t>(i);
        grad[idx] += g;
        hess[idx] += std::max(h, 0.0);
        return;
      }
    }
  }
};

// One cost for solve, score, and logs. kin == nullptr evaluates only the
// joint-space terms (motion / smooth / neutral / limit / pitch comfort).
PathCostBreakdown accumulate_cost(
    const JointCurve& samples, const Vec18& q_ref, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics* kin, bool left, const WholeBodyGuards* guards,
    LinAccum* lin) {
  PathCostBreakdown out;
  out.calibrated = weights.calibrated;
  // The height switch is gone for good. This flag must never flip back on.
  out.depends_on_target_height = false;
  const std::size_t n = samples.size();
  if (n == 0 || !q_neutral.allFinite()) {
    return out;
  }
  const double inv_n = 1.0 / static_cast<double>(n);

  // J_move: mean squared scaled step.
  if (n >= 2) {
    const double inv_steps = 1.0 / static_cast<double>(n - 1);
    for (std::size_t i = 1; i < n; ++i) {
      out.motion += scaled_sq(samples[i] - samples[i - 1], weights.motion_scale);
    }
    out.motion *= inv_steps;
    if (lin != nullptr && lin->on()) {
      const double w = weights.move * 2.0 * inv_steps;
      for (std::size_t i = 0; i < n; ++i) {
        for (int j = 0; j < kNDof; ++j) {
          const double s2 = weights.motion_scale[j] * weights.motion_scale[j];
          double g = 0.0;
          double hdiag = 0.0;
          if (i > 0) {
            g += s2 * (samples[i][j] - samples[i - 1][j]);
            hdiag += s2;
          }
          if (i + 1 < n) {
            g += s2 * (samples[i][j] - samples[i + 1][j]);
            hdiag += s2;
          }
          lin->add(i, j, w * g, w * hdiag);
        }
      }
    }
  }

  // J_smooth: mean squared scaled second difference.
  if (n >= 3) {
    const double inv = 1.0 / static_cast<double>(n - 2);
    for (std::size_t i = 1; i + 1 < n; ++i) {
      const Vec18 acc = samples[i + 1] - 2.0 * samples[i] + samples[i - 1];
      out.smooth += scaled_sq(acc, weights.motion_scale);
      if (lin != nullptr && lin->on()) {
        for (int j = 0; j < kNDof; ++j) {
          const double s2 = weights.motion_scale[j] * weights.motion_scale[j];
          const double r = acc[j];
          const double w = weights.smooth * 2.0 * inv * s2;
          lin->add(i - 1, j, w * r, w);
          lin->add(i, j, w * (-2.0 * r), w * 4.0);
          lin->add(i + 1, j, w * r, w);
        }
      }
    }
    out.smooth *= inv;
  }

  // J_neutral + J_joint_limit (barrier -log) + J_pitch_comfort per sample.
  const double pitch_scale =
      weights.pitch_scale_rad > 1e-9 ? weights.pitch_scale_rad : 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    const Vec18& q = samples[i];
    out.neutral += scaled_sq(q - q_neutral, Vec18::Ones());
    if (lin != nullptr && lin->on()) {
      const double w = weights.neutral * 2.0 * inv_n;
      for (int j = 0; j < kNDof; ++j) {
        lin->add(i, j, w * (q[j] - q_neutral[j]), w);
      }
    }
    for (int j = 0; j < kNDof; ++j) {
      const double range = policy.q_max[j] - policy.q_min[j];
      if (!std::isfinite(q[j]) || !(range > 0.0)) {
        out.limit_outside_hard_box = true;
        out.limit += 1.0e6;
        continue;
      }
      const double margin =
          std::min(q[j] - policy.q_min[j], policy.q_max[j] - q[j]);
      const double side =
          (q[j] - policy.q_min[j]) <= (policy.q_max[j] - q[j]) ? 1.0 : -1.0;
      if (margin < -kHardBoxSlackRad) {
        out.limit_outside_hard_box = true;
        out.limit += 1.0e6;
        if (lin != nullptr && lin->on()) {
          // Pull back inside with the steepest barrier slope we admit.
          lin->add(i, j, -weights.joint_limit * inv_n * side / kLimitEpsRad,
                   weights.joint_limit * inv_n / (kLimitEpsRad * kLimitEpsRad));
        }
        continue;
      }
      const double m = std::clamp(margin, kLimitEpsRad, range);
      out.limit += -std::log(m / range);
      if (lin != nullptr && lin->on()) {
        // b = -log(m/range); db/dq = -(1/m) dm/dq; d2b = 1/m^2.
        lin->add(i, j, -weights.joint_limit * inv_n * side / m,
                 weights.joint_limit * inv_n / (m * m));
      }
    }
    // J_pitch_comfort: keep the waist pitch sum near level.
    const double pitch = q[0] + q[1] + q[2];
    const double rp = pitch / pitch_scale;
    out.pitch += rp * rp;
    if (lin != nullptr && lin->on()) {
      const double w = weights.pitch_comfort * inv_n;
      const double g = w * 2.0 * pitch / (pitch_scale * pitch_scale);
      const double h = w * 2.0 / (pitch_scale * pitch_scale);
      for (int j = 0; j < 3; ++j) {
        lin->add(i, j, g, h);
      }
    }
  }
  out.neutral *= inv_n;
  out.limit *= inv_n;
  out.pitch *= inv_n;

  // Terms that need FK / CoM.
  if (kin != nullptr) {
    MotionEvaluator eval(*kin);
    std::vector<MotionSnapshot, Eigen::aligned_allocator<MotionSnapshot>> snaps;
    eval.batch(samples, left, &snaps);
    MotionSnapshot ref;
    eval.evaluate(q_ref, left, &ref);
    const double ell = weights.body_forward_len_m > 1e-9
                           ? weights.body_forward_len_m
                           : 0.05;
    const double d0 = ref.body_minus_leg_x;
    const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
    const int elbow = left ? kIdxJ4L : kIdxJ4R;
    for (std::size_t i = 0; i < n; ++i) {
      const MotionSnapshot& s = snaps[i];
      // J_body: softplus(d_x/ell)^2 with d_x the (Bx-Lx) growth relative to
      // start. d_z is only reported; there is no negative reward on it.
      const double dx = s.body_minus_leg_x - d0;
      const double z = dx / ell;
      const double sp = softplus(z);
      out.body_forward += sp * sp;
      if (lin != nullptr && lin->on()) {
        const double sg = sigmoid(z);
        const double gd = weights.body_forward * 2.0 * sp * sg / ell * inv_n;
        const double hd =
            weights.body_forward * 2.0 * (sg / ell) * (sg / ell) * inv_n;
        for (int j = 0; j < kNDof; ++j) {
          const double dd = s.body_jac(0, j) - s.leg_jac(0, j);
          if (dd != 0.0) {
            lin->add(i, j, gd * dd, hd * dd * dd);
          }
        }
      }
      // J_com_margin: softplus barrier on the CoM xy work walls.
      if (guards != nullptr) {
        const double m_xlo = s.com.x() - guards->com_work_xmin();
        const double m_xhi = guards->com_work_xmax() - s.com.x();
        const double m_ylo = s.com.y() - guards->com_work_ymin();
        const double m_yhi = guards->com_work_ymax() - s.com.y();
        double m = m_xlo;
        int axis = 0;
        double dir = 1.0;
        if (m_xhi < m) {
          m = m_xhi;
          axis = 0;
          dir = -1.0;
        }
        if (m_ylo < m) {
          m = m_ylo;
          axis = 1;
          dir = 1.0;
        }
        if (m_yhi < m) {
          m = m_yhi;
          axis = 1;
          dir = -1.0;
        }
        const double zc = -m / kComMarginRefM;
        const double spc = softplus(zc);
        out.com_margin += spc * spc;
        if (lin != nullptr && lin->on()) {
          const double sgc = sigmoid(zc);
          const double gm = weights.com_margin * 2.0 * spc * sgc *
                            (-1.0 / kComMarginRefM) * inv_n;
          const double hm = weights.com_margin * 2.0 *
                            (sgc / kComMarginRefM) * (sgc / kComMarginRefM) *
                            inv_n;
          for (int j = 0; j < kNDof; ++j) {
            const double dm = dir * s.j_com(axis, j);
            if (dm != 0.0) {
              lin->add(i, j, gm * dm, hm * dm * dm);
            }
          }
        }
      }
      // J_arm_manip (approximate): wrist + elbow distance to mid-range.
      double manip = 0.0;
      const int joints[4] = {elbow, arm + 4, arm + 5, arm + 6};
      for (int j : joints) {
        const double half = 0.5 * (policy.q_max[j] - policy.q_min[j]);
        if (!(half > 1e-6)) {
          continue;
        }
        const double mid = 0.5 * (policy.q_max[j] + policy.q_min[j]);
        const double r = (samples[i][j] - mid) / half;
        manip += r * r;
        if (lin != nullptr && lin->on()) {
          const double w = weights.arm_manip * 0.25 * inv_n;
          lin->add(i, j, w * 2.0 * r / half, w * 2.0 / (half * half));
        }
      }
      out.arm_manip += 0.25 * manip;
    }
    out.body_forward *= inv_n;
    out.com_margin *= inv_n;
    out.arm_manip *= inv_n;
    const MotionSnapshot& first = snaps.front();
    const MotionSnapshot& last = snaps.back();
    out.body_dx_m = last.body.x() - first.body.x();
    out.body_dz_m = last.body.z() - first.body.z();
    out.body_minus_leg_dx_m = last.body_minus_leg_x - first.body_minus_leg_x;
  }

  out.motion *= weights.move;
  out.smooth *= weights.smooth;
  out.body_forward *= weights.body_forward;
  out.neutral *= weights.neutral;
  out.limit *= weights.joint_limit;
  out.pitch *= weights.pitch_comfort;
  out.com_margin *= weights.com_margin;
  out.arm_manip *= weights.arm_manip;
  out.total = out.motion + out.smooth + out.body_forward + out.neutral +
              out.limit + out.pitch + out.com_margin + out.arm_manip;
  return out;
}

}  // namespace

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights) {
  if (samples.empty()) {
    PathCostBreakdown out;
    out.calibrated = weights.calibrated;
    return out;
  }
  return accumulate_cost(samples, samples.front(), q_neutral, policy, weights,
                         nullptr, true, nullptr, nullptr);
}

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards) {
  if (samples.empty()) {
    PathCostBreakdown out;
    out.calibrated = weights.calibrated;
    return out;
  }
  return accumulate_cost(samples, samples.front(), q_neutral, policy, weights,
                         &kin, left, guards, nullptr);
}

CostLinearization value_and_linearize(
    const JointCurve& knots, const Vec18& q_start, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards,
    const std::vector<int>& active_joints) {
  CostLinearization out;
  LinAccum lin;
  lin.reset(knots.size(), &active_joints);
  out.value = accumulate_cost(knots, q_start, q_neutral, policy, weights, &kin,
                              left, guards, &lin);
  out.grad = std::move(lin.grad);
  out.hess_diag = std::move(lin.hess);
  return out;
}

TrajectoryCostWeights default_cost_weights(const Vec18& dq_max) {
  TrajectoryCostWeights weights;
  for (int i = 0; i < kNDof; ++i) {
    weights.motion_scale[i] = dq_max[i] > 1e-6 ? 1.0 / dq_max[i] : 1.0;
  }
  weights.calibrated = false;
  weights.calibration_version = "default_uncalibrated";
  return weights;
}

TrajectoryCostWeights load_trajectory_quality_weights(
    const std::string& yaml_path, const Vec18& dq_max, std::string* why) {
  TrajectoryCostWeights weights = default_cost_weights(dq_max);
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    const YAML::Node tq = root["trajectory_quality"]
                              ? root["trajectory_quality"]
                              : root;
    if (tq["calibration_version"]) {
      weights.calibration_version = tq["calibration_version"].as<std::string>();
    }
    if (tq["calibrated"]) {
      weights.calibrated = tq["calibrated"].as<bool>();
    } else {
      weights.calibrated = false;
    }
    const YAML::Node w = tq["weights"] ? tq["weights"] : tq;
    auto rd = [&](const char* key, double* dst) {
      if (w[key]) {
        *dst = w[key].as<double>();
      }
    };
    rd("move", &weights.move);
    rd("smooth", &weights.smooth);
    rd("body_forward", &weights.body_forward);
    rd("neutral", &weights.neutral);
    rd("joint_limit", &weights.joint_limit);
    rd("pitch_comfort", &weights.pitch_comfort);
    rd("com_margin", &weights.com_margin);
    rd("arm_manip", &weights.arm_manip);
    if (tq["normalization"]) {
      const YAML::Node n = tq["normalization"];
      if (n["body_forward_len_m"]) {
        weights.body_forward_len_m = n["body_forward_len_m"].as<double>();
      }
      if (n["pitch_scale_rad"]) {
        weights.pitch_scale_rad = n["pitch_scale_rad"].as<double>();
      }
    }
    if (why != nullptr) {
      *why = "ok";
    }
  } catch (const std::exception& e) {
    if (why != nullptr) {
      *why = std::string("quality_yaml_fail:") + e.what();
    }
    weights = default_cost_weights(dq_max);
    weights.calibrated = false;
  }
  return weights;
}

}  // namespace kmpc
