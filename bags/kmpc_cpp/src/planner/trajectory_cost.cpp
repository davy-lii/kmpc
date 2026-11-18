#include "kmpc/trajectory_cost.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kmpc/motion_evaluator.hpp"

namespace kmpc {

double joint_path_foldback_rad(const JointCurve& samples, int j) {
  if (samples.size() < 2) {
    return 0.0;
  }
  double travel = 0.0;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    travel += std::abs(samples[i][j] - samples[i - 1][j]);
  }
  const double net = std::abs(samples.back()[j] - samples.front()[j]);
  return std::max(0.0, travel - net);
}

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
    LinAccum* lin, const std::vector<int>* stage_knot_ends = nullptr) {
  PathCostBreakdown out;
  out.calibrated = weights.calibrated;
  // The height switch is gone for good. This flag must never flip back on.
  out.depends_on_target_height = false;
  const std::size_t n = samples.size();
  if (n == 0 || !q_neutral.allFinite()) {
    return out;
  }
  const double inv_n = 1.0 / static_cast<double>(n);
  Vec18 work_lo = Vec18::Zero();
  Vec18 work_hi = Vec18::Zero();
  joint_work_limits(policy.q_min, policy.q_max, policy.q_limit_margin_rad,
                    work_lo, work_hi);

  // J_move: mean squared scaled step, with directional chan_gain (IRWLS).
  // Gain is frozen from the current iterate; inward exit stays at 1.
  if (n >= 2) {
    const double steps = static_cast<double>(n - 1);
    const double inv_steps = 1.0 / steps;
    for (std::size_t i = 1; i < n; ++i) {
      for (int j = 0; j < kNDof; ++j) {
        const double dq = samples[i][j] - samples[i - 1][j];
        const double gain =
            joint_chan_gain(samples[i][j], work_lo[j], work_hi[j], dq);
        const double v = gain * weights.motion_scale[j] * dq;
        out.motion += v * v;
      }
    }
    out.motion *= inv_steps;
    if (lin != nullptr && lin->on()) {
      const double w = weights.move * 2.0 * inv_steps;
      for (std::size_t i = 0; i < n; ++i) {
        for (int j = 0; j < kNDof; ++j) {
          double g = 0.0;
          double hdiag = 0.0;
          if (i > 0) {
            const double dq = samples[i][j] - samples[i - 1][j];
            const double gain =
                joint_chan_gain(samples[i][j], work_lo[j], work_hi[j], dq);
            const double s2 = (gain * weights.motion_scale[j]) *
                              (gain * weights.motion_scale[j]);
            g += s2 * dq;
            hdiag += s2;
          }
          if (i + 1 < n) {
            const double dq = samples[i + 1][j] - samples[i][j];
            const double gain = joint_chan_gain(samples[i + 1][j], work_lo[j],
                                                work_hi[j], dq);
            const double s2 = (gain * weights.motion_scale[j]) *
                              (gain * weights.motion_scale[j]);
            g -= s2 * dq;
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
    // J_pitch_comfort: mild deviation from start pitch (not pull toward 0°).
    const double pitch = q[0] + q[1] + q[2];
    const double pitch_ref = q_ref[0] + q_ref[1] + q_ref[2];
    const double e = (pitch - pitch_ref) / pitch_scale;
    out.pitch += e * e;
    if (lin != nullptr && lin->on()) {
      const double w = weights.pitch_comfort * inv_n;
      const double g = w * 2.0 * e / (pitch_scale * pitch_scale);
      const double h = w * 2.0 / (pitch_scale * pitch_scale);
      for (int j = 0; j < 3; ++j) {
        lin->add(i, j, g, h);
      }
    }
  }
  out.neutral *= inv_n;
  out.limit *= inv_n;
  out.pitch *= inv_n;

  // J_j4_twist: disabled at weight 0; foldback cost still penalizes J4 reversals.
  if (weights.j4_twist > 1e-9) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double e = samples[i][kIdxYaw] - q_ref[kIdxYaw];
      acc += e * e;
    }
    out.j4_twist = acc * inv_n;
    if (lin != nullptr && lin->on()) {
      const double w = weights.j4_twist * 2.0 * inv_n;
      for (std::size_t i = 0; i < n; ++i) {
        const double e = samples[i][kIdxYaw] - q_ref[kIdxYaw];
        lin->add(i, kIdxYaw, w * e, w);
      }
    }
  }

  // J_foldback: whole-path (travel − |net|)^2 on waist J4 + active wrist J6/J7.
  // Same ruler as joint_path_foldback_rad (severe / logs / gtests). Other DOFs
  // stay off foldback so GN attributes cross-stage reversals to these joints.
  (void)stage_knot_ends;
  if (n >= 2) {
    const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
    const int wrist6 = arm + 5;
    const int wrist7 = arm + 6;
    const int fold_joints[3] = {kIdxYaw, wrist6, wrist7};
    const std::size_t i0 = 0;
    const std::size_t i1 = n - 1;
    double fold_sum = 0.0;
    for (int j : fold_joints) {
      double travel = 0.0;
      for (std::size_t i = i0 + 1; i <= i1; ++i) {
        travel += std::abs(samples[i][j] - samples[i - 1][j]);
      }
      const double net = samples[i1][j] - samples[i0][j];
      const double abs_net = std::abs(net);
      const double excess = std::max(0.0, travel - abs_net);
      const double wj = (j == wrist6 || j == wrist7) ? weights.wrist_foldback
                                                     : weights.foldback;
      fold_sum += wj * excess * excess;
      if (lin != nullptr && lin->on() && excess > 1e-9) {
        const double g_ex = wj * 2.0 * excess;
        for (std::size_t i = i0 + 1; i <= i1; ++i) {
          const double dq = samples[i][j] - samples[i - 1][j];
          if (std::abs(dq) < 1e-12) {
            continue;
          }
          const double s = dq > 0.0 ? 1.0 : -1.0;
          lin->add(i, j, g_ex * s, wj * 2.0);
          lin->add(i - 1, j, -g_ex * s, wj * 2.0);
        }
        if (abs_net > 1e-12) {
          const double sn = net > 0.0 ? 1.0 : -1.0;
          lin->add(i1, j, -g_ex * sn, wj * 2.0);
          lin->add(i0, j, g_ex * sn, wj * 2.0);
        }
      }
    }
    out.foldback = fold_sum;
  }

  // J_wrist_span: softplus(span/soft - 1)^2 on J6/J7 peak-to-peak.
  // Catches monotone overshoot that foldback misses when flips≈0.
  {
    const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
    const int wrists[2] = {arm + 5, arm + 6};
    const double soft = weights.wrist_span_soft_rad > 1e-6
                            ? weights.wrist_span_soft_rad
                            : 0.40;
    for (int j : wrists) {
      double lo = samples[0][j];
      double hi = samples[0][j];
      std::size_t ilo = 0;
      std::size_t ihi = 0;
      for (std::size_t i = 1; i < n; ++i) {
        if (samples[i][j] < lo) {
          lo = samples[i][j];
          ilo = i;
        }
        if (samples[i][j] > hi) {
          hi = samples[i][j];
          ihi = i;
        }
      }
      const double span = hi - lo;
      const double z = span / soft - 1.0;
      const double sp = softplus(z);
      out.wrist_span += sp * sp;
      if (lin != nullptr && lin->on() && span > 1e-9) {
        const double sg = sigmoid(z);
        const double g = weights.wrist_span * 2.0 * sp * sg / soft;
        const double h =
            weights.wrist_span * 2.0 * (sg / soft) * (sg / soft);
        // Push extremes toward each other.
        lin->add(ihi, j, g, h);
        lin->add(ilo, j, -g, h);
      }
    }
  }

  // J_pitch_near_barrier: softplus(|pitch|/scale - 0.85)^2 — allows useful pitch
  // but grows near the hard ±32° comfort scale (hard box remains separate).
  {
    const double barrier = 0.85;
    for (std::size_t i = 0; i < n; ++i) {
      const double pitch = samples[i][0] + samples[i][1] + samples[i][2];
      const double z = std::abs(pitch) / pitch_scale - barrier;
      const double sp = softplus(z);
      out.pitch_near_barrier += sp * sp;
      if (lin != nullptr && lin->on()) {
        const double sg = sigmoid(z);
        const double sign = pitch >= 0.0 ? 1.0 : -1.0;
        const double g = weights.pitch_near_barrier * 2.0 * sp * sg *
                         (sign / pitch_scale) * inv_n;
        const double h = weights.pitch_near_barrier * 2.0 *
                         (sg / pitch_scale) * (sg / pitch_scale) * inv_n;
        for (int j = 0; j < 3; ++j) {
          lin->add(i, j, g, h);
        }
      }
    }
    out.pitch_near_barrier *= inv_n;
  }

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

    // J_height_coop: when the TCP task needs a drop, prefer matching Body_Link
    // drop. Continuous soft mismatch — not a height-switch hard mode.
    // task_tcp_dz_m is filled by the planner when known; else infer from EE.
    double need_dz = out.task_tcp_dz_m;
    if (!std::isfinite(need_dz) || need_dz == 0.0) {
      const Vec3 ee0 = kin->fk_pos(samples.front(), left ? std::string(kEeFrameLeft)
                                                         : std::string(kEeFrameRight));
      const Vec3 ee1 = kin->fk_pos(samples.back(), left ? std::string(kEeFrameLeft)
                                                        : std::string(kEeFrameRight));
      need_dz = ee1.z() - ee0.z();
    }
    const double ell_h = weights.height_coop_len_m > 1e-9
                             ? weights.height_coop_len_m
                             : 0.08;
    if (need_dz < -0.02) {
      // Want body_dz roughly in [need_dz, 0.25*need_dz] (drop with TCP).
      const double want = 0.55 * need_dz;
      const double miss = (out.body_dz_m - want) / ell_h;
      // Penalize both insufficient drop (body stays high) and oversquat.
      const double sp = softplus(miss);  // body_dz > want (less drop) → cost
      const double over = softplus((-out.body_dz_m + need_dz) / ell_h);
      out.height_coop = sp * sp + 0.35 * over * over;
      if (lin != nullptr && lin->on()) {
        const MotionSnapshot& last_s = snaps.back();
        const double sg = sigmoid(miss);
        const double g = weights.height_coop * 2.0 * sp * sg / ell_h;
        const double h =
            weights.height_coop * 2.0 * (sg / ell_h) * (sg / ell_h);
        for (int j = 0; j < kNDof; ++j) {
          const double dz = last_s.body_jac(2, j);
          if (dz != 0.0) {
            lin->add(n - 1, j, g * dz, h * dz * dz);
          }
        }
      }
    }
  }

  out.motion *= weights.move;
  out.smooth *= weights.smooth;
  out.body_forward *= weights.body_forward;
  out.neutral *= weights.neutral;
  out.limit *= weights.joint_limit;
  out.pitch *= weights.pitch_comfort;
  out.com_margin *= weights.com_margin;
  out.arm_manip *= weights.arm_manip;
  out.j4_twist *= weights.j4_twist;
  // foldback already includes per-joint weights.
  out.wrist_span *= weights.wrist_span;
  out.height_coop *= weights.height_coop;
  out.pitch_near_barrier *= weights.pitch_near_barrier;
  out.total = out.motion + out.smooth + out.body_forward + out.neutral +
              out.limit + out.pitch + out.com_margin + out.arm_manip +
              out.j4_twist + out.foldback + out.wrist_span + out.height_coop +
              out.pitch_near_barrier;
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
                         nullptr, true, nullptr, nullptr, nullptr);
}

PathCostBreakdown evaluate_path_cost(
    const JointCurve& samples, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards,
    const std::vector<int>* stage_knot_ends) {
  if (samples.empty()) {
    PathCostBreakdown out;
    out.calibrated = weights.calibrated;
    return out;
  }
  return accumulate_cost(samples, samples.front(), q_neutral, policy, weights,
                         &kin, left, guards, nullptr, stage_knot_ends);
}

CostLinearization value_and_linearize(
    const JointCurve& knots, const Vec18& q_start, const Vec18& q_neutral,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    const Kinematics& kin, bool left, const WholeBodyGuards* guards,
    const std::vector<int>& active_joints,
    const std::vector<int>* stage_knot_ends) {
  CostLinearization out;
  LinAccum lin;
  lin.reset(knots.size(), &active_joints);
  out.value = accumulate_cost(knots, q_start, q_neutral, policy, weights, &kin,
                              left, guards, &lin, stage_knot_ends);
  out.grad = std::move(lin.grad);
  out.hess_diag = std::move(lin.hess);
  return out;
}

PathQualityMetrics evaluate_path_quality(
    const JointCurve& samples, const Vec18& q0, bool left,
    const MotionConstraintPolicy& policy, const PathCostBreakdown& soft,
    double pitch_cap_rad, const std::vector<int>* stage_knot_ends) {
  PathQualityMetrics m;
  if (samples.size() < 2) {
    return m;
  }
  const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
  const int j6 = arm + 5;
  const int j7 = arm + 6;
  double travel[kNDof] = {};
  for (std::size_t i = 1; i < samples.size(); ++i) {
    for (int j = 0; j < kNDof; ++j) {
      travel[j] += std::abs(samples[i][j] - samples[i - 1][j]);
    }
  }
  for (int j = 0; j < kNDof; ++j) {
    m.total_travel_rad += travel[j];
  }
  m.j4_travel_rad = travel[kIdxYaw];
  m.wrist_j6_travel_rad = travel[j6];
  m.wrist_j7_travel_rad = travel[j7];
  const double j6_fold_full = joint_path_foldback_rad(samples, j6);
  const double j7_fold_full = joint_path_foldback_rad(samples, j7);
  m.wrist_j6_fold_rad = j6_fold_full;
  m.wrist_j7_fold_rad = j7_fold_full;
  double lo6 = samples.front()[j6];
  double hi6 = lo6;
  double lo7 = samples.front()[j7];
  double hi7 = lo7;
  for (const auto& q : samples) {
    m.j4_peak_from_start_rad =
        std::max(m.j4_peak_from_start_rad, std::abs(q[kIdxYaw] - q0[kIdxYaw]));
    const double pitch = std::abs(q[0] + q[1] + q[2]);
    m.pitch_peak_abs_rad = std::max(m.pitch_peak_abs_rad, pitch);
    lo6 = std::min(lo6, q[j6]);
    hi6 = std::max(hi6, q[j6]);
    lo7 = std::min(lo7, q[j7]);
    hi7 = std::max(hi7, q[j7]);
  }
  m.wrist_j6_span_rad = hi6 - lo6;
  m.wrist_j7_span_rad = hi7 - lo7;
  const double cap = pitch_cap_rad > 0.0 ? pitch_cap_rad : 1e9;
  m.pitch_min_margin_rad = cap - m.pitch_peak_abs_rad;
  m.body_dz_m = soft.body_dz_m;
  m.body_dx_m = soft.body_dx_m;
  m.soft_cost = soft.total;
  // Severe = real foldback (travel − |net|), the r54 ~716° J6 class.
  // Net J4 twist / modest wrist span stay in the score: a 0.5 m drop can
  // need ~50° yaw without reversing. Do not hard-reject that as severe.
  const double j4_fold_full = joint_path_foldback_rad(samples, kIdxYaw);
  // Planner reject matches r54 gtest (full-path travel − |net|).
  constexpr double kR54J6FoldMaxRad = 90.0 * kPi / 180.0;
  constexpr double kR54J7FoldMaxRad = 120.0 * kPi / 180.0;
  constexpr double kR54J4FoldMaxRad = 40.0 * kPi / 180.0;
  m.severe_foldback = (j6_fold_full > kR54J6FoldMaxRad) ||
                      (j7_fold_full > kR54J7FoldMaxRad) ||
                      (j4_fold_full > kR54J4FoldMaxRad);
  // Score: foldback / span / J4 first (offline J7 gate is 0.50 rad).
  constexpr double kJ7Soft = 0.40;
  constexpr double kJ6Soft = 0.55;
  m.score = 8.0 * j6_fold_full + 6.0 * j7_fold_full +
            12.0 * std::max(0.0, m.wrist_j7_span_rad - kJ7Soft) +
            6.0 * std::max(0.0, m.wrist_j6_span_rad - kJ6Soft) +
            10.0 * m.j4_peak_from_start_rad + 2.0 * m.j4_travel_rad +
            0.15 * m.total_travel_rad + 0.02 * m.soft_cost +
            0.5 * std::max(0.0, -m.pitch_min_margin_rad) +
            1.5 * std::max(0.0, 0.15 - std::max(0.0, -m.body_dz_m));
  (void)policy;
  (void)stage_knot_ends;
  return m;
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
    rd("j4_twist", &weights.j4_twist);
    rd("foldback", &weights.foldback);
    rd("wrist_foldback", &weights.wrist_foldback);
    rd("wrist_span", &weights.wrist_span);
    rd("height_coop", &weights.height_coop);
    rd("pitch_near_barrier", &weights.pitch_near_barrier);
    if (tq["normalization"]) {
      const YAML::Node n = tq["normalization"];
      if (n["body_forward_len_m"]) {
        weights.body_forward_len_m = n["body_forward_len_m"].as<double>();
      }
      if (n["pitch_scale_rad"]) {
        weights.pitch_scale_rad = n["pitch_scale_rad"].as<double>();
      }
      if (n["height_coop_len_m"]) {
        weights.height_coop_len_m = n["height_coop_len_m"].as<double>();
      }
      if (n["wrist_span_soft_rad"]) {
        weights.wrist_span_soft_rad = n["wrist_span_soft_rad"].as<double>();
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
