#include "kmpc/unified_multistage_planner.hpp"

#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_cost.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Cholesky>

namespace kmpc {
namespace {

struct StageGeom {
  Vec3 a = Vec3::Zero();
  Vec3 b = Vec3::Zero();
  Vec4 qa = Vec4(0, 0, 0, 1);
  Vec4 qb = Vec4(0, 0, 0, 1);
  bool ori = false;
};

struct KnotTarget {
  Vec3 p = Vec3::Zero();
  Vec4 quat = Vec4(0, 0, 0, 1);
  bool ori = false;
};

uint32_t path_id_of(const JointCurve& samples) {
  uint32_t h = 2166136261u;
  for (const auto& q : samples) {
    for (int i = 0; i < kNDof; ++i) {
      const auto bits = static_cast<uint32_t>(std::llround(q[i] * 1e6));
      h ^= bits;
      h *= 16777619u;
    }
  }
  return h;
}

class UnifiedSolver {
 public:
  UnifiedSolver(const Kinematics& kin, const WholeBodyGuards& guards,
                const UnifiedPlanRequest& req)
      : kin_(kin), guards_(guards), req_(req) {
    q0_ = req.q0;
    q_neutral_ = req.q_neutral.allFinite() && req.q_neutral.norm() > 1e-3
                     ? req.q_neutral
                     : req.q0;
    arm_ = req.left ? kSliceLeft.begin : kSliceRight.begin;
    idle_ = req.left ? kSliceRight.begin : kSliceLeft.begin;
    active_full_.reserve(11);
    if (!req.waist_locked) {
      for (int j = 0; j < kNWaist; ++j) {
        active_full_.push_back(j);
      }
    }
    for (int j = 0; j < kNArm; ++j) {
      active_full_.push_back(arm_ + j);
    }
    active_arm_ = active_full_;
    if (!req.waist_locked) {
      active_arm_.clear();
      active_arm_.push_back(kIdxYaw);
      for (int j = 0; j < kNArm; ++j) {
        active_arm_.push_back(arm_ + j);
      }
    }
    active_mask_.fill(false);
    for (int j : active_full_) {
      active_mask_[static_cast<std::size_t>(j)] = true;
    }
    for (int j = 0; j < kNDof; ++j) {
      const double dq = req.dq_max[j];
      motion_scale_[j] = dq > 1e-6 ? 1.0 / dq : 1.0;
    }
    pitch_cap_ = req.policy.torso_pitch_max_rad > 0.0
                     ? req.policy.torso_pitch_max_rad
                     : 1e9;
    start_pitch_abs_ = std::abs(q0_.head<3>().sum());
  }

  UnifiedPlanResult run() {
    const auto t0 = std::chrono::steady_clock::now();
    UnifiedPlanResult out;
    std::string why;
    if (!build_stages(&why)) {
      out.reason = why;
      finish_report(out, t0);
      return out;
    }
    JointCurve knots;
    std::vector<KnotTarget> targets;
    if (!solve_knots(&knots, &targets, &why)) {
      out.reason = why;
      out.iterations = iterations_;
      finish_report(out, t0);
      return out;
    }
    const auto t_opt = std::chrono::steady_clock::now();
    out.optimize_ms =
        std::chrono::duration<double, std::milli>(t_opt - t0).count();
    out.iterations = iterations_;

    JointCurve segment;
    JointCurve successor;
    bool have_succ = false;
    double worst = 0.0;
    if (!densify(knots, targets, /*monotone_waist=*/true, &segment, &successor,
                 &have_succ, &worst, &why)) {
      if (!densify(knots, targets, /*monotone_waist=*/false, &segment,
                   &successor, &have_succ, &worst, &why)) {
        out.reason = why.empty() ? "unified_tube" : why;
        out.tcp_worst_m = worst;
        finish_report(out, t_opt);
        return out;
      }
    }
    const auto t_cert = std::chrono::steady_clock::now();
    out.certify_ms =
        std::chrono::duration<double, std::milli>(t_cert - t_opt).count();
    out.tcp_worst_m = worst;
    if (!admit_direction(segment, &out.contract, &out.direction, &why)) {
      out.reason = why;
      out.segment = std::move(segment);
      finish_report(out, t_cert);
      return out;
    }
    if (have_succ) {
      std::string succ_why;
      if (!admit_direction(successor, &out.successor_contract,
                           &out.successor_direction, &succ_why)) {
        out.reason = std::string("cascade_") + succ_why;
        finish_report(out, t_cert);
        return out;
      }
      out.successor = std::move(successor);
      out.has_successor = true;
    }
    out.segment = std::move(segment);
    out.ok = true;
    out.reason = "unified_multistage";
    fill_metrics(out);
    finish_report(out, t_cert);
    return out;
  }

 private:
  const Kinematics& kin_;
  const WholeBodyGuards& guards_;
  UnifiedPlanRequest req_;
  Vec18 q0_ = Vec18::Zero();
  Vec18 q_neutral_ = Vec18::Zero();
  Vec18 motion_scale_ = Vec18::Ones();
  int arm_ = 4;
  int idle_ = 11;
  std::vector<int> active_full_;
  std::vector<int> active_arm_;
  std::array<bool, kNDof> active_mask_{};
  double pitch_cap_ = 1e9;
  double start_pitch_abs_ = 0.0;
  int iterations_ = 0;
  std::vector<StageGeom> stages_;
  int boundary_knot_ = 0;

  bool build_stages(std::string* why) {
    if (!req_.q0.allFinite() || !req_.goal.xyz.allFinite() ||
        req_.frame.empty()) {
      *why = "unified_ik";
      return false;
    }
    if (joint_violates_hard_box(req_.q0, req_.policy.q_min, req_.policy.q_max)) {
      *why = "unified_hard_box";
      return false;
    }
    StageGeom first;
    first.a = kin_.fk_pos(req_.q0, req_.frame);
    first.b = req_.goal.xyz;
    first.ori = req_.goal.ori_on;
    first.qa = rot_to_quat_xyzw(kin_.fk(req_.q0, req_.frame).block<3, 3>(0, 0));
    first.qb = req_.goal.quat;
    if ((first.b - first.a).norm() > static_cast<double>(req_.max_samples) *
                                         std::max(req_.sample_max_m, 1e-3)) {
      *why = "unified_sample_limit";
      return false;
    }
    stages_.push_back(first);
    if (req_.has_next) {
      if (!req_.next.xyz.allFinite()) {
        *why = "cascade_unreachable";
        return false;
      }
      const double reach = (req_.next.xyz - req_.goal.xyz).norm();
      if (reach > static_cast<double>(req_.max_samples) *
                      std::max(req_.sample_max_m, 1e-3)) {
        *why = "cascade_unreachable";
        return false;
      }
      StageGeom second;
      second.a = req_.goal.xyz;
      second.b = req_.next.xyz;
      second.ori = req_.next.ori_on;
      second.qa = req_.goal.ori_on ? req_.goal.quat : first.qa;
      second.qb = req_.next.quat;
      stages_.push_back(second);
    }
    return true;
  }

  static int spans_for(double dist) {
    return std::clamp(static_cast<int>(std::ceil(dist / 0.04)), 1, 5);
  }

  bool solve_knots(JointCurve* knots, std::vector<KnotTarget>* targets,
                   std::string* why) {
    std::vector<int> knot_of_stage;
    knot_of_stage.push_back(0);
    int K = 1;
    for (const auto& st : stages_) {
      K += spans_for((st.b - st.a).norm());
      knot_of_stage.push_back(K - 1);
    }
    boundary_knot_ = knot_of_stage.size() > 1 ? knot_of_stage[1] : K - 1;
    if (stages_.size() == 1) {
      boundary_knot_ = K - 1;
    } else {
      boundary_knot_ = knot_of_stage[1];
    }
    knots->assign(static_cast<std::size_t>(K), q0_);
    targets->assign(static_cast<std::size_t>(K), KnotTarget{});
    (*targets)[0].p = stages_[0].a;
    (*targets)[0].quat = stages_[0].qa;
    (*targets)[0].ori = stages_[0].ori;
    for (std::size_t s = 0; s < stages_.size(); ++s) {
      const int k0 = knot_of_stage[s];
      const int k1 = knot_of_stage[s + 1];
      const int n = std::max(1, k1 - k0);
      for (int k = k0; k <= k1; ++k) {
        const double t = static_cast<double>(k - k0) / static_cast<double>(n);
        (*targets)[static_cast<std::size_t>(k)].p =
            (1.0 - t) * stages_[s].a + t * stages_[s].b;
        (*targets)[static_cast<std::size_t>(k)].quat =
            quat_slerp_xyzw(stages_[s].qa, stages_[s].qb, t);
        (*targets)[static_cast<std::size_t>(k)].ori = stages_[s].ori;
      }
    }
    for (int k = 1; k < K; ++k) {
      (*knots)[static_cast<std::size_t>(k)] = (*knots)[static_cast<std::size_t>(k - 1)];
      const auto& tgt = (*targets)[static_cast<std::size_t>(k)];
      improve_pose(&(*knots)[static_cast<std::size_t>(k)], active_full_, tgt.p,
                   tgt.quat, tgt.ori,
                   (*knots)[static_cast<std::size_t>(k - 1)], 16);
      ++iterations_;
    }
    gauss_newton(knots, *targets);
    double worst = 0.0;
    for (int k = 1; k < K; ++k) {
      const double err =
          ((*targets)[static_cast<std::size_t>(k)].p -
           kin_.fk_pos((*knots)[static_cast<std::size_t>(k)], req_.frame))
              .norm();
      worst = std::max(worst, err);
    }
    if (worst > req_.pos_tol_m) {
      std::ostringstream os;
      os << std::fixed << std::setprecision(4) << "unified_ik:m=" << worst;
      for (int k = 1; k < K; ++k) {
        const double err =
            ((*targets)[static_cast<std::size_t>(k)].p -
             kin_.fk_pos((*knots)[static_cast<std::size_t>(k)], req_.frame))
                .norm();
        os << " k" << k << "=" << err;
      }
      *why = os.str();
      return false;
    }
    if (!req_.waist_locked) {
      JointCurve mono = *knots;
      const int cuts[] = {0, boundary_knot_, K - 1};
      const int ncuts = stages_.size() == 1 ? 2 : 3;
      for (int c = 0; c + 1 < ncuts; ++c) {
        const int a = cuts[c];
        const int b = cuts[c + 1];
        const Vec18 qa = (*knots)[static_cast<std::size_t>(a)];
        const Vec18 qb = (*knots)[static_cast<std::size_t>(b)];
        for (int k = a; k <= b; ++k) {
          const double t = b == a
                               ? 1.0
                               : static_cast<double>(k - a) /
                                     static_cast<double>(b - a);
          for (int j = 0; j < 3; ++j) {
            mono[static_cast<std::size_t>(k)][j] =
                (1.0 - t) * qa[j] + t * qb[j];
          }
        }
      }
      bool mono_ok = true;
      for (int k = 1; k < K; ++k) {
        const bool endpoint =
            k == boundary_knot_ || k == K - 1;
        if (endpoint) {
          continue;
        }
        const auto& tgt = (*targets)[static_cast<std::size_t>(k)];
        improve_pose(&mono[static_cast<std::size_t>(k)], active_arm_, tgt.p,
                     tgt.quat, tgt.ori, mono[static_cast<std::size_t>(k - 1)],
                     6);
        const double err =
            (tgt.p - kin_.fk_pos(mono[static_cast<std::size_t>(k)], req_.frame))
                .norm();
        if (err > req_.pos_tol_m) {
          mono_ok = false;
          break;
        }
      }
      if (mono_ok) {
        *knots = std::move(mono);
      }
    }
    (void)why;
    return true;
  }

  void gauss_newton(JointCurve* knots, const std::vector<KnotTarget>& targets) {
    const int K = static_cast<int>(knots->size());
    const int na = static_cast<int>(active_full_.size());
    if (K < 2 || na < 1) {
      return;
    }
    const int nvar = (K - 1) * na;
    const double w_pos = 400.0;
    for (int iter = 0; iter < 4; ++iter) {
      ++iterations_;
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nvar, nvar);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(nvar);
      for (int k = 1; k < K; ++k) {
        const Vec18& q = (*knots)[static_cast<std::size_t>(k)];
        const auto& tgt = targets[static_cast<std::size_t>(k)];
        const auto pose = kin_.fk(q, req_.frame);
        const Vec3 ep = tgt.p - pose.col(3).head<3>();
        const Mat6x18 J = kin_.jacobian_spatial(q, req_.frame);
        const int col = (k - 1) * na;
        // Position only. Orientation is a nullspace polish after the hand is on the line.
        add_task_block(&H, &g, col, J, ep, nullptr, w_pos, 0.0);
        add_secondary_block(&H, &g, col, q, (*knots)[static_cast<std::size_t>(k - 1)]);
        if (k + 1 < K) {
          const Vec18& qn = (*knots)[static_cast<std::size_t>(k + 1)];
          const Vec18 qmid = 0.5 * (q + qn);
          const Vec3 pm =
              0.5 * (tgt.p + targets[static_cast<std::size_t>(k + 1)].p);
          const Vec3 em = pm - kin_.fk_pos(qmid, req_.frame);
          const Mat6x18 Jm = kin_.jacobian_spatial(qmid, req_.frame);
          add_mid_block(&H, &g, col, col + na, Jm, em, 0.5 * w_pos);
        }
      }
      H.diagonal().array() += 1e-4;
      const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
      if (ldlt.info() != Eigen::Success) {
        return;
      }
      const Eigen::VectorXd dq = ldlt.solve(g);
      if (!dq.allFinite()) {
        return;
      }
      if (dq.cwiseAbs().maxCoeff() < 1e-4) {
        return;
      }
      const double base = knot_pos_cost(*knots, targets);
      double worst_base = 0.0;
      for (int k = 1; k < K; ++k) {
        worst_base = std::max(
            worst_base,
            (targets[static_cast<std::size_t>(k)].p -
             kin_.fk_pos((*knots)[static_cast<std::size_t>(k)], req_.frame))
                .norm());
      }
      bool improved = false;
      for (double alpha : {1.0, 0.5, 0.25}) {
        JointCurve trial = *knots;
        apply_delta(&trial, dq, alpha);
        const double cost = knot_pos_cost(trial, targets);
        double worst_trial = 0.0;
        for (int k = 1; k < K; ++k) {
          worst_trial = std::max(
              worst_trial,
              (targets[static_cast<std::size_t>(k)].p -
               kin_.fk_pos(trial[static_cast<std::size_t>(k)], req_.frame))
                  .norm());
        }
        if (cost < base - 1e-8 && worst_trial <= worst_base + 1e-9 &&
            worst_trial <= req_.pos_tol_m) {
          *knots = std::move(trial);
          improved = true;
          break;
        }
      }
      if (!improved) {
        return;
      }
    }
  }

  double knot_pos_cost(const JointCurve& knots,
                       const std::vector<KnotTarget>& targets) const {
    double cost = 0.0;
    const int K = static_cast<int>(knots.size());
    for (int k = 1; k < K; ++k) {
      const Vec3 e = targets[static_cast<std::size_t>(k)].p -
                     kin_.fk_pos(knots[static_cast<std::size_t>(k)], req_.frame);
      cost += e.squaredNorm();
      const Vec3 pm =
          0.5 * (targets[static_cast<std::size_t>(k - 1)].p +
                 targets[static_cast<std::size_t>(k)].p);
      const Vec18 qmid = 0.5 * (knots[static_cast<std::size_t>(k - 1)] +
                                knots[static_cast<std::size_t>(k)]);
      const Vec3 em = pm - kin_.fk_pos(qmid, req_.frame);
      cost += em.squaredNorm();
    }
    return cost;
  }

  void add_task_block(Eigen::MatrixXd* H, Eigen::VectorXd* g, int col,
                      const Mat6x18& J, const Vec3& ep, const Vec3* eo,
                      double w_pos, double w_ori) const {
    const int na = static_cast<int>(active_full_.size());
    for (int i = 0; i < na; ++i) {
      const int ai = active_full_[static_cast<std::size_t>(i)];
      double gi = 0.0;
      for (int r = 0; r < 3; ++r) {
        gi += w_pos * J(r, ai) * ep[r];
      }
      if (eo != nullptr) {
        for (int r = 0; r < 3; ++r) {
          gi += w_ori * J(3 + r, ai) * (*eo)[r];
        }
      }
      (*g)[col + i] += gi;
      for (int j = i; j < na; ++j) {
        const int aj = active_full_[static_cast<std::size_t>(j)];
        double hij = 0.0;
        for (int r = 0; r < 3; ++r) {
          hij += w_pos * J(r, ai) * J(r, aj);
        }
        if (eo != nullptr) {
          for (int r = 0; r < 3; ++r) {
            hij += w_ori * J(3 + r, ai) * J(3 + r, aj);
          }
        }
        (*H)(col + i, col + j) += hij;
        if (i != j) {
          (*H)(col + j, col + i) += hij;
        }
      }
    }
  }

  void add_mid_block(Eigen::MatrixXd* H, Eigen::VectorXd* g, int col_l,
                     int col_r, const Mat6x18& J, const Vec3& em,
                     double w) const {
    const int na = static_cast<int>(active_full_.size());
    Eigen::MatrixXd Jl(3, na);
    Eigen::MatrixXd Jr(3, na);
    for (int i = 0; i < na; ++i) {
      const int ai = active_full_[static_cast<std::size_t>(i)];
      Jl.col(i) = 0.5 * J.col(ai).head<3>();
      Jr.col(i) = Jl.col(i);
    }
    const int cols[2] = {col_l, col_r};
    const Eigen::MatrixXd* Js[2] = {&Jl, &Jr};
    for (int a = 0; a < 2; ++a) {
      const Eigen::VectorXd ga = w * Js[a]->transpose() * em;
      g->segment(cols[a], na) += ga;
      for (int b = 0; b < 2; ++b) {
        const Eigen::MatrixXd Hab = w * Js[a]->transpose() * (*Js[b]);
        H->block(cols[a], cols[b], na, na) += Hab;
      }
    }
  }

  void add_secondary_block(Eigen::MatrixXd* H, Eigen::VectorXd* g, int col,
                           const Vec18& q, const Vec18& anchor) const {
    const int na = static_cast<int>(active_full_.size());
    constexpr double w_sec = 0.08;
    for (int i = 0; i < na; ++i) {
      const int ai = active_full_[static_cast<std::size_t>(i)];
      const double desire = std::clamp(-secondary_grad(q, anchor, ai), -0.03, 0.03);
      (*H)(col + i, col + i) += w_sec;
      (*g)[col + i] += w_sec * desire;
    }
  }

  double secondary_grad(const Vec18& q, const Vec18& anchor, int j) const {
    constexpr double eps = 1e-3;
    const double m_hi = req_.policy.q_max[j] - q[j];
    const double m_lo = q[j] - req_.policy.q_min[j];
    double grad = eps / ((m_hi + eps) * (m_hi + eps));
    grad -= eps / ((m_lo + eps) * (m_lo + eps));
    grad += 0.02 * motion_scale_[j] * (q[j] - q_neutral_[j]);
    grad += 0.35 * (q[j] - anchor[j]);
    if (j < 3 && req_.policy.torso_pitch_max_rad > 0.0) {
      const double pitch = q[0] + q[1] + q[2];
      const double sign = pitch >= 0.0 ? 1.0 : -1.0;
      const double m = req_.policy.torso_pitch_max_rad - std::abs(pitch);
      const double den = std::max(m, 1e-6) + eps;
      const double dcdm = -eps / (den * den);
      grad += 0.4 * dcdm * (-sign);
    }
    return grad;
  }

  void apply_delta(JointCurve* knots, const Eigen::VectorXd& dq,
                   double alpha) const {
    const int na = static_cast<int>(active_full_.size());
    const int K = static_cast<int>(knots->size());
    for (int k = 1; k < K; ++k) {
      Vec18& q = (*knots)[static_cast<std::size_t>(k)];
      for (int i = 0; i < na; ++i) {
        q[active_full_[static_cast<std::size_t>(i)]] +=
            alpha * dq[(k - 1) * na + i];
      }
      restore_hold(&q);
      clamp_hard(&q);
      pull_pitch(&q);
    }
  }

  bool project_position_nullspace(const Vec18& q, const std::vector<int>& active,
                                  Eigen::VectorXd* step) const {
    const int na = static_cast<int>(active.size());
    if (step->size() != na) {
      return false;
    }
    const Mat6x18 J = kin_.jacobian_spatial(q, req_.frame);
    Eigen::MatrixXd Jp(3, na);
    for (int i = 0; i < na; ++i) {
      Jp.col(i) = J.col(active[static_cast<std::size_t>(i)]).head<3>();
    }
    Eigen::Matrix3d gram = Jp * Jp.transpose();
    gram.diagonal().array() += 1e-6;
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(gram);
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    const Eigen::Vector3d corr = ldlt.solve(Jp * (*step));
    *step -= Jp.transpose() * corr;
    return step->allFinite();
  }

  void polish_orientation(Vec18* q, const std::vector<int>& active,
                          const Vec3& p_des, const Vec4& quat_des) const {
    const int na = static_cast<int>(active.size());
    for (int it = 0; it < 6; ++it) {
      const auto pose = kin_.fk(*q, req_.frame);
      const Vec3 eo =
          ori_error_xyzw(quat_des, rot_to_quat_xyzw(pose.block<3, 3>(0, 0)));
      if (eo.norm() < 0.04) {
        return;
      }
      const Mat6x18 J = kin_.jacobian_spatial(*q, req_.frame);
      const Vec3 ep = p_des - pose.col(3).head<3>();
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(na, na);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(na);
      constexpr double w_keep = 1.0;
      for (int i = 0; i < na; ++i) {
        const int ai = active[static_cast<std::size_t>(i)];
        for (int r = 0; r < 3; ++r) {
          g[i] += J(3 + r, ai) * eo[r];
          g[i] += w_keep * J(r, ai) * ep[r];
        }
        for (int j = i; j < na; ++j) {
          const int aj = active[static_cast<std::size_t>(j)];
          double hij = 0.0;
          for (int r = 0; r < 3; ++r) {
            hij += J(3 + r, ai) * J(3 + r, aj);
            hij += w_keep * J(r, ai) * J(r, aj);
          }
          H(i, j) += hij;
          if (i != j) {
            H(j, i) += hij;
          }
        }
        H(i, i) += 1e-3;
      }
      const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
      if (ldlt.info() != Eigen::Success) {
        return;
      }
      Eigen::VectorXd step = ldlt.solve(g);
      const double peak = step.cwiseAbs().maxCoeff();
      if (peak < 1e-5) {
        return;
      }
      if (peak > 0.05) {
        step *= 0.05 / peak;
      }
      bool improved = false;
      for (double alpha : {1.0, 0.5, 0.25}) {
        Vec18 trial = *q;
        for (int i = 0; i < na; ++i) {
          trial[active[static_cast<std::size_t>(i)]] += alpha * step[i];
        }
        restore_hold(&trial);
        clamp_hard(&trial);
        pull_pitch(&trial);
        const auto posed = kin_.fk(trial, req_.frame);
        const double pos = (p_des - posed.col(3).head<3>()).norm();
        const double ori_now =
            ori_error_xyzw(quat_des, rot_to_quat_xyzw(posed.block<3, 3>(0, 0)))
                .norm();
        if (pos <= req_.pos_tol_m - 2e-4 && ori_now < eo.norm() - 1e-4) {
          *q = trial;
          improved = true;
          break;
        }
      }
      if (!improved) {
        return;
      }
    }
  }

  void polish_margin(Vec18* q, const std::vector<int>& active, const Vec3& p_des,
                     const Vec18& anchor) const {
    const int na = static_cast<int>(active.size());
    for (int it = 0; it < 4; ++it) {
      const double pos0 = (p_des - kin_.fk_pos(*q, req_.frame)).norm();
      Eigen::VectorXd step(na);
      for (int i = 0; i < na; ++i) {
        const int ai = active[static_cast<std::size_t>(i)];
        step[i] = std::clamp(-secondary_grad(*q, anchor, ai), -0.02, 0.02);
      }
      if (!project_position_nullspace(*q, active, &step)) {
        return;
      }
      if (step.cwiseAbs().maxCoeff() < 1e-5) {
        return;
      }
      Vec18 trial = *q;
      for (int i = 0; i < na; ++i) {
        trial[active[static_cast<std::size_t>(i)]] += step[i];
      }
      restore_hold(&trial);
      clamp_hard(&trial);
      if (moves_pitch(active)) {
        pull_pitch(&trial);
      }
      const double pos = (p_des - kin_.fk_pos(trial, req_.frame)).norm();
      if (pos > std::min(pos0 + 1e-4, req_.pos_tol_m - 2e-4) ||
          joint_violates_hard_box(trial, req_.policy.q_min, req_.policy.q_max)) {
        return;
      }
      *q = trial;
    }
  }

  static bool moves_pitch(const std::vector<int>& active) {
    for (int j : active) {
      if (j < 3) {
        return true;
      }
    }
    return false;
  }

  void improve_pose(Vec18* q, const std::vector<int>& active, const Vec3& p_des,
                    const Vec4& quat_des, bool ori, const Vec18& anchor,
                    int iters) const {
    const double w_pos = 400.0;
    const int na = static_cast<int>(active.size());
    if (na < 1) {
      return;
    }
    for (int it = 0; it < iters; ++it) {
      const auto pose = kin_.fk(*q, req_.frame);
      const Vec3 ep = p_des - pose.col(3).head<3>();
      if (ep.norm() <= req_.pos_tol_m * 0.5) {
        break;
      }
      const Mat6x18 J = kin_.jacobian_spatial(*q, req_.frame);
      const double base = ep.norm();
      auto solve_cols = [&](const Mat6x18& Js, const Vec3& e,
                            const std::vector<int>& cols) -> Eigen::VectorXd {
        const int n = static_cast<int>(cols.size());
        Eigen::VectorXd full = Eigen::VectorXd::Zero(na);
        if (n < 1) {
          return full;
        }
        Eigen::MatrixXd Hs = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd gs = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i) {
          const int ai = active[static_cast<std::size_t>(cols[static_cast<std::size_t>(i)])];
          double gi = 0.0;
          for (int r = 0; r < 3; ++r) {
            gi += w_pos * Js(r, ai) * e[r];
          }
          gs[i] = gi;
          for (int j = i; j < n; ++j) {
            const int aj =
                active[static_cast<std::size_t>(cols[static_cast<std::size_t>(j)])];
            double hij = 0.0;
            for (int r = 0; r < 3; ++r) {
              hij += w_pos * Js(r, ai) * Js(r, aj);
            }
            Hs(i, j) += hij;
            if (i != j) {
              Hs(j, i) += hij;
            }
          }
          const double w_anchor = 0.4;
          gs[i] += w_anchor * (anchor[ai] - (*q)[ai]);
          Hs(i, i) += 1e-4 * motion_scale_[ai] * motion_scale_[ai] + w_anchor;
        }
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(Hs);
        if (ldlt.info() != Eigen::Success) {
          return full;
        }
        const Eigen::VectorXd solved = ldlt.solve(gs);
        if (!solved.allFinite()) {
          return full;
        }
        for (int i = 0; i < n; ++i) {
          full[cols[static_cast<std::size_t>(i)]] = solved[i];
        }
        const double step_cap = base < 0.01 ? 0.04 : 0.08;
        const double peak = full.cwiseAbs().maxCoeff();
        if (peak > step_cap) {
          full *= step_cap / peak;
        }
        return full;
      };
      auto accept = [&](const Eigen::VectorXd& step) -> bool {
        for (double alpha : {1.0, 0.5, 0.25, 0.1}) {
          Vec18 trial = *q;
          for (int i = 0; i < na; ++i) {
            trial[active[static_cast<std::size_t>(i)]] += alpha * step[i];
          }
          restore_hold(&trial);
          clamp_hard(&trial);
          if (moves_pitch(active)) {
            pull_pitch(&trial);
          }
          const double err = (p_des - kin_.fk_pos(trial, req_.frame)).norm();
          const double need =
              base <= 2.0 * req_.pos_tol_m
                  ? 1e-4
                  : std::max(2e-4, 0.2 * std::max(0.0, base - 0.5 * req_.pos_tol_m));
          if (err <= req_.pos_tol_m * 0.5 || err <= base - need) {
            *q = trial;
            return true;
          }
        }
        return false;
      };
      std::vector<int> all(static_cast<std::size_t>(na));
      std::iota(all.begin(), all.end(), 0);
      const Eigen::VectorXd full = solve_cols(J, ep, all);
      bool improved = accept(full);
      if (!improved && moves_pitch(active)) {
        std::vector<int> arm_cols;
        arm_cols.reserve(static_cast<std::size_t>(na));
        for (int i = 0; i < na; ++i) {
          if (active[static_cast<std::size_t>(i)] >= 3) {
            arm_cols.push_back(i);
          }
        }
        Vec18 seed = *q;
        for (int i = 0; i < na; ++i) {
          seed[active[static_cast<std::size_t>(i)]] += full[i];
        }
        restore_hold(&seed);
        clamp_hard(&seed);
        pull_pitch(&seed);
        const double need =
            base <= 2.0 * req_.pos_tol_m
                ? 1e-4
                : std::max(2e-4, 0.2 * std::max(0.0, base - 0.5 * req_.pos_tol_m));
        for (int n = 0; n < 8 && !arm_cols.empty(); ++n) {
          const Vec3 e2 = p_des - kin_.fk_pos(seed, req_.frame);
          const double err = e2.norm();
          if (err <= req_.pos_tol_m * 0.5 || err <= base - need) {
            *q = seed;
            improved = true;
            break;
          }
          const Mat6x18 Js = kin_.jacobian_spatial(seed, req_.frame);
          const Eigen::VectorXd arm = solve_cols(Js, e2, arm_cols);
          Vec18 trial = seed;
          for (int i = 0; i < na; ++i) {
            trial[active[static_cast<std::size_t>(i)]] += arm[i];
          }
          restore_hold(&trial);
          clamp_hard(&trial);
          pull_pitch(&trial);
          const double err2 = (p_des - kin_.fk_pos(trial, req_.frame)).norm();
          if (err2 > err - 2e-4) {
            break;
          }
          seed = trial;
        }
      }
      if (!improved) {
        break;
      }
    }
    polish_margin(q, active, p_des, anchor);
    repair_com(q, active, p_des);
    if (ori) {
      polish_orientation(q, active, p_des, quat_des);
    }
  }

  void repair_com(Vec18* q, const std::vector<int>& active, const Vec3& p_des) const {
    const double cx =
        0.5 * (guards_.com_work_xmin() + guards_.com_work_xmax());
    const double cy =
        0.5 * (guards_.com_work_ymin() + guards_.com_work_ymax());
    const int na = static_cast<int>(active.size());
    if (na < 1) {
      return;
    }
    for (int it = 0; it < 4; ++it) {
      const Vec3 com = kin_.com(*q);
      if (guards_.com_xy_inside(com)) {
        return;
      }
      const Vec3 before = kin_.fk_pos(*q, req_.frame);
      const Eigen::Vector2d err(cx - com.x(), cy - com.y());
      const Mat3x18 Jc = kin_.jacobian_com(*q);
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(na, na);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(na);
      for (int i = 0; i < na; ++i) {
        const int ai = active[static_cast<std::size_t>(i)];
        g[i] = Jc(0, ai) * err[0] + Jc(1, ai) * err[1];
        for (int j = i; j < na; ++j) {
          const int aj = active[static_cast<std::size_t>(j)];
          const double hij = Jc(0, ai) * Jc(0, aj) + Jc(1, ai) * Jc(1, aj);
          H(i, j) += hij;
          if (i != j) {
            H(j, i) += hij;
          }
        }
        H(i, i) += 1e-3;
      }
      const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
      if (ldlt.info() != Eigen::Success) {
        return;
      }
      Eigen::VectorXd step = ldlt.solve(g);
      if (!step.allFinite()) {
        return;
      }
      const double peak = step.cwiseAbs().maxCoeff();
      if (peak > 0.02) {
        step *= 0.02 / std::max(peak, 1e-12);
      }
      Vec18 trial = *q;
      for (int i = 0; i < na; ++i) {
        trial[active[static_cast<std::size_t>(i)]] += step[i];
      }
      restore_hold(&trial);
      clamp_hard(&trial);
      if (moves_pitch(active)) {
        pull_pitch(&trial);
      }
      const double tcp = (p_des - kin_.fk_pos(trial, req_.frame)).norm();
      const double tcp0 = (p_des - before).norm();
      if (tcp > std::min(tcp0 + 1e-4, req_.pos_tol_m - 2e-4)) {
        return;
      }
      *q = trial;
    }
  }

  bool densify(const JointCurve& knots, const std::vector<KnotTarget>& targets,
               bool monotone_waist, JointCurve* segment, JointCurve* successor,
               bool* have_succ, double* worst, std::string* why) const {
    (void)targets;
    const int K = static_cast<int>(knots.size());
    const int b = std::clamp(boundary_knot_, 0, K - 1);
    if (!densify_range(knots, 0, b, stages_[0], monotone_waist, segment, worst,
                       why)) {
      return false;
    }
    if (stages_.size() < 2) {
      *have_succ = false;
      return true;
    }
    JointCurve tip_knots = knots;
    if (!segment->empty()) {
      tip_knots[static_cast<std::size_t>(b)] = segment->back();
    }
    if (!densify_range(tip_knots, b, K - 1, stages_[1], monotone_waist, successor,
                       worst, why)) {
      if (why->empty()) {
        *why = "cascade_unreachable";
      }
      return false;
    }
    *have_succ = successor->size() >= 2;
    return *have_succ;
  }

  bool densify_range(const JointCurve& knots, int k0, int k1,
                     const StageGeom& stage, bool monotone_waist,
                     JointCurve* out, double* worst, std::string* why) const {
    const double dist = (stage.b - stage.a).norm();
    double h = std::min(std::max(req_.sample_max_m, 1e-4), 0.005);
    if (dist < 1e-5) {
      out->clear();
      out->push_back(knots[static_cast<std::size_t>(k0)]);
      out->push_back(knots[static_cast<std::size_t>(k1)]);
      return certify_curve(*out, stage, worst, why);
    }
    for (int attempt = 0; attempt < 4; ++attempt) {
      const int n = std::max(1, static_cast<int>(std::ceil(dist / h)));
      if (n + 1 > req_.max_samples) {
        if (why->empty()) {
          *why = "unified_sample_limit";
        }
        return false;
      }
      JointCurve trial(static_cast<std::size_t>(n + 1), q0_);
      trial.front() = knots[static_cast<std::size_t>(k0)];
      trial.back() = knots[static_cast<std::size_t>(k1)];
      bool reached = true;
      for (int i = 1; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const Vec3 p = (1.0 - t) * stage.a + t * stage.b;
        const Vec4 quat = quat_slerp_xyzw(stage.qa, stage.qb, t);
        const auto& active = monotone_waist ? active_arm_ : active_full_;
        Vec18 lerp_q = trial[static_cast<std::size_t>(i - 1)];
        if (k1 > k0) {
          const double u = t * static_cast<double>(k1 - k0);
          const int i0 =
              std::min(static_cast<int>(std::floor(u)), std::max(0, k1 - k0 - 1));
          const double a = u - static_cast<double>(i0);
          lerp_q = (1.0 - a) * knots[static_cast<std::size_t>(k0 + i0)] +
                   a * knots[static_cast<std::size_t>(k0 + i0 + 1)];
          if (monotone_waist && !req_.waist_locked) {
            for (int j = 0; j < 3; ++j) {
              lerp_q[j] = (1.0 - t) * knots[static_cast<std::size_t>(k0)][j] +
                          t * knots[static_cast<std::size_t>(k1)][j];
            }
          }
        }
        const bool lerp_ok =
            (p - kin_.fk_pos(lerp_q, req_.frame)).norm() <= req_.pos_tol_m &&
            (lerp_q - trial[static_cast<std::size_t>(i - 1)]).cwiseAbs().maxCoeff() <=
                0.45;
        Vec18 q = lerp_ok ? lerp_q : trial[static_cast<std::size_t>(i - 1)];
        if (!lerp_ok) {
          if (monotone_waist && !req_.waist_locked) {
            for (int j = 0; j < 3; ++j) {
              q[j] = (1.0 - t) * knots[static_cast<std::size_t>(k0)][j] +
                     t * knots[static_cast<std::size_t>(k1)][j];
            }
          }
          improve_pose(&q, active, p, quat, stage.ori,
                       trial[static_cast<std::size_t>(i - 1)], 8);
          if (monotone_waist && !req_.waist_locked) {
            for (int j = 0; j < 3; ++j) {
              q[j] = (1.0 - t) * knots[static_cast<std::size_t>(k0)][j] +
                     t * knots[static_cast<std::size_t>(k1)][j];
            }
            if (stage.ori) {
              polish_orientation(&q, active, p, quat);
            }
            pull_pitch(&q);
          }
        }
        const double err = (p - kin_.fk_pos(q, req_.frame)).norm();
        if (err > req_.pos_tol_m) {
          std::ostringstream os;
          os << std::fixed << std::setprecision(4) << "unified_ik:m=" << err;
          *why = os.str();
          reached = false;
          break;
        }
        for (int cut = 0; cut < 2; ++cut) {
          bool chord_ok = true;
          for (int s = 1; s <= 9; ++s) {
            const double a = static_cast<double>(s) / 10.0;
            const Vec18 mid =
                (1.0 - a) * trial[static_cast<std::size_t>(i - 1)] + a * q;
            const double tt =
                (static_cast<double>(i - 1) + a) / static_cast<double>(n);
            double ignored = 0.0;
            std::string ignore;
            if (!sample_ok(mid, stage, tt, &ignored, &ignore)) {
              chord_ok = false;
              break;
            }
          }
          if (chord_ok) {
            break;
          }
          q = 0.5 * (trial[static_cast<std::size_t>(i - 1)] + q);
          if (monotone_waist && !req_.waist_locked) {
            for (int j = 0; j < 3; ++j) {
              q[j] = (1.0 - t) * knots[static_cast<std::size_t>(k0)][j] +
                     t * knots[static_cast<std::size_t>(k1)][j];
            }
          }
          improve_pose(&q, active, p, quat, stage.ori,
                       trial[static_cast<std::size_t>(i - 1)], 6);
          if (monotone_waist && !req_.waist_locked) {
            for (int j = 0; j < 3; ++j) {
              q[j] = (1.0 - t) * knots[static_cast<std::size_t>(k0)][j] +
                     t * knots[static_cast<std::size_t>(k1)][j];
            }
            if (stage.ori) {
              polish_orientation(&q, active, p, quat);
            }
            pull_pitch(&q);
          }
        }
        if ((p - kin_.fk_pos(q, req_.frame)).norm() > req_.pos_tol_m) {
          *why = "unified_tube";
          reached = false;
          break;
        }
        trial[static_cast<std::size_t>(i)] = q;
      }
      if (reached && n >= 2) {
        const Vec18& prev = trial[static_cast<std::size_t>(n - 1)];
        Vec18 q = prev;
        const auto& active = monotone_waist ? active_arm_ : active_full_;
        improve_pose(&q, active, stage.b, stage.qb, stage.ori, prev, 8);
        const double err = (stage.b - kin_.fk_pos(q, req_.frame)).norm();
        const double step = (q - prev).cwiseAbs().maxCoeff();
        if (err <= req_.pos_tol_m && step <= 0.45) {
          trial.back() = q;
        }
      }
      if (reached && pitch_cap_ < 1e8) {
        for (auto& sample : trial) {
          const double excess =
              std::abs(sample[0] + sample[1] + sample[2]) - pitch_cap_;
          if (excess > 1e-9 && excess < 0.002) {
            pull_pitch(&sample);
          }
        }
      }
      if (reached && certify_curve(trial, stage, worst, why)) {
        *out = std::move(trial);
        return true;
      }
      h *= 0.5;
    }
    if (why->empty()) {
      *why = "unified_tube";
    }
    return false;
  }

  bool certify_curve(const JointCurve& curve, const StageGeom& stage,
                     double* worst, std::string* why) const {
    if (curve.size() < 2) {
      *why = "unified_tube";
      return false;
    }
    const int n = static_cast<int>(curve.size()) - 1;
    for (int i = 0; i < n; ++i) {
      const double jump =
          (curve[static_cast<std::size_t>(i + 1)] - curve[static_cast<std::size_t>(i)])
              .cwiseAbs()
              .maxCoeff();
      if (jump > 0.50) {
        *why = "unified_jump";
        return false;
      }
      for (int s = 0; s <= 10; ++s) {
        const double a = static_cast<double>(s) / 10.0;
        const Vec18 q = (1.0 - a) * curve[static_cast<std::size_t>(i)] +
                        a * curve[static_cast<std::size_t>(i + 1)];
        const double t =
            (static_cast<double>(i) + a) / static_cast<double>(n);
        if (!sample_ok(q, stage, t, worst, why)) {
          return false;
        }
      }
    }
    return true;
  }

  bool sample_ok(const Vec18& q, const StageGeom& stage, double t,
                 double* worst, std::string* why) const {
    if (!q.allFinite() || joint_violates_hard_box(q, req_.policy.q_min, req_.policy.q_max)) {
      *why = "unified_hard_box";
      return false;
    }
    // A start already outside the hard cap may be blended back in. Never
    // certify a pitch further outside than that start.
    const double admit = std::max(pitch_cap_, start_pitch_abs_);
    if (req_.policy.torso_pitch_max_rad > 0.0 &&
        std::abs(q.head<3>().sum()) > admit + 1e-6) {
      *why = "unified_pitch";
      return false;
    }
    if (!guards_.com_xy_inside(kin_.com(q))) {
      *why = "unified_com";
      return false;
    }
    const auto pose = kin_.fk(q, req_.frame);
    const Vec3 p = pose.col(3).head<3>();
    const Vec3 want = (1.0 - t) * stage.a + t * stage.b;
    const double err = (p - want).norm();
    *worst = std::max(*worst, err);
    if (err > req_.pos_tol_m) {
      std::ostringstream os;
      os << std::fixed << std::setprecision(4) << "unified_tube:m=" << err
         << " t=" << t;
      *why = os.str();
      return false;
    }
    if (stage.ori) {
      const Vec4 qdes = quat_slerp_xyzw(stage.qa, stage.qb, t);
      const double oe =
          ori_error_xyzw(qdes, rot_to_quat_xyzw(pose.block<3, 3>(0, 0))).norm();
      if (oe > 0.08) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3) << "unified_orientation:rad=" << oe
           << " t=" << t;
        *why = os.str();
        return false;
      }
    }
    for (int j = 0; j < kNArm; ++j) {
      if (std::abs(q[idle_ + j] - q0_[idle_ + j]) > 1e-6) {
        *why = "unified_idle_arm";
        return false;
      }
    }
    return true;
  }

  bool admit_direction(const JointCurve& curve, DirectionContract* contract,
                       DirectionCheckResult* check, std::string* why) const {
    *contract = make_path_direction_contract(curve);
    *check = check_path_direction(curve, *contract);
    if (check->ok) {
      return true;
    }
    if (smooth_reversal(curve)) {
      check->ok = true;
      std::snprintf(check->reason, sizeof(check->reason), "smooth_reversal");
      return true;
    }
    *why = std::string("unified_direction:") + check->reason;
    return false;
  }

  static bool smooth_reversal(const JointCurve& curve) {
    if (curve.size() < 4) {
      return false;
    }
    auto axis = [](const Vec18& q, int id) {
      if (id < 3) {
        return q[id];
      }
      return q[0] + q[1] + q[2];
    };
    for (int id = 0; id < 4; ++id) {
      int changes = 0;
      int last = 0;
      for (std::size_t i = 1; i < curve.size(); ++i) {
        const double d = axis(curve[i], id) - axis(curve[i - 1], id);
        if (std::abs(d) < 1e-4) {
          continue;
        }
        const int s = d > 0.0 ? 1 : -1;
        if (last != 0 && s != last) {
          ++changes;
        }
        last = s;
      }
      if (changes >= 2) {
        return false;
      }
      double peak_d2 = 0.0;
      for (std::size_t i = 1; i + 1 < curve.size(); ++i) {
        const double d2 = axis(curve[i + 1], id) - 2.0 * axis(curve[i], id) +
                          axis(curve[i - 1], id);
        peak_d2 = std::max(peak_d2, std::abs(d2));
      }
      if (peak_d2 > 0.08) {
        return false;
      }
    }
    return true;
  }

  void fill_metrics(UnifiedPlanResult& out) const {
    if (out.segment.size() < 2) {
      return;
    }
    const Vec18& qN = out.segment.back();
    double margin = 1e9;
    for (const auto& q : out.segment) {
      margin = std::min(margin, joint_hard_margin_rad(q, req_.policy.q_min,
                                                      req_.policy.q_max));
    }
    out.min_hard_margin_rad = margin;
    const double z0 = kin_.fk(out.segment.front(), "Body_Link")(2, 3);
    const double z1 = kin_.fk(qN, "Body_Link")(2, 3);
    out.body_drop_m = z0 - z1;
    Vec18 qw = q0_;
    qw.head<kNWaist>() = qN.head<kNWaist>();
    Vec18 qa = q0_;
    qa.segment<kNArm>(arm_) = qN.segment<kNArm>(arm_);
    const Vec3 p0 = kin_.fk_pos(q0_, req_.frame);
    out.waist_tcp_m = (kin_.fk_pos(qw, req_.frame) - p0).norm();
    out.arm_tcp_m = (kin_.fk_pos(qa, req_.frame) - p0).norm();
    out.path_id = path_id_of(out.segment);
  }

  void finish_report(UnifiedPlanResult& out,
                     std::chrono::steady_clock::time_point) const {
    if (out.path_id == 0 && !out.segment.empty()) {
      out.path_id = path_id_of(out.segment);
    }
    TrajectoryCostWeights weights;
    for (int i = 0; i < kNDof; ++i) {
      weights.motion_scale[i] = motion_scale_[i];
    }
    const auto cost =
        evaluate_path_cost(out.segment, q_neutral_, req_.policy, weights);
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "planner=unified_multistage"
       << " path_id=" << std::hex << out.path_id << std::dec
       << " stages=" << stages_.size()
       << " opt=" << (out.ok ? "ok" : "fail")
       << " accept=" << (out.ok ? "pass" : "fail")
       << " tcp_mm=" << (out.tcp_worst_m * 1000.0)
       << " margin_deg=" << (out.min_hard_margin_rad * 180.0 / kPi)
       << " body_drop_mm=" << (out.body_drop_m * 1000.0)
       << " waist_tcp_mm=" << (out.waist_tcp_m * 1000.0)
       << " arm_tcp_mm=" << (out.arm_tcp_m * 1000.0)
       << " iters=" << out.iterations
       << " opt_ms=" << out.optimize_ms
       << " cert_ms=" << out.certify_ms
       << " cost=" << cost.total
       << " height_switch=" << (cost.depends_on_target_height ? 1 : 0)
       << " dir=" << (out.direction.reason[0] ? out.direction.reason : "none");
    out.report = os.str();
    if (out.reason.empty()) {
      out.reason = out.ok ? "unified_multistage" : "unified_ik";
    }
  }

  void restore_hold(Vec18* q) const {
    for (int j = 0; j < kNDof; ++j) {
      if (!active_mask_[static_cast<std::size_t>(j)]) {
        (*q)[j] = q0_[j];
      }
    }
    for (int j = 0; j < kNArm; ++j) {
      (*q)[idle_ + j] = q0_[idle_ + j];
    }
  }

  void clamp_hard(Vec18* q) const {
    for (int j = 0; j < kNDof; ++j) {
      (*q)[j] = std::clamp((*q)[j], req_.policy.q_min[j], req_.policy.q_max[j]);
    }
  }

  void pull_pitch(Vec18* q) const {
    if (!(req_.policy.torso_pitch_max_rad > 0.0)) {
      return;
    }
    for (int n = 0; n < 6; ++n) {
      const double pitch = (*q)[0] + (*q)[1] + (*q)[2];
      const double excess = std::abs(pitch) - pitch_cap_;
      if (excess <= 1e-9) {
        return;
      }
      const double sign = pitch > 0.0 ? 1.0 : -1.0;
      int best = 0;
      double room_best = -1.0;
      for (int a = 0; a < 3; ++a) {
        const double room = sign > 0.0 ? ((*q)[a] - req_.policy.q_min[a])
                                       : (req_.policy.q_max[a] - (*q)[a]);
        if (room > room_best) {
          room_best = room;
          best = a;
        }
      }
      const double step = std::min(excess, std::max(0.0, room_best));
      (*q)[best] -= sign * step;
      (*q)[best] = std::clamp((*q)[best], req_.policy.q_min[best], req_.policy.q_max[best]);
    }
  }
};

}  // namespace

UnifiedPlanResult plan_unified_multistage(const Kinematics& kin,
                                          const WholeBodyGuards& guards,
                                          const UnifiedPlanRequest& req) {
  return UnifiedSolver(kin, guards, req).run();
}

bool recertify_anchored_start(
    const Kinematics& kin, const WholeBodyGuards& guards,
    const MotionConstraintPolicy& policy, const JointCurve& samples,
    const Vec18& q_meas, const Vec3& target, const std::string& frame,
    double pos_tol_m, std::string* why) {
  auto fail = [&](const char* reason) {
    if (why != nullptr) {
      *why = reason;
    }
    return false;
  };
  if (samples.size() < 2) {
    return fail("empty");
  }
  if (!q_meas.allFinite() || joint_violates_hard_box(q_meas, policy.q_min, policy.q_max)) {
    return fail("hard_box");
  }
  if (!guards.com_xy_inside(kin.com(q_meas))) {
    return fail("com");
  }
  const double cap = policy.torso_pitch_max_rad > 0.0
                         ? std::max(policy.torso_pitch_max_rad,
                                    std::abs(samples.front().head<3>().sum()))
                         : 1e9;
  if (std::abs(q_meas.head<3>().sum()) > cap + 1e-6) {
    return fail("pitch");
  }
  const double delta = (q_meas - samples.front()).cwiseAbs().maxCoeff();
  if (delta <= 1e-4) {
    return true;
  }
  const Vec3 p0 = kin.fk_pos(samples.front(), frame);
  const Vec3 ab = target - p0;
  const double ab2 = std::max(ab.squaredNorm(), 1e-12);
  auto cross = [&](const Vec3& p) {
    const double u = std::clamp((p - p0).dot(ab) / ab2, 0.0, 1.0);
    return (p - (p0 + u * ab)).norm();
  };
  const double admit = std::max(pos_tol_m, 0.008);
  if (cross(kin.fk_pos(q_meas, frame)) > admit) {
    return fail("tube");
  }
  const Vec18 mid = 0.5 * (q_meas + samples[1]);
  if (joint_violates_hard_box(mid, policy.q_min, policy.q_max) ||
      !guards.com_xy_inside(kin.com(mid))) {
    return fail("chord");
  }
  if (std::abs(mid.head<3>().sum()) > cap + 1e-6) {
    return fail("pitch");
  }
  if (cross(kin.fk_pos(mid, frame)) > admit) {
    return fail("chord");
  }
  return true;
}

}  // namespace kmpc
