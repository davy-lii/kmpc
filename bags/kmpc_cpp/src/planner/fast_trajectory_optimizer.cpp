// unified_multistage_v2 optimizer. Differences from the retired v1 chain:
//   - solved knots are final: densify never re-poses them (joint interp only);
//   - no direction sign hard gate and no smooth_reversal patch;
//   - the QP subproblem carries the hard rows (joint box, pitch cap, TCP
//     stage targets) instead of relying on after-the-fact patches;
//   - the executed samples are exactly what validate_actual_q_of_t certifies.

#include "kmpc/fast_trajectory_optimizer.hpp"

#include "kmpc/motion_evaluator.hpp"
#include "kmpc/time_parameterizer.hpp"
#include "kmpc/tracker_math.hpp"

#include <algorithm>
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <proxsuite/proxqp/dense/dense.hpp>
#pragma GCC diagnostic pop

namespace kmpc {
namespace {

// Field PRE→TIP approaches are ~0.6–0.8 m. With K=16 the layout coarsens to
// ~50 mm spans; joint densify then bows past a 3 mm LIN tube under orientation.
// K=32 keeps ~12–25 mm spans on those reaches without relaxing the tube.
// Short reaches (≤6 cm) intentionally use fewer knots — never coarsen the
// TCP tube / hard box / pitch / CoM / admit checks to buy latency.
constexpr int kMaxKnots = 32;
constexpr double kKnotSpacingM = 0.012;
constexpr double kShortPathM = 0.06;
constexpr double kDenseSampleM = 0.005;
// Per-sample joint step stays well under the stream QP trust region (the
// tracker accepts roughly 4 x per-cycle dq cap, ~0.023 rad on arm joints).
constexpr double kDenseJointStepRad = 0.02;
constexpr int kMaxDenseSamples = 400;
// Same reach envelope as the retired planner (400 samples x 5 mm).
constexpr double kReachLimitM = 2.0;
constexpr double kWPos = 400.0;
constexpr double kWOri = 40.0;
constexpr int kMaxGnIters = 14;
// After the knots already satisfy hard geometry (accept_knots), only a few
// quality iters remain — do not polish soft cost forever (§9.3).
constexpr int kMaxGnItersAfterCert = 3;
constexpr int kMaxGnFeasIters = 8;
// Stream-compatibility endgame: the tracker stream QP switches to a 3 mm
// euclidean arrival tube (per-axis box 3/sqrt(3) mm) for the last chord
// fraction (line_qp.cpp kQpTubeEndM / kQpTubeEndFrac, not modifiable here).
// A stage endpoint that only satisfies the task tolerance (e.g. 3 mm
// euclidean, up to 2.4 mm per axis inside the QP rows) can still sit outside
// that arrival box and the stream refuses the line mid-approach. Stage
// endpoint rows/checks therefore use the tighter of both boxes.
constexpr double kStreamEndTubeM = 0.003;
constexpr double kStreamEndBoxM =
    kStreamEndTubeM / 1.7320508075688772;  // ~1.732 mm per axis
constexpr double kStreamEndRowM = 0.9 * kStreamEndBoxM;

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

uint64_t digest_of_string(const std::string& s) {
  uint64_t h = 14695981039346656037ull;
  for (char c : s) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
    h *= 1099511628211ull;
  }
  return h;
}

struct StageGeom {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 a = Vec3::Zero();
  Vec3 b = Vec3::Zero();
  Vec4 qa = Vec4(0, 0, 0, 1);
  Vec4 qb = Vec4(0, 0, 0, 1);
  bool ori = false;
  TaskGeometry geo = TaskGeometry::Lin;
  double tol = 0.003;
  double ori_tol = 0.05;
  uint32_t stage_id = 0;
  bool stop_required = false;
  bool wait_ack = false;
};

struct KnotTarget {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 p = Vec3::Zero();
  Vec4 quat = Vec4(0, 0, 0, 1);
  bool ori = false;
  bool stage_endpoint = false;
  double tol = 0.003;
  double ori_tol = 0.05;
};

class V2Solver {
 public:
  enum class SeedKind { RemainingPath = 0, Neutral = 1, ArmHeavy = 2 };

  V2Solver(const Kinematics& kin, const WholeBodyGuards* guards,
           const MotionTask& task, const RobotState& state,
           const MotionConstraintPolicy& policy,
           const TrajectoryCostWeights& weights, double deadline_ms)
      : kin_(kin),
        guards_(guards),
        task_(task),
        policy_(policy),
        weights_(weights),
        deadline_ms_(deadline_ms),
        t0_(std::chrono::steady_clock::now()) {
    q0_ = state.q.allFinite() ? state.q : task.q;
    start_dq_ = state.dq;
    start_dq_known_ = state.dq_known;
    left_ = task.left;
    frame_ = !task.frame.empty()
                 ? task.frame
                 : (left_ ? std::string(kEeFrameLeft)
                          : std::string(kEeFrameRight));
    arm_ = left_ ? kSliceLeft.begin : kSliceRight.begin;
    active_.reserve(11);
    if (!task.waist_locked) {
      for (int j = 0; j < kNWaist; ++j) {
        active_.push_back(j);
      }
    }
    for (int j = 0; j < kNArm; ++j) {
      active_.push_back(arm_ + j);
    }
    // Neutral posture model (mid hard-box) — NOT the measured start.
    // Start keeping is enforced by freezing samples.front()=q0_ and move cost.
    for (int j = 0; j < kNDof; ++j) {
      q_neutral_[j] = 0.5 * (policy.q_min[j] + policy.q_max[j]);
    }
    pitch_cap_ = policy.torso_pitch_max_rad > 0.0
                     ? policy.torso_pitch_max_rad
                     : 1e9;
    start_pitch_abs_ = std::abs(q0_.head<3>().sum());
  }

  FastPlanResult run() {
    FastPlanResult out;
    std::string why;
    if (!preflight(&why)) {
      out.reason = why;
      finish_report(out);
      return out;
    }
    JointCurve knots;
    std::vector<KnotTarget> targets;
    if (!solve_knots(&knots, &targets, &why)) {
      out.reason = why;
      out.iterations = iterations_;
      finish_report(out);
      return out;
    }
    out.optimize_ms = elapsed_ms();
    out.iterations = iterations_;

    JointCurve samples;
    std::vector<StageBoundary> boundaries;
    if (!densify(knots, &samples, &boundaries, &why)) {
      out.reason = why;
      finish_report(out);
      return out;
    }
    const TimeParameterizer parameterizer;
    const TimeLawTable law =
        parameterizer.compute(samples, policy_, boundaries);

    TrajectoryPackage pkg;
    pkg.generation = task_.generation;
    pkg.task_digest = motion_task_digest(task_);
    pkg.model_digest = digest_of_string(task_.model_digest);
    pkg.limits_digest = limits_digest_of(policy_);
    pkg.path_id = path_id_of(samples);
    pkg.start_q = q0_;
    pkg.start_dq = start_dq_;
    pkg.start_dq_known = start_dq_known_;
    pkg.left = left_;
    pkg.samples = samples;
    pkg.time_law = law;
    pkg.stage_boundaries = boundaries;
    pkg.planner = "unified_multistage_v2";

    const TrajectoryCertificate cert =
        validate_actual_q_of_t(pkg, task_, policy_, kin_, guards_);
    out.certify_ms = elapsed_ms() - out.optimize_ms;
    out.certificate = cert;
    out.package = pkg;
    out.package.certificate.passed = cert.passed;
    out.package.certificate.continuous_certified = cert.continuous_certified;
    out.package.certificate.sampled_only = true;
    out.package.certificate.coverage = "sampled_only";
    out.package.certificate.min_joint_margin_rad = cert.min_joint_margin_rad;
    out.package.certificate.min_pitch_margin_rad = cert.min_pitch_margin_rad;
    out.package.certificate.worst_tcp_cross_track_m =
        cert.worst_tcp_cross_track_m;
    out.package.certificate.worst_tcp_endpoint_m = cert.worst_tcp_endpoint_m;
    out.package.certificate.first_fault = cert.first_fault;

    out.ok = cert.passed;
    out.reason = cert.passed ? "unified_multistage_v2" : cert.first_fault;
    out.tcp_worst_m = std::max(cert.worst_tcp_cross_track_m,
                               cert.worst_tcp_endpoint_m);
    out.min_hard_margin_rad = cert.min_joint_margin_rad;
    out.duration_s = law.duration_s;
    out.software_limited = law.software_limited;
    samples_ = std::move(samples);
    fill_body_metrics(&out);
    finish_report(out);
    out.package.quality_report = out.report;
    return out;
  }

 private:
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0_)
        .count();
  }

  bool deadline_hit() {
    if (deadline_ms_ > 0.0 && elapsed_ms() > deadline_ms_) {
      deadline_hit_ = true;
      return true;
    }
    return false;
  }

  bool preflight(std::string* why) {
    if (!q0_.allFinite()) {
      *why = "unified_ik";
      return false;
    }
    if (task_.stages.empty()) {
      *why = "stage_count_not_supported";
      return false;
    }
    if (task_.stages.size() > static_cast<std::size_t>(kMaxKnots - 1)) {
      *why = "too_many_stages";
      return false;
    }
    if (joint_violates_hard_box(q0_, policy_.q_min, policy_.q_max)) {
      *why = "unified_hard_box";
      return false;
    }
    const auto pose0 = kin_.fk(q0_, frame_);
    Vec3 prev = pose0.col(3).head<3>();
    Vec4 prev_quat = rot_to_quat_xyzw(pose0.block<3, 3>(0, 0));
    for (const TaskStage& stage : task_.stages) {
      StageGeom g;
      g.a = prev;
      g.b = stage.position;
      g.geo = stage.geometry;
      g.tol = stage.position_tol_m;
      g.ori_tol = stage.orientation_tol_rad;
      g.ori = stage.has_orientation;
      g.stage_id = stage.stage_id;
      g.stop_required = stage.stop == StageStop::StopRequired;
      g.wait_ack = stage.wait_for_external_ack;
      g.qa = prev_quat;
      g.qb = stage.has_orientation ? stage.quat : prev_quat;
      if (g.geo == TaskGeometry::FollowCurve) {
        *why = "follow_curve_not_implemented";
        return false;
      }
      if (!g.b.allFinite() || !(g.tol > 0.0) || !std::isfinite(g.tol)) {
        *why = "bad_task_tol";
        return false;
      }
      if (g.geo == TaskGeometry::Lin && g.tol >= 0.20) {
        *why = "lin_tol_is_legacy_hard_tube";
        return false;
      }
      if ((g.b - g.a).norm() > kReachLimitM) {
        *why = "unified_sample_limit";
        return false;
      }
      prev = g.b;
      prev_quat = g.qb;
      stages_.push_back(g);
    }
    return true;
  }

  bool solve_knots(JointCurve* knots, std::vector<KnotTarget>* targets,
                   std::string* why) {
    // Knot layout: ~12 mm spacing, K <= kMaxKnots (spans across all stages).
    // Short single-stage reaches keep few knots (e.g. 5 cm → ~2–3 spans).
    double path_len = 0.0;
    for (const StageGeom& g : stages_) {
      path_len += (g.b - g.a).norm();
    }
    std::vector<int> spans;
    double spacing = kKnotSpacingM;
    if (path_len <= kShortPathM && stages_.size() == 1) {
      // Prefer ~2 interior spans so mid-chord bow is still checked, without
      // allocating a long-path knot budget for a 5 cm reach.
      spacing = std::max(kKnotSpacingM, path_len / 2.0);
    }
    int total = 0;
    for (int attempt = 0; attempt < 10; ++attempt) {
      spans.clear();
      total = 0;
      for (const StageGeom& g : stages_) {
        const double dist = (g.b - g.a).norm();
        const int n = std::clamp(
            static_cast<int>(std::ceil(dist / spacing)), 1, kMaxKnots - 1);
        spans.push_back(n);
        total += n;
      }
      if (total <= kMaxKnots - 1) {
        break;
      }
      spacing *= 1.5;
    }
    if (total > kMaxKnots - 1) {
      // Last resort: one span per stage, extras by distance share.
      const int extra = (kMaxKnots - 1) - static_cast<int>(stages_.size());
      double dist_sum = 0.0;
      for (const StageGeom& g : stages_) {
        dist_sum += (g.b - g.a).norm();
      }
      total = 0;
      for (std::size_t s = 0; s < stages_.size(); ++s) {
        const double dist = (stages_[s].b - stages_[s].a).norm();
        int n = 1;
        if (dist_sum > 1e-9 && extra > 0) {
          n += std::min(extra, static_cast<int>(std::floor(
                                   extra * dist / dist_sum + 0.5)));
        }
        spans[static_cast<std::size_t>(s)] = n;
        total += n;
      }
      while (total > kMaxKnots - 1) {
        auto it = std::max_element(spans.begin(), spans.end());
        if (*it <= 1) {
          *why = "too_many_stages";
          return false;
        }
        --(*it);
        --total;
      }
    }
    knot_of_stage_.clear();
    knot_of_stage_.push_back(0);
    int K = 1;
    for (int n : spans) {
      K += n;
      knot_of_stage_.push_back(K - 1);
    }

    if (trace_gn()) {
      std::fprintf(stderr, "v2gn layout K=%d spacing=%.4g spans=", K, spacing);
      for (int n : spans) {
        std::fprintf(stderr, "%d,", n);
      }
      std::fprintf(stderr, "\n");
    }
    knots->assign(static_cast<std::size_t>(K), q0_);
    targets->assign(static_cast<std::size_t>(K), KnotTarget{});
    (*targets)[0].p = stages_[0].a;
    (*targets)[0].quat = stages_[0].qa;
    for (std::size_t s = 0; s < stages_.size(); ++s) {
      const int k0 = knot_of_stage_[s];
      const int k1 = knot_of_stage_[s + 1];
      const int n = std::max(1, k1 - k0);
      for (int k = k0; k <= k1; ++k) {
        const double t = static_cast<double>(k - k0) / static_cast<double>(n);
        KnotTarget& tgt = (*targets)[static_cast<std::size_t>(k)];
        if (k == k0 && s > 0) {
          // Shared stage boundary: it is the INCOMING stage's endpoint. Keep
          // that stage's target/tolerance and the endpoint flag — the next
          // stage's loop must not erase them.
          tgt.stage_endpoint = true;
          continue;
        }
        tgt.p = (1.0 - t) * stages_[s].a + t * stages_[s].b;
        tgt.quat = quat_slerp_xyzw(stages_[s].qa, stages_[s].qb, t);
        tgt.ori = stages_[s].ori;
        tgt.stage_endpoint = (k == k1);
        tgt.tol = stages_[s].tol;
        tgt.ori_tol = stages_[s].ori_tol;
      }
    }

    // Multi-seed: same cost + hard constraints, different warm starts.
    // Order prefers arm motion when waist already sits on a hard wall.
    const bool waist_near_wall = waist_near_hard_wall();
    SeedKind order[3];
    if (waist_near_wall) {
      order[0] = SeedKind::ArmHeavy;
      order[1] = SeedKind::Neutral;
      order[2] = SeedKind::RemainingPath;
    } else {
      order[0] = SeedKind::RemainingPath;
      order[1] = SeedKind::Neutral;
      order[2] = SeedKind::ArmHeavy;
    }

    std::string last_why = "unified_ik";
    JointCurve best_knots;
    bool have_best = false;
    double best_end_axis = 1e9;

    for (SeedKind kind : order) {
      if (deadline_hit()) {
        *why = "deadline";
        return false;
      }
      JointCurve trial = *knots;  // already q0-filled layout
      seed_curve(&trial, *targets, kind);
      std::string accept_why;
      // First seed that already hard-certifies: light quality polish only,
      // then stop — do not burn the remaining seed budget (§9.3).
      const bool seed_certified = accept_knots(trial, *targets, &accept_why);
      if (seed_certified) {
        JointCurve polished = trial;
        gauss_newton(&polished, *targets, /*already_certified=*/true);
        if (!deadline_hit_ && accept_knots(polished, *targets, &accept_why)) {
          trial = std::move(polished);
        }
        // Soft polish must not undo hard geometry; keep the certified seed.
        *knots = std::move(trial);
        return true;
      }
      gauss_newton(&trial, *targets, /*already_certified=*/false);
      if (deadline_hit_) {
        last_why = "deadline";
        continue;
      }
      if (!accept_knots(trial, *targets, &accept_why)) {
        polish_knots(&trial, *targets);
      }
      const double end_axis = endpoint_axis_error(trial, *targets);
      if (accept_knots(trial, *targets, &accept_why)) {
        *knots = std::move(trial);
        return true;
      }
      last_why = accept_why;
      if (end_axis < best_end_axis) {
        best_end_axis = end_axis;
        best_knots = std::move(trial);
        have_best = true;
      }
      if (trace_gn()) {
        std::fprintf(stderr, "v2gn seed=%d fail %s end_axis=%.5g\n",
                     static_cast<int>(kind), accept_why.c_str(), end_axis);
      }
    }
    if (have_best) {
      *knots = std::move(best_knots);
    }
    *why = last_why;
    return false;
  }

  bool waist_near_hard_wall(double margin_rad = 0.05) const {
    if (task_.waist_locked) {
      return false;
    }
    for (int j = 0; j < kNWaist; ++j) {
      const double lo = q0_[j] - policy_.q_min[j];
      const double hi = policy_.q_max[j] - q0_[j];
      if (lo <= margin_rad || hi <= margin_rad) {
        return true;
      }
    }
    return false;
  }

  Vec18 unbiased_neutral() const {
    // Height-independent mid-range prior — not a fixed squat pose.
    Vec18 q = q0_;
    for (int j : active_) {
      if (std::isfinite(policy_.q_min[j]) && std::isfinite(policy_.q_max[j])) {
        q[j] = 0.5 * (policy_.q_min[j] + policy_.q_max[j]);
      }
    }
    // Keep start waist if already near a wall so the prior does not invent a
    // fold; arm DOFs still open the redundancy space.
    if (waist_near_hard_wall()) {
      q.head<kNWaist>() = q0_.head<kNWaist>();
    }
    clamp_hard(&q);
    pull_pitch(&q);
    return q;
  }

  void seed_curve(JointCurve* knots, const std::vector<KnotTarget>& targets,
                  SeedKind kind) {
    const int K = static_cast<int>(knots->size());
    const double waist_w =
        (kind == SeedKind::ArmHeavy) ? 80.0 : 0.4;
    const Vec18 neutral = unbiased_neutral();
    for (int k = 1; k < K; ++k) {
      (*knots)[static_cast<std::size_t>(k)] =
          (*knots)[static_cast<std::size_t>(k - 1)];
      Vec18 anchor = (*knots)[static_cast<std::size_t>(k - 1)];
      if (kind == SeedKind::Neutral) {
        // Blend previous knot with the unbiased mid-range prior.
        anchor = 0.5 * (anchor + neutral);
      } else if (kind == SeedKind::ArmHeavy) {
        // Pin waist to start; arm carries the Cartesian step.
        anchor.head<kNWaist>() = q0_.head<kNWaist>();
      }
      seed_ik(&(*knots)[static_cast<std::size_t>(k)],
              targets[static_cast<std::size_t>(k)], anchor, 16, waist_w);
      if (kind == SeedKind::ArmHeavy) {
        // Re-assert waist near start after the seed step so the GN warm
        // start does not begin already leaning into the hard wall.
        (*knots)[static_cast<std::size_t>(k)].head<kNWaist>() =
            q0_.head<kNWaist>();
        clamp_hard(&(*knots)[static_cast<std::size_t>(k)]);
        pull_pitch(&(*knots)[static_cast<std::size_t>(k)]);
      }
      ++iterations_;
    }
  }

  void polish_knots(JointCurve* knots,
                    const std::vector<KnotTarget>& targets) {
    // Phase C: block coordinate descent. The global GN direction degrades
    // near the constraint boundary; sweeping single knots with damped 6D IK
    // (same lerped targets, self-anchored) converges the curve onto the
    // chord. Still inside the solve — densify/validate certify the result,
    // nothing is re-posed afterwards.
    const int K = static_cast<int>(knots->size());
    const double waist_w = waist_near_hard_wall() ? 80.0 : 0.4;
    for (int pass = 0; pass < 2; ++pass) {
      for (int kk = 1; kk < K; ++kk) {
        const int k = pass == 0 ? kk : K - kk;  // forward, then backward
        const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
        Vec18& qk = (*knots)[static_cast<std::size_t>(k)];
        const auto pose = kin_.fk(qk, frame_);
        const double err = (tgt.p - pose.col(3).head<3>()).norm();
        const double axis =
            (tgt.p - pose.col(3).head<3>()).cwiseAbs().maxCoeff();
        double oe = 0.0;
        if (tgt.ori && tgt.stage_endpoint) {
          oe = ori_error_xyzw(tgt.quat,
                              rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                   .norm();
        }
        const bool tight =
            tgt.stage_endpoint
                ? (axis <= kStreamEndRowM && oe <= 0.5 * tgt.ori_tol)
                : (err <= 0.5 * min_tol());
        if (tight) {
          continue;
        }
        seed_ik(&qk, tgt, qk, tgt.stage_endpoint ? 24 : 8, waist_w);
      }
    }
  }

  double endpoint_axis_error(const JointCurve& knots,
                             const std::vector<KnotTarget>& targets) const {
    double axis = 0.0;
    for (std::size_t k = 1; k < knots.size(); ++k) {
      if (!targets[k].stage_endpoint) {
        continue;
      }
      axis = std::max(
          axis, (targets[k].p - kin_.fk_pos(knots[k], frame_))
                    .cwiseAbs()
                    .maxCoeff());
    }
    return axis;
  }

  bool accept_knots(const JointCurve& knots,
                    const std::vector<KnotTarget>& targets,
                    std::string* why) const {
    const int K = static_cast<int>(knots.size());
    double worst = 0.0;
    double tol_max = 0.0;
    for (int k = 1; k < K; ++k) {
      const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
      const Vec3 pe = kin_.fk_pos(knots[static_cast<std::size_t>(k)], frame_);
      const double err = (tgt.p - pe).norm();
      worst = std::max(worst, err);
      tol_max = std::max(tol_max, tgt.tol);
      if (k + 1 < K) {
        // Inter-knot FK bow: the densifier interpolates in joint space, so
        // the executed midpoint can leave the chord even when both knots sit
        // inside. Certify what execution would see.
        const Vec18 qmid =
            0.5 * (knots[static_cast<std::size_t>(k)] +
                   knots[static_cast<std::size_t>(k + 1)]);
        const Vec3 pmid =
            0.5 * (tgt.p + targets[static_cast<std::size_t>(k + 1)].p);
        const double merr = (pmid - kin_.fk_pos(qmid, frame_)).norm();
        worst = std::max(worst, merr);
        if (trace_gn() && merr > 0.8 * tgt.tol) {
          std::fprintf(stderr, "v2gn accept mid k=%d merr=%.5g (knot %.5g)\n",
                       k, merr, err);
        }
      }
      if (tgt.stage_endpoint) {
        if (trace_gn()) {
          const auto pose =
              kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
          const double oe =
              tgt.ori ? ori_error_xyzw(
                            tgt.quat,
                            rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                            .norm()
                      : 0.0;
          std::fprintf(stderr, "v2gn accept k=%d pos=%.5g axis=%.5g ori=%.5g\n",
                       k, err, (tgt.p - pe).cwiseAbs().maxCoeff(), oe);
        }
        // Stream arrival box: per-axis, fixed 3 mm tube, not the task tol.
        const double axis = (tgt.p - pe).cwiseAbs().maxCoeff();
        if (axis > kStreamEndBoxM) {
          std::ostringstream os;
          os << std::fixed << std::setprecision(4)
             << "unified_ik:end_axis=" << axis;
          *why = os.str();
          return false;
        }
        if (tgt.ori) {
          const auto pose =
              kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
          const double oe = ori_error_xyzw(
                                tgt.quat,
                                rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                                .norm();
          if (oe > tgt.ori_tol) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(4)
               << "unified_ik:end_ori=" << oe;
            *why = os.str();
            return false;
          }
        }
      }
    }
    if (worst > tol_max) {
      std::ostringstream os;
      os << std::fixed << std::setprecision(4) << "unified_ik:m=" << worst;
      *why = os.str();
      return false;
    }
    return true;
  }

  void seed_ik(Vec18* q, const KnotTarget& tgt, const Vec18& anchor,
               int iters, double waist_anchor_w = 0.4) const {
    // 6D damped IK seed: position AND (when the stage asks for it) the
    // slerped orientation. A position-only seed can strand the wrist in a
    // branch from which the GN solve cannot reach the endpoint orientation
    // within the stage tolerance. waist_anchor_w >> arm soft-anchor prefers
    // arm motion when the waist is already against a hard wall.
    const int na = static_cast<int>(active_.size());
    const bool ori_now = tgt.ori;
    const double ori_goal = 0.5 * tgt.ori_tol;
    for (int it = 0; it < iters; ++it) {
      const auto pose = kin_.fk(*q, frame_);
      const Vec3 ep = tgt.p - pose.col(3).head<3>();
      Vec3 eo = Vec3::Zero();
      if (ori_now) {
        eo = ori_error_xyzw(tgt.quat,
                            rot_to_quat_xyzw(pose.block<3, 3>(0, 0)));
      }
      const double base = ep.norm();
      if (base <= 0.5 * min_tol() && (!ori_now || eo.norm() <= ori_goal)) {
        break;
      }
      const double merit =
          kWPos * base * base + (ori_now ? kWOri * eo.squaredNorm() : 0.0);
      const Mat6x18 J = kin_.jacobian_spatial(*q, frame_);
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(na, na);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(na);
      constexpr double w_arm_anchor = 0.4;
      for (int i = 0; i < na; ++i) {
        const int ai = active_[static_cast<std::size_t>(i)];
        const double w_anchor =
            (ai < kNWaist) ? waist_anchor_w : w_arm_anchor;
        g[i] = kWPos * J.col(ai).head<3>().dot(ep) +
               w_anchor * (anchor[ai] - (*q)[ai]);
        if (ori_now) {
          g[i] += kWOri * J.col(ai).tail<3>().dot(eo);
        }
        for (int j = i; j < na; ++j) {
          const int aj = active_[static_cast<std::size_t>(j)];
          double hij = kWPos * J.col(ai).head<3>().dot(J.col(aj).head<3>());
          if (ori_now) {
            hij += kWOri * J.col(ai).tail<3>().dot(J.col(aj).tail<3>());
          }
          H(i, j) += hij;
          if (i != j) {
            H(j, i) += hij;
          }
        }
        H(i, i) += 1e-4 + w_anchor;
      }
      const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
      if (ldlt.info() != Eigen::Success) {
        return;
      }
      Eigen::VectorXd step = ldlt.solve(g);
      if (!step.allFinite()) {
        return;
      }
      const double cap = base < 0.01 ? 0.04 : 0.08;
      const double peak = step.cwiseAbs().maxCoeff();
      if (peak > cap) {
        step *= cap / peak;
      }
      bool improved = false;
      for (double alpha : {1.0, 0.5, 0.25, 0.1}) {
        Vec18 trial = *q;
        for (int i = 0; i < na; ++i) {
          trial[active_[static_cast<std::size_t>(i)]] += alpha * step[i];
        }
        clamp_hard(&trial);
        pull_pitch(&trial);
        const auto tpose = kin_.fk(trial, frame_);
        const double err = (tgt.p - tpose.col(3).head<3>()).norm();
        const double oe =
            ori_now ? ori_error_xyzw(
                          tgt.quat,
                          rot_to_quat_xyzw(tpose.block<3, 3>(0, 0)))
                        .norm()
                    : 0.0;
        const double trial_merit =
            kWPos * err * err + (ori_now ? kWOri * oe * oe : 0.0);
        const double need = base <= 2.0 * min_tol()
                                ? 1e-4
                                : std::max(2e-4, 0.2 * (base - 0.5 * min_tol()));
        if ((err <= 0.5 * min_tol() && oe <= ori_goal) ||
            trial_merit <= merit - kWPos * need * need) {
          *q = trial;
          improved = true;
          break;
        }
      }
      if (!improved) {
        break;
      }
    }
    repair_com(q, tgt.p);
  }

  void repair_com(Vec18* q, const Vec3& p_des) const {
    if (guards_ == nullptr) {
      return;
    }
    const double cx = 0.5 * (guards_->com_work_xmin() + guards_->com_work_xmax());
    const double cy = 0.5 * (guards_->com_work_ymin() + guards_->com_work_ymax());
    const int na = static_cast<int>(active_.size());
    for (int it = 0; it < 4; ++it) {
      const Vec3 com = kin_.com(*q);
      if (guards_->com_xy_inside(com)) {
        return;
      }
      const double tcp0 = (p_des - kin_.fk_pos(*q, frame_)).norm();
      const Eigen::Vector2d err(cx - com.x(), cy - com.y());
      const Mat3x18 Jc = kin_.jacobian_com(*q);
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(na, na);
      Eigen::VectorXd g = Eigen::VectorXd::Zero(na);
      for (int i = 0; i < na; ++i) {
        const int ai = active_[static_cast<std::size_t>(i)];
        g[i] = Jc(0, ai) * err[0] + Jc(1, ai) * err[1];
        for (int j = i; j < na; ++j) {
          const int aj = active_[static_cast<std::size_t>(j)];
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
        trial[active_[static_cast<std::size_t>(i)]] += step[i];
      }
      clamp_hard(&trial);
      pull_pitch(&trial);
      const double tcp = (p_des - kin_.fk_pos(trial, frame_)).norm();
      if (tcp > std::min(tcp0 + 1e-4, min_tol() - 2e-4)) {
        return;
      }
      *q = trial;
    }
  }

  double min_tol() const {
    double tol = 1e9;
    for (const StageGeom& g : stages_) {
      tol = std::min(tol, g.tol);
    }
    return tol;
  }

  double objective(const JointCurve& knots,
                   const std::vector<KnotTarget>& targets,
                   double* worst) const {
    double pos = 0.0;
    double worst_err = 0.0;
    const int K = static_cast<int>(knots.size());
    for (int k = 1; k < K; ++k) {
      const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
      const auto pose = kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
      const Vec3 epv = tgt.p - pose.col(3).head<3>();
      const double e = epv.norm();
      pos += e * e;
      worst_err = std::max(worst_err, e);
      if (tgt.stage_endpoint) {
        // Line search must never trade the stream arrival box against the
        // soft cost: outside the row box the penalty dwarfs every pull.
        const double axis = epv.cwiseAbs().maxCoeff();
        const double viol = std::max(0.0, axis - kStreamEndRowM);
        pos += 1.0e6 * viol * viol;
      }
      if (tgt.ori && tgt.stage_endpoint) {
        const Vec3 eo = ori_error_xyzw(
            tgt.quat, rot_to_quat_xyzw(pose.block<3, 3>(0, 0)));
        pos += (kWOri / kWPos) * eo.squaredNorm();
      }
      if (k + 1 < K) {
        const Vec18 qmid = 0.5 * (knots[static_cast<std::size_t>(k)] +
                                  knots[static_cast<std::size_t>(k + 1)]);
        const Vec3 pm =
            0.5 * (tgt.p + targets[static_cast<std::size_t>(k + 1)].p);
        const Vec3 em = pm - kin_.fk_pos(qmid, frame_);
        pos += em.squaredNorm();
      }
    }
    if (worst != nullptr) {
      *worst = worst_err;
    }
    // Same waist ramp glue as in the QP model (keeps line search honest).
    double ramp = 0.0;
    if (!task_.waist_locked) {
      constexpr double kWRamp = 1.0e4;
      for (std::size_t s = 0; s + 1 < knot_of_stage_.size(); ++s) {
        const int k0 = knot_of_stage_[s];
        const int k1 = knot_of_stage_[s + 1];
        for (int k = k0 + 1; k < k1; ++k) {
          const double t =
              static_cast<double>(k - k0) / static_cast<double>(k1 - k0);
          for (int j = 0; j < 3; ++j) {
            const double q0j =
                k0 == 0 ? q0_[j] : knots[static_cast<std::size_t>(k0)][j];
            const double r = knots[static_cast<std::size_t>(k)][j] -
                             (1.0 - t) * q0j -
                             t * knots[static_cast<std::size_t>(k1)][j];
            ramp += kWRamp * r * r;
          }
        }
      }
    }
    const PathCostBreakdown cost = evaluate_path_cost(
        knots, q_neutral_, policy_, weights_, kin_, left_, guards_);
    return kWPos * pos + ramp + cost.total;
  }

  void gauss_newton(JointCurve* knots, const std::vector<KnotTarget>& targets,
                    bool already_certified) {
    // Phase A: merged objective (task + TrajectoryCost + ramp glue).
    // When the seed already hard-certifies, only a few quality iters run.
    gn_phase(knots, targets, true,
             already_certified ? kMaxGnItersAfterCert : kMaxGnIters);
    std::string why;
    if (already_certified || accept_knots(*knots, targets, &why)) {
      return;  // certified: skip feasibility polish forever (§9.3)
    }
    // Phase B: feasibility-only polish. The merged objective can equilibrate
    // with interior knots a few millimetres off the chord and the endpoint
    // just outside the stream arrival box; a short task-only GN drives the
    // same solved knots onto the same targets. This is still inside the
    // solve — nothing re-poses the result afterwards.
    gn_phase(knots, targets, false, kMaxGnFeasIters);
  }

  void gn_phase(JointCurve* knots, const std::vector<KnotTarget>& targets,
                bool use_cost, int max_iters) {
    const int K = static_cast<int>(knots->size());
    const int na = static_cast<int>(active_.size());
    if (K < 2 || na < 1 || max_iters <= 0) {
      return;
    }
    const int nvar = (K - 1) * na;
    const bool waist_rows = !task_.waist_locked;
    int nin = 0;
    for (int k = 1; k < K; ++k) {
      if (waist_rows) {
        ++nin;  // pitch row
      }
      if (targets[static_cast<std::size_t>(k)].stage_endpoint) {
        nin += 3;  // TCP stage target rows
        if (targets[static_cast<std::size_t>(k)].ori) {
          nin += 3;  // orientation stage target rows
        }
      }
    }

    proxsuite::proxqp::dense::QP<double> qp(nvar, 0, nin, true);
    qp.settings.eps_abs = 1e-9;
    qp.settings.eps_rel = 1e-9;
    qp.settings.verbose = false;
    qp.settings.max_iter = 200;
    qp.settings.initial_guess =
        proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    bool qp_ready = false;

    // Fixed workspaces — resize once, setZero each iter (no realloc churn).
    Eigen::MatrixXd H(nvar, nvar);
    Eigen::VectorXd g(nvar);
    Eigen::VectorXd lb(nvar);
    Eigen::VectorXd ub(nvar);
    Eigen::MatrixXd C(nin, nvar);
    Eigen::VectorXd l(nin);
    Eigen::VectorXd u(nin);
    Eigen::VectorXd dq(nvar);
    Eigen::VectorXd g_qp(nvar);
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
        poses(static_cast<std::size_t>(K));
    std::vector<Mat6x18, Eigen::aligned_allocator<Mat6x18>> jacs(
        static_cast<std::size_t>(K));

    int post_cert_left = -1;
    for (int iter = 0; iter < max_iters; ++iter) {
      if (deadline_hit()) {
        return;
      }
      ++iterations_;
      H.setZero();
      g.setZero();
      // One FK + spatial Jac per knot; hard rows reuse the same snapshots.
      for (int k = 1; k < K; ++k) {
        const Vec18& q = (*knots)[static_cast<std::size_t>(k)];
        poses[static_cast<std::size_t>(k)] = kin_.fk(q, frame_);
        jacs[static_cast<std::size_t>(k)] = kin_.jacobian_spatial(q, frame_);
      }
      for (int k = 1; k < K; ++k) {
        const Vec18& q = (*knots)[static_cast<std::size_t>(k)];
        const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
        const auto& pose = poses[static_cast<std::size_t>(k)];
        const Mat6x18& J = jacs[static_cast<std::size_t>(k)];
        const Vec3 ep = tgt.p - pose.col(3).head<3>();
        const int col = (k - 1) * na;
        const bool ori_now = tgt.ori && tgt.stage_endpoint;
        Vec3 eo = Vec3::Zero();
        if (ori_now) {
          eo = ori_error_xyzw(tgt.quat,
                              rot_to_quat_xyzw(pose.block<3, 3>(0, 0)));
        }
        // Feasibility phase: endpoint tracking dominates so the GN direction
        // prioritizes the arrival box over mid-line cosmetics.
        const double w_mult =
            (!use_cost && tgt.stage_endpoint) ? 10.0 : 1.0;
        add_task_block(&H, &g, col, J, ep, ori_now ? &eo : nullptr, w_mult);
        if (k + 1 < K) {
          const Vec18& qn = (*knots)[static_cast<std::size_t>(k + 1)];
          const Vec18 qmid = 0.5 * (q + qn);
          const Vec3 pm =
              0.5 * (tgt.p + targets[static_cast<std::size_t>(k + 1)].p);
          const Vec3 em = pm - kin_.fk_pos(qmid, frame_);
          const Mat6x18 Jm = kin_.jacobian_spatial(qmid, frame_);
          add_mid_block(&H, &g, col, col + na, Jm, em, 0.5 * kWPos);
        }
      }
      // Same TrajectoryCost that scores the package, linearized at the knots.
      // Skipped in the feasibility-only phase. MotionEvaluator batch reuse
      // lives inside value_and_linearize (one FK/CoM/Jac pass per knot).
      if (use_cost) {
        const CostLinearization cl = value_and_linearize(
            *knots, q0_, q_neutral_, policy_, weights_, kin_, left_, guards_,
            active_);
        for (int k = 1; k < K; ++k) {
          for (int i = 0; i < na; ++i) {
            const std::size_t src =
                static_cast<std::size_t>(k) * static_cast<std::size_t>(na) +
                static_cast<std::size_t>(i);
            const int dst = (k - 1) * na + i;
            g[dst] -= cl.grad[src];
            H(dst, dst) += cl.hess_diag[src];
          }
        }
      }
      // Waist ramp glue: interior knot waist J1/J2/J3 are pulled onto the
      // linear ramp between the owning stage's boundary knots, inside the
      // same QP. The committed-sign stream contract then sees a monotone
      // waist by construction — v2 keeps no sign hard gate and no post-solve
      // straightening patch. Stage endpoints stay free (full active set).
      constexpr double kWRamp = 1.0e4;
      for (std::size_t s = 0; s + 1 < knot_of_stage_.size(); ++s) {
        const int k0 = knot_of_stage_[s];
        const int k1 = knot_of_stage_[s + 1];
        for (int k = k0 + 1; k < k1; ++k) {
          const double t =
              static_cast<double>(k - k0) / static_cast<double>(k1 - k0);
          for (int j = 0; j < 3; ++j) {
            if (task_.waist_locked) {
              continue;  // waist joints are not in the active set
            }
            const double q0j =
                k0 == 0 ? q0_[j] : (*knots)[static_cast<std::size_t>(k0)][j];
            const double qkj = (*knots)[static_cast<std::size_t>(k)][j];
            const double q1j = (*knots)[static_cast<std::size_t>(k1)][j];
            const double r = qkj - (1.0 - t) * q0j - t * q1j;
            const int ck = (k - 1) * na + j;  // active_[0..3] == waist 0..3
            const int c1 = (k1 - 1) * na + j;
            g[ck] -= 2.0 * kWRamp * r;
            H(ck, ck) += 2.0 * kWRamp;
            g[c1] += 2.0 * kWRamp * r * t;
            H(c1, c1) += 2.0 * kWRamp * t * t;
            H(ck, c1) -= 2.0 * kWRamp * t;
            H(c1, ck) -= 2.0 * kWRamp * t;
            if (k0 >= 1) {
              const int c0 = (k0 - 1) * na + j;
              g[c0] += 2.0 * kWRamp * r * (1.0 - t);
              H(c0, c0) += 2.0 * kWRamp * (1.0 - t) * (1.0 - t);
              H(ck, c0) -= 2.0 * kWRamp * (1.0 - t);
              H(c0, ck) -= 2.0 * kWRamp * (1.0 - t);
              H(c0, c1) += 2.0 * kWRamp * (1.0 - t) * t;
              H(c1, c0) += 2.0 * kWRamp * (1.0 - t) * t;
            }
          }
        }
      }
      H.diagonal().array() += 1e-4;

      // Hard subproblem rows: joint box, pitch cap, TCP stage targets.
      for (int k = 1; k < K; ++k) {
        const Vec18& q = (*knots)[static_cast<std::size_t>(k)];
        for (int i = 0; i < na; ++i) {
          const int j = active_[static_cast<std::size_t>(i)];
          lb[(k - 1) * na + i] = policy_.q_min[j] - q[j];
          ub[(k - 1) * na + i] = policy_.q_max[j] - q[j];
        }
      }
      C.setZero();
      // Recovery envelope: a start already outside the cap may keep |start|,
      // matching the validator. Never wider than that, never growing.
      const double pitch_row_cap = pitch_envelope();
      int row = 0;
      for (int k = 1; k < K; ++k) {
        const Vec18& q = (*knots)[static_cast<std::size_t>(k)];
        if (waist_rows) {
          const double pitch = q[0] + q[1] + q[2];
          for (int i = 0; i < na; ++i) {
            if (active_[static_cast<std::size_t>(i)] < 3) {
              C(row, (k - 1) * na + i) = 1.0;
            }
          }
          l[row] = -pitch_row_cap - pitch;
          u[row] = pitch_row_cap - pitch;
          ++row;
        }
        const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
        if (tgt.stage_endpoint) {
          const auto& pose = poses[static_cast<std::size_t>(k)];
          const Vec3 p = pose.col(3).head<3>();
          const Mat6x18& Js = jacs[static_cast<std::size_t>(k)];
          const double tol = std::min(0.8 * tgt.tol, kStreamEndRowM);
          for (int r = 0; r < 3; ++r) {
            for (int i = 0; i < na; ++i) {
              C(row + r, (k - 1) * na + i) =
                  Js(r, active_[static_cast<std::size_t>(i)]);
            }
            l[row + r] = tgt.p[r] - tol - p[r];
            u[row + r] = tgt.p[r] + tol - p[r];
          }
          row += 3;
          if (tgt.ori) {
            // Orientation hard row, per-axis box inscribed in the validator
            // norm tolerance (mirrors the stream's inscribed boxes).
            const Vec3 eo = ori_error_xyzw(
                tgt.quat, rot_to_quat_xyzw(pose.block<3, 3>(0, 0)));
            const double ot = 0.8 * tgt.ori_tol / 1.7320508075688772;
            for (int r = 0; r < 3; ++r) {
              for (int i = 0; i < na; ++i) {
                C(row + r, (k - 1) * na + i) =
                    Js(3 + r, active_[static_cast<std::size_t>(i)]);
              }
              l[row + r] = eo[r] - ot;
              u[row + r] = eo[r] + ot;
            }
            row += 3;
          }
        }
      }

      dq.setZero();
      bool solved = false;
      try {
        g_qp.noalias() = -g;
        using proxsuite::nullopt;
        if (!qp_ready) {
          qp.init(H, g_qp, nullopt, nullopt, C, l, u, lb, ub, true);
          qp_ready = true;
        } else {
          qp.update(H, g_qp, nullopt, nullopt, C, l, u, lb, ub, true);
        }
        qp.solve();
        const auto st = qp.results.info.status;
        if (st == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED ||
            st == proxsuite::proxqp::QPSolverOutput::
                       PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE) {
          dq = qp.results.x;
          solved = dq.allFinite();
        }
      } catch (...) {
        solved = false;
      }
      if (!solved) {
        // Dense LDLT fallback on the same normal equations; the box/pitch
        // clamps below re-apply the hard rows after the fact.
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
        if (ldlt.info() != Eigen::Success) {
          return;
        }
        dq = ldlt.solve(g);
        if (!dq.allFinite()) {
          return;
        }
      }
      if (dq.cwiseAbs().maxCoeff() < 1e-4) {
        return;
      }
      if (!use_cost) {
        // Trust region for the undamped task-only normal equations: without
        // the cost Hessian the raw step can be enormous near joint limits.
        const double peak = dq.cwiseAbs().maxCoeff();
        if (peak > 0.05) {
          dq *= 0.05 / peak;
        }
        // Feasibility metric: worst per-knot line error relative to the task
        // tolerance, and the endpoint arrival box relative to the stream row.
        double tol_max_f = 0.0;
        for (const StageGeom& gs : stages_) {
          tol_max_f = std::max(tol_max_f, gs.tol);
        }
        const auto feas_metric = [&](const JointCurve& ks) {
          double w = 0.0;
          double ea = 0.0;
          double eo_max = 0.0;
          double eo_tol = 1.0;
          for (std::size_t k = 1; k < ks.size(); ++k) {
            const auto pose = kin_.fk(ks[k], frame_);
            const Vec3 ev = targets[k].p - pose.col(3).head<3>();
            w = std::max(w, ev.norm());
            if (k + 1 < ks.size()) {
              const Vec18 qmid = 0.5 * (ks[k] + ks[k + 1]);
              const Vec3 pmid = 0.5 * (targets[k].p + targets[k + 1].p);
              w = std::max(w,
                           (pmid - kin_.fk_pos(qmid, frame_)).norm());
            }
            if (targets[k].stage_endpoint) {
              ea = std::max(ea, ev.cwiseAbs().maxCoeff());
              if (targets[k].ori) {
                eo_max = std::max(
                    eo_max,
                    ori_error_xyzw(
                        targets[k].quat,
                        rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                        .norm());
                eo_tol = std::max(1e-9, targets[k].ori_tol);
              }
            }
          }
          return std::max(std::max(w / std::max(tol_max_f, 1e-9),
                                   ea / kStreamEndRowM),
                          eo_max / eo_tol);
        };
        const double m_cur = feas_metric(*knots);
        if (trace_gn()) {
          const auto pose =
              kin_.fk((*knots).back(), frame_);
          const KnotTarget& te = targets.back();
          std::fprintf(
              stderr,
              "v2gn feas iter m=%.5g end_axis=%.5g end_ori=%.5g\n", m_cur,
              (te.p - pose.col(3).head<3>()).cwiseAbs().maxCoeff(),
              te.ori ? ori_error_xyzw(
                           te.quat,
                           rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                           .norm()
                     : 0.0);
        }
        if (m_cur <= 0.95) {
          return;  // inside both boxes with margin: phase done
        }
        bool improved = false;
        for (double alpha : {1.0, 0.5, 0.25, 0.1, 0.05}) {
          JointCurve trial = *knots;
          apply_delta(&trial, dq, alpha);
          const double mt = feas_metric(trial);
          if (trace_gn()) {
            std::fprintf(stderr, "v2gn feas alpha=%.2g m=%.5g (cur %.5g)\n",
                         alpha, mt, m_cur);
          }
          if (mt < m_cur - 1e-4) {
            *knots = std::move(trial);
            improved = true;
            break;
          }
        }
        if (!improved) {
          return;
        }
        continue;
      }
      double worst_base = 0.0;
      const double base = objective(*knots, targets, &worst_base);
      double tol_max = 0.0;
      for (const StageGeom& gs : stages_) {
        tol_max = std::max(tol_max, gs.tol);
      }
      bool improved = false;
      double alpha_used = 0.0;
      for (double alpha : {1.0, 0.5, 0.25, 0.1, 0.05}) {
        JointCurve trial = *knots;
        apply_delta(&trial, dq, alpha);
        double worst_trial = 0.0;
        const double cost = objective(trial, targets, &worst_trial);
        if (cost < base - 1e-10 &&
            worst_trial <= std::max(worst_base + 1e-9, 0.98 * tol_max)) {
          *knots = std::move(trial);
          improved = true;
          alpha_used = alpha;
          break;
        }
      }
      if (!improved) {
        // Feasibility restoration: while a stage endpoint sits outside the
        // stream arrival box, accept a step that shrinks the violation even
        // when the merged objective rises (the soft cost pulls against the
        // box; the QP direction already encodes the constraint). Standard
        // SQP restore phase — the solve, not a post-pass, brings the
        // endpoint inside.
        const double viol_cur = end_violation(*knots, targets);
        if (viol_cur <= 0.0) {
          return;
        }
        double best_vt = viol_cur;
        JointCurve best_trial;
        double best_alpha = 0.0;
        for (double alpha : {1.0, 0.5, 0.25, 0.1, 0.05}) {
          JointCurve trial = *knots;
          apply_delta(&trial, dq, alpha);
          double worst_trial = 0.0;
          objective(trial, targets, &worst_trial);
          const double vt = end_violation(trial, targets);
          if (trace_gn()) {
            std::fprintf(stderr,
                         "v2gn restore alpha=%.2g vt=%.5g worst=%.5g\n", alpha,
                         vt, worst_trial);
          }
          if (vt < best_vt &&
              worst_trial <= std::max(worst_base + 1e-9, 0.98 * tol_max)) {
            best_vt = vt;
            best_trial = std::move(trial);
            best_alpha = alpha;
          }
        }
        if (best_alpha > 0.0 && best_vt <= 0.85 * viol_cur) {
          *knots = std::move(best_trial);
          improved = true;
          alpha_used = best_alpha;
        }
        if (!improved) {
          return;
        }
      }
      if (trace_gn()) {
        double end_axis = 0.0;
        for (int kk = 1; kk < K; ++kk) {
          if (targets[static_cast<std::size_t>(kk)].stage_endpoint) {
            end_axis = std::max(
                end_axis,
                (targets[static_cast<std::size_t>(kk)].p -
                 kin_.fk_pos((*knots)[static_cast<std::size_t>(kk)], frame_))
                    .cwiseAbs()
                    .maxCoeff());
          }
        }
        std::fprintf(stderr,
                     "v2gn iter=%d solved=%d dq=%.4g alpha=%.2g base=%.6g "
                     "worst=%.4g end_axis=%.5g\n",
                     iter, solved ? 1 : 0, dq.cwiseAbs().maxCoeff(), alpha_used,
                     base, worst_base, end_axis);
      }
      // After becoming hard-feasible mid-phase, only a few more quality
      // iters — do not keep polishing soft cost for the full budget.
      if (use_cost && max_iters > kMaxGnItersAfterCert) {
        std::string why;
        if (accept_knots(*knots, targets, &why)) {
          if (post_cert_left < 0) {
            post_cert_left = kMaxGnItersAfterCert;
          } else if (--post_cert_left <= 0) {
            return;
          }
        }
      }
    }
  }

  static bool trace_gn() {
    static const bool on = std::getenv("KMPC_V2_TRACE") != nullptr;
    return on;
  }

  // Worst stage-endpoint per-axis excess over the stream arrival row box.
  // Zero means every endpoint satisfies the stream QP arrival tube.
  double end_violation(const JointCurve& knots,
                       const std::vector<KnotTarget>& targets) const {
    double viol = 0.0;
    for (std::size_t k = 1; k < knots.size(); ++k) {
      if (!targets[k].stage_endpoint) {
        continue;
      }
      const double axis =
          (targets[k].p - kin_.fk_pos(knots[k], frame_))
              .cwiseAbs()
              .maxCoeff();
      viol = std::max(viol, axis - kStreamEndRowM);
    }
    return std::max(0.0, viol);
  }

  void add_task_block(Eigen::MatrixXd* H, Eigen::VectorXd* g, int col,
                      const Mat6x18& J, const Vec3& ep, const Vec3* eo,
                      double w_mult = 1.0) const {
    const int na = static_cast<int>(active_.size());
    const double wp = kWPos * w_mult;
    for (int i = 0; i < na; ++i) {
      const int ai = active_[static_cast<std::size_t>(i)];
      double gi = 0.0;
      for (int r = 0; r < 3; ++r) {
        gi += wp * J(r, ai) * ep[r];
      }
      if (eo != nullptr) {
        for (int r = 0; r < 3; ++r) {
          gi += kWOri * w_mult * J(3 + r, ai) * (*eo)[r];
        }
      }
      (*g)[col + i] += gi;
      for (int j = i; j < na; ++j) {
        const int aj = active_[static_cast<std::size_t>(j)];
        double hij = 0.0;
        for (int r = 0; r < 3; ++r) {
          hij += wp * J(r, ai) * J(r, aj);
        }
        if (eo != nullptr) {
          for (int r = 0; r < 3; ++r) {
            hij += kWOri * w_mult * J(3 + r, ai) * J(3 + r, aj);
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
    const int na = static_cast<int>(active_.size());
    Eigen::MatrixXd Jl(3, na);
    for (int i = 0; i < na; ++i) {
      Jl.col(i) = 0.5 * J.col(active_[static_cast<std::size_t>(i)]).head<3>();
    }
    const int cols[2] = {col_l, col_r};
    for (int a = 0; a < 2; ++a) {
      const Eigen::VectorXd ga = w * Jl.transpose() * em;
      g->segment(cols[a], na) += ga;
      for (int b = 0; b < 2; ++b) {
        const Eigen::MatrixXd Hab = w * Jl.transpose() * Jl;
        H->block(cols[a], cols[b], na, na) += Hab;
      }
    }
  }

  void apply_delta(JointCurve* knots, const Eigen::VectorXd& dq,
                   double alpha) const {
    const int na = static_cast<int>(active_.size());
    const int K = static_cast<int>(knots->size());
    for (int k = 1; k < K; ++k) {
      Vec18& q = (*knots)[static_cast<std::size_t>(k)];
      for (int i = 0; i < na; ++i) {
        q[active_[static_cast<std::size_t>(i)]] += alpha * dq[(k - 1) * na + i];
      }
      clamp_hard(&q);
      pull_pitch(&q);
    }
  }

  bool densify(const JointCurve& knots, JointCurve* samples,
               std::vector<StageBoundary>* boundaries, std::string* why) const {
    // Joint-linear interpolation only. The solved knots are the plan; these
    // samples exist so validation sees what execution would interpolate.
    // Adaptive densify budget: short spans keep ~5 mm; longer knot spans use
    // up to 8 mm (never widen the task TCP tube / skip CoM / skip admit).
    const int K = static_cast<int>(knots.size());
    samples->clear();
    boundaries->clear();
    if (K < 2) {
      *why = "unified_ik";
      return false;
    }
    samples->reserve(std::min(kMaxDenseSamples, K * 8));
    boundaries->reserve(stages_.size());
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> knot_tcp(
        static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
      knot_tcp[static_cast<std::size_t>(k)] =
          kin_.fk_pos(knots[static_cast<std::size_t>(k)], frame_);
    }
    samples->push_back(knots.front());
    for (int k = 0; k + 1 < K; ++k) {
      const Vec18& qa = knots[static_cast<std::size_t>(k)];
      const Vec18& qb = knots[static_cast<std::size_t>(k + 1)];
      const double tcp_dist =
          (knot_tcp[static_cast<std::size_t>(k + 1)] -
           knot_tcp[static_cast<std::size_t>(k)])
              .norm();
      const double joint_step = (qb - qa).cwiseAbs().maxCoeff();
      const double sample_m =
          tcp_dist <= 2.0 * kDenseSampleM
              ? kDenseSampleM
              : std::clamp(tcp_dist / 3.0, kDenseSampleM, 0.008);
      const int n = std::max(
          {1, static_cast<int>(std::ceil(tcp_dist / sample_m)),
           static_cast<int>(std::ceil(joint_step / kDenseJointStepRad))});
      for (int i = 1; i <= n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        samples->push_back((1.0 - t) * qa + t * qb);
      }
      for (std::size_t s = 0; s + 1 < knot_of_stage_.size(); ++s) {
        if (k + 1 == knot_of_stage_[s + 1]) {
          StageBoundary b;
          b.sample_index = samples->size() - 1;
          b.stage_id = stages_[s].stage_id;
          b.stop_required = stages_[s].stop_required;
          b.wait_for_external_ack = stages_[s].wait_ack;
          boundaries->push_back(b);
        }
      }
      if (samples->size() > static_cast<std::size_t>(kMaxDenseSamples)) {
        *why = "unified_sample_limit";
        return false;
      }
    }
    project_waist_monotone(samples, *boundaries);
    return samples->size() >= 2;
  }

  // Per-stage monotone projection on waist J1/J2/J3 (and the pitch sum via
  // the same-axis pass): the solved boundary knots stay authoritative, only
  // interior samples move. The GN solve leaves micro dents (~1e-5 rad) on
  // top of an otherwise monotone waist profile; the committed-sign stream
  // envelope is tighter than those dents and would refuse the line. This is
  // NOT a re-posing pass: endpoints are pinned, adjustments are bounded by
  // the dent depth, and validation re-certifies the projected samples (TCP
  // cross-track, stage targets, guards) before anything ships.
  void project_waist_monotone(JointCurve* samples,
                              const std::vector<StageBoundary>& boundaries)
      const {
    if (task_.waist_locked || samples->size() < 3) {
      return;
    }
    std::size_t lo = 0;
    std::size_t hi = samples->size() - 1;
    std::size_t next_b = 0;
    const std::size_t nseg =
        boundaries.empty() ? 1 : boundaries.size() + 1;
    for (std::size_t seg = 0; seg < nseg; ++seg) {
      if (next_b < boundaries.size()) {
        hi = boundaries[next_b].sample_index;
        ++next_b;
      } else {
        hi = samples->size() - 1;
      }
      if (hi > lo + 1 && hi < samples->size()) {
        for (int j = 0; j < 3; ++j) {
          monotone_axis_pass(samples, lo, hi, j);
        }
      }
      lo = hi;
    }
  }

  static void monotone_axis_pass(JointCurve* samples, std::size_t lo,
                                 std::size_t hi, int j) {
    const double v0 = (*samples)[lo][j];
    const double v1 = (*samples)[hi][j];
    const double cap_lo = std::min(v0, v1);
    const double cap_hi = std::max(v0, v1);
    double run = v0;
    for (std::size_t i = lo + 1; i < hi; ++i) {
      double v = std::clamp((*samples)[i][j], cap_lo, cap_hi);
      run = v1 >= v0 ? std::max(run, v) : std::min(run, v);
      (*samples)[i][j] = run;
    }
  }

  void fill_body_metrics(FastPlanResult* out) const {
    if (samples_.size() < 2) {
      return;
    }
    MotionEvaluator eval(kin_);
    MotionSnapshot first;
    MotionSnapshot last;
    eval.evaluate(samples_.front(), left_, &first);
    eval.evaluate(samples_.back(), left_, &last);
    out->body_dx_m = last.body.x() - first.body.x();
    out->body_dz_m = last.body.z() - first.body.z();
    out->body_minus_leg_dx_m =
        last.body_minus_leg_x - first.body_minus_leg_x;
  }

  void finish_report(FastPlanResult& out) {
    out.deadline_hit = deadline_hit_;
    const PathCostBreakdown cost = evaluate_path_cost(
        samples_, q_neutral_, policy_, weights_, kin_, left_, guards_);
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "planner=unified_multistage_v2"
       << " path_id=" << std::hex << out.package.path_id << std::dec
       << " stages=" << stages_.size()
       << " opt=" << (out.ok ? "ok" : "fail")
       << " accept=" << (out.ok ? "pass" : "fail")
       << " tcp_mm=" << (out.tcp_worst_m * 1000.0)
       << " margin_deg=" << (out.min_hard_margin_rad * 180.0 / kPi)
       << " body_dx_mm=" << (out.body_dx_m * 1000.0)
       << " body_dz_mm=" << (out.body_dz_m * 1000.0)
       << " body_leg_dx_mm=" << (out.body_minus_leg_dx_m * 1000.0)
       << " iters=" << out.iterations
       << " opt_ms=" << out.optimize_ms
       << " cert_ms=" << out.certify_ms
       << " cost=" << cost.total
       << " height_switch=0"
       << " dur_s=" << out.duration_s
       << " sw_limited=" << (out.software_limited ? 1 : 0)
       << " deadline=" << (out.deadline_hit ? 1 : 0);
    out.report = os.str();
    if (out.reason.empty()) {
      out.reason = out.ok ? "unified_multistage_v2" : "unified_ik";
    }
  }

  void clamp_hard(Vec18* q) const {
    for (int j = 0; j < kNDof; ++j) {
      (*q)[j] = std::clamp((*q)[j], policy_.q_min[j], policy_.q_max[j]);
    }
  }

  // Same recovery envelope as the QP pitch row / validator: a start already
  // outside ±pitch_cap may keep |start|, never grow. Forcing ±cap on every
  // seed trial would yank waist (and TCP) when J1 is near the hard wall and
  // start pitch already exceeds the ordinary task limit.
  double pitch_envelope() const {
    return std::max(pitch_cap_, start_pitch_abs_);
  }

  void pull_pitch(Vec18* q) const {
    if (!(policy_.torso_pitch_max_rad > 0.0)) {
      return;
    }
    const double cap = pitch_envelope();
    for (int n = 0; n < 6; ++n) {
      const double pitch = (*q)[0] + (*q)[1] + (*q)[2];
      const double excess = std::abs(pitch) - cap;
      if (excess <= 1e-9) {
        return;
      }
      const double sign = pitch > 0.0 ? 1.0 : -1.0;
      int best = 0;
      double room_best = -1.0;
      for (int a = 0; a < 3; ++a) {
        const double room = sign > 0.0
                                ? ((*q)[a] - policy_.q_min[a])
                                : (policy_.q_max[a] - (*q)[a]);
        if (room > room_best) {
          room_best = room;
          best = a;
        }
      }
      const double step = std::min(excess, std::max(0.0, room_best));
      (*q)[best] -= sign * step;
      (*q)[best] = std::clamp((*q)[best], policy_.q_min[best],
                              policy_.q_max[best]);
    }
  }

  const Kinematics& kin_;
  const WholeBodyGuards* guards_;
  MotionTask task_;
  MotionConstraintPolicy policy_;
  TrajectoryCostWeights weights_;
  double deadline_ms_ = 0.0;
  std::chrono::steady_clock::time_point t0_;
  Vec18 q0_ = Vec18::Zero();
  Vec18 start_dq_ = Vec18::Zero();
  bool start_dq_known_ = false;
  Vec18 q_neutral_ = Vec18::Zero();
  bool left_ = true;
  std::string frame_;
  int arm_ = 4;
  std::vector<int> active_;
  std::vector<StageGeom, Eigen::aligned_allocator<StageGeom>> stages_;
  std::vector<int> knot_of_stage_;
  double pitch_cap_ = 1e9;
  double start_pitch_abs_ = 0.0;
  int iterations_ = 0;
  bool deadline_hit_ = false;
  JointCurve samples_;
};

}  // namespace

FastPlanResult FastTrajectoryOptimizer::plan(
    const MotionTask& task, const RobotState& state,
    const MotionConstraintPolicy& policy, const TrajectoryCostWeights& weights,
    double deadline_ms) const {
  return V2Solver(kin_, guards_, task, state, policy, weights, deadline_ms)
      .run();
}

}  // namespace kmpc
