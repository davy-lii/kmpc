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
#include "kmpc/trajectory_validator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
// Production budget is 1 s (not 10 ms): do NOT thin multi-stage knot spacing
// for P99 latency; success/certify first, then tune wall time from logs.
struct FoldRulerProbe {
  double j4_rad = 0.0;
  double j6_rad = 0.0;
  double j7_rad = 0.0;
};

FoldRulerProbe measure_fold_ruler(const JointCurve& curve, bool left) {
  FoldRulerProbe out;
  if (curve.size() < 2) {
    return out;
  }
  const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
  out.j4_rad = joint_path_foldback_rad(curve, kIdxYaw);
  out.j6_rad = joint_path_foldback_rad(curve, arm + 5);
  out.j7_rad = joint_path_foldback_rad(curve, arm + 6);
  return out;
}

void append_fold_ruler_probe(std::ostringstream& os, const char* tag,
                             const FoldRulerProbe& p) {
  os << std::fixed << std::setprecision(3) << " " << tag << "_j4f_deg="
     << (p.j4_rad * 180.0 / kPi) << " " << tag << "_j6f_deg="
     << (p.j6_rad * 180.0 / kPi) << " " << tag << "_j7f_deg="
     << (p.j7_rad * 180.0 / kPi);
}

constexpr int kMaxKnots = 32;
// Default layout budget; retry may widen via layout_dense_retry_.
constexpr int kLayoutBudgetKnots = 32;
constexpr double kJ4StageMonoEpsRad = 5e-4;
// Coarse mid-tube admit only during layout/refine; not a time_law pass gate.
constexpr double kKnotSpacingM = 0.012;
// After a free PTP PREP, the LIN approach needs denser knots so joint-span
// midpoints stay inside the 3 mm tube under orientation.
constexpr double kLinAfterPtpSpacingM = 0.008;
constexpr double kShortPathM = 0.06;
constexpr double kDenseSampleM = 0.005;
// Per-sample joint step stays well under the stream QP trust region (the
// tracker accepts roughly 4 x per-cycle dq cap, ~0.023 rad on arm joints).
constexpr double kDenseJointStepRad = 0.02;
// Hard densify memory/compute budget. Demand is computed up-front from TCP +
// joint spacing; never coarsen mid-span to "fit" a soft cap. Beyond this
// ceiling report resource_sample_limit (not a fake geometry fault).
constexpr int kMaxDenseSamplesHard = 3000;
// Same reach envelope as the retired planner (legacy 400 x 5 mm = 2 m).
constexpr double kReachLimitM = 2.0;
constexpr double kWPos = 400.0;
constexpr double kWOri = 40.0;
constexpr int kMaxGnIters = 14;
// After the knots already satisfy hard geometry (accept_knots), only a few
// quality iters remain — do not polish soft cost forever (§9.3).
constexpr int kMaxGnItersAfterCert = 8;
constexpr int kMaxGnFeasIters = 8;
constexpr int kMaxEndpointPolishIters = 6;
// V2 production settles on task euclidean tol via TrajectoryFollower, not the
// retired line_qp per-axis arrival box. Inscribe a per-axis soft/hard row in
// the task sphere so QP endpoint rows stay consistent with the validator.
inline double end_axis_tol_m(double task_tol_m) {
  return task_tol_m / 1.7320508075688772;
}
inline double end_row_tol_m(double task_tol_m) {
  return 0.9 * end_axis_tol_m(task_tol_m);
}

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
  double tol = kDefaultTaskPositionTolM;
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
  double tol = kDefaultTaskPositionTolM;
  double ori_tol = 0.05;
  // Owning stage for this knot (endpoint or LIN waypoint). Span k→k+1 uses
  // span_stage(k), not only the left knot's geometry.
  int stage_index = 0;
  TaskGeometry geometry = TaskGeometry::Lin;
};

class V2Solver {
 public:
  enum class SeedKind {
    RemainingPath = 0,
    Neutral = 1,
    ArmHeavy = 2,
    ZSquatMild = 3,
    ZSquatStrong = 4,
    KeepYaw = 5,
    // Production sole posture family: task-adaptive waist–arm coordination.
    Coordinated = 6
  };

  struct KnotCandidate {
    JointCurve knots;
    std::vector<int> knot_of_stage;  // may grow if LIN mid-insert refined
    SeedKind kind = SeedKind::RemainingPath;
    PathQualityMetrics quality;
    PathCostBreakdown soft{};
    double seed_ms = 0.0;
    double sqp_ms = 0.0;
    double densify_ms = 0.0;
    double time_ms = 0.0;
    double cert_ms = 0.0;
    int gn_iters = 0;
    bool knots_ok = false;
    bool densify_ok = false;
    bool time_ok = false;
    bool cert_ok = false;
    bool selected = false;
    std::string fail_stage;
    std::string fail_reason;
    // joint_path_foldback_rad along coarse/dense curves (plan regression probes).
    FoldRulerProbe fold_gn{};
    FoldRulerProbe fold_ep{};
    FoldRulerProbe fold_pre_den{};
    FoldRulerProbe fold_den_span{};
    FoldRulerProbe fold_den{};
  };

  static const char* seed_name(SeedKind kind) {
    switch (kind) {
      case SeedKind::RemainingPath:
        return "RemainingPath";
      case SeedKind::Neutral:
        return "Neutral";
      case SeedKind::ArmHeavy:
        return "ArmHeavy";
      case SeedKind::ZSquatMild:
        return "ZSquatMild";
      case SeedKind::ZSquatStrong:
        return "ZSquatStrong";
      case SeedKind::KeepYaw:
        return "KeepYaw";
      case SeedKind::Coordinated:
        return "Coordinated";
    }
    return "Unknown";
  }

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
    q0_body_x_ = kin_.fk_pos(q0_, "Body_Link").x();
    joint_effective_work_limits(q0_, policy_, work_lo_, work_hi_);
    joint_effective_recovery_limits(q0_, policy_, rec_lo_, rec_hi_);
  }

  void try_pin_j4_hold() {
    // J4 always stays in active_; "少动腰" is trajectory_cost j4_twist only.
    j4_pinned_ = false;
    j4_hold_probe_ = policy_.yaw_limited ? "soft_yaw" : "unlimited";
  }

  bool knot_target_met(const Vec18& q, const KnotTarget& tgt) const {
    if (!q.allFinite()) {
      return false;
    }
    if (joint_violates_effective_work_box(q, q0_, policy_) ||
        joint_violates_yaw_task_box(q, policy_)) {
      return false;
    }
    const Vec3 pe = kin_.fk_pos(q, frame_);
    if (tgt.geometry == TaskGeometry::Lin && !tgt.stage_endpoint) {
      // LIN mid-chord: soft runway only (see kDefaultLinRunwaySoftM).
    } else if ((tgt.p - pe).norm() > tgt.tol) {
      return false;
    }
    if (tgt.ori) {
      const auto pose = kin_.fk(q, frame_);
      const double oe = ori_error_xyzw(
                            tgt.quat,
                            rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                            .norm();
      if (oe > tgt.ori_tol) {
        return false;
      }
    }
    if (guards_ != nullptr && !guards_->com_xy_inside(kin_.com(q))) {
      return false;
    }
    const double pitch = q[0] + q[1] + q[2];
    if (std::abs(pitch) > pitch_envelope() + 1e-6) {
      return false;
    }
    return true;
  }

  bool knot_densify_sample_ok(const Vec18& q, const KnotTarget& tgt) const {
    if (!q.allFinite()) {
      return false;
    }
    if (joint_violates_effective_work_box(q, q0_, policy_) ||
        joint_violates_yaw_task_box(q, policy_)) {
      return false;
    }
    if (j4_pinned_ &&
        std::abs(q[kIdxYaw] - j4_pin_q_) > kJ4StageMonoEpsRad) {
      return false;
    }
    const Vec3 pe = kin_.fk_pos(q, frame_);
    if (tgt.geometry == TaskGeometry::Lin && !tgt.stage_endpoint) {
      return cross_track_to_stage(pe, tgt.stage_index) <= kCoarseLinMidAdmitM;
    }
    return knot_target_met(q, tgt);
  }

  void release_j4_for_retry() {
    if (!j4_pinned_) {
      return;
    }
    j4_pinned_ = false;
    j4_release_reason_ = "hold_geometry_retry";
    if (std::find(active_.begin(), active_.end(), kIdxYaw) == active_.end()) {
      auto it = std::find_if(active_.begin(), active_.end(),
                             [](int j) { return j > kIdxYaw; });
      active_.insert(it, kIdxYaw);
    }
  }

  void pin_j4(Vec18* q) const {
    if (j4_pinned_) {
      (*q)[kIdxYaw] = j4_pin_q_;
    }
  }

  // GN j4_twist may be 0; densify LIN IK still must not spin J4 away from the
  // joint-lerp sample (ik_anchor is a frozen copy — never pass q as both q and anchor).
  double densify_j4_yaw_anchor_w() const {
    if (j4_pinned_) {
      return -1.0;
    }
    if (weights_.j4_twist > 1e-9) {
      return weights_.j4_twist;
    }
    return -1.0;
  }

  void clamp_densify_j4_to_lerp(Vec18* q, const Vec18& ik_anchor,
                                const KnotTarget& nt) const {
    if (j4_pinned_) {
      pin_j4(q);
      return;
    }
    Vec18 trial = *q;
    trial[kIdxYaw] = ik_anchor[kIdxYaw];
    clamp_hard(&trial);
    pull_pitch(&trial);
    if (knot_densify_sample_ok(trial, nt)) {
      *q = trial;
      return;
    }
    if (knot_densify_sample_ok(ik_anchor, nt)) {
      *q = ik_anchor;
      clamp_hard(q);
      pull_pitch(q);
    }
  }

  bool boost_lin_knots_hold(JointCurve* knots) {
    if (!j4_pinned_ || knots->size() < 2 ||
        static_cast<int>(knots->size()) >= kMaxKnots ||
        lin_boost_count_ >= kMaxLinBoostHold) {
      return false;
    }
    int worst_k = -1;
    double worst_m = 0.0;
    const int K = static_cast<int>(knots->size());
    for (int k = 0; k + 1 < K; ++k) {
      if (!tracks_cartesian_span(k)) {
        continue;
      }
      const int sid = span_stage(k);
      const Vec18 qmid =
          0.5 * ((*knots)[static_cast<std::size_t>(k)] +
                 (*knots)[static_cast<std::size_t>(k + 1)]);
      const double merr =
          cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid);
      if (merr > worst_m) {
        worst_m = merr;
        worst_k = k;
      }
    }
    if (worst_k < 0 ||
        worst_m <= stages_[static_cast<std::size_t>(span_stage(worst_k))].tol) {
      return false;
    }
    const int sid = span_stage(worst_k);
    const StageGeom& sg = stages_[static_cast<std::size_t>(sid)];
    const Vec18 qmid =
        0.5 * ((*knots)[static_cast<std::size_t>(worst_k)] +
               (*knots)[static_cast<std::size_t>(worst_k + 1)]);
    const Vec3 pm = kin_.fk_pos(qmid, frame_);
    const Vec3 ab = sg.b - sg.a;
    const double ab2 = ab.squaredNorm();
    double t = 0.5;
    Vec3 closest = sg.b;
    if (ab2 >= 1e-16) {
      t = std::clamp((pm - sg.a).dot(ab) / ab2, 0.0, 1.0);
      closest = sg.a + t * ab;
    }
    KnotTarget nt;
    nt.p = closest;
    nt.quat = quat_slerp_xyzw(sg.qa, sg.qb, t);
    nt.ori = false;
    nt.stage_endpoint = false;
    nt.tol = sg.tol;
    nt.ori_tol = sg.ori_tol;
    nt.stage_index = sid;
    nt.geometry = TaskGeometry::Lin;
    Vec18 qnew = qmid;
    const double ww = waist_near_hard_wall() ? 80.0 : 0.4;
    seed_ik(&qnew, nt, qmid, 40, ww, -1.0);
    pin_j4(&qnew);
    knots->insert(knots->begin() + worst_k + 1, qnew);
    for (std::size_t s = 0; s < knot_of_stage_.size(); ++s) {
      if (knot_of_stage_[s] > worst_k) {
        ++knot_of_stage_[s];
      }
    }
    ++lin_boost_count_;
    ++iterations_;
    return true;
  }

  void enforce_j4_stage_contract(JointCurve* knots) const {
    if (knots->empty() || !j4_pinned_) {
      return;
    }
    for (Vec18& q : *knots) {
      q[kIdxYaw] = j4_pin_q_;
    }
  }

  FastPlanResult run() {
    FastPlanResult out;
    std::string why;
    if (!preflight(&why)) {
      out.reason = why;
      out.total_ms = elapsed_ms();
      finish_report(out);
      return out;
    }
    std::vector<KnotTarget> targets;
    const double t_pre = elapsed_ms();
    FastPlanResult best_ok;
    std::string last_fail;
    std::string cand_digest;
    if (!solve_certify_serial(&targets, &best_ok, &last_fail, &cand_digest)) {
      out.reason = last_fail.empty() ? "unified_ik" : last_fail;
      out.iterations = iterations_;
      out.seed_ik_ms = seed_ik_ms_;
      out.sqp_ms = sqp_ms_;
      out.densify_ms = densify_ms_;
      out.time_law_ms = time_law_ms_;
      out.certify_ms = certify_ms_;
      out.optimize_ms = elapsed_ms() - t_pre;
      out.candidate_digest = cand_digest;
      out.candidates_tried = main_attempt_count_;
      out.main_attempt_count = main_attempt_count_;
      out.retry_count = retry_count_;
      out.retry_reason = retry_reason_;
      out.chosen_seed = chosen_seed_;
      out.total_ms = elapsed_ms();
      finish_report(out);
      if (!cand_digest.empty()) {
        out.report += " | ";
        out.report += cand_digest;
      }
      return out;
    }

    out = best_ok;
    out.densify_ms = densify_ms_;
    out.time_law_ms = time_law_ms_;
    out.certify_ms = certify_ms_;
    out.optimize_ms = elapsed_ms() - t_pre;
    out.seed_ik_ms = seed_ik_ms_;
    out.sqp_ms = sqp_ms_;
    out.iterations = iterations_;
    out.candidate_digest = cand_digest;
    out.candidates_tried = main_attempt_count_;
    out.main_attempt_count = main_attempt_count_;
    out.retry_count = retry_count_;
    out.retry_reason = retry_reason_;
    out.chosen_seed = chosen_seed_;
    samples_ = out.package.samples;
    out.total_ms = elapsed_ms();
    const double t_rep0 = elapsed_ms();
    finish_report(out);
    out.report_ms = elapsed_ms() - t_rep0;
    out.total_ms = elapsed_ms();
    {
      std::ostringstream tail;
      tail << std::fixed << std::setprecision(3)
           << " report_ms=" << out.report_ms
           << " total_ms=" << out.total_ms
           << " over_10ms="
           << ((out.total_ms > kPlannerPerfStatMs) ? 1 : 0)
           << " seed=" << chosen_seed_
           << " main_attempt=" << main_attempt_count_
           << " retry_count=" << retry_count_
           << " cands=" << main_attempt_count_
           << " qscore=" << chosen_quality_.score
           << " j4pk_deg="
           << (chosen_quality_.j4_peak_from_start_rad * 180.0 / kPi)
           << " j6fold_deg="
           << (chosen_quality_.wrist_j6_fold_rad * 180.0 / kPi)
           << " j7fold_deg="
           << (chosen_quality_.wrist_j7_fold_rad * 180.0 / kPi)
           << " severe_fold=" << (chosen_quality_.severe_foldback ? 1 : 0)
           << " body_dz_mm=" << (out.body_dz_m * 1000.0)
           << " pitch_mrg_deg="
           << (chosen_quality_.pitch_min_margin_rad * 180.0 / kPi)
           << " law_before_s=" << out.package.time_law.duration_before_s
           << " law_scale_iters=" << out.package.time_law.scale_iters
           << " law_src=" << out.package.time_law.limit_source;
      out.report += tail.str();
      if (!out.candidate_digest.empty()) {
        out.report += " | ";
        out.report += out.candidate_digest;
      }
    }
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
    if (yaw_start_outside_task_box(q0_, policy_)) {
      *why = "yaw_start_outside_task_box";
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
    task_tcp_dz_m_ =
        task_stages_tcp_dz_m(stages_.front().a.z(), task_.stages);
    try_pin_j4_hold();
    return true;
  }

  // Span k→k+1 belongs to the stage whose knot range contains k.
  int span_stage(int k) const {
    for (std::size_t s = 0; s + 1 < knot_of_stage_.size(); ++s) {
      if (k >= knot_of_stage_[s] && k < knot_of_stage_[s + 1]) {
        return static_cast<int>(s);
      }
    }
    return stages_.empty() ? 0 : static_cast<int>(stages_.size()) - 1;
  }

  // Stage endpoints always track Cartesian; LIN interiors do; PTP interiors do not.
  static bool tracks_cartesian_knot(const KnotTarget& tgt) {
    return tgt.stage_endpoint || tgt.geometry == TaskGeometry::Lin;
  }

  bool tracks_cartesian_span(int k) const {
    const int s = span_stage(k);
    if (s < 0 || s >= static_cast<int>(stages_.size())) {
      return false;
    }
    return stages_[static_cast<std::size_t>(s)].geo == TaskGeometry::Lin;
  }

  double cross_track_to_stage(const Vec3& p, int stage_idx) const {
    if (stage_idx < 0 || stage_idx >= static_cast<int>(stages_.size())) {
      return 0.0;
    }
    const StageGeom& g = stages_[static_cast<std::size_t>(stage_idx)];
    const Vec3 ab = g.b - g.a;
    const double ab2 = ab.squaredNorm();
    if (ab2 < 1e-16) {
      return (p - g.b).norm();
    }
    const double t = std::clamp((p - g.a).dot(ab) / ab2, 0.0, 1.0);
    return (p - (g.a + t * ab)).norm();
  }

  bool layout_knots(JointCurve* knots, std::vector<KnotTarget>* targets,
                    std::string* why) {
    // LIN: ~12 mm Cartesian spacing. PTP: few knots for hard-constraint
    // sampling (joint-span layout), not a dense Cartesian chord.
    double path_len = 0.0;
    for (const StageGeom& g : stages_) {
      path_len += (g.b - g.a).norm();
    }
    std::vector<int> spans;
    double spacing = kKnotSpacingM;
    if (path_len <= kShortPathM && stages_.size() == 1 &&
        stages_.front().geo == TaskGeometry::Lin) {
      spacing = std::max(kKnotSpacingM, path_len / 2.0);
    }
    constexpr double kPtpSpacingM = 0.20;
    int total = 0;
    for (int attempt = 0; attempt < 10; ++attempt) {
      spans.clear();
      total = 0;
      for (const StageGeom& g : stages_) {
        const double dist = (g.b - g.a).norm();
        int n = 1;
        if (g.geo == TaskGeometry::Ptp) {
          n = std::clamp(static_cast<int>(std::ceil(dist / kPtpSpacingM)), 1,
                         4);
        } else {
          double lin_sp = spacing;
          // stages_[s] not yet indexed here — use geo of previous stage via
          // spans.size() as the current stage index once pushed.
          const std::size_t si = spans.size();
          if (si > 0 && stages_[si - 1].geo == TaskGeometry::Ptp) {
            lin_sp = std::min(lin_sp, kLinAfterPtpSpacingM);
          }
          n = std::clamp(static_cast<int>(std::ceil(dist / lin_sp)), 1,
                         kMaxKnots - 1);
        }
        spans.push_back(n);
        total += n;
      }
      const int layout_cap =
          layout_dense_retry_ ? kMaxKnots : kLayoutBudgetKnots;
      if (total <= layout_cap - 1) {
        break;
      }
      spacing *= 1.5;
    }
    if (total > kMaxKnots - 1) {
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
        if (stages_[s].geo == TaskGeometry::Ptp) {
          n = std::min(n, 4);
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
    (*targets)[0].geometry = stages_[0].geo;
    (*targets)[0].stage_index = 0;
    for (std::size_t s = 0; s < stages_.size(); ++s) {
      const int k0 = knot_of_stage_[s];
      const int k1 = knot_of_stage_[s + 1];
      const int n = std::max(1, k1 - k0);
      for (int k = k0; k <= k1; ++k) {
        const double t = static_cast<double>(k - k0) / static_cast<double>(n);
        KnotTarget& tgt = (*targets)[static_cast<std::size_t>(k)];
        if (k == k0 && s > 0) {
          // Shared PREP: already filled as previous stage endpoint.
          continue;
        }
        tgt.p = (1.0 - t) * stages_[s].a + t * stages_[s].b;
        tgt.quat = quat_slerp_xyzw(stages_[s].qa, stages_[s].qb, t);
        // Lock flange orientation only at stage endpoints; mid LIN knots are
        // position-only so the arm can use wrist redundancy along the drop.
        tgt.ori = stages_[s].ori && (k == k1);
        tgt.stage_endpoint = (k == k1);
        tgt.tol = stages_[s].tol;
        tgt.ori_tol = stages_[s].ori_tol;
        tgt.stage_index = static_cast<int>(s);
        tgt.geometry = stages_[s].geo;
      }
    }
    // Task TCP descent intent (LIN segment drops, not only net q0→final).
    task_tcp_dz_m_ = task_stages_tcp_dz_m(stages_.front().a.z(), task_.stages);
    return true;
  }

  double coordinated_drop_frac_for_attempt(int attempt_idx) const {
    if (task_tcp_dz_m_ >= -0.02) {
      return 0.0;
    }
    const double base = (task_tcp_dz_m_ < -0.03) ? 0.45 : 0.30;
    if (attempt_idx <= 0) {
      return base;
    }
    return std::min(0.85, base + 0.12);
  }

  static bool plan_failure_retryable(const std::string& stage,
                                     const std::string& reason) {
    if (reason == "deadline") {
      return false;
    }
    (void)stage;
    (void)reason;
    return true;
  }

  // Single posture family: seed → GN/polish/refine → knots ready for densify.
  bool solve_coordinated_knots(int attempt_idx, KnotCandidate* cand,
                               std::vector<KnotTarget>* targets,
                               const JointCurve& layout,
                               const std::vector<KnotTarget>& layout_targets,
                               const std::vector<int>& layout_kos,
                               const JointCurve* warm_knots,
                               const std::vector<int>* warm_kos,
                               bool warm_continue_gn,
                               std::ostringstream& digest,
                               std::string* last_why) {
    *targets = layout_targets;
    knot_of_stage_ = layout_kos;
    cand->kind = SeedKind::Coordinated;
    coordinated_drop_frac_ = coordinated_drop_frac_for_attempt(attempt_idx);
    const char* tag = (attempt_idx == 0) ? "main" : "retry";

    JointCurve trial = layout;
    bool use_warm =
        warm_continue_gn && warm_knots != nullptr &&
        warm_knots->size() == layout.size();
    if (attempt_idx > 0 && retry_force_reseed_init()) {
      use_warm = false;
    }
    if (use_warm) {
      trial = *warm_knots;
      if (warm_kos != nullptr && !warm_kos->empty()) {
        knot_of_stage_ = *warm_kos;
      }
    }
    gn_trace_tag_ = tag;
    if (trace_gn() && attempt_idx > 0) {
      std::fprintf(stderr, "v2gn retry_init cand=%s mode=%s\n", tag,
                   use_warm ? "warm" : "reseed");
    }

    const int iters0 = iterations_;
    const double t_seed0 = elapsed_ms();
    if (!use_warm) {
      seed_curve(&trial, *targets, SeedKind::Coordinated);
    }
    if (attempt_idx > 0) {
      trace_fold_stages(tag, "retry_entry", trial);
    }
    enforce_j4_stage_contract(&trial);
    harmonize_oriented_wrists(&trial, *targets);
    double seed_j4_peak_rad = 0.0;
    for (const Vec18& qk : trial) {
      seed_j4_peak_rad =
          std::max(seed_j4_peak_rad, std::abs(qk[kIdxYaw] - q0_[kIdxYaw]));
    }
    std::string accept_why;
    const bool seed_certified =
        accept_knots(trial, *targets, &accept_why, 0.0);
    const bool seed_coarse_ok =
        accept_knots(trial, *targets, &accept_why, kCoarseLinMidAdmitM);
    cand->seed_ms = elapsed_ms() - t_seed0;
    seed_ik_ms_ += cand->seed_ms;

    const double t_sqp0 = elapsed_ms();
    if (attempt_idx == 0 && (seed_certified || seed_coarse_ok)) {
      // Endpoints are on the 3 mm sphere; LIN mids ≤10 mm are densify-repair
      // work. Do not run Phase-B feasibility that trades J4/wrist for a
      // coarse 3 mm mid the interpolator will fix.
      const JointCurve pre_harmonize = trial;
      harmonize_oriented_wrists(&trial, *targets);
      if (!accept_knots(trial, *targets, &accept_why, kCoarseLinMidAdmitM,
                        /*gate_lin_mid=*/false)) {
        trial = pre_harmonize;
      }
      const JointCurve feasible_seed = trial;
      const std::vector<KnotTarget> feasible_tgt = *targets;
      const std::vector<int> feasible_kos = knot_of_stage_;
      gauss_newton(&trial, *targets, /*already_certified=*/true);
      if (!accept_knots(trial, *targets, &accept_why, kCoarseLinMidAdmitM,
                        /*gate_lin_mid=*/false)) {
        trial = feasible_seed;
        *targets = feasible_tgt;
        knot_of_stage_ = feasible_kos;
      }
    } else {
      gauss_newton(&trial, *targets, /*already_certified=*/false);
      if (!accept_knots(trial, *targets, &accept_why, 0.0)) {
        polish_knots(&trial, *targets);
      }
      harmonize_oriented_wrists(&trial, *targets);
    }
    cand->fold_gn = measure_fold_ruler(trial, left_);
    trace_fold_stages(tag, "post_gn", trial);
    if (!accept_knots(trial, *targets, &accept_why, 0.0)) {
      refine_lin_mids(&trial, targets);
    }
    // Endpoints stay on the 3 mm task sphere; LIN mids may still be coarse
    // here and are repaired by densify (10 mm admit, not a skip of refine).
    if (!accept_knots(trial, *targets, &accept_why, kCoarseLinMidAdmitM,
                      /*gate_lin_mid=*/false)) {
      polish_knots(&trial, *targets);
    }
    polish_endpoints_only(&trial, *targets);
    cand->fold_ep = measure_fold_ruler(trial, left_);
    trace_fold_stages(tag, "post_geom", trial);
    cand->sqp_ms = elapsed_ms() - t_sqp0;
    sqp_ms_ += cand->sqp_ms;
    cand->gn_iters = iterations_ - iters0;

    if (!accept_knots(trial, *targets, &accept_why, kCoarseLinMidAdmitM,
                      /*gate_lin_mid=*/false)) {
      cand->fail_stage = "knots";
      cand->fail_reason = accept_why;
      *last_why = accept_why;
      cand->knots = std::move(trial);
      cand->knot_of_stage = knot_of_stage_;
      digest << " | " << tag << ":fail=knots:" << accept_why
             << " drop_frac=" << coordinated_drop_frac_
             << " seed_ms=" << cand->seed_ms << " sqp_ms=" << cand->sqp_ms
             << " gn=" << cand->gn_iters
             << " seed_j4pk_deg=" << (seed_j4_peak_rad * 180.0 / kPi);
      return false;
    }
    cand->knots_ok = true;
    cand->knots = std::move(trial);
    cand->knot_of_stage = knot_of_stage_;
    PathCostBreakdown soft = evaluate_path_cost(
        cand->knots, q_neutral_, policy_, weights_, kin_, left_, guards_,
        &cand->knot_of_stage);
    soft.task_tcp_dz_m = task_tcp_dz_m_;
    cand->soft = soft;
    cand->quality = evaluate_path_quality(
        cand->knots, q0_, left_, policy_, soft, pitch_envelope(),
        &cand->knot_of_stage);
    digest << " | " << tag << ":knots_ok"
           << " drop_frac=" << coordinated_drop_frac_
           << (attempt_idx > 0 ? (use_warm ? " retry_init=warm" : " retry_init=reseed")
                               : "")
           << " seed_ms=" << cand->seed_ms << " sqp_ms=" << cand->sqp_ms
           << " gn=" << cand->gn_iters << " qscore=" << cand->quality.score
           << " seed_j4pk_deg=" << (seed_j4_peak_rad * 180.0 / kPi)
           << " j4pk_deg="
           << (cand->quality.j4_peak_from_start_rad * 180.0 / kPi)
           << " severe=" << (cand->quality.severe_foldback ? 1 : 0);
    append_fold_ruler_probe(digest, "fold_gn", cand->fold_gn);
    append_fold_ruler_probe(digest, "fold_ep", cand->fold_ep);
    return true;
  }

  bool certify_coordinated_package(KnotCandidate* cand,
                                   std::vector<KnotTarget>* targets,
                                   FastPlanResult* best_ok,
                                   std::ostringstream& digest,
                                   std::string* last_why) {
    const char* tag =
        (retry_count_ > 0 && cand->kind == SeedKind::Coordinated) ? "retry"
                                                                  : "main";
    if (deadline_hit()) {
      *last_why = "deadline";
      digest << " | " << tag << ":deadline_pkg";
      return false;
    }
    knot_of_stage_ = cand->knot_of_stage;
    {
      const double ww = waist_near_hard_wall() ? 80.0 : 0.4;
      for (std::size_t ki = 1; ki + 1 < cand->knots.size(); ++ki) {
        const int sid = span_stage(static_cast<int>(ki));
        if (sid < 0 || sid >= static_cast<int>(stages_.size()) ||
            stages_[static_cast<std::size_t>(sid)].geo != TaskGeometry::Lin) {
          continue;
        }
        bool is_end = false;
        for (int ke : knot_of_stage_) {
          if (ke == static_cast<int>(ki)) {
            is_end = true;
            break;
          }
        }
        if (is_end) {
          continue;
        }
        const StageGeom& sg = stages_[static_cast<std::size_t>(sid)];
        Vec18& qk = cand->knots[ki];
        const Vec3 pe = kin_.fk_pos(qk, frame_);
        if (cross_track_to_stage(pe, sid) <= sg.tol) {
          continue;
        }
        const Vec3 ab = sg.b - sg.a;
        const double ab2 = ab.squaredNorm();
        KnotTarget nt;
        nt.p = sg.b;
        double tt = 0.5;
        if (ab2 >= 1e-16) {
          tt = std::clamp((pe - sg.a).dot(ab) / ab2, 0.0, 1.0);
          nt.p = sg.a + tt * ab;
        }
        nt.quat = quat_slerp_xyzw(sg.qa, sg.qb, tt);
        nt.ori = false;
        nt.tol = sg.tol;
        nt.ori_tol = sg.ori_tol;
        nt.geometry = TaskGeometry::Lin;
        nt.stage_index = sid;
        const Vec18 ik_anchor = qk;
        seed_ik(&qk, nt, ik_anchor, j4_pinned_ ? 32 : 20, ww,
                densify_j4_yaw_anchor_w());
        pin_j4(&qk);
      }
    }
    cand->fold_pre_den = measure_fold_ruler(cand->knots, left_);
    JointCurve samples;
    std::vector<StageBoundary> boundaries;
    TimeLawTable law;
    TrajectoryCertificate cert;
    FastPlanResult plan_trial;
    cand->densify_ms = 0.0;
    cand->time_ms = 0.0;
    cand->cert_ms = 0.0;
    const double t_den0 = elapsed_ms();
    std::string den_why;
    if (!densify(cand->knots, &samples, &boundaries, &den_why,
                 &cand->fold_den_span)) {
      cand->densify_ms = elapsed_ms() - t_den0;
      densify_ms_ += elapsed_ms() - t_den0;
      cand->fail_stage = "densify";
      cand->fail_reason = den_why;
      *last_why = den_why;
      digest << " | " << tag << ":fail=densify:" << den_why
             << " den_ms=" << cand->densify_ms;
      return false;
    }
    cand->densify_ok = true;
    cand->densify_ms = elapsed_ms() - t_den0;
    densify_ms_ += elapsed_ms() - t_den0;
    cand->fold_den = measure_fold_ruler(samples, left_);
    trace_fold_stages(tag, "post_densify", samples);

    const double t_law0 = elapsed_ms();
    const TimeParameterizer parameterizer;
    law = parameterizer.compute(samples, policy_, boundaries);
    cand->time_ms = elapsed_ms() - t_law0;
    time_law_ms_ += elapsed_ms() - t_law0;
    if (!law.ok) {
      cand->fail_stage = "time_law";
      cand->fail_reason = law.reason;
      *last_why = law.reason;
      digest << " | " << tag << ":fail=time:" << law.reason
             << " time_ms=" << cand->time_ms;
      return false;
    }
    cand->time_ok = true;

    const double t_cert0 = elapsed_ms();
    TrajectoryPackage pkg_probe;
    pkg_probe.generation = task_.generation;
    pkg_probe.start_q = q0_;
    pkg_probe.samples = samples;
    pkg_probe.time_law = law;
    pkg_probe.stage_boundaries = boundaries;
    pkg_probe.left = left_;
    cert = validate_actual_q_of_t(pkg_probe, task_, policy_, kin_, guards_);
    cand->cert_ms = elapsed_ms() - t_cert0;
    certify_ms_ += elapsed_ms() - t_cert0;
    if (!cert.passed) {
      cand->fail_stage = "certify";
      cand->fail_reason = !cert.fault_detail.empty()
                              ? std::string(cert.first_fault) + ":" +
                                    cert.fault_detail
                              : std::string(cert.first_fault);
      *last_why = cand->fail_reason;
      digest << " | " << tag << ":fail=cert:" << cand->fail_reason
             << " cert_ms=" << cand->cert_ms;
      return false;
    }
    cand->cert_ok = true;

    std::vector<int> sample_ends;
    sample_ends.push_back(0);
    for (const StageBoundary& b : boundaries) {
      sample_ends.push_back(static_cast<int>(b.sample_index));
    }
    PathCostBreakdown dens_soft = evaluate_path_cost(
        samples, q_neutral_, policy_, weights_, kin_, left_, guards_,
        &sample_ends);
    dens_soft.task_tcp_dz_m = task_tcp_dz_m_;
    cand->soft = dens_soft;
    cand->quality = evaluate_path_quality(
        samples, q0_, left_, policy_, dens_soft, pitch_envelope(),
        &sample_ends);

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

    plan_trial.ok = true;
    plan_trial.reason = "unified_multistage_v2";
    plan_trial.certificate = cert;
    plan_trial.package = pkg;
    plan_trial.package.certificate.passed = true;
    plan_trial.package.certificate.continuous_certified = false;
    plan_trial.package.certificate.sampled_only = true;
    plan_trial.package.certificate.coverage =
        cert.coverage == PathCoverage::AdaptiveSampledWithOnlineGuard
            ? "sampled+online_guard"
            : "sampled_only";
    plan_trial.package.certificate.min_joint_margin_rad =
        cert.min_joint_margin_rad;
    plan_trial.package.certificate.min_work_margin_rad =
        cert.min_work_margin_rad;
    plan_trial.package.certificate.min_pitch_margin_rad =
        cert.min_pitch_margin_rad;
    plan_trial.package.certificate.worst_tcp_cross_track_m =
        cert.worst_tcp_cross_track_m;
    plan_trial.package.certificate.worst_tcp_endpoint_m =
        cert.worst_tcp_endpoint_m;
    plan_trial.package.certificate.first_fault = cert.first_fault;
    // Endpoint hard gate only; 10 mm LIN runway is cert/diagnostic, not tcp_worst.
    plan_trial.tcp_worst_m = cert.worst_tcp_endpoint_m;
    plan_trial.min_hard_margin_rad = cert.min_joint_margin_rad;
    plan_trial.min_work_margin_rad = cert.min_work_margin_rad;
    plan_trial.duration_s = law.duration_s;
    plan_trial.software_limited = law.software_limited;
    samples_ = samples;
    fill_body_metrics(&plan_trial);

    digest << " | " << tag << ":ok"
           << " den_ms=" << cand->densify_ms << " time_ms=" << cand->time_ms
           << " cert_ms=" << cand->cert_ms
           << " qscore=" << cand->quality.score
           << " body_dz_mm=" << (plan_trial.body_dz_m * 1000.0)
           << " plan_duration_s=" << law.duration_s
           << " law_before_s=" << law.duration_before_s
           << " law_scale_iters=" << law.scale_iters
           << " law_src=" << law.limit_source
           << " j4pk_deg="
           << (cand->quality.j4_peak_from_start_rad * 180.0 / kPi)
           << " j6f_deg=" << (cand->quality.wrist_j6_fold_rad * 180.0 / kPi)
           << " j7f_deg=" << (cand->quality.wrist_j7_fold_rad * 180.0 / kPi)
           << " j6s_deg=" << (cand->quality.wrist_j6_span_rad * 180.0 / kPi)
           << " j7s_deg=" << (cand->quality.wrist_j7_span_rad * 180.0 / kPi)
           << " severe=" << (cand->quality.severe_foldback ? 1 : 0);
    append_fold_ruler_probe(digest, "fold_pre_den", cand->fold_pre_den);
    append_fold_ruler_probe(digest, "fold_den_span", cand->fold_den_span);
    append_fold_ruler_probe(digest, "fold_den", cand->fold_den);

    if (cand->quality.severe_foldback) {
      cand->fail_stage = "quality";
      cand->fail_reason = "quality_rejected";
      *last_why = "quality_rejected";
      digest << " selected=0 quality_rejected=1";
      return false;
    }
    cand->selected = true;
    *best_ok = plan_trial;
    chosen_seed_ = static_cast<int>(SeedKind::Coordinated);
    chosen_quality_ = cand->quality;
    digest << " selected=1";
  (void)targets;
    return true;
  }

  // One CoordinatedWaistArm main solve; at most one same-direction retry under
  // the shared 1 s budget. No multi-posture scoring or severe backup.
  bool solve_certify_serial(std::vector<KnotTarget>* targets,
                            FastPlanResult* best_ok, std::string* why,
                            std::string* digest_out) {
    seed_ik_ms_ = 0.0;
    sqp_ms_ = 0.0;
    densify_ms_ = 0.0;
    time_law_ms_ = 0.0;
    certify_ms_ = 0.0;
    main_attempt_count_ = 0;
    retry_count_ = 0;
    retry_reason_.clear();
    candidates_tried_ = 0;
    chosen_seed_ = static_cast<int>(SeedKind::Coordinated);

    std::string last_why = "unified_ik";
    std::ostringstream digest;
    digest << std::fixed << std::setprecision(3);
    digest << "attempt_task left=" << (left_ ? 1 : 0)
           << " stages=" << stages_.size()
           << " tcp_dz_mm=" << (task_tcp_dz_m_ * 1000.0)
           << " gen=" << task_.generation
           << " posture=CoordinatedWaistArm"
           << " j4_mode=" << j4_hold_probe_;

    JointCurve warm_knots;
    std::vector<int> warm_kos;
    bool have_warm = false;
    bool warm_continue_gn = false;

    for (int attempt = 0; attempt < 2; ++attempt) {
      if (deadline_hit()) {
        last_why = "deadline";
        digest << " | attempt=" << attempt << ":deadline";
        break;
      }
      ++main_attempt_count_;
      JointCurve layout;
      if (!layout_knots(&layout, targets, why)) {
        return false;
      }
      if (layout_dense_retry_) {
        digest << " | layout_dense=1";
      }
      const std::vector<KnotTarget> layout_targets = *targets;
      const std::vector<int> layout_kos = knot_of_stage_;
      KnotCandidate cand;
      const bool knots_ok = solve_coordinated_knots(
          attempt, &cand, targets, layout, layout_targets, layout_kos,
          have_warm ? &warm_knots : nullptr, have_warm ? &warm_kos : nullptr,
          warm_continue_gn, digest, &last_why);
      if (!knots_ok) {
        if (attempt == 0 &&
            plan_failure_retryable(cand.fail_stage, cand.fail_reason) &&
            !deadline_hit()) {
          last_fail_stage_ = cand.fail_stage;
          last_fail_reason_ = cand.fail_reason;
          retry_reason_ = cand.fail_reason;
          if (cand.fail_stage == "geometry" && !j4_pinned_) {
            layout_dense_retry_ = true;
            have_warm = false;
          } else if (cand.fail_stage == "densify") {
            have_warm = false;
          } else {
            warm_knots = cand.knots;
            warm_kos = cand.knot_of_stage;
            have_warm = !warm_knots.empty();
          }
          warm_continue_gn = false;
          retry_count_ = 1;
          continue;
        }
        *why = last_why;
        if (digest_out != nullptr) {
          digest << " main_attempt=" << main_attempt_count_
                 << " retry_count=" << retry_count_;
          *digest_out = digest.str();
        }
        return false;
      }
      if (certify_coordinated_package(&cand, targets, best_ok, digest,
                                      &last_why)) {
        if (digest_out != nullptr) {
          digest << " main_attempt=" << main_attempt_count_
                 << " retry_count=" << retry_count_;
          if (!retry_reason_.empty()) {
            digest << " retry_reason=" << retry_reason_;
          }
          *digest_out = digest.str();
        }
        return true;
      }
      if (attempt == 0 &&
          plan_failure_retryable(cand.fail_stage, cand.fail_reason) &&
          !deadline_hit()) {
        last_fail_stage_ = cand.fail_stage;
        last_fail_reason_ = cand.fail_reason;
        retry_reason_ = cand.fail_stage + ":" + cand.fail_reason;
        if (cand.fail_stage == "geometry" && !j4_pinned_) {
          layout_dense_retry_ = true;
          have_warm = false;
        } else if (cand.fail_stage == "densify") {
          have_warm = false;
        } else {
          warm_knots = cand.knots;
          warm_kos = cand.knot_of_stage;
          have_warm = !warm_knots.empty();
        }
        warm_continue_gn = true;
        retry_count_ = 1;
        continue;
      }
      *why = last_why;
      if (digest_out != nullptr) {
        digest << " main_attempt=" << main_attempt_count_
               << " retry_count=" << retry_count_;
        *digest_out = digest.str();
      }
      return false;
    }
    *why = last_why.empty() ? "unified_ik" : last_why;
    if (digest_out != nullptr) {
      digest << " main_attempt=" << main_attempt_count_
             << " retry_count=" << retry_count_;
      *digest_out = digest.str();
    }
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

  double descent_pitch_target() const {
    const double pitch0 = q0_[0] + q0_[1] + q0_[2];
    if (task_tcp_dz_m_ >= -kMinTcpDropForDescentPitchM) {
      return pitch0;
    }
    const double work_cap = pitch_planning_cap_rad(
        pitch_cap_, start_pitch_abs_, task_tcp_dz_m_);
    const double drop_m = std::abs(task_tcp_dz_m_);
    const double frac = coordinated_drop_frac_ > 1e-6 ? coordinated_drop_frac_
                                                      : 1.0;
    constexpr double kLeanPer150mmRad = 12.0 * kPi / 180.0;
    constexpr double kRefDropM = 0.15;
    double extra = (drop_m * frac / kRefDropM) * kLeanPer150mmRad;
    const double room = std::max(0.0, work_cap - std::abs(pitch0));
    extra = std::min(extra, room);
    const double sign = pitch0 != 0.0 ? (pitch0 > 0.0 ? 1.0 : -1.0) : -1.0;
    return pitch0 + sign * extra;
  }

  // Task-adaptive Z-squat prior on J1–J3 via Body_Link z Jacobian.
  // drop_frac in (0,1]: share of TCP height drop sought from torso height.
  // Not a fixed squat table — scaled by current q0 and task_tcp_dz_m_.
  Vec18 z_squat_prior(double drop_frac) const {
    Vec18 q = q0_;
    if (task_.waist_locked || !(task_tcp_dz_m_ < -0.02)) {
      return unbiased_neutral();
    }
    const double z0 = kin_.fk_pos(q0_, "Body_Link").z();
    const double z_goal = z0 + drop_frac * task_tcp_dz_m_;  // tcp_dz < 0
    // Prefer opening a Z-shape: J1↑ / J2↓ (opposing) lowers height with less
    // pitch growth than pure lean. Solve damped LS on [dz, dpitch, dx].
    const double pitch_target = descent_pitch_target();
    for (int it = 0; it < 12; ++it) {
      const Vec3 p = kin_.fk_pos(q, "Body_Link");
      const Mat6x18 J = kin_.jacobian_spatial(q, "Body_Link");
      const double pitch = q[0] + q[1] + q[2];
      Eigen::Matrix<double, 3, 3> H = Eigen::Matrix<double, 3, 3>::Zero();
      Eigen::Matrix<double, 3, 1> g = Eigen::Matrix<double, 3, 1>::Zero();
      // Height error.
      const double ez = z_goal - p.z();
      constexpr double wz = 42.0;
      constexpr double wp = 4.0;  // track descent lean; Z drop dominates
      constexpr double wx = 6.0;
      constexpr double w_reg = 0.4;
      for (int i = 0; i < 3; ++i) {
        g[i] += wz * J(2, i) * ez;
        g[i] += wp * (pitch_target - pitch);
        g[i] += wx * J(0, i) * (q0_body_x_ - p.x());  // limit forward arch
        g[i] += w_reg * (q0_[i] - q[i]);
        for (int j = 0; j < 3; ++j) {
          H(i, j) += wz * J(2, i) * J(2, j);
          H(i, j) += wp;
          H(i, j) += wx * J(0, i) * J(0, j);
        }
        H(i, i) += w_reg + 1e-3;
      }
      // Mild Z bias: encourage J1/J2 opposite signs relative to start.
      const double z_bias = 0.22 * drop_frac;
      g[0] += z_bias;
      g[1] -= z_bias;
      const Eigen::LDLT<Eigen::Matrix<double, 3, 3>> ldlt(H);
      if (ldlt.info() != Eigen::Success) {
        break;
      }
      Eigen::Matrix<double, 3, 1> step = ldlt.solve(g);
      if (!step.allFinite()) {
        break;
      }
      const double peak = step.cwiseAbs().maxCoeff();
      if (peak > 0.08) {
        step *= 0.08 / peak;
      }
      if (peak < 1e-4 && std::abs(ez) < 0.005) {
        break;
      }
      for (int i = 0; i < 3; ++i) {
        q[i] += step[i];
      }
      clamp_hard(&q);
      pull_pitch(&q);
    }
    clamp_hard(&q);
    pull_pitch(&q);
    return q;
  }

  // Level place: tip ori often dumps onto J7 vs retract start. Bias tip (+prep)
  // wrists toward segment start, then re-IK — cuts offline wrist_span.
  void harmonize_oriented_wrists(JointCurve* knots,
                                 const std::vector<KnotTarget>& targets) const {
    if (knots->size() < 3 || targets.size() != knots->size()) {
      return;
    }
    int tip = -1;
    for (int k = static_cast<int>(knots->size()) - 1; k >= 1; --k) {
      if (targets[static_cast<std::size_t>(k)].ori &&
          targets[static_cast<std::size_t>(k)].stage_endpoint) {
        tip = k;
        break;
      }
    }
    if (tip < 2) {
      return;
    }
    const KnotTarget& tip_tgt = targets[static_cast<std::size_t>(tip)];
    int prep = -1;
    for (int k = 1; k < tip; ++k) {
      if (targets[static_cast<std::size_t>(k)].stage_endpoint &&
          targets[static_cast<std::size_t>(k)].ori) {
        prep = k;
        break;
      }
    }
    if (prep < 0 ||
        std::abs(targets[static_cast<std::size_t>(prep)].p.z() - tip_tgt.p.z()) >
            0.05) {
      return;
    }
    const int j6 = arm_ + 5;
    const int j7 = arm_ + 6;
    const Vec18& q_start = (*knots)[0];
    auto bias_toward_start = [&](int idx) {
      const KnotTarget& tgt = targets[static_cast<std::size_t>(idx)];
      if (!tgt.ori) {
        return;
      }
      Vec18& q = (*knots)[static_cast<std::size_t>(idx)];
      q[j6] = 0.55 * q_start[j6] + 0.45 * q[j6];
      q[j7] = 0.55 * q_start[j7] + 0.45 * q[j7];
      seed_ik(&q, tgt, q, 14, 0.35);
      // Keep a residual pull so GN soft terms do not fully undo.
      q[j6] = 0.35 * q_start[j6] + 0.65 * q[j6];
      q[j7] = 0.35 * q_start[j7] + 0.65 * q[j7];
      clamp_hard(&q);
      pull_pitch(&q);
      seed_ik(&q, tgt, q, 8, 0.4);
    };
    bias_toward_start(tip);
    bias_toward_start(prep);
  }

  void apply_seed_anchor_bias(Vec18* anchor, SeedKind kind, double u,
                              const Vec18& neutral, const Vec18& squat) const {
    if (kind == SeedKind::Neutral) {
      *anchor = 0.5 * (*anchor + neutral);
    } else if (kind == SeedKind::ArmHeavy) {
      anchor->head<kNWaist>() = q0_.head<kNWaist>();
    } else if (kind == SeedKind::ZSquatMild || kind == SeedKind::ZSquatStrong) {
      for (int j = 0; j < 3; ++j) {
        (*anchor)[j] = (1.0 - u) * q0_[j] + u * squat[j];
      }
    } else if (kind == SeedKind::KeepYaw) {
      (*anchor)[kIdxYaw] = q0_[kIdxYaw];
    } else if (kind == SeedKind::Coordinated) {
      if (coordinated_drop_frac_ > 1e-6) {
        for (int j = 0; j < 3; ++j) {
          (*anchor)[j] = (1.0 - u) * q0_[j] + u * squat[j];
        }
      } else if (waist_near_hard_wall()) {
        anchor->head<kNWaist>() = q0_.head<kNWaist>();
      } else {
        *anchor = 0.5 * (*anchor + neutral);
      }
    }
  }

  void post_seed_endpoint(Vec18* q, SeedKind kind) const {
    if (kind == SeedKind::ArmHeavy) {
      q->head<kNWaist>() = q0_.head<kNWaist>();
      clamp_hard(q);
      pull_pitch(q);
    } else if (kind == SeedKind::KeepYaw) {
      (*q)[kIdxYaw] = 0.85 * q0_[kIdxYaw] + 0.15 * (*q)[kIdxYaw];
      clamp_hard(q);
      pull_pitch(q);
    } else if (kind == SeedKind::Coordinated) {
      // Mild J4 helper already lives in seed_ik yaw_anchor / j4_twist.
      // Do not smash yaw after Cartesian IK — that walks endpoints off the
      // 3 mm sphere and yanks LIN mids off the tube.
      clamp_hard(q);
      pull_pitch(q);
    }
  }

  void seed_curve(JointCurve* knots, const std::vector<KnotTarget>& targets,
                  SeedKind kind) {
    const int K = static_cast<int>(knots->size());
    double waist_w = 0.4;
    if (kind == SeedKind::ArmHeavy) {
      waist_w = 80.0;
    } else if (kind == SeedKind::KeepYaw) {
      waist_w = 0.4;
    } else if (kind == SeedKind::ZSquatMild || kind == SeedKind::ZSquatStrong) {
      waist_w = 0.25;  // freer waist for squat prior
    } else if (kind == SeedKind::Coordinated) {
      waist_w = (task_tcp_dz_m_ < -0.02) ? 0.25 : 0.4;
      if (waist_near_hard_wall()) {
        waist_w = 80.0;
      }
    }
    const Vec18 neutral = unbiased_neutral();
    Vec18 squat = q0_;
    if (kind == SeedKind::ZSquatMild) {
      squat = z_squat_prior(0.45);
    } else if (kind == SeedKind::ZSquatStrong) {
      squat = z_squat_prior(0.85);
    } else if (kind == SeedKind::Coordinated) {
      squat = coordinated_drop_frac_ > 1e-6
                  ? z_squat_prior(coordinated_drop_frac_)
                  : neutral;
    }
    const double yaw_w =
        (kind == SeedKind::ZSquatMild || kind == SeedKind::Coordinated)
            ? std::max(0.0, weights_.j4_twist)
            : -1.0;
    const int n_end = std::max(1, static_cast<int>(stages_.size()));

    // 1) Stage endpoints first (shared PREP stored once, inherited by next).
    for (std::size_t s = 0; s < stages_.size(); ++s) {
      const int k0 = knot_of_stage_[s];
      const int k1 = knot_of_stage_[s + 1];
      Vec18 anchor = (*knots)[static_cast<std::size_t>(k0)];
      const double u =
          static_cast<double>(s + 1) / static_cast<double>(n_end);
      apply_seed_anchor_bias(&anchor, kind, u, neutral, squat);
      seed_ik(&(*knots)[static_cast<std::size_t>(k1)],
              targets[static_cast<std::size_t>(k1)], anchor, 24, waist_w,
              yaw_w);
      post_seed_endpoint(&(*knots)[static_cast<std::size_t>(k1)], kind);
      ++iterations_;
    }

    // 2) Interiors: PTP = joint lerp; LIN = Cartesian IK along the chord.
    for (std::size_t s = 0; s < stages_.size(); ++s) {
      const int k0 = knot_of_stage_[s];
      const int k1 = knot_of_stage_[s + 1];
      if (k1 - k0 <= 1) {
        continue;
      }
      if (stages_[s].geo == TaskGeometry::Ptp) {
        for (int k = k0 + 1; k < k1; ++k) {
          const double t = static_cast<double>(k - k0) /
                           static_cast<double>(k1 - k0);
          (*knots)[static_cast<std::size_t>(k)] =
              (1.0 - t) * (*knots)[static_cast<std::size_t>(k0)] +
              t * (*knots)[static_cast<std::size_t>(k1)];
          clamp_hard(&(*knots)[static_cast<std::size_t>(k)]);
          pull_pitch(&(*knots)[static_cast<std::size_t>(k)]);
        }
        continue;
      }
      // LIN: sequential Cartesian IK from shared start through tip so the
      // final span does not bow off the tube after a free PTP PREP posture.
      for (int k = k0 + 1; k <= k1; ++k) {
        (*knots)[static_cast<std::size_t>(k)] =
            (*knots)[static_cast<std::size_t>(k - 1)];
        Vec18 anchor = (*knots)[static_cast<std::size_t>(k - 1)];
        const double u =
            static_cast<double>(k) / static_cast<double>(std::max(1, K - 1));
        apply_seed_anchor_bias(&anchor, kind, u, neutral, squat);
        seed_ik(&(*knots)[static_cast<std::size_t>(k)],
                targets[static_cast<std::size_t>(k)], anchor, 16, waist_w,
                yaw_w);
        post_seed_endpoint(&(*knots)[static_cast<std::size_t>(k)], kind);
        ++iterations_;
      }
    }
  }

  // Tighten stage endpoints only; keep waist near the coarse solution.
  void polish_endpoints_only(JointCurve* knots,
                             const std::vector<KnotTarget>& targets) {
    const int K = static_cast<int>(knots->size());
    if (K < 2) {
      return;
    }
    JointCurve backup = *knots;
    std::string why_before;
    const bool before_ok = accept_knots(backup, targets, &why_before, 0.0);
    constexpr double kWaistW = 0.12;
    for (int pass = 0; pass < 3; ++pass) {
      for (int k = 1; k < K; ++k) {
        const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
        if (!tgt.stage_endpoint || !tracks_cartesian_knot(tgt)) {
          continue;
        }
        Vec18& qk = (*knots)[static_cast<std::size_t>(k)];
        Vec18 anchor = qk;
        anchor.head<kNWaist>() = (*knots)[0].head<kNWaist>();
        seed_ik(&qk, tgt, anchor, 36, kWaistW, -1.0);
      }
    }
  JointCurve after_ik = *knots;
    gn_phase(knots, targets, /*use_cost=*/false, kMaxEndpointPolishIters);
    std::string why;
    if (!accept_knots(*knots, targets, &why, 0.0)) {
      if (accept_knots(after_ik, targets, &why, 0.0)) {
        *knots = std::move(after_ik);
      } else if (before_ok) {
        *knots = std::move(backup);
      }
    }
  }

  void polish_knots(JointCurve* knots,
                    const std::vector<KnotTarget>& targets) {
    // Controlled repair of Cartesian-tracked knots only. PTP interiors are
    // joint free variables — do not chase a fictional chord. Keep a full
    // backup and reject the polish if any previously-ok stage endpoint or
    // LIN tube regresses past tolerance.
    const int K = static_cast<int>(knots->size());
    const double waist_w = waist_near_hard_wall() ? 80.0 : 0.4;
    JointCurve backup = *knots;
    std::string why_before;
    const bool before_ok = accept_knots(backup, targets, &why_before);
    for (int pass = 0; pass < 2; ++pass) {
      for (int kk = 1; kk < K; ++kk) {
        const int k = pass == 0 ? kk : K - kk;  // forward, then backward
        const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
        if (!tracks_cartesian_knot(tgt)) {
          continue;
        }
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
                ? (axis <= end_row_tol_m(tgt.tol) && oe <= 0.5 * tgt.ori_tol)
                : (err <= 0.5 * tgt.tol);
        if (tight) {
          continue;
        }
        KnotTarget ik_tgt = tgt;
        if (!ik_tgt.stage_endpoint) {
          ik_tgt.ori = false;
        }
        seed_ik(&qk, ik_tgt, qk, tgt.stage_endpoint ? 24 : 8, waist_w,
                std::max(0.0, weights_.j4_twist));
      }
    }
    std::string why_after;
    if (!accept_knots(*knots, targets, &why_after)) {
      if (before_ok) {
        *knots = std::move(backup);
      }
    }
  }

  // Bounded LIN mid insert: if joint-span mid leaves the tube, insert a knot
  // at the joint mid, IK it onto the owning segment, and retry accept.
  // Keeps a best-so-far backup so a bad insert cannot regress a near-miss.
  bool refine_lin_mids(JointCurve* knots, std::vector<KnotTarget>* targets) {
    constexpr int kMaxInsertPasses = 8;
    const double waist_w = waist_near_hard_wall() ? 80.0 : 0.4;
    auto lin_worst = [&](const JointCurve& ks,
                         const std::vector<KnotTarget>& tg) {
      double w = 0.0;
      for (int k = 1; k < static_cast<int>(ks.size()); ++k) {
        const KnotTarget& t = tg[static_cast<std::size_t>(k)];
        if (t.geometry == TaskGeometry::Lin && !t.stage_endpoint) {
          w = std::max(
              w, cross_track_to_stage(kin_.fk_pos(ks[static_cast<std::size_t>(k)],
                                                  frame_),
                                      t.stage_index));
        }
        if (k + 1 < static_cast<int>(ks.size()) &&
            tracks_cartesian_span(k)) {
          const int sid = span_stage(k);
          const Vec18 qmid =
              0.5 * (ks[static_cast<std::size_t>(k)] +
                     ks[static_cast<std::size_t>(k + 1)]);
          w = std::max(w, cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid));
        }
      }
      return w;
    };
    JointCurve best_knots = *knots;
    std::vector<KnotTarget> best_tgt = *targets;
    std::vector<int> best_kos = knot_of_stage_;
    double best_w = lin_worst(best_knots, best_tgt);
    for (int pass = 0; pass < kMaxInsertPasses; ++pass) {
      std::string why;
      if (accept_knots(*knots, *targets, &why, 0.0)) {
        return true;
      }
      if (static_cast<int>(knots->size()) >= kMaxKnots) {
        return accept_knots(*knots, *targets, &why, kCoarseLinMidAdmitM);
      }
      int worst_k = -1;
      double worst_m = 0.0;
      const int K = static_cast<int>(knots->size());
      for (int k = 0; k + 1 < K; ++k) {
        if (!tracks_cartesian_span(k)) {
          continue;
        }
        const int sid = span_stage(k);
        const Vec18 qmid =
            0.5 * ((*knots)[static_cast<std::size_t>(k)] +
                   (*knots)[static_cast<std::size_t>(k + 1)]);
        const double merr =
            cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid);
        const double stol = stages_[static_cast<std::size_t>(sid)].tol;
        if (merr > stol && merr > worst_m) {
          worst_m = merr;
          worst_k = k;
        }
      }
      if (worst_k < 0) {
        // Interior LIN knot off the tube: re-IK onto the segment projection.
        bool pulled = false;
        for (int k = 1; k < K; ++k) {
          KnotTarget tgt = (*targets)[static_cast<std::size_t>(k)];
          if (tgt.stage_endpoint || tgt.geometry != TaskGeometry::Lin) {
            continue;
          }
          Vec18& qk = (*knots)[static_cast<std::size_t>(k)];
          const Vec3 pe = kin_.fk_pos(qk, frame_);
          const double ct = cross_track_to_stage(pe, tgt.stage_index);
          if (ct <= kDefaultLinRunwaySoftM) {
            continue;
          }
          const StageGeom& sg =
              stages_[static_cast<std::size_t>(tgt.stage_index)];
          const Vec3 ab = sg.b - sg.a;
          const double ab2 = ab.squaredNorm();
          if (ab2 >= 1e-16) {
            const double tt =
                std::clamp((pe - sg.a).dot(ab) / ab2, 0.0, 1.0);
            tgt.p = sg.a + tt * ab;
            tgt.quat = quat_slerp_xyzw(sg.qa, sg.qb, tt);
            (*targets)[static_cast<std::size_t>(k)] = tgt;
          }
          tgt.ori = false;
          seed_ik(&qk, tgt, qk, 28, waist_w, std::max(0.0, weights_.j4_twist));
          pulled = true;
        }
        if (!pulled) {
          return accept_knots(*knots, *targets, &why, kCoarseLinMidAdmitM);
        }
        continue;
      }
      const int sid = span_stage(worst_k);
      const StageGeom& sg = stages_[static_cast<std::size_t>(sid)];
      const Vec18 qmid =
          0.5 * ((*knots)[static_cast<std::size_t>(worst_k)] +
                 (*knots)[static_cast<std::size_t>(worst_k + 1)]);
      const Vec3 pm = kin_.fk_pos(qmid, frame_);
      const Vec3 ab = sg.b - sg.a;
      const double ab2 = ab.squaredNorm();
      double t = 0.5;
      Vec3 closest = sg.b;
      if (ab2 >= 1e-16) {
        t = std::clamp((pm - sg.a).dot(ab) / ab2, 0.0, 1.0);
        closest = sg.a + t * ab;
      }
      KnotTarget nt;
      nt.p = closest;
      nt.quat = quat_slerp_xyzw(sg.qa, sg.qb, t);
      nt.ori = false;
      nt.stage_endpoint = false;
      nt.tol = sg.tol;
      nt.ori_tol = sg.ori_tol;
      nt.stage_index = sid;
      nt.geometry = TaskGeometry::Lin;
      Vec18 qnew = qmid;
      seed_ik(&qnew, nt, qmid, 20, waist_w, std::max(0.0, weights_.j4_twist));
      knots->insert(knots->begin() + worst_k + 1, qnew);
      targets->insert(targets->begin() + worst_k + 1, nt);
      for (std::size_t s = 0; s < knot_of_stage_.size(); ++s) {
        if (knot_of_stage_[s] > worst_k) {
          ++knot_of_stage_[s];
        }
      }
      ++iterations_;
      const double now_w = lin_worst(*knots, *targets);
      if (now_w + 1e-9 < best_w) {
        best_w = now_w;
        best_knots = *knots;
        best_tgt = *targets;
        best_kos = knot_of_stage_;
      } else {
        // Insert did not help — revert and stop thrashing.
        *knots = best_knots;
        *targets = best_tgt;
        knot_of_stage_ = best_kos;
        break;
      }
    }
    *knots = best_knots;
    *targets = best_tgt;
    knot_of_stage_ = best_kos;
    std::string why;
    return accept_knots(*knots, *targets, &why, kCoarseLinMidAdmitM);
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
                    const std::vector<KnotTarget>& targets, std::string* why,
                    double mid_admit_m = 0.0,
                    bool gate_lin_mid = true) const {
    const int K = static_cast<int>(knots.size());
    double lin_worst = 0.0;
    double lin_tol = 0.0;
    double ptp_chord_diag = 0.0;  // diagnostic only; never fails accept
    for (int k = 0; k < K; ++k) {
      const Vec18& qk = knots[static_cast<std::size_t>(k)];
      if (joint_violates_effective_work_box(qk, q0_, policy_)) {
        std::ostringstream os;
        os << "unified_ik:work_box:k=" << k;
        *why = os.str();
        return false;
      }
      if (joint_violates_yaw_task_box(qk, policy_)) {
        std::ostringstream os;
        os << "unified_ik:yaw_task:k=" << k;
        *why = os.str();
        return false;
      }
    }
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
        knot_pose(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
      knot_pose[static_cast<std::size_t>(k)] =
          kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
    }
    for (int k = 1; k < K; ++k) {
      const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
      const auto& pose = knot_pose[static_cast<std::size_t>(k)];
      const Vec3 pe = pose.col(3).head<3>();
      const double err = (tgt.p - pe).norm();

      if (tgt.stage_endpoint) {
        if (trace_gn()) {
          const double oe =
              tgt.ori ? ori_error_xyzw(
                            tgt.quat,
                            rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                            .norm()
                      : 0.0;
          std::fprintf(stderr,
                       "v2gn accept end stage=%d geo=%d k=%d pos=%.5g "
                       "axis=%.5g ori=%.5g\n",
                       tgt.stage_index, static_cast<int>(tgt.geometry), k, err,
                       (tgt.p - pe).cwiseAbs().maxCoeff(), oe);
        }
        if (err > tgt.tol) {
          std::ostringstream os;
          os << std::fixed << std::setprecision(4)
             << "unified_ik:endpoint_position:stage=" << tgt.stage_index
             << ":end_m=" << err;
          *why = os.str();
          return false;
        }
        if (tgt.ori) {
          const double oe = ori_error_xyzw(
                                tgt.quat,
                                rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                                .norm();
          if (oe > tgt.ori_tol) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(4)
               << "unified_ik:endpoint_orientation:stage=" << tgt.stage_index
               << ":end_ori=" << oe;
            *why = os.str();
            return false;
          }
        }
      } else if (tgt.geometry == TaskGeometry::Lin) {
        const double ct = cross_track_to_stage(pe, tgt.stage_index);
        const double lim =
            mid_admit_m > 0.0 ? mid_admit_m : kDefaultLinRunwaySoftM;
        lin_worst = std::max(lin_worst, ct);
        lin_tol = std::max(lin_tol, tgt.tol);
        if (gate_lin_mid && ct > lim) {
          std::ostringstream os;
          os << std::fixed << std::setprecision(4)
             << "unified_ik:lin_cross_track:stage=" << tgt.stage_index
             << ":m=" << ct;
          *why = os.str();
          return false;
        }
      } else {
        ptp_chord_diag = std::max(ptp_chord_diag, err);
      }

      if (k + 1 < K && tracks_cartesian_span(k)) {
        const int sid = span_stage(k);
        const Vec18 qmid =
            0.5 * (knots[static_cast<std::size_t>(k)] +
                   knots[static_cast<std::size_t>(k + 1)]);
        const double merr =
            cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid);
        const double lim = mid_admit_m > 0.0 ? mid_admit_m : kDefaultLinRunwaySoftM;
        lin_worst = std::max(lin_worst, merr);
        lin_tol = std::max(lin_tol, kDefaultLinRunwaySoftM);
        if (trace_gn() && merr > 0.8 * kDefaultLinRunwaySoftM) {
          std::fprintf(stderr,
                       "v2gn accept lin_mid k=%d stage=%d merr=%.5g\n", k, sid,
                       merr);
        }
        if (gate_lin_mid && merr > lim) {
          std::ostringstream os;
          os << std::fixed << std::setprecision(4)
             << "unified_ik:lin_cross_track:stage=" << sid << ":mid_m=" << merr;
          *why = os.str();
          return false;
        }
      }
    }
    if (trace_gn() && ptp_chord_diag > 0.0) {
      std::fprintf(stderr, "v2gn accept ptp_chord_diag_m=%.5g (not gated)\n",
                   ptp_chord_diag);
    }
    (void)lin_worst;
    (void)lin_tol;
    return true;
  }

  // Descent tasks: secondary waist goals in the null space of the EE position
  // Jacobian (Z-squat > hold pitch sum > J4 near q0), not a sequential script.
  void apply_descent_waist_nullspace(Vec18* q, const Mat6x18& J_ee) const {
    if (task_tcp_dz_m_ >= -kMinTcpDropForDescentPitchM) {
      return;
    }
    const int na = static_cast<int>(active_.size());
    if (na < 4) {
      return;
    }
    Eigen::MatrixXd Jp(3, na);
    for (int i = 0; i < na; ++i) {
      Jp.col(i) = J_ee.col(active_[static_cast<std::size_t>(i)]).head<3>();
    }
    constexpr double kDamp = 0.025;
    Eigen::Matrix3d jjt = Jp * Jp.transpose();
    jjt.diagonal().array() += kDamp * kDamp;
    const Eigen::MatrixXd P =
        Eigen::MatrixXd::Identity(na, na) -
        Jp.transpose() * jjt.ldlt().solve(Jp);

    const Mat6x18 Jb = kin_.jacobian_spatial(*q, "Body_Link");
    const double z0 = kin_.fk_pos(q0_, "Body_Link").z();
    const double frac = coordinated_drop_frac_ > 1e-6 ? coordinated_drop_frac_
                                                      : 0.45;
    const double z_goal = z0 + frac * task_tcp_dz_m_;
    const double z_now = kin_.fk_pos(*q, "Body_Link").z();
    const double ez = z_goal - z_now;
    const double pitch = (*q)[0] + (*q)[1] + (*q)[2];
    const double pitch_target = descent_pitch_target();
    const double ep = pitch_target - pitch;

    constexpr double wz = 12.0;
    constexpr double wp = 3.0;

    Eigen::VectorXd g = Eigen::VectorXd::Zero(na);
    for (int i = 0; i < na; ++i) {
      const int ai = active_[static_cast<std::size_t>(i)];
      if (ai < kNWaist) {
        g[i] += wz * Jb(2, ai) * ez;
        g[i] += wp * ep;
      }
    }
    Eigen::VectorXd dq = P * g;
    const double peak = dq.cwiseAbs().maxCoeff();
    if (peak > 0.022) {
      dq *= 0.022 / peak;
    }
    if (!dq.allFinite()) {
      return;
    }
    for (int i = 0; i < na; ++i) {
      (*q)[active_[static_cast<std::size_t>(i)]] += dq[i];
    }
    clamp_hard(q);
    pull_pitch(q);
  }

  void seed_ik(Vec18* q, const KnotTarget& tgt, const Vec18& anchor,
               int iters, double waist_anchor_w = 0.4,
               double yaw_anchor_w = -1.0) const {
    // 6D damped IK seed: position AND (when the stage asks for it) the
    // slerped orientation. A position-only seed can strand the wrist in a
    // branch from which the GN solve cannot reach the endpoint orientation
    // within the stage tolerance. waist_anchor_w >> arm soft-anchor prefers
    // arm motion when the waist is already against a hard wall.
    // yaw_anchor_w >= 0 overrides waist_anchor_w for J4 only (Mild uses the
    // same j4_twist weight as GN); <0 keeps waist_w for all waist axes.
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
            (ai == kIdxYaw && yaw_anchor_w >= 0.0)
                ? yaw_anchor_w
                : ((ai < kNWaist) ? waist_anchor_w : w_arm_anchor);
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
          Vec18 saved = trial;
          apply_descent_waist_nullspace(&trial, J);
          const double err2 = (tgt.p - kin_.fk_pos(trial, frame_)).norm();
          const double lim = std::max(tgt.tol, min_tol());
          if (err2 <= std::max(err, lim)) {
            *q = trial;
          } else {
            *q = saved;
          }
          improved = true;
          break;
        }
      }
      if (!improved) {
        break;
      }
    }
    {
      const Vec18 saved = *q;
      const double err0 = (tgt.p - kin_.fk_pos(*q, frame_)).norm();
      const Mat6x18 J0 = kin_.jacobian_spatial(*q, frame_);
      apply_descent_waist_nullspace(q, J0);
      const double err1 = (tgt.p - kin_.fk_pos(*q, frame_)).norm();
      const double lim = std::max(tgt.tol, min_tol());
      if (err1 > std::max(err0, lim) * 1.05) {
        *q = saved;
      }
    }
    repair_com(q, tgt.p);
    pin_j4(q);
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
      if (!tracks_cartesian_knot(tgt)) {
        continue;
      }
      const auto pose = kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
      const Vec3 epv = tgt.p - pose.col(3).head<3>();
      const double e = epv.norm();
      pos += e * e;
      worst_err = std::max(worst_err, e);
      if (tgt.stage_endpoint) {
        // Soft cost must not trade away the task endpoint sphere.
        const double viol = std::max(0.0, e - tgt.tol);
        pos += 1.0e6 * viol * viol;
      }
      if (tgt.ori && tgt.stage_endpoint) {
        const Vec3 eo = ori_error_xyzw(
            tgt.quat, rot_to_quat_xyzw(pose.block<3, 3>(0, 0)));
        pos += (kWOri / kWPos) * eo.squaredNorm();
      }
      if (k + 1 < K && tracks_cartesian_span(k)) {
        const int sid = span_stage(k);
        const Vec18 qmid = 0.5 * (knots[static_cast<std::size_t>(k)] +
                                  knots[static_cast<std::size_t>(k + 1)]);
        const double merr =
            cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid);
        pos += merr * merr;
        const double runway_viol =
            std::max(0.0, merr - kDefaultLinRunwaySoftM);
        pos += 5.0e3 * runway_viol * runway_viol;
        worst_err = std::max(worst_err, merr);
      }
    }
    if (worst != nullptr) {
      *worst = worst_err;
    }
    const PathCostBreakdown cost = evaluate_path_cost(
        knots, q_neutral_, policy_, weights_, kin_, left_, guards_,
        &knot_of_stage_);
    return kWPos * pos + cost.total;
  }

  struct WorstBreakdown {
    double worst = 0.0;
    double stage_end_pos = 0.0;
    double lin_knot_pos = 0.0;
    double lin_span_mid = 0.0;
    int stage_end_k = -1;
    int stage_end_stage = -1;
    int lin_knot_k = -1;
    int lin_knot_stage = -1;
    int lin_mid_k = -1;
    int lin_mid_stage = -1;
  };

  WorstBreakdown worst_breakdown(const JointCurve& knots,
                                 const std::vector<KnotTarget>& targets) const {
    WorstBreakdown b;
    const int K = static_cast<int>(knots.size());
    for (int k = 1; k < K; ++k) {
      const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
      if (!tracks_cartesian_knot(tgt)) {
        continue;
      }
      const auto pose = kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
      const Vec3 epv = tgt.p - pose.col(3).head<3>();
      const double e = epv.norm();
      if (tgt.stage_endpoint) {
        if (e >= b.stage_end_pos) {
          b.stage_end_pos = e;
          b.stage_end_k = k;
          b.stage_end_stage = tgt.stage_index;
        }
      } else if (tgt.geometry == TaskGeometry::Lin) {
        if (e >= b.lin_knot_pos) {
          b.lin_knot_pos = e;
          b.lin_knot_k = k;
          b.lin_knot_stage = tgt.stage_index;
        }
      }
      if (k + 1 < K && tracks_cartesian_span(k)) {
        const int sid = span_stage(k);
        const Vec18 qmid = 0.5 * (knots[static_cast<std::size_t>(k)] +
                                  knots[static_cast<std::size_t>(k + 1)]);
        const double merr =
            cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid);
        if (merr >= b.lin_span_mid) {
          b.lin_span_mid = merr;
          b.lin_mid_k = k;
          b.lin_mid_stage = sid;
        }
      }
    }
    b.worst = std::max(b.stage_end_pos,
                       std::max(b.lin_knot_pos, b.lin_span_mid));
    return b;
  }

  static void worst_binding_label(const WorstBreakdown& t, const char** kind,
                                  int* k, int* stage, double* v) {
    const double w = t.worst;
    const double eps = 1e-12;
    if (t.stage_end_pos >= w - eps) {
      *kind = "stage_end_pos";
      *k = t.stage_end_k;
      *stage = t.stage_end_stage;
      *v = t.stage_end_pos;
      return;
    }
    if (t.lin_knot_pos >= w - eps) {
      *kind = "lin_knot_pos";
      *k = t.lin_knot_k;
      *stage = t.lin_knot_stage;
      *v = t.lin_knot_pos;
      return;
    }
    *kind = "lin_span_mid";
    *k = t.lin_mid_k;
    *stage = t.lin_mid_stage;
    *v = t.lin_span_mid;
  }

  bool trial_geom_soft_ok(const JointCurve& knots,
                          const std::vector<KnotTarget>& targets) const {
    const int K = static_cast<int>(knots.size());
    for (int k = 1; k < K; ++k) {
      const KnotTarget& tgt = targets[static_cast<std::size_t>(k)];
      const auto pose = kin_.fk(knots[static_cast<std::size_t>(k)], frame_);
      const Vec3 pe = pose.col(3).head<3>();
      const double err = (tgt.p - pe).norm();
      if (tgt.stage_endpoint) {
        if (err > tgt.tol) {
          return false;
        }
        if (tgt.ori) {
          const double oe = ori_error_xyzw(
                                tgt.quat,
                                rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                                .norm();
          if (oe > tgt.ori_tol) {
            return false;
          }
        }
      } else if (tgt.geometry == TaskGeometry::Lin) {
        const double ct = cross_track_to_stage(pe, tgt.stage_index);
        if (ct > kDefaultLinRunwaySoftM) {
          return false;
        }
      }
      if (k + 1 < K && tracks_cartesian_span(k)) {
        const int sid = span_stage(k);
        const Vec18 qmid = 0.5 * (knots[static_cast<std::size_t>(k)] +
                                  knots[static_cast<std::size_t>(k + 1)]);
        const double merr =
            cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid);
        if (merr > kDefaultLinRunwaySoftM) {
          return false;
        }
      }
    }
    return true;
  }

  void trace_quality_backtrack(int iter, double base, double /*worst_base*/,
                               double worst_lim,
                               const WorstBreakdown& wb,
                               const FoldRulerProbe& fold_cur,
                               double alpha, const JointCurve& trial,
                               const std::vector<KnotTarget>& targets,
                               double cost) const {
    if (!trace_gn()) {
      return;
    }
    const FoldRulerProbe fold_t = measure_fold_ruler(trial, left_);
    const double dj6 = fold_t.j6_rad - fold_cur.j6_rad;
    const bool cost_ok = cost < base - 1e-10;
    const WorstBreakdown wt = worst_breakdown(trial, targets);
    const bool worst_ok = wt.worst <= worst_lim;
    const bool split_attr =
        (dj6 < -1e-9) && cost_ok && !worst_ok;
    const char* kind_cur = "none";
    const char* kind_trial = "none";
    int k_cur = -1;
    int st_cur = -1;
    int k_trial = -1;
    int st_trial = -1;
    double v_cur = 0.0;
    double v_trial = 0.0;
    worst_binding_label(wb, &kind_cur, &k_cur, &st_cur, &v_cur);
    worst_binding_label(wt, &kind_trial, &k_trial, &st_trial, &v_trial);
    const bool geom_ok = split_attr ? trial_geom_soft_ok(trial, targets) : false;
    std::fprintf(stderr,
                 "v2gn bt cand=%s iter=%d alpha=%.2g j6f_cur=%.5g j6f_tr=%.5g "
                 "dj6f=%.5g cost=%.6g base=%.6g cost_ok=%d worst_lim=%.5g "
                 "worst_cur=%.5g worst_tr=%.5g worst_ok=%d bind_cur=%s k_cur=%d "
                 "st_cur=%d v_cur=%.5g bind_tr=%s k_tr=%d st_tr=%d v_tr=%.5g "
                 "worst_attr=%d geom_soft_ok=%d\n",
                 gn_trace_tag_, iter, alpha, fold_cur.j6_rad, fold_t.j6_rad,
                 dj6, cost, base, cost_ok ? 1 : 0, worst_lim, wb.worst, wt.worst,
                 worst_ok ? 1 : 0, kind_cur, k_cur, st_cur, v_cur, kind_trial,
                 k_trial, st_trial, v_trial, split_attr ? 1 : 0,
                 geom_ok ? 1 : 0);
  }

  void trace_fold_stages(const char* cand, const char* phase,
                         const JointCurve& knots) const {
    if (!trace_gn() || knots.empty()) {
      return;
    }
    const FoldRulerProbe fr = measure_fold_ruler(knots, left_);
    const int j6 = arm_ + 5;
    const int j4 = kIdxYaw;
    const int j7 = arm_ + 6;
    std::fprintf(stderr,
                 "v2gn fold cand=%s phase=%s j4f_deg=%.3f j6f_deg=%.3f "
                 "j7f_deg=%.3f\n",
                 cand, phase, fr.j4_rad * 180.0 / kPi, fr.j6_rad * 180.0 / kPi,
                 fr.j7_rad * 180.0 / kPi);
    for (std::size_t k = 1; k < knots.size(); ++k) {
      std::fprintf(stderr,
                   "v2gn seam cand=%s phase=%s k=%zu j4_deg=%.3f j6_deg=%.3f "
                   "j7_deg=%.3f\n",
                   cand, phase, k, (knots[k][j4] - q0_[j4]) * 180.0 / kPi,
                   knots[k][j6] * 180.0 / kPi, knots[k][j7] * 180.0 / kPi);
    }
  }

  void gauss_newton(JointCurve* knots, const std::vector<KnotTarget>& targets,
                    bool already_certified) {
    // Explicit feasibility vs quality: polish soft cost after hard-feasible,
    // but roll back if quality iters break accept_knots.
    if (already_certified) {
      JointCurve backup = *knots;
      gn_phase(knots, targets, true, kMaxGnItersAfterCert);
      std::string why;
      if (!accept_knots(*knots, targets, &why, kCoarseLinMidAdmitM)) {
        *knots = std::move(backup);
      }
      return;
    }
    // Phase A: task + TrajectoryCost (no J1–J3 angle ramp).
    gn_phase(knots, targets, true, kMaxGnIters);
    std::string why;
    if (accept_knots(*knots, targets, &why, kCoarseLinMidAdmitM)) {
      JointCurve backup = *knots;
      gn_phase(knots, targets, true, kMaxGnItersAfterCert);
      if (!accept_knots(*knots, targets, &why, kCoarseLinMidAdmitM)) {
        *knots = std::move(backup);
      }
      return;
    }
    // Phase B: feasibility-only polish, then a short quality recovery that
    // still shares the 1 s budget. Geometric backup cannot skip quality.
    gn_phase(knots, targets, false, kMaxGnFeasIters);
    if (accept_knots(*knots, targets, &why, kCoarseLinMidAdmitM) &&
        !deadline_hit()) {
      JointCurve backup = *knots;
      gn_phase(knots, targets, true, kMaxGnItersAfterCert);
      if (!accept_knots(*knots, targets, &why, kCoarseLinMidAdmitM)) {
        *knots = std::move(backup);
      }
    }
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
        const int col = (k - 1) * na;
        if (tracks_cartesian_knot(tgt)) {
          const Vec3 ep = tgt.p - pose.col(3).head<3>();
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
        }
        if (k + 1 < K && tracks_cartesian_span(k)) {
          const int sid = span_stage(k);
          const Vec18& qn = (*knots)[static_cast<std::size_t>(k + 1)];
          const Vec18 qmid = 0.5 * (q + qn);
          const Vec3 pm_fk = kin_.fk_pos(qmid, frame_);
          // Cross-track residual to owning LIN segment (not chord midpoint).
          const StageGeom& sg = stages_[static_cast<std::size_t>(sid)];
          const Vec3 ab = sg.b - sg.a;
          const double ab2 = ab.squaredNorm();
          Vec3 closest = sg.b;
          if (ab2 >= 1e-16) {
            const double tt =
                std::clamp((pm_fk - sg.a).dot(ab) / ab2, 0.0, 1.0);
            closest = sg.a + tt * ab;
          }
          const Vec3 em = closest - pm_fk;
          const Mat6x18 Jm = kin_.jacobian_spatial(qmid, frame_);
          // Stronger mid tube pull after free PTP PREP (feas + quality).
          add_mid_block(&H, &g, col, col + na, Jm, em, 1.0 * kWPos);
        }
      }
      // Same TrajectoryCost that scores the package, linearized at the knots.
      // Skipped in the feasibility-only phase. MotionEvaluator batch reuse
      // lives inside value_and_linearize (one FK/CoM/Jac pass per knot).
      if (use_cost) {
        const CostLinearization cl = value_and_linearize(
            *knots, q0_, q_neutral_, policy_, weights_, kin_, left_, guards_,
            active_, &knot_of_stage_);
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
      H.diagonal().array() += 1e-4;

      // Hard subproblem rows: joint box, pitch cap, TCP stage targets.
      for (int k = 1; k < K; ++k) {
        const Vec18& q = (*knots)[static_cast<std::size_t>(k)];
        for (int i = 0; i < na; ++i) {
          const int j = active_[static_cast<std::size_t>(i)];
          lb[(k - 1) * na + i] = rec_lo_[j] - q[j];
          ub[(k - 1) * na + i] = rec_hi_[j] - q[j];
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
          const double tol = end_row_tol_m(tgt.tol);
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
        if (st == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
          dq = qp.results.x;
          solved = dq.allFinite();
        }
      } catch (...) {
        solved = false;
      }
      if (!solved) {
        return;
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
        // Feasibility metric: real endpoint / LIN tube violations only.
        // PTP interior chord distance is not a feasibility residual.
        const auto feas_metric = [&](const JointCurve& ks) {
          double lin_w = 0.0;
          double end_w = 0.0;
          double eo_max = 0.0;
          double eo_tol = 1.0;
          double end_axis_ratio = 0.0;
          for (std::size_t k = 1; k < ks.size(); ++k) {
            const KnotTarget& tgt = targets[k];
            const auto pose = kin_.fk(ks[k], frame_);
            const Vec3 pe = pose.col(3).head<3>();
            if (tgt.stage_endpoint) {
              const Vec3 ev = tgt.p - pe;
              end_w = std::max(end_w, ev.norm());
              const double row = std::max(end_row_tol_m(tgt.tol), 1e-9);
              end_axis_ratio =
                  std::max(end_axis_ratio, ev.cwiseAbs().maxCoeff() / row);
              if (tgt.ori) {
                eo_max = std::max(
                    eo_max,
                    ori_error_xyzw(tgt.quat,
                                   rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                        .norm());
                eo_tol = std::max(1e-9, tgt.ori_tol);
              }
            } else if (tgt.geometry == TaskGeometry::Lin) {
              lin_w = std::max(lin_w, cross_track_to_stage(pe, tgt.stage_index));
            }
            if (k + 1 < ks.size() && tracks_cartesian_span(static_cast<int>(k))) {
              const int sid = span_stage(static_cast<int>(k));
              const Vec18 qmid = 0.5 * (ks[k] + ks[k + 1]);
              lin_w = std::max(
                  lin_w, cross_track_to_stage(kin_.fk_pos(qmid, frame_), sid));
            }
          }
          const double lin_ratio = lin_w / std::max(kDefaultLinRunwaySoftM, 1e-9);
          const double end_ratio =
              end_w / std::max(stages_.back().tol, 1e-9);
          return std::max(
              std::max(lin_ratio, end_ratio),
              std::max(end_axis_ratio, eo_max / eo_tol));
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
      const WorstBreakdown wb = worst_breakdown(*knots, targets);
      double tol_max = 0.0;
      for (const StageGeom& gs : stages_) {
        tol_max = std::max(tol_max, gs.tol);
      }
      const double worst_lim = std::max(worst_base + 1e-9, 0.98 * tol_max);
      const FoldRulerProbe fold_cur = measure_fold_ruler(*knots, left_);
      bool improved = false;
      double alpha_used = 0.0;
      for (double alpha : {1.0, 0.5, 0.25, 0.1, 0.05}) {
        JointCurve trial = *knots;
        apply_delta(&trial, dq, alpha);
        double worst_trial = 0.0;
        const double cost = objective(trial, targets, &worst_trial);
        if (trace_gn()) {
          trace_quality_backtrack(iter, base, worst_base, worst_lim, wb,
                                  fold_cur, alpha, trial, targets, cost);
        }
        if (cost < base - 1e-10 && worst_trial <= worst_lim) {
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
        if (accept_knots(*knots, targets, &why, kCoarseLinMidAdmitM)) {
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

  static bool retry_force_reseed_init() {
    const char* v = std::getenv("KMPC_RETRY_INIT");
    if (v == nullptr) {
      return false;
    }
    return std::strcmp(v, "reseed") == 0 || std::strcmp(v, "B") == 0 ||
           std::strcmp(v, "b") == 0;
  }

  // Worst stage-endpoint excess over the task euclidean tol.
  double end_violation(const JointCurve& knots,
                       const std::vector<KnotTarget>& targets) const {
    double viol = 0.0;
    for (std::size_t k = 1; k < knots.size(); ++k) {
      if (!targets[k].stage_endpoint) {
        continue;
      }
      const double err =
          (targets[k].p - kin_.fk_pos(knots[k], frame_)).norm();
      viol = std::max(viol, err - targets[k].tol);
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
      pin_j4(&q);
    }
  }

  bool densify(const JointCurve& knots, JointCurve* samples,
               std::vector<StageBoundary>* boundaries, std::string* why,
               FoldRulerProbe* fold_after_span = nullptr) const {
    // Joint-linear interpolation only. The solved knots are the plan; these
    // samples are Hermite control points for execution + certification.
    // Compute every span's true sample demand first, then allocate — never
    // coarsen a late span because an early span ate the budget.
    const int K = static_cast<int>(knots.size());
    samples->clear();
    boundaries->clear();
    if (K < 2) {
      *why = "unified_ik";
      return false;
    }
    boundaries->reserve(stages_.size());
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> knot_tcp(
        static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
      knot_tcp[static_cast<std::size_t>(k)] =
          kin_.fk_pos(knots[static_cast<std::size_t>(k)], frame_);
    }
    std::vector<int> n_span(static_cast<std::size_t>(K - 1), 1);
    int demand = 1;  // include start knot
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
      n_span[static_cast<std::size_t>(k)] = n;
      demand += n;
    }
    if (demand > kMaxDenseSamplesHard) {
      std::ostringstream os;
      os << "resource_sample_limit:demand=" << demand
         << ":allow=" << kMaxDenseSamplesHard << ":phase=densify:knots=" << K;
      *why = os.str();
      return false;
    }
    samples->reserve(static_cast<std::size_t>(demand));
    samples->push_back(knots.front());
    const double densify_waist_w = waist_near_hard_wall() ? 80.0 : 0.4;
    for (int k = 0; k + 1 < K; ++k) {
      const Vec18& qa = knots[static_cast<std::size_t>(k)];
      const Vec18& qb = knots[static_cast<std::size_t>(k + 1)];
      const int n = n_span[static_cast<std::size_t>(k)];
      const bool lin_span = tracks_cartesian_span(k);
      const int sid = span_stage(k);
      for (int i = 1; i <= n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        Vec18 q = (1.0 - t) * qa + t * qb;
        // LIN span: joint lerp and coarse interior knots may sit off the
        // tube (10 mm admit). Project position onto the segment; keep
        // orientation only at stage endpoints.
        bool lin_stage_end = false;
        if (lin_span && i == n) {
          for (int ke : knot_of_stage_) {
            if (ke == k + 1) {
              lin_stage_end = true;
              break;
            }
          }
        }
        if (lin_span && !lin_stage_end) {
          const StageGeom& sg = stages_[static_cast<std::size_t>(sid)];
          const Vec3 pe = kin_.fk_pos(q, frame_);
          const double admit = 0.85 * kDefaultLinRunwaySoftM;
          if (cross_track_to_stage(pe, sid) > admit) {
            const Vec3 ab = sg.b - sg.a;
            const double ab2 = ab.squaredNorm();
            KnotTarget nt;
            nt.p = sg.b;
            double tt = t;
            if (ab2 >= 1e-16) {
              tt = std::clamp((pe - sg.a).dot(ab) / ab2, 0.0, 1.0);
              nt.p = sg.a + tt * ab;
            }
            nt.quat = quat_slerp_xyzw(sg.qa, sg.qb, tt);
            nt.ori = false;
            nt.tol = sg.tol;
            nt.ori_tol = sg.ori_tol;
            nt.geometry = TaskGeometry::Lin;
            nt.stage_index = sid;
            const Vec18 ik_anchor = q;
            seed_ik(&q, nt, ik_anchor, 28, densify_waist_w,
                    densify_j4_yaw_anchor_w());
            clamp_densify_j4_to_lerp(&q, ik_anchor, nt);
            if (!knot_densify_sample_ok(q, nt)) {
              std::ostringstream os;
              os << "densify:lin_projection:span=" << k << ":i=" << i
                 << ":stage=" << sid;
              *why = os.str();
              return false;
            }
          }
        }
        samples->push_back(q);
      }
      for (std::size_t s = 0; s + 1 < knot_of_stage_.size(); ++s) {
        if (k + 1 == knot_of_stage_[s + 1]) {
          StageBoundary b;
          b.sample_index = samples->size() - 1;
          b.stage_id = stages_[s].stage_id;
          b.stop_required = stages_[s].stop_required;
          b.wait_for_external_ack = stages_[s].wait_ack;
          b.position_tol_m = stages_[s].tol;
          b.orientation_tol_rad = stages_[s].ori_tol;
          b.has_orientation = stages_[s].ori;
          boundaries->push_back(b);
        }
      }
    }
    if (fold_after_span != nullptr) {
      *fold_after_span = measure_fold_ruler(*samples, left_);
    }
    // PCHIP is C1 across knots; corner rounding can leave the LIN tube even
    // when densified chord samples sit inside. Insert joint-linear midpoints
    // only where Hermite mid TCP exceeds the stage tol (bounded passes), then
    // the full package is re-certified by validate_actual_q_of_t.
    // When span densify already keeps J4 fold under the quality gate, skip
    // Hermite TCP refine — it was the main source of waist fold amplification.
    constexpr double kJ4FoldSkipHermiteRad = 40.0 * kPi / 180.0;
    const bool j4_span_within_gate =
        fold_after_span == nullptr ||
        fold_after_span->j4_rad <= kJ4FoldSkipHermiteRad + 1e-9;
    if (!j4_span_within_gate && !refine_hermite_tcp(samples, boundaries, why)) {
      return false;
    }
    return samples->size() >= 2;
  }

  bool refine_hermite_tcp(JointCurve* samples,
                          std::vector<StageBoundary>* boundaries,
                          std::string* why) const {
    if (samples->size() < 2 || stages_.empty()) {
      return true;
    }
    const std::size_t nstage = stages_.size();
    auto seg_dist = [](const Vec3& p, const Vec3& a, const Vec3& b) {
      const Vec3 ab = b - a;
      const double ab2 = ab.squaredNorm();
      if (ab2 < 1e-16) {
        return (p - b).norm();
      }
      const double t = std::clamp((p - a).dot(ab) / ab2, 0.0, 1.0);
      return (p - (a + t * ab)).norm();
    };
    // Peak TCP bow from corner rounding is often off-midpoint; probe quarters.
    // Also densify where PCHIP overshoots wrist J6/J7 past the local chord
    // envelope — offline wrist_span gates measure Hermite q(t), not knots.
    constexpr double kProbeU[] = {0.25, 0.5, 0.75};
    const int kMaxRefinePasses = j4_pinned_ ? 3 : 8;
    constexpr double kWristOvershootRad = 0.005;  // ~0.3°
    const int arm0 = left_ ? kSliceLeft.begin : kSliceRight.begin;
    const int j_wrist[2] = {arm0 + 5, arm0 + 6};
    for (int pass = 0; pass < kMaxRefinePasses; ++pass) {
      std::vector<std::pair<std::size_t, bool>> insert_after;
      insert_after.reserve(samples->size());
      for (std::size_t i = 0; i + 1 < samples->size(); ++i) {
        const std::size_t seg =
            stage_for_sample_span(i, nstage, boundaries);
        // LIN: Cartesian tube refine. All geometries: wrist Hermite overshoot.
        // Do not skip wrist densify when stage is PTP.
        const bool lin = stages_[seg].geo == TaskGeometry::Lin;
        const double refine_tol = 0.95 * stages_[seg].tol;
        bool tcp_bad = false;
        bool wrist_bad = false;
        const Vec18& qa = (*samples)[i];
        const Vec18& qb = (*samples)[i + 1];
        for (double u : kProbeU) {
          const Vec18 q =
              hermite_joint_at(*samples, static_cast<double>(i) + u);
          if (lin) {
            const Vec3 p = kin_.fk_pos(q, frame_);
            if (seg_dist(p, stages_[seg].a, stages_[seg].b) > refine_tol) {
              tcp_bad = true;
            }
          }
          for (int jw : j_wrist) {
            const double lo = std::min(qa[jw], qb[jw]) - kWristOvershootRad;
            const double hi = std::max(qa[jw], qb[jw]) + kWristOvershootRad;
            if (q[jw] < lo || q[jw] > hi) {
              wrist_bad = true;
            }
          }
        }
        if (tcp_bad || wrist_bad) {
          // LIN TCP projection re-solves waist; wrist-only gets joint lerp.
          insert_after.emplace_back(i, tcp_bad);
        }
      }
      if (insert_after.empty()) {
        return true;
      }
      for (auto it = insert_after.rbegin(); it != insert_after.rend(); ++it) {
        const std::size_t i = it->first;
        const bool tcp_bad = it->second;
        if (static_cast<int>(samples->size()) >= kMaxDenseSamplesHard) {
          if (why != nullptr) {
            *why = "hermite_refine_capped";
          }
          return true;
        }
        const std::size_t seg =
            stage_for_sample_span(i, nstage, boundaries);
        const bool lin = stages_[seg].geo == TaskGeometry::Lin;
        Vec18 mid = 0.5 * ((*samples)[i] + (*samples)[i + 1]);
        pin_j4(&mid);
        if (lin && tcp_bad) {
          const StageGeom& sg = stages_[seg];
          const Vec3 pe = kin_.fk_pos(mid, frame_);
          const Vec3 ab = sg.b - sg.a;
          const double ab2 = ab.squaredNorm();
          KnotTarget nt;
          nt.p = sg.b;
          double tt = 0.5;
          if (ab2 >= 1e-16) {
            tt = std::clamp((pe - sg.a).dot(ab) / ab2, 0.0, 1.0);
            nt.p = sg.a + tt * ab;
          }
          nt.quat = quat_slerp_xyzw(sg.qa, sg.qb, tt);
          nt.ori = false;
          nt.tol = sg.tol;
          nt.ori_tol = sg.ori_tol;
          nt.geometry = TaskGeometry::Lin;
          nt.stage_index = static_cast<int>(seg);
          const double ww = waist_near_hard_wall() ? 80.0 : 0.4;
          const Vec18 ik_anchor = mid;
          seed_ik(&mid, nt, ik_anchor, j4_pinned_ ? 36 : 24, ww,
                  densify_j4_yaw_anchor_w());
          clamp_densify_j4_to_lerp(&mid, ik_anchor, nt);
          if (!knot_densify_sample_ok(mid, nt)) {
            mid = 0.5 * ((*samples)[i] + (*samples)[i + 1]);
            pin_j4(&mid);
          }
        }
        samples->insert(samples->begin() + static_cast<std::ptrdiff_t>(i + 1),
                        mid);
        for (StageBoundary& b : *boundaries) {
          if (b.sample_index > i) {
            ++b.sample_index;
          }
        }
      }
    }
    return true;
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
    // Do not re-run evaluate_path_cost on densified samples here — that is a
    // second full FK/feature pass after certification and was dominating the
    // post-cert tail on Jetson. Soft cost already ran inside GN; report the
    // certificate metrics instead.
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "planner=unified_multistage_v2"
       << " path_id=" << std::hex << out.package.path_id << std::dec
       << " stages=" << stages_.size()
       << " opt=" << (out.ok ? "ok" : "fail")
       << " accept=" << (out.ok ? "pass" : "fail")
       << " tcp_mm=" << (out.tcp_worst_m * 1000.0)
       << " margin_deg=" << (out.min_hard_margin_rad * 180.0 / kPi)
       << " work_mrg_deg=" << (out.min_work_margin_rad * 180.0 / kPi)
       << " j1_work_lo_deg=" << (work_lo_[0] * 180.0 / kPi)
       << " j1_work_hi_deg=" << (work_hi_[0] * 180.0 / kPi)
       << " body_dx_mm=" << (out.body_dx_m * 1000.0)
       << " body_dz_mm=" << (out.body_dz_m * 1000.0)
       << " body_leg_dx_mm=" << (out.body_minus_leg_dx_m * 1000.0)
       << " iters=" << out.iterations
       << " seed_ms=" << out.seed_ik_ms
       << " sqp_ms=" << out.sqp_ms
       << " densify_ms=" << out.densify_ms
       << " time_ms=" << out.time_law_ms
       << " opt_ms=" << out.optimize_ms
       << " cert_ms=" << out.certify_ms
       << " total_ms=" << out.total_ms
       << " over_10ms="
       << ((out.total_ms > kPlannerPerfStatMs) ? 1 : 0)
       << " cost=cert_metrics"
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
      (*q)[j] = std::clamp((*q)[j], rec_lo_[j], rec_hi_[j]);
    }
  }

  // Same recovery envelope as the QP pitch row / validator: a start already
  // outside ±pitch_cap may keep |start|, never grow. Forcing ±cap on every
  // seed trial would yank waist (and TCP) when J1 is near the hard wall and
  // start pitch already exceeds the ordinary task limit.
  double pitch_envelope() const {
    const double cap = pitch_planning_cap_rad(pitch_cap_, start_pitch_abs_,
                                              task_tcp_dz_m_);
    double env = std::max(cap, start_pitch_abs_);
    // Leave a sliver below the planning cap so PCHIP/exec ticks do not kiss the
    // cert rail (≈0.1 mrad overshoot seen on r54 left drop).
    if (task_tcp_dz_m_ < -kMinTcpDropForDescentPitchM && cap > start_pitch_abs_) {
      env = std::max(start_pitch_abs_, cap - 0.25 * kPitchDescentWorkMarginRad);
    }
    return env;
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
        const double room = sign > 0.0 ? ((*q)[a] - rec_lo_[a])
                                       : (rec_hi_[a] - (*q)[a]);
        if (room > room_best) {
          room_best = room;
          best = a;
        }
      }
      const double step = std::min(excess, std::max(0.0, room_best));
      (*q)[best] -= sign * step;
      (*q)[best] = std::clamp((*q)[best], rec_lo_[best], rec_hi_[best]);
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
  Vec18 work_lo_ = Vec18::Zero();
  Vec18 work_hi_ = Vec18::Zero();
  Vec18 rec_lo_ = Vec18::Zero();
  Vec18 rec_hi_ = Vec18::Zero();
  bool left_ = true;
  std::string frame_;
  int arm_ = 4;
  std::vector<int> active_;
  std::vector<StageGeom, Eigen::aligned_allocator<StageGeom>> stages_;
  std::vector<int> knot_of_stage_;
  double pitch_cap_ = 1e9;
  double start_pitch_abs_ = 0.0;
  double q0_body_x_ = 0.0;
  double task_tcp_dz_m_ = 0.0;
  int chosen_seed_ = -1;
  int candidates_tried_ = 0;
  int main_attempt_count_ = 0;
  int retry_count_ = 0;
  std::string retry_reason_;
  double coordinated_drop_frac_ = 0.0;
  PathQualityMetrics chosen_quality_;
  int iterations_ = 0;
  bool deadline_hit_ = false;
  double seed_ik_ms_ = 0.0;
  double sqp_ms_ = 0.0;
  double densify_ms_ = 0.0;
  double time_law_ms_ = 0.0;
  double certify_ms_ = 0.0;
  JointCurve samples_;
  bool j4_pinned_ = false;
  double j4_pin_q_ = 0.0;
  std::string j4_release_reason_;
  std::string last_fail_stage_;
  std::string last_fail_reason_;
  int lin_boost_count_ = 0;
  static constexpr int kMaxLinBoostHold = 3;
  bool layout_dense_retry_ = false;
  std::string j4_hold_probe_;
  const char* gn_trace_tag_ = "main";
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
