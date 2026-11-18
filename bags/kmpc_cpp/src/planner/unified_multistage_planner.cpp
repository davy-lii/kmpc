// v1 entry points on top of the unified_multistage_v2 chain.
// plan_unified_multistage is an adapter: MotionTask in, certified package
// out, segment/successor split at the stage boundary. The DirectionContract
// fields are diagnostics only — the sign hard gate and the smooth_reversal
// patch are retired on this chain (direction.ok is always true with
// reason "v2_no_sign_hard_gate").

#include "kmpc/unified_multistage_planner.hpp"

#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_cost.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

namespace kmpc {
namespace {

// Orientation gate the retired planner certified against.
constexpr double kStageOriTolRad = 0.08;

std::string precheck_report(const char* reason, std::size_t stages) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(3);
  os << "planner=unified_multistage_v2"
     << " path_id=0"
     << " stages=" << stages
     << " opt=fail accept=fail"
     << " tcp_mm=0 margin_deg=0"
     << " body_dx_mm=0 body_dz_mm=0 body_leg_dx_mm=0"
     << " iters=0 opt_ms=0 cert_ms=0 cost=0"
     << " height_switch=0"
     << " reason=" << reason;
  return os.str();
}

void fill_direction_diagnostics(const JointCurve& curve,
                                DirectionContract* contract,
                                DirectionCheckResult* check) {
  *contract = make_path_direction_contract(curve);
  *check = check_path_direction(curve, *contract);
  // v2: the planner-level sign hard gate is retired. Axis stats above stay
  // for diagnostics and for the stream-side envelope; admission here is free.
  check->ok = true;
  check->violated_axis = -1;
  std::snprintf(check->reason, sizeof(check->reason), "v2_no_sign_hard_gate");
}

}  // namespace

UnifiedPlanResult plan_unified_multistage(
    const Kinematics& kin, const WholeBodyGuards& guards,
    const UnifiedPlanRequest& req) {
  UnifiedPlanResult out;
  const std::size_t nstage = req.has_next ? 2u : 1u;

  // Legacy reason strings preserved for the tracker and logs.
  if (!req.q0.allFinite() || !req.goal.xyz.allFinite() || req.frame.empty()) {
    out.reason = "unified_ik";
    out.report = precheck_report("unified_ik", nstage);
    return out;
  }
  if (joint_violates_hard_box(req.q0, req.policy.q_min, req.policy.q_max)) {
    out.reason = "unified_hard_box";
    out.report = precheck_report("unified_hard_box", nstage);
    return out;
  }
  const Vec3 p0 = kin.fk_pos(req.q0, req.frame);
  const double reach0 = (req.goal.xyz - p0).norm();
  if (reach0 > static_cast<double>(req.max_samples) *
                   std::max(req.sample_max_m, 1e-3)) {
    out.reason = "unified_sample_limit";
    out.report = precheck_report("unified_sample_limit", nstage);
    return out;
  }
  if (req.has_next) {
    const double reach1 = (req.next.xyz - req.goal.xyz).norm();
    if (!req.next.xyz.allFinite() ||
        reach1 > static_cast<double>(req.max_samples) *
                     std::max(req.sample_max_m, 1e-3)) {
      out.reason = "cascade_unreachable";
      out.report = precheck_report("cascade_unreachable", nstage);
      return out;
    }
  }

  MotionTask task;
  task.q = req.q0;
  task.left = req.left;
  task.waist_locked = req.waist_locked;
  task.frame = req.frame;
  task.stages.resize(nstage);
  TaskStage& s0 = task.stages[0];
  // Cascade PREP→TIP mirrors legacy two-point: PTP then LIN. Lone goal stays LIN.
  s0.geometry = req.has_next ? TaskGeometry::Ptp : TaskGeometry::Lin;
  s0.position = req.goal.xyz;
  s0.has_orientation = req.goal.ori_on;
  s0.quat = req.goal.quat;
  s0.position_tol_m = req.pos_tol_m;
  s0.orientation_tol_rad = kStageOriTolRad;
  s0.stop = req.has_next ? StageStop::Continuous : StageStop::StopRequired;
  s0.stage_id = 0;
  if (req.has_next) {
    TaskStage& s1 = task.stages[1];
    s1.geometry = TaskGeometry::Lin;
    s1.position = req.next.xyz;
    s1.has_orientation = req.next.ori_on;
    s1.quat = req.next.quat;
    s1.position_tol_m = req.pos_tol_m;
    s1.orientation_tol_rad = kStageOriTolRad;
    s1.stop = StageStop::StopRequired;
    s1.stage_id = 1;
  }

  RobotState state;
  state.q = req.q0;
  state.dq = Vec18::Zero();
  state.dq_known = false;
  state.fresh = true;

  const TrajectoryCostWeights weights = default_cost_weights(req.dq_max);
  const FastPlanResult planned = solve_to_verified_package(
      kin, guards, task, req.policy, weights);

  out.ok = planned.ok;
  out.reason = planned.ok ? "unified_multistage_v2" : planned.reason;
  out.report = planned.report;
  out.iterations = planned.iterations;
  out.optimize_ms = planned.optimize_ms;
  out.certify_ms = planned.certify_ms;
  out.tcp_worst_m = planned.tcp_worst_m;
  out.min_hard_margin_rad = planned.min_hard_margin_rad;
  out.path_id = planned.package.path_id;
  if (planned.ok) {
    out.package = planned.package;
    out.has_package = true;
  }

  if (planned.ok) {
    const JointCurve& samples = planned.package.samples;
    if (req.has_next && !planned.package.stage_boundaries.empty()) {
      const std::size_t b = std::min(
          planned.package.stage_boundaries.front().sample_index,
          samples.size() - 1);
      out.segment.assign(samples.begin(), samples.begin() + b + 1);
      out.successor.assign(samples.begin() + b, samples.end());
      out.has_successor = out.successor.size() >= 2;
    } else {
      out.segment = samples;
      out.has_successor = false;
    }
    fill_direction_diagnostics(out.segment, &out.contract, &out.direction);
    if (out.has_successor) {
      fill_direction_diagnostics(out.successor, &out.successor_contract,
                                 &out.successor_direction);
    }
  }

  // Body metrics kept from the v1 report contract.
  if (out.segment.size() >= 2) {
    const Vec18& qN = out.segment.back();
    const double z0 = kin.fk(out.segment.front(), "Body_Link")(2, 3);
    const double z1 = kin.fk(qN, "Body_Link")(2, 3);
    out.body_drop_m = z0 - z1;
    const int arm = req.left ? kSliceLeft.begin : kSliceRight.begin;
    Vec18 qw = req.q0;
    qw.head<kNWaist>() = qN.head<kNWaist>();
    Vec18 qa = req.q0;
    qa.segment<kNArm>(arm) = qN.segment<kNArm>(arm);
    out.waist_tcp_m = (kin.fk_pos(qw, req.frame) - p0).norm();
    out.arm_tcp_m = (kin.fk_pos(qa, req.frame) - p0).norm();
  }
  return out;
}

FastPlanResult solve_to_verified_package(
    const Kinematics& kin, const WholeBodyGuards& guards,
    const MotionTask& task, const MotionConstraintPolicy& policy,
    const TrajectoryCostWeights& weights) {
  RobotState state;
  state.q = task.q;
  state.dq = task.dq;
  state.dq_known = task.dq_known;
  state.fresh = true;
  const FastTrajectoryOptimizer optimizer(kin, &guards);
  return optimizer.plan(task, state, policy, weights, 0.0);
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
  if (!q_meas.allFinite() ||
      joint_violates_hard_box(q_meas, policy.q_min, policy.q_max)) {
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
