// 单臂够点：整段直线一次求完，之后只沿这条路径推进度。
// 变量是四个独立腰关节加这条臂。另一条臂保持出发角度。
// 步长用直线自己的速度比例。单臂解不出就停住，不进跟踪。双臂、标定仍走 QP。

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/unified_multistage_planner.hpp"
#include "kmpc/path_progress.hpp"
#include "kmpc/path_smooth_verify.hpp"
#include "kmpc/unified_time_law.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <limits>

#include <Eigen/Cholesky>

#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr double kLineSampleM = 0.005;
constexpr double kLineReplanM = 0.008;
// Freeze / abort-near-goal distance. Brake distance is a separate runtime param.
constexpr double kLineFreezeM = 0.08;
constexpr double kLineSampleTolM = 0.003;
// Stream safety tube. The planner certifies a tighter line; this is the
// independent reject for a chord that has left the segment.
constexpr double kLineTubeHardM = 0.20;
constexpr double kLineAbortM = 0.05;
constexpr double kLineAbortRad = 15.0 * kPi / 180.0;
constexpr int kLineMaxSamples = 400;
std::string line_frame(bool left, std::string_view given) {
  if (!given.empty()) {
    return std::string(given);
  }
  return std::string(left ? kEeFrameLeft : kEeFrameRight);
}

double line_pitch_turnback(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path) {
  if (path.size() < 2) return 0.0;
  const auto pitch=[](const Vec18& q){return q.head<3>().sum();};
  const double net=pitch(path.back())-pitch(path.front());
  if (std::abs(net)<1e-4) {
    double lo=pitch(path.front()),hi=lo;
    for (const auto& q:path) {lo=std::min(lo,pitch(q));hi=std::max(hi,pitch(q));}
    return hi-lo;
  }
  const double dir=net>0 ? 1.0 : -1.0;
  double best=dir*pitch(path.front()),worst=0.0;
  for (const auto& q:path) {
    const double p=dir*pitch(q);
    best=std::max(best,p);
    worst=std::max(worst,best-p);
  }
  return worst;
}

}  // namespace

void CartesianTracker::set_line_pace(double speed_scale, double accel_s) {
  if (!std::isfinite(speed_scale)) {
    line_speed_scale_ = 1.0;
  } else {
    line_speed_scale_ = std::clamp(speed_scale, 0.0, 1.0);
  }
  if (!std::isfinite(accel_s) || accel_s < 0.0) {
    line_accel_s_ = 0.0;
  } else {
    line_accel_s_ = accel_s;
  }
}

void CartesianTracker::set_line_jerk_s(double jerk_s) {
  if (!std::isfinite(jerk_s) || jerk_s <= 0.0) {
    line_jerk_s_ = 0.08;
  } else {
    line_jerk_s_ = jerk_s;
  }
}

bool CartesianTracker::set_line_brake_distance_m(double meters) {
  if (!std::isfinite(meters) || !(meters > 0.0)) {
    return false;
  }
  line_brake_distance_m_ = meters;
  return true;
}

bool CartesianTracker::set_line_brake_floor(double floor) {
  if (!std::isfinite(floor) || !(floor > 0.0) || !(floor <= 1.0)) {
    return false;
  }
  line_brake_floor_ = floor;
  return true;
}

bool CartesianTracker::set_line_qp_max_iter(int max_iter) {
  if (max_iter < 1 || max_iter > 10000) {
    return false;
  }
  line_qp_max_iter_ = max_iter;
  return true;
}

void CartesianTracker::set_line_chain_profile(LineChainProfile profile) {
  line_chain_profile_ = profile;
  if (profile == LineChainProfile::NewUnified) {
    // NewUnified: unified s(t) time law + package sample follow.
    // Do not use line_path_qp as a pose selector (see follow: v2_sample).
    line_pace_mode_ = LinePaceMode::UnifiedTimeLaw;
    line_qp_follow_mode_ = LineQpFollowMode::B;  // unused on v2 follow path
    line_path_interp_ = LinePathInterp::Linear;  // promoted after verify
  } else {
    // Legacy must preserve already-loaded line_pace_mode_ / line_qp_follow_mode_
    // (YAML/node setters). Only reset interpolant/smooth state.
    line_path_interp_ = LinePathInterp::Linear;
  }
  last_path_smooth_report_ = PathSmoothReport{};
  last_line_geometry_c1_ = false;
  stream_commit_feedback_scale_ = 1.0;
}

void CartesianTracker::line_clear() {
  // Drop any unconfirmed propose so a later generation cannot auto-adopt it.
  discard_line_stream_commit();
  line_ = LineRun{};
  line_q_.setZero();
  line_rate_ = 0.0;
  line_rate_acc_ = 0.0;
  line_path_interp_ = LinePathInterp::Linear;
  last_line_geometry_c1_ = false;
  last_path_smooth_report_ = PathSmoothReport{};
  line_tip_.reset();
  line_reference_.reset();
  last_line_cache_committed_ = false;
  // SuccessorPlan is a chain commitment across rearm/generation. Cleared only
  // when consumed, mismatched, or explicitly invalidated — not on every segment
  // reset.
  reverse_decision_ = reverse_last_line_enable_ ? "skipped" : "disabled";
  line_seg_start_s_ = -1.0;
  line_tail_enter_brake_s_ = -1.0;
  line_tail_enter_2cm_s_ = -1.0;
  line_tail_hold_s_ = -1.0;
  last_line_qp_fail_ = LineQpFailSnapshot{};
  // Pending matrix dumps must survive nail/empty-pose reset(); the 100ms
  // dump timer usually loses the race otherwise (~10ms to poses=0).
  if (!last_line_qp_fail_dump_.pending_write) {
    last_line_qp_fail_dump_ = LineQpFailDump{};
  }
}

void CartesianTracker::clear_last_line_cache() {
  last_line_cache_ = LastLineCache{};
  last_line_cache_committed_ = false;
}

void CartesianTracker::set_line_diag_enable(bool enable) {
  line_diag_enable_ = enable;
  if (enable && line_diag_ring_.empty()) {
    line_diag_ring_.assign(kLineDiagCap, LineDiagEvent{});
  }
  if (!enable) {
    line_diag_begin_ = 0;
    line_diag_count_ = 0;
    line_diag_overflows_ = 0;
  }
}

size_t CartesianTracker::line_diag_size() const { return line_diag_count_; }

void CartesianTracker::line_diag_clear() {
  line_diag_begin_ = 0;
  line_diag_count_ = 0;
  line_diag_overflows_ = 0;
}

std::vector<CartesianTracker::LineDiagEvent> CartesianTracker::line_diag_snapshot() const {
  std::vector<LineDiagEvent> out;
  out.reserve(line_diag_count_);
  for (size_t i = 0; i < line_diag_count_; ++i) {
    out.push_back(line_diag_ring_[(line_diag_begin_ + i) % kLineDiagCap]);
  }
  return out;
}

void CartesianTracker::line_diag_push(
    LineDiagLayer layer, const Vec18& q, const Vec18& q_aux, double prog,
    double build_ms, const char* candidate, const char* decision) {
  if (!line_diag_enable_) return;
  if (line_diag_ring_.empty()) {
    line_diag_ring_.assign(kLineDiagCap, LineDiagEvent{});
  }
  LineDiagEvent ev;
  ev.layer = layer;
  ev.goal_generation = goal_generation_;
  ev.line_gen = line_gen_;
  ev.tick_id = control_tick_id_.load(std::memory_order_acquire);
  ev.t_steady_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
  ev.prog = prog;
  ev.s_dot = line_rate_;
  ev.build_ms = build_ms;
  ev.solve_us = last_line_qp_solve_us_;
  ev.commit_scale = 1.0;
  ev.qp_dx_norm = last_line_qp_dx_norm_;
  ev.dq_proj_max = last_line_qp_proj_max_;
  ev.qp_mode = static_cast<uint8_t>(line_qp_follow_mode_);
  ev.field_mask = 0;
  if (layer == LineDiagLayer::FollowQp || layer == LineDiagLayer::FollowCommit) {
    ev.field_mask = static_cast<uint8_t>(
        ev.field_mask | (1u << 0) | (1u << 6));  // s_dot + qp_mode
    if (last_line_qp_solve_us_ > 0.0) {
      ev.field_mask = static_cast<uint8_t>(ev.field_mask | (1u << 1));
    }
    if (last_line_qp_dx_norm_ > 0.0 || last_line_qp_proj_max_ > 0.0) {
      ev.field_mask = static_cast<uint8_t>(ev.field_mask | (1u << 5));
    }
    ev.q_nominal = last_line_qp_nominal_;
    ev.q_pre = last_line_qp_pre_;
    if (last_line_qp_nominal_.allFinite()) {
      ev.field_mask = static_cast<uint8_t>(ev.field_mask | (1u << 3));
    }
    if (last_line_qp_pre_.allFinite()) {
      ev.field_mask = static_cast<uint8_t>(ev.field_mask | (1u << 4));
    }
  }
  ev.q = q;
  ev.q_aux = q_aux;
  ev.mono_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  ev.recv_mono_ns = ev.mono_ns;
  ev.source_stamp_ns = line_diag_source_ns_;
  ev.run_id = line_diag_run_id_;
  ev.seq = ++line_diag_seq_;
  line_diag_source_ns_ = -1;
  if (candidate) {
    std::snprintf(ev.candidate, sizeof(ev.candidate), "%s", candidate);
  }
  if (decision) {
    std::snprintf(ev.decision, sizeof(ev.decision), "%s", decision);
  }
  if (line_diag_count_ < kLineDiagCap) {
    line_diag_ring_[(line_diag_begin_ + line_diag_count_) % kLineDiagCap] = ev;
    ++line_diag_count_;
  } else {
    line_diag_ring_[line_diag_begin_] = ev;
    line_diag_begin_ = (line_diag_begin_ + 1) % kLineDiagCap;
    ++line_diag_overflows_;
  }
}

bool CartesianTracker::confirm_line_stream_commit(const Vec18& accepted,
                                                 double scale) {
  if (!line_commit_pending_ || !accepted.allFinite()) {
    last_progress_recover_.ok = false;
    last_progress_recover_.unknown = true;
    std::snprintf(last_progress_recover_.reason,
                  sizeof(last_progress_recover_.reason),
                  line_commit_pending_ ? "nonfinite" : "not_pending");
    return false;
  }
  // Bind progress to the accepted joint on the path — never treat stream chord
  // scale as an s fraction when raw≠endpoint / QP reshape / knot crossing.
  auto at = [this](double s) { return line_at(s); };
  // Exact full admit of this propose: the path parameter of the propose is the
  // committed progress (QP may sit off the nominal sample; the chord was already
  // stream-checked before confirm).
  if ((accepted - line_commit_proposed_).cwiseAbs().maxCoeff() <= 1e-12) {
    last_progress_recover_.ok = true;
    last_progress_recover_.unknown = false;
    last_progress_recover_.prog = line_commit_new_prog_;
    last_progress_recover_.residual_rad =
        (at(line_commit_new_prog_) - accepted).cwiseAbs().maxCoeff();
    std::snprintf(last_progress_recover_.reason,
                  sizeof(last_progress_recover_.reason), "full_propose");
  } else {
    last_progress_recover_ = recover_path_progress(
        line_commit_old_prog_, line_commit_new_prog_, accepted, at);
  }
  if (last_progress_recover_.unknown || !last_progress_recover_.ok) {
    // Chord-fraction fallback only when accepted lies on the propose chord AND
    // the mapped path sample itself is near accepted. Never accept because the
    // proposed endpoint alone is near the path while accepted is not.
    constexpr double kChordTol = 0.01;   // joint rad
    constexpr double kPathTol = 0.015;  // joint rad vs line_at(cand)
    const Vec18 delta = line_commit_proposed_ - line_commit_from_;
    const double denom = delta.squaredNorm();
    bool chord_ok = false;
    double alpha = std::clamp(scale, 0.0, 1.0);
    if (denom > 1e-16) {
      alpha = std::clamp(delta.dot(accepted - line_commit_from_) / denom, 0.0,
                         1.0);
      const Vec18 on_chord = line_commit_from_ + alpha * delta;
      chord_ok = (on_chord - accepted).cwiseAbs().maxCoeff() <= kChordTol;
    } else {
      chord_ok =
          (accepted - line_commit_from_).cwiseAbs().maxCoeff() <= kChordTol;
      alpha = 0.0;
    }
    const double cand =
        line_commit_old_prog_ +
        alpha * (line_commit_new_prog_ - line_commit_old_prog_);
    const double path_res = (at(cand) - accepted).cwiseAbs().maxCoeff();
    if (chord_ok && path_res <= kPathTol) {
      last_progress_recover_.ok = true;
      last_progress_recover_.unknown = false;
      last_progress_recover_.prog = cand;
      last_progress_recover_.residual_rad = path_res;
      std::snprintf(last_progress_recover_.reason,
                    sizeof(last_progress_recover_.reason),
                    "chord_fraction_verified");
    } else {
      line_commit_pending_ = false;
      line_q_ = line_commit_from_;
      line_.prog = line_commit_old_prog_;
      stream_commit_feedback_scale_ = 0.0;
      last_progress_recover_.unknown = true;
      if (!chord_ok) {
        std::snprintf(last_progress_recover_.reason,
                      sizeof(last_progress_recover_.reason),
                      "off_chord_unverified");
      } else {
        std::snprintf(last_progress_recover_.reason,
                      sizeof(last_progress_recover_.reason),
                      "chord_path_residual");
      }
      if (line_diag_enable_) {
        char note[48];
        std::snprintf(note, sizeof(note), "commit_unknown:%.28s",
                      last_progress_recover_.reason);
        line_diag_push(LineDiagLayer::FollowCommit, accepted, line_commit_from_,
                       line_.prog, 0.0, line_.candidate.c_str(), note);
      }
      return false;
    }
  }
  line_q_ = accepted;
  line_.prog = last_progress_recover_.prog;
  line_commit_pending_ = false;
  stream_commit_feedback_scale_ = std::clamp(scale, 0.0, 1.0);
  // Freeze only after an accepted near-goal command — never on proposal.
  if ((line_.target - kin_.fk_pos(accepted, line_.frame)).norm() <=
      kLineFreezeM) {
    line_.frozen = true;
  }
  if (line_diag_enable_) {
    LineDiagEvent ev;
    ev.layer = LineDiagLayer::FollowCommit;
    ev.goal_generation = goal_generation_;
    ev.line_gen = line_gen_;
    ev.tick_id = line_commit_tick_id_;
    ev.prog = line_.prog;
    ev.s_dot = line_rate_;
    ev.commit_scale = scale;
    ev.field_mask = static_cast<uint8_t>((1u << 0) | (1u << 2));
    ev.q = accepted;
    ev.q_aux = line_commit_from_;
    ev.q_nominal = line_commit_proposed_;
    std::snprintf(ev.decision, sizeof(ev.decision), "commit_ok:%.28s",
                  last_progress_recover_.reason);
    std::snprintf(ev.candidate, sizeof(ev.candidate), "%s",
                  line_.candidate.c_str());
    ev.mono_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    ev.recv_mono_ns = ev.mono_ns;
    ev.run_id = line_diag_run_id_;
    ev.seq = ++line_diag_seq_;
    if (line_diag_ring_.empty()) {
      line_diag_ring_.assign(kLineDiagCap, LineDiagEvent{});
    }
    if (line_diag_count_ < kLineDiagCap) {
      line_diag_ring_[(line_diag_begin_ + line_diag_count_) % kLineDiagCap] = ev;
      ++line_diag_count_;
    } else {
      line_diag_ring_[line_diag_begin_] = ev;
      line_diag_begin_ = (line_diag_begin_ + 1) % kLineDiagCap;
      ++line_diag_overflows_;
    }
  }
  return true;
}

void CartesianTracker::abort_line_stream_commit() {
  if (!line_commit_pending_) {
    return;
  }
  line_.prog = line_commit_old_prog_;
  line_q_ = line_commit_from_;
  line_commit_pending_ = false;
  if (line_diag_enable_) {
    line_diag_push(LineDiagLayer::FollowCommit, line_commit_from_,
                   line_commit_proposed_, line_.prog, 0.0,
                   line_.candidate.c_str(), "commit_abort");
  }
}

void CartesianTracker::discard_line_stream_commit() {
  line_commit_pending_ = false;
}

std::vector<CartesianTracker::LineDiagEvent> CartesianTracker::line_diag_take() {
  auto out = line_diag_snapshot();
  line_diag_clear();
  return out;
}

void CartesianTracker::line_diag_splice(const std::vector<LineDiagEvent>& events) {
  if (!line_diag_enable_ || events.empty()) return;
  if (line_diag_ring_.empty()) {
    line_diag_ring_.assign(kLineDiagCap, LineDiagEvent{});
  }
  for (const LineDiagEvent& ev : events) {
    if (line_diag_count_ < kLineDiagCap) {
      line_diag_ring_[(line_diag_begin_ + line_diag_count_) % kLineDiagCap] = ev;
      ++line_diag_count_;
    } else {
      line_diag_ring_[line_diag_begin_] = ev;
      line_diag_begin_ = (line_diag_begin_ + 1) % kLineDiagCap;
      ++line_diag_overflows_;
    }
  }
}

void CartesianTracker::line_diag_push_feedback(const Vec18& q_meas, double source_stamp_s) {
  if (!line_diag_enable_) return;
  if (source_stamp_s >= 0.0 && std::isfinite(source_stamp_s)) {
    line_diag_source_ns_ = static_cast<int64_t>(std::llround(source_stamp_s * 1e9));
  } else {
    line_diag_source_ns_ = -1;
  }
  if (!q_meas.allFinite()) {
    line_diag_push(LineDiagLayer::Feedback, Vec18::Zero(), Vec18::Zero(), line_.prog,
                   0.0, line_.candidate.c_str(), "invalid");
    return;
  }
  Vec18 aux = Vec18::Zero();
  aux[0] = source_stamp_s;
  line_diag_push(LineDiagLayer::Feedback, q_meas, aux, line_.prog, 0.0,
                 line_.candidate.c_str(), "feedback");
}

void CartesianTracker::maybe_commit_last_line_cache(const Vec18& q_meas) {
  if (!reverse_last_line_enable_ || last_line_cache_committed_ || !line_.on ||
      line_.samples.size() < 2 || !line_.hold) {
    return;
  }
  if (!q_meas.allFinite()) return;
  // Only cache after the measured tip is inside the tube of the commanded end.
  if (!line_measured_arrived(q_meas) &&
      (kin_.fk_pos(q_meas, line_.frame) - line_.target).norm() > reverse_xyz_tol_) {
    return;
  }
  LastLineCache cache;
  cache.valid = true;
  cache.left = line_.left;
  cache.ori_on = line_.ori_on;
  cache.frame = line_.frame;
  cache.start_xyz = kin_.fk_pos(line_.samples.front(), line_.frame);
  cache.end_xyz = line_.target;
  cache.start_quat = rot_to_quat_xyzw(
      kin_.fk(line_.samples.front(), line_.frame).block<3, 3>(0, 0));
  cache.end_quat = line_.quat;
  cache.q_start = line_.samples.front();
  cache.q_end = line_.samples.back();
  cache.samples = line_.samples;
  cache.generation = goal_generation_;
  cache.candidate_source = line_.candidate;
  cache.direction_contract = line_.direction_contract;
  cache.direction_check = line_.direction_check;
  cache.direction_ok = line_.direction_check.ok;
  last_line_cache_ = std::move(cache);
  last_line_cache_committed_ = true;
}

bool CartesianTracker::line_target_far(const Vec3& target, const Vec4* quat) const {
  if ((target - line_.target).norm() >= kLineReplanM) {
    return true;
  }
  if (!line_.ori_on) {
    return false;
  }
  if (quat == nullptr || quat_is_identity_xyzw(quat)) {
    return false;
  }
  return ori_error_xyzw(*quat, line_.quat).norm() >= kTargetJumpOriRad;
}

Vec18 CartesianTracker::line_at(double prog) const {
  // Default: piecewise-linear. VerifiedHermite is live only after full
  // revalidation in the NewUnified profile (see line_build).
  if (line_path_interp_ == LinePathInterp::VerifiedHermite) {
    return hermite_joint_at(line_.samples, prog);
  }
  const int n = static_cast<int>(line_.samples.size());
  if (n <= 0) {
    return Vec18::Zero();
  }
  if (n == 1) {
    return line_.samples[0];
  }
  const double hi = static_cast<double>(n - 1);
  double p = prog;
  if (p < 0.0) {
    p = 0.0;
  } else if (p > hi) {
    p = hi;
  }
  const int i = static_cast<int>(std::floor(p));
  if (i >= n - 1) {
    return line_.samples[static_cast<std::size_t>(n - 1)];
  }
  const double a = p - static_cast<double>(i);
  return (1.0 - a) * line_.samples[static_cast<std::size_t>(i)] +
         a * line_.samples[static_cast<std::size_t>(i + 1)];
}

double max_arm_fold(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples, int arm) {
  if (samples.size() < 2) {
    return 0.0;
  }
  double worst = 0.0;
  for (int j = 0; j < kNArm; ++j) {
    const int idx = arm + j;
    double lo = samples.front()[idx];
    double hi = lo;
    for (const Vec18& q : samples) {
      lo = std::min(lo, q[idx]);
      hi = std::max(hi, q[idx]);
    }
    const double extra =
        (hi - lo) - std::abs(samples.back()[idx] - samples.front()[idx]);
    worst = std::max(worst, extra);
  }
  return worst;
}

bool CartesianTracker::recertify_line_start(LineRun& plan, const Vec18& q_meas) const {
  if (!plan.on || plan.samples.size() < 2) {
    return false;
  }
  std::string why;
  MotionConstraintPolicy policy;
  policy.q_min = limits_.q_min;
  policy.q_max = limits_.q_max;
  policy.torso_pitch_max_rad = torso_pitch_max_;
  policy.q_limit_margin_rad = q_limit_margin_;
  const bool ok = recertify_anchored_start(
      kin_, guards_, policy, plan.samples, q_meas, plan.target, plan.frame,
      kLineSampleTolM, &why);
  if (!ok) {
    plan.reason = std::string("anchor_recertify:") + why;
  }
  return ok;
}

bool CartesianTracker::rearm_line(uint64_t generation, const std::optional<LineTarget>& tip) {
  if (generation == 0 || generation <= goal_generation_) return false;
  line_clear();
  goal_generation_ = generation;
  line_tip_ = tip;
  last_arrived_ = false;
  return true;
}

void CartesianTracker::refuse_line(const std::string& reason) {
  // Refuse drops the active line; do not restore old prog into a cleared run.
  discard_line_stream_commit();
  line_.on = false;
  line_.hold = false;
  line_.refused = true;
  line_.reason = reason;
  last_arrived_ = false;
  line_rate_ = 0.0;
  line_rate_acc_ = 0.0;
  clear_last_line_cache();
  successor_plan_ = SuccessorPlan{};
}

bool CartesianTracker::line_needs_build(bool left, const Vec3& target, const Vec4* quat) const {
  if (line_.refused) return false;
  if (!line_.on) return true;
  return !line_.frozen && (line_.left != left || line_target_far(target, quat));
}

bool CartesianTracker::line_measured_arrived(const Vec18& q) const {
  if(!line_.hold || !q.allFinite()) return false;
  const auto pose=kin_.fk(q,line_.frame);
  return (line_.target-pose.col(3).head<3>()).norm()<=kLineSampleTolM &&
      (!line_.ori_on || ori_error_xyzw(line_.quat,
          rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm()<=arrived_ori_rad_);
}

CartesianTracker::LineStreamCheck CartesianTracker::line_stream_segment_check(
    const Vec18& from, const Vec18& to) const {
  LineStreamCheck result;
  if (!line_.on || line_.samples.empty() || !from.allFinite() || !to.allFinite())
    return result;
  result.ok = true;
  result.reason = "ok";
  const double inf = std::numeric_limits<double>::infinity();
  result.joint_margin_rad = inf;
  result.com_margin_m = inf;
  result.pitch_margin_rad = inf;
  result.tube_margin_m = inf;
  result.progress_margin_m = inf;
  const Vec3 start=kin_.fk_pos(line_.samples.front(),line_.frame);
  const Vec3 ab=line_.target-start;
  double previous=(kin_.fk_pos(from,line_.frame)-start).dot(ab);
  // Same legal pitch domain as line_validate / stream_filtered_fault:
  // ordinary start-inside → hard + round eps only (never +kStartPitchAdmitRad);
  // recovery start-outside → |start| envelope, no growth. QP must project into
  // this domain; gates do not permanently widen the box for float crumbs.
  const double start_abs=std::abs(line_.samples.front().head<3>().sum());
  const double pitch_bound=torso_pitch_max_>0
      ? pitch_abs_bound(torso_pitch_max_,start_abs,/*for_gate=*/true)
      : inf;
  // r33 field: recovery ProxQP pulls waist → EE retreats ~8e-6 m and the
  // strict progress gate killed the chord (stream_progress). Allow a bounded
  // reverse only while live pitch is outside the hard box.
  const bool pitch_recovering =
      torso_pitch_max_ > 0.0 &&
      std::abs(from.head<3>().sum()) > torso_pitch_max_ + 1e-9;
  const double progress_min =
      pitch_recovering
          ? (-kStreamRecoverProgressSlackM / std::max(ab.norm(), 1e-6))
          : 0.0;
  for(int i=1;i<=5;++i) {
    const Vec18 q=from+(i/5.0)*(to-from);
    // Hard box only. q_limit_margin_ stays a planner clamp / soft-growth rule
    // inside line_fit and line_validate. It must not reject a chord that QP
    // and the final gate already treat as inside q_min/q_max. r40 rise:
    // J1 ≈ 57.70° (soft wall) plus ~0.004° was stream_joint while still
    // ~0.3° short of the 58° hard stop.
    const double joint_margin =
        joint_hard_margin_rad(q, limits_.q_min, limits_.q_max);
    const Vec3 com = kin_.com(q);
    const double com_margin = std::min({
        com.x()-(com_xmin_+com_margin_),
        (com_xmax_-com_margin_)-com.x(),
        com.y()-(com_ymin_+com_margin_),
        (com_ymax_-com_margin_)-com.y()});
    const double pitch_margin = torso_pitch_max_>0
        ? pitch_bound-std::abs(q.head<3>().sum()) : inf;
    result.joint_margin_rad = std::min(result.joint_margin_rad,joint_margin);
    result.com_margin_m = std::min(result.com_margin_m,com_margin);
    result.pitch_margin_rad = std::min(result.pitch_margin_rad,pitch_margin);
    if (joint_margin < -kHardBoxSlackRad || !guards_.com_xy_inside(com) ||
        pitch_margin < -1e-7) {
      result.ok = false;
      result.reason = joint_margin < -kHardBoxSlackRad ? "joint" :
          !guards_.com_xy_inside(com) ? "com" : "pitch";
      result.sample_index = i;
      return result;
    }
    const Vec3 position=kin_.fk_pos(q,line_.frame);
    const double along=(position-start).dot(ab);
    const double t=std::clamp(along/std::max(ab.squaredNorm(),1e-12),0.0,1.0);
    const double tube_margin=kLineTubeHardM-(position-(start+t*ab)).norm();
    // Convert the historical dot-product progress tolerance to meters.
    const double progress_margin=(along-previous+1e-6*std::max(ab.norm(),1e-6)) /
        std::max(ab.norm(),1e-6);
    result.tube_margin_m=std::min(result.tube_margin_m,tube_margin);
    result.progress_margin_m=std::min(result.progress_margin_m,progress_margin);
    if(tube_margin<0 || progress_margin<progress_min) {
      result.ok=false;
      result.reason=tube_margin<0 ? "tube" : "progress";
      result.sample_index=i;
      return result;
    }
    previous=along;
  }
  return result;
}

bool CartesianTracker::line_stream_segment_ok(const Vec18& from,const Vec18& to) const {
  return line_stream_segment_check(from,to).ok;
}

LineFoldContrast CartesianTracker::probe_fold_contrast(
    const Vec18& q0, bool left, const Vec3& target, const Vec4* quat,
    const LineTarget* successor) const {
  (void)q0;
  (void)left;
  (void)target;
  (void)quat;
  (void)successor;
  LineFoldContrast out;
  out.fold_reason = "removed";
  out.baseline_reason = "not_a_selector";
  return out;
}

bool CartesianTracker::line_build(
    const Vec18& q0, bool left, const Vec3& target, const Vec4* quat,
    const std::string& frame) {
  const auto started = std::chrono::steady_clock::now();
  const Vec3 p0 = kin_.fk_pos(q0, frame);
  const double dist = (target-p0).norm();
  const bool ori_on = quat != nullptr && !quat_is_identity_xyzw(quat);
  const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
  line_.reason = "geometry_or_orientation";
  line_.cascade_checked=line_tip_.has_value();
  line_.cascade_passed=false;
  reverse_decision_=reverse_last_line_enable_ ? "skipped" : "disabled";
  if(torso_pitch_max_>0 &&
     std::abs(q0.head<3>().sum())>torso_pitch_max_+kStartPitchAdmitRad) {
    line_.reason="start_pitch_outside"; return false;
  }
  if (dist > kLineMaxSamples*kLineSampleM) { line_.reason="sample_limit"; return false; }
  if (line_tip_ && (line_tip_->frame != kBaseFrame || !line_tip_->xyz.allFinite() ||
      (line_tip_->ori_on && (!line_tip_->quat_xyzw.allFinite() || line_tip_->quat_xyzw.norm()<1e-6)))) {
    line_.reason="invalid_cascade_target"; return false;
  }
  if (line_tip_) {
    const double reach = (line_tip_->xyz - target).norm();
    if (reach > static_cast<double>(kLineMaxSamples) * kLineSampleM) {
      line_.reason = "cascade_unreachable";
      line_.cascade_passed = false;
      return false;
    }
  }
  std::vector<Vec18,Eigen::aligned_allocator<Vec18>> samples;
  bool locked=true;
  std::string candidate="none";
  std::string pose_report="none";
  std::string pose_decision="disabled";
  std::string reverse_decision=reverse_last_line_enable_ ? "skipped" : "disabled";
  const bool recovering_pitch=torso_pitch_max_>0 && std::abs(q0.head<3>().sum())>torso_pitch_max_+1e-6;
  if (recovering_pitch && upper_.waist_locked) { line_.reason="waist_locked_outside"; return false; }
  // A new preparation+tip probe replaces any prior chain commitment.
  if(line_tip_.has_value()) successor_plan_=SuccessorPlan{};

  bool ok = false;
  bool straighten_axis_reverted = false;
  bool have_accepted_dir = false;
  DirectionContract accepted_contract{};
  DirectionCheckResult accepted_dir{};
  bool forced = false;
  bool cascade_full_body = false;
  samples.clear();
  locked = upper_.waist_locked;
  // V2_CHAIN: production pose selector is unified_multistage_v2 only.
  // Old fold_15 / pose_pref / reverse_cache / line_path_qp posture reshape
  // are bypassed on this candidate (see docs/v2_bypass_call_sites.md).
  candidate = "unified_multistage_v2";
  pose_decision = "unified_only";
  if (reverse_last_line_enable_) {
    reverse_decision = "not_a_selector";
  }

  UnifiedPlanRequest plan_req;
  plan_req.q0 = q0;
  plan_req.q_neutral = line_q_pref_;
  plan_req.dq_max = limits_.dq_max;
  plan_req.left = left;
  plan_req.waist_locked = upper_.waist_locked;
  plan_req.frame = frame;
  plan_req.goal.xyz = target;
  plan_req.goal.ori_on = ori_on;
  if (ori_on && quat != nullptr) {
    plan_req.goal.quat = *quat;
  }
  plan_req.policy.q_min = limits_.q_min;
  plan_req.policy.q_max = limits_.q_max;
  plan_req.policy.torso_pitch_max_rad = torso_pitch_max_;
  plan_req.policy.q_limit_margin_rad = q_limit_margin_;
  plan_req.pos_tol_m = kLineSampleTolM;
  plan_req.sample_max_m = kLineSampleM;
  plan_req.max_samples = kLineMaxSamples;
  if (line_tip_) {
    plan_req.has_next = true;
    plan_req.next.xyz = line_tip_->xyz;
    plan_req.next.ori_on = line_tip_->ori_on;
    plan_req.next.quat = line_tip_->quat_xyzw;
  }
  const UnifiedPlanResult planned = plan_unified_multistage(kin_, guards_, plan_req);
  pose_report = planned.report;
  if (!planned.ok || planned.segment.size() < 2) {
    ok = false;
    line_.reason = planned.reason.empty() ? "unified_ik" : planned.reason;
    line_.failure_category = "unified_plan";
  } else {
    samples = planned.segment;
    ok = true;
    line_.reason = "unified_multistage_v2";
    accepted_contract = planned.contract;
    accepted_dir = planned.direction;
    have_accepted_dir = true;
    line_qp_follow_mode_ = LineQpFollowMode::B;
    if (planned.has_package) {
      line_.package = planned.package;
      line_.has_package = true;
    }
    if (!accepted_dir.ok) {
      ok = false;
      line_.reason = std::string("direction_search_exhausted;") + accepted_dir.reason;
      line_.failure_category = "direction_search_exhausted";
    }
    if (planned.has_successor && line_tip_) {
      successor_plan_ = SuccessorPlan{};
      successor_plan_.valid = true;
      successor_plan_.tip_xyz = line_tip_->xyz;
      successor_plan_.tip_quat = line_tip_->quat_xyzw;
      successor_plan_.tip_ori_on = line_tip_->ori_on;
      successor_plan_.tip_samples = planned.successor;
      successor_plan_.tip_contract = planned.successor_contract;
      successor_plan_.tip_check = planned.successor_direction;
      successor_plan_.generation = goal_generation_;
      successor_plan_.left = left;
      successor_plan_.frame = frame;
      successor_plan_.q_boundary = samples.back();
      cascade_full_body = true;
      forced = false;
    }
  }

  constexpr double kPitchTurnbackMax = 0.05 * kPi / 180.0;
  if (ok && !samples.empty()) {
    if (line_direction_strict_enable_) {
      if (!have_accepted_dir || !accepted_dir.ok) {
        ok = false;
        have_accepted_dir = true;
        line_.reason = std::string("direction_search_exhausted;") + accepted_dir.reason;
      }
    } else if (line_pitch_straighten_enable_ && !straighten_axis_reverted &&
               line_pitch_turnback(samples) > kPitchTurnbackMax) {
      ok = false;
      line_.reason = "pitch_turnback_unresolved";
    }
  }

  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  reverse_decision_=reverse_decision;
  if (!ok || samples.empty()) {
    // pitch_recovery_infeasible is a category only — never overwrite root_reason.
    // Keep pool_empty category so refuse logs stay searchable.
    if(line_.failure_category!="pool_empty") {
      if(recovering_pitch) {
        line_.failure_category="pitch_recovery_infeasible";
      } else if(line_.reason.find("direction_")!=std::string::npos ||
                line_.reason.find("candidate_direction_")!=std::string::npos) {
        line_.failure_category="direction_search_exhausted";
      }
    }
    if(line_.reason.empty() || line_.reason=="geometry_or_orientation") {
      if(!accepted_dir.ok && accepted_dir.reason[0]!='\0' &&
         std::strcmp(accepted_dir.reason,"ok")!=0) {
        line_.reason=std::string("direction_search_exhausted;")+accepted_dir.reason;
      }
    }
    line_.build_ms=ms;
    line_.reverse_decision=reverse_decision;
    line_.direction_contract=accepted_contract;
    line_.direction_check=accepted_dir;
    line_diag_push(LineDiagLayer::Build,q0,Vec18::Zero(),0.0,ms,candidate.c_str(),
                   reverse_decision.c_str());
    return false;
  }
  // Consumed or failed reverse intent must not be reused for a later segment.
  if(candidate=="reverse_cache" || candidate=="qa_seed" ||
     reverse_decision.rfind("rejected_",0)==0) {
    clear_last_line_cache();
  }
  LineRun built;
  built.on=true; built.left=left; built.ori_on=ori_on; built.waist_locked=locked;
  built.target=target; built.quat=ori_on?*quat:Vec4(0,0,0,1); built.frame=frame;
  built.length_m=dist; built.samples=std::move(samples); built.build_ms=ms;
  built.cascade_checked=line_tip_.has_value(); built.cascade_passed=line_tip_.has_value();
  built.cascade_forced_waist=forced;
  built.cascade_full_body=cascade_full_body;
  built.candidate=candidate;
  built.pose_report=pose_report;
  built.pose_pref_decision=pose_decision;
  built.reverse_decision=reverse_decision;
  built.direction_contract=accepted_contract;
  built.direction_check=accepted_dir;
  built.package=line_.package;
  built.has_package=line_.has_package;
  built.max_arm_fold=max_arm_fold(built.samples,arm);
  built.frozen=dist<=kLineFreezeM;
  last_path_corner_metrics_=measure_path_corners(built.samples);
  last_path_smooth_report_ = PathSmoothReport{};
  last_line_geometry_c1_ = false;
  line_path_interp_ = LinePathInterp::Linear;
  // NewUnified: verify the *actual* Hermite curve (adaptive samples), not a
  // densified polyline. Failure keeps linear + corner stop in s(t).
  if (line_chain_profile_ == LineChainProfile::NewUnified &&
      built.samples.size() >= 3) {
    const Vec3 start_xyz = kin_.fk_pos(built.samples.front(), frame);
    const Vec4 start_ori =
        rot_to_quat_xyzw(kin_.fk(built.samples.front(), frame).block<3, 3>(0, 0));
    const double start_pitch_abs =
        std::abs(built.samples.front().head<3>().sum());
    auto check_q = [&](const Vec18& q, std::string* why) -> bool {
      if (!q.allFinite()) {
        if (why) {
          *why = "nonfinite";
        }
        return false;
      }
      for (int k = 0; k < 18; ++k) {
        if (q[k] < limits_.q_min[k] - 1e-9 || q[k] > limits_.q_max[k] + 1e-9) {
          if (why) {
            *why = "joint_limit";
          }
          return false;
        }
      }
      if (!guards_.com_xy_inside(kin_.com(q))) {
        if (why) {
          *why = "com";
        }
        return false;
      }
      if (torso_pitch_max_ > 0.0) {
        const double start_ex =
            std::max(0.0, start_pitch_abs - torso_pitch_max_);
        const double ex =
            std::max(0.0, std::abs(q.head<3>().sum()) - torso_pitch_max_);
        if (ex > std::max(start_ex, kStartPitchAdmitRad) + kPitchRoundEpsRad) {
          if (why) {
            *why = "pitch";
          }
          return false;
        }
      }
      const auto pose = kin_.fk(q, frame);
      const Vec3 p = pose.col(3).head<3>();
      const Vec3 ab = target - start_xyz;
      const double ab_norm = std::max(ab.norm(), 1e-9);
      const double along_m = (p - start_xyz).dot(ab) / ab_norm;
      const double u = std::clamp(along_m / ab_norm, 0.0, 1.0);
      const double cross = (p - (start_xyz + u * ab)).norm();
      if (u > 0.98) {
        if ((p - target).norm() > kLineSampleTolM) {
          if (why) *why = "tube";
          return false;
        }
      } else if (cross > kLineSampleTolM) {
        if (why) {
          *why = "tube";
        }
        return false;
      }
      // NewUnified / v2: Hermite mid-sample ori-vs-slerp is not a hard gate.
      // Endpoint/task ori stays certified by the package; TCP tube above remains.
      if (line_chain_profile_ != LineChainProfile::NewUnified && ori_on &&
          quat &&
          ori_error_xyzw(quat_slerp_xyzw(start_ori, *quat, u),
                         rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                  .norm() > arrived_ori_rad_) {
        if (why) {
          *why = "orientation";
        }
        return false;
      }
      return true;
    };
    last_path_smooth_report_ =
        try_verified_hermite_smooth(built.samples, check_q);
    if (last_path_smooth_report_.hermite_verified &&
        last_path_smooth_report_.applied) {
      // Legacy: also require direction contract on Hermite-sampled dense knots.
      // V2_CHAIN / NewUnified: sign hard gate is retired (diagnostics only).
      bool dir_ok = true;
      if (line_direction_strict_enable_ && accepted_dir.ok &&
          line_chain_profile_ != LineChainProfile::NewUnified) {
        JointPath probe;
        const int n = static_cast<int>(built.samples.size());
        probe.reserve(static_cast<std::size_t>((n - 1) * 4 + 1));
        for (int i = 0; i < n - 1; ++i) {
          for (int k = 0; k < 4; ++k) {
            const double s =
                static_cast<double>(i) + static_cast<double>(k) / 4.0;
            probe.push_back(hermite_joint_at(built.samples, s));
          }
        }
        probe.push_back(built.samples.back());
        const auto chk = check_path_direction(probe, accepted_contract);
        if (!chk.ok) {
          dir_ok = false;
          last_path_smooth_report_.hermite_verified = false;
          last_path_smooth_report_.applied = false;
          std::snprintf(last_path_smooth_report_.reason,
                        sizeof(last_path_smooth_report_.reason),
                        "hermite_direction_fail:%s", chk.reason);
        }
      }
      if (dir_ok) {
        line_path_interp_ = LinePathInterp::VerifiedHermite;
        last_line_geometry_c1_ = last_path_smooth_report_.geometry_c1;
        last_path_corner_metrics_ = last_path_smooth_report_.after;
      }
    }
  }
  // Pack smooth/interp into LineRun so async adopt executes the same curve.
  built.path_interp = line_path_interp_;
  built.path_smooth = last_path_smooth_report_;
  built.corners = last_path_corner_metrics_;
  built.geometry_c1 = last_line_geometry_c1_;
  built.chain_profile = line_chain_profile_;
  if(line_diag_enable_ && line_shadow_diag_enable_) {
    char corner[64];
    std::snprintf(corner, sizeof(corner), "corner_j%.3f_w%.3f_k%d_%.20s",
                  last_path_corner_metrics_.max_tangent_jump_rad,
                  last_path_corner_metrics_.max_waist_turn_rad,
                  last_path_corner_metrics_.worst_knot,
                  last_path_smooth_report_.reason);
    line_diag_push(LineDiagLayer::Build, q0, built.samples.back(), 0.0, ms,
                   candidate.c_str(), corner);
  }
  line_=std::move(built); line_q_=q0; line_rate_=0.0; line_rate_acc_=0.0; ++line_gen_;
  last_line_cache_committed_=false;
  line_diag_push(LineDiagLayer::Build,q0,line_.samples.back(),0.0,ms,candidate.c_str(),
                 reverse_decision.c_str());
  return true;
}

bool CartesianTracker::follow_line(
    const Vec18& q0, const Vec3* left, const Vec3* right, const Vec4* quat_left,
    const Vec4* quat_right, std::string_view frame_left,
    std::string_view frame_right, TrackerStep& out) {
  const bool has_left = left != nullptr;
  const bool has_right = right != nullptr;
  if (has_left == has_right) {
    line_clear();
    return false;
  }
  const bool is_left = has_left;
  const Vec3& target = is_left ? *left : *right;
  const Vec4* quat = is_left ? quat_left : quat_right;
  const std::string frame = line_frame(is_left, is_left ? frame_left : frame_right);
  if (!target.allFinite() || !q0.allFinite()) {
    return false;
  }

  // Abort a stale propose before freeze/jump checks so an unconfirmed
  // line_q_ cannot latch frozen and then survive abort.
  if (line_commit_pending_) {
    abort_line_stream_commit();
  }

  if (line_.on && !line_.frozen &&
      (line_.target - kin_.fk_pos(line_q_,line_.frame)).norm() <= kLineFreezeM) line_.frozen=true;
  if (line_.frozen && !line_.refused) {
    const double pos_jump=(target-line_.target).norm();
    const double ori_jump=(quat && line_.ori_on) ? ori_error_xyzw(*quat,line_.quat).norm():0.0;
    if (pos_jump >= kLineAbortM-1e-12 || ori_jump >= kLineAbortRad-1e-12) {
      line_.abort_latched=true;
      refuse_line(pos_jump >= kLineAbortM-1e-12 ? "frozen_position_jump" : "frozen_orientation_jump");
    }
  }
  if (line_.refused) {
    last_e0_=(target-kin_.fk_pos(q0,frame)).norm();
    last_arrived_=false; out.dq.setZero(); return false;
  }
  const bool same = !line_needs_build(is_left,target,quat);
  if (!same) {
    const Vec18 start=line_reference_.value_or(q0);
    if (!line_build(start,is_left,target,quat,frame)) {
      refuse_line(line_.reason);
      out.dq.setZero(); return false;
    }
  }

  const Eigen::Matrix4d posed = kin_.fk(q0, line_.frame);
  const Vec3 p_now = posed.col(3).head<3>();
  const Vec3 e_p = line_.target - p_now;
  Vec3 e_o = Vec3::Zero();
  if (line_.ori_on) {
    e_o = ori_error_xyzw(line_.quat, rot_to_quat_xyzw(posed.block<3, 3>(0, 0)));
  }
  const double remain = e_p.norm();
  const bool ori_ok =
      !line_.ori_on || e_o.norm() < arrived_ori_rad_;

  last_e0_ = remain;
  last_e_ori_.setZero();
  if (is_left) {
    last_e_ori_.head<3>() = e_o;
    out.ee_error.head<3>() = e_p;
  } else {
    last_e_ori_.tail<3>() = e_o;
    out.ee_error.tail<3>() = e_p;
  }
  last_ori_on_ = line_.ori_on;
  last_com_ = kin_.com(q0);
  last_pitch_ = q0[0] + q0[1] + q0[2];
  last_qp_status_ = QpStatus::Success;

  if (line_.hold && same) {
    last_arrived_ = true;
    maybe_commit_last_line_cache(q0);
    out.dq.setZero();
    return true;
  }

  const double nseg =
      line_.samples.size() <= 1
          ? 0.0
          : static_cast<double>(line_.samples.size() - 1);
  // 步长看路径自身的增量，不看 q0。q0 偏了也不把方向拉去追它。
  // 末端时间律：linear 旧式线性；sqrt 在相同 floor 下平均更快、仍连续降到 floor。
  // 进度已到终点后必须 want=0，否则 floor 会把 rate 顶在 hold 门槛之上永远到不了点。
  double brake = 1.0;
  const double path_remain = line_.length_m * (1.0-line_.prog/std::max(nseg,1.0));
  const double brake_m = line_brake_distance_m_;
  const bool prog_at_end = nseg <= 0.0 || line_.prog >= nseg - 1e-4;
  const double now_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
  if (line_seg_start_s_ < 0.0) {
    line_seg_start_s_ = now_s;
  }
  if (prog_at_end) {
    brake = 0.0;
  } else if (path_remain < brake_m) {
    double scale = path_remain / std::max(brake_m, 1e-9);
    if (line_brake_profile_ == LineBrakeProfile::Sqrt) {
      scale = std::sqrt(std::max(0.0, scale));
    }
    brake = std::max(line_brake_floor_, std::min(1.0, scale));
    if (line_tail_enter_brake_s_ < 0.0) {
      line_tail_enter_brake_s_ = now_s;
    }
  }
  if (path_remain < 0.02 && line_tail_enter_2cm_s_ < 0.0) {
    line_tail_enter_2cm_s_ = now_s;
  }
  const double speed = line_speed_scale_;
  double want = speed * brake;
  last_line_path_remain_m_ = path_remain;
  last_line_brake_ = brake;
  last_line_want_ = want;
  // 加减速只乘在路径进度这一个标量上，再去采样已经求好的折线。不能按关节各做一条 S 曲线，
  // 否则手会离开直线，预备点/抓取点上会重新出现走过再折返。
  if (speed <= 1e-4) {
    line_rate_ = 0.0;
    line_rate_acc_ = 0.0;
  } else if (line_pace_mode_ == LinePaceMode::UnifiedTimeLaw) {
    TimeLawLimits lim;
    lim.step_cap = effective_dq_max() * constraints_.dt();
    lim.dq_max = effective_dq_max();
    lim.speed_scale = speed;
    lim.path_remain_m = path_remain;
    lim.brake_distance_m = line_brake_distance_m_;
    lim.brake_floor = line_brake_floor_;
    lim.corner_remain_s = next_corner_remain(
        line_.prog, last_path_corner_metrics_, /*jump_thresh_rad=*/0.02);
    {
      const double end_s =
          static_cast<double>(std::max<std::size_t>(1, line_.samples.size()) - 1);
      lim.end_remain_s = std::max(0.0, end_s - line_.prog);
    }
    lim.feedback_scale = stream_commit_feedback_scale_;
    lim.rate = line_rate_;
    lim.rate_acc = line_rate_acc_;
    lim.accel_s = line_accel_s_;
    lim.jerk_s = line_jerk_s_;
    lim.dt = constraints_.dt();
    lim.geometry_c1 = last_line_geometry_c1_;
    if (prog_at_end) {
      line_rate_ = 0.0;
      line_rate_acc_ = 0.0;
      want = 0.0;
    } else {
      const auto sample = unified_time_law_step(
          [this](double s) { return line_at(s); }, line_.prog, lim);
      line_rate_ = sample.rate;
      line_rate_acc_ = sample.rate_acc;
      want = sample.want;
      last_line_brake_ = sample.brake;
    }
    last_line_want_ = want;
  } else if (line_pace_mode_ == LinePaceMode::R10bFirstOrder) {
    // Local restore of 08e16ce first-order tracking of want. rate_acc is N/A.
    line_rate_acc_ = 0.0;
    if (!(line_accel_s_ > 1e-3)) {
      line_rate_ = want;
    } else {
      const double du = constraints_.dt() / line_accel_s_;
      if (line_rate_ < want) {
        line_rate_ = std::min(want, line_rate_ + du);
      } else {
        line_rate_ = std::max(want, line_rate_ - du);
      }
    }
  } else if (!(line_accel_s_ > 1e-3)) {
    line_rate_ = want;
    line_rate_acc_ = 0.0;
  } else {
    const double dt = std::max(constraints_.dt(), 1e-4);
    const double a_bar = 1.0 / line_accel_s_;
    const double j_bar = a_bar / std::max(line_jerk_s_, 1e-3);
    const auto clip = [](double x, double lim) { return std::clamp(x, -lim, lim); };
    const double a_des = clip((want - line_rate_) / dt, a_bar);
    const double j_cmd = clip((a_des - line_rate_acc_) / dt, j_bar);
    line_rate_acc_ = clip(line_rate_acc_ + j_cmd * dt, a_bar);
    double v = line_rate_ + line_rate_acc_ * dt;
    if (v < 0.0) {
      v = 0.0;
      line_rate_acc_ = 0.0;
    }
    if (v > line_speed_scale_) {
      v = line_speed_scale_;
      if (line_rate_acc_ > 0.0) {
        line_rate_acc_ = 0.0;
      }
    }
    line_rate_ = v;
  }
  // Stale propose already aborted at follow_line entry (before freeze latch).
  const Vec18 q_from = line_reference_.value_or(line_q_);
  const Vec18 step_lim = effective_dq_max() * constraints_.dt() * line_rate_;
  const double old_prog = line_.prog;
  if (line_rate_ > 1e-4 && nseg > line_.prog) {
    const Vec18 origin=line_at(old_prog)-line_q_+q_from;
    line_.prog=bounded_path_progress(old_prog,nseg,origin,step_lim,
        [this](double s){return line_at(s);});
  }

  const bool at_end=nseg<=0.0 || line_.prog>=nseg-1e-4;
  // If the previously accepted command already reaches the target, wait for
  // measured feedback to catch up instead of running a new QP at the final
  // path point. That QP may have no forward progress left and can propose a
  // chord outside the tube while the servo is merely lagging its command.
  const auto ref_pose=at_end?kin_.fk(q_from,line_.frame):Eigen::Matrix4d::Identity();
  const bool ref_at_goal=at_end &&
      (ref_pose.col(3).head<3>()-line_.target).norm()<=kLineSampleTolM &&
      (!line_.ori_on ||
       ori_error_xyzw(line_.quat,
          rot_to_quat_xyzw(ref_pose.block<3,3>(0,0))).norm()<=arrived_ori_rad_);
  if(ref_at_goal) {
    // Path parameter is exhausted and the command is on the goal. 3.3 mm vs the
    // 3 mm sample tol must finish the line, not refuse it (r28).
    constexpr double kPathDoneArriveM = 0.005;
    if (remain > kPathDoneArriveM || !ori_ok) {
      constexpr int kTailFeedbackWaitTicks=50; // 1 s at the 50 Hz tracker rate.
      if(++line_.tail_feedback_wait_ticks>kTailFeedbackWaitTicks) {
        refuse_line("tail_feedback_not_arrived");
        out.dq.setZero();return false;
      }
      line_q_=q_from;
      out.dq.setZero();return true;
    }
    line_.tail_feedback_wait_ticks=0;
    // 人已经在终点公差里。速度还没收完也不再解 QP，避免在预备点/抓取点上折返微调。
    if(line_rate_<=0.1*line_speed_scale_+1e-9) {
      line_.hold=true;line_q_=q_from;line_rate_=0;line_rate_acc_=0;last_arrived_=true;
      if (line_tail_hold_s_ < 0.0) {
        line_tail_hold_s_ = std::chrono::duration<double>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
      }
      maybe_commit_last_line_cache(q0);
      line_diag_push(LineDiagLayer::FollowCommit,q_from,q0,line_.prog,0.0,
                     line_.candidate.c_str(),"hold");
    } else {
      line_q_=q_from;
    }
    out.dq.setZero();return true;
  }
  line_.tail_feedback_wait_ticks=0;
  Vec18 next = q_from;
  // V2_CHAIN: NewUnified follow uses path samples + time law only.
  // Do NOT call line_path_qp as a posture reshape selector (QP-A/B/C).
  // Legacy offline/profile still uses line_path_qp.
  const bool v2_package_follow =
      line_chain_profile_ == LineChainProfile::NewUnified;
  if (v2_package_follow) {
    if (line_rate_ > 1e-4) {
      next = line_at(line_.prog);
      if (!next.allFinite()) {
        line_.prog = old_prog;
        line_commit_pending_ = false;
        line_diag_push(LineDiagLayer::FollowQp, q_from, q_from, line_.prog, 0.0,
                       line_.candidate.c_str(), "v2_sample_nonfinite");
        refuse_line("v2_sample_nonfinite");
        out.dq.setZero();
        return false;
      }
    }
    line_diag_push(LineDiagLayer::FollowQp, next, q_from, line_.prog, 0.0,
                   line_.candidate.c_str(), "v2_sample");
  } else if (line_rate_ > 1e-4 && !line_path_qp(q_from,line_.prog,step_lim,next)) {
    line_.prog=old_prog;
    line_commit_pending_=false;
    line_diag_push(LineDiagLayer::FollowQp,q_from,q_from,line_.prog,0.0,
                   line_.candidate.c_str(),"qp_fail");
    refuse_line(line_.reason.empty()?"path_qp_or_tube":line_.reason); out.dq.setZero(); return false;
  } else {
    line_diag_push(LineDiagLayer::FollowQp,next,q_from,line_.prog,0.0,
                   line_.candidate.c_str(),"qp");
  }
  const Vec18 dq=next-q_from;
  // P2: propose → stream accept → confirm. With requires_stream, progress stays
  // tentative until the export confirms; offline compute confirms immediately.
  // Do NOT freeze on proposal — abort/partial must not leave an early freeze.
  line_commit_pending_ = true;
  line_commit_tick_id_ = control_tick_id_.load(std::memory_order_acquire);
  line_commit_old_prog_ = old_prog;
  line_commit_new_prog_ = line_.prog;
  line_commit_from_ = q_from;
  line_commit_proposed_ = next;
  line_q_ = next;
  last_arrived_ = false;
  line_diag_push(LineDiagLayer::FollowCommit,next,q0,line_.prog,0.0,
                 line_.candidate.c_str(),"propose");
  if (!line_commit_requires_stream_) {
    if (!confirm_line_stream_commit(next, 1.0)) {
      line_.prog = old_prog;
      line_q_ = q_from;
      refuse_line(std::string("commit_progress_") +
                  last_progress_recover_.reason);
      out.dq.setZero();
      return false;
    }
  }
  out.dq = dq;
  return true;
}

}  // namespace kmpc
