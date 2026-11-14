// 单臂够点：整段直线一次求完，之后只沿这条路径推进度。
// 变量是四个独立腰关节加这条臂。另一条臂保持出发角度。
// 步长用直线自己的速度比例。单臂解不出就停住，不进跟踪。双臂、标定仍走 QP。

#include "kmpc/cartesian_tracker.hpp"
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
// Candidate-pool spacing (2×). Winner is densified once at kLineSampleM.
constexpr double kLineSamplePoolM = 0.010;
constexpr double kLineReplanM = 0.008;
// Freeze / abort-near-goal distance. Brake distance is a separate runtime param.
constexpr double kLineFreezeM = 0.08;
constexpr double kLineSampleTolM = 0.003;
// Mid-path line is a soft tube. The hand may leave the straight segment while
// the waist folds, then rejoin by the endpoint. It must not be pulled back by
// reversing an arm joint. Endpoint accuracy stays kLineSampleTolM.
constexpr double kLineTubeM = 0.020;
// Past this, the bow is no longer a soft leave-and-rejoin. Inside it, a
// smooth path is kept even if the arm would have to reverse to get tighter.
constexpr double kLineTubeHardM = 0.20;
// 中间点姿态相对投影点的上限。端点仍用 arrived_ori_rad_。
// R18 预备在软管里测到的中间姿态偏差约 0.31 rad，留一点余量。
constexpr double kLineOriHardRad = 0.40;
constexpr double kLineSolveTolM = 0.0015;
// Proximal arm (shoulder/elbow) pull back to the joint chord, in the same
// units as kWArm. Large enough that a Cartesian residual cannot walk those
// joints off the chord and back. Wrist stays soft for orientation.
constexpr double kArmChordW = 0.64;
constexpr double kLineAbortM = 0.05;
constexpr double kLineAbortRad = 15.0 * kPi / 180.0;
constexpr double kLineJointJump = 0.80;
constexpr int kLineMaxSamples = 400;
constexpr int kRed = kNWaist + kNArm;
constexpr double kChanCap = 50.0;
constexpr double kWPitch = 0.20 * 0.20;
constexpr double kWYaw = 0.40 * 0.40;
constexpr double kWArm = 0.04 * 0.04;
constexpr double kWWrist = 0.12 * 0.12;

using Red = Eigen::Matrix<double, kRed, 1>;
using MatR = Eigen::Matrix<double, kRed, kRed>;
template <typename T>
using Aligned = std::vector<T, Eigen::aligned_allocator<T>>;

double chan_gain(double q, double lo, double hi, double dq) {
  const double span = hi - lo;
  if (!(span > 1e-4) || !std::isfinite(dq)) {
    return 1.0;
  }
  constexpr double kPad = 1e-4;
  const double qc = std::min(hi - kPad, std::max(lo + kPad, q));
  const double den = (hi - qc) * (qc - lo);
  if (!(den > 1e-8)) {
    return kChanCap;
  }
  const double dH = 0.25 * span * span * (2.0 * qc - hi - lo) / (den * den);
  if (!(dH * dq > 0.0)) {
    return 1.0;
  }
  return std::min(1.0 + std::abs(dH), kChanCap);
}

bool solve_block(Aligned<MatR>& diag, Aligned<MatR>& upper, Aligned<Red>& rhs,
                 Aligned<Red>& step) {
  const int n = static_cast<int>(diag.size());
  if (n <= 0 || static_cast<int>(upper.size()) != n - 1 ||
      static_cast<int>(rhs.size()) != n) {
    return false;
  }
  Aligned<MatR> cond(static_cast<std::size_t>(n));
  Aligned<MatR> cinv_u(static_cast<std::size_t>(std::max(n - 1, 0)));
  Aligned<Red> d = rhs;
  cond[0] = diag[0];
  for (int i = 0; i < n - 1; ++i) {
    const Eigen::LDLT<MatR> ldlt(cond[static_cast<std::size_t>(i)]);
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    cinv_u[static_cast<std::size_t>(i)] = ldlt.solve(upper[static_cast<std::size_t>(i)]);
    cond[static_cast<std::size_t>(i + 1)] =
        diag[static_cast<std::size_t>(i + 1)] -
        upper[static_cast<std::size_t>(i)].transpose() * cinv_u[static_cast<std::size_t>(i)];
    d[static_cast<std::size_t>(i + 1)] -=
        upper[static_cast<std::size_t>(i)].transpose() *
        ldlt.solve(d[static_cast<std::size_t>(i)]);
  }
  step.resize(static_cast<std::size_t>(n));
  {
    const Eigen::LDLT<MatR> ldlt(cond[static_cast<std::size_t>(n - 1)]);
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    step[static_cast<std::size_t>(n - 1)] = ldlt.solve(d[static_cast<std::size_t>(n - 1)]);
  }
  for (int i = n - 2; i >= 0; --i) {
    const Eigen::LDLT<MatR> ldlt(cond[static_cast<std::size_t>(i)]);
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    step[static_cast<std::size_t>(i)] =
        ldlt.solve(d[static_cast<std::size_t>(i)]) -
        cinv_u[static_cast<std::size_t>(i)] * step[static_cast<std::size_t>(i + 1)];
  }
  return true;
}

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
    // NewUnified owns pace/QP: unified s(t) + QP-B. Explicit profile switch
    // overrides prior single-parameter settings.
    line_pace_mode_ = LinePaceMode::UnifiedTimeLaw;
    line_qp_follow_mode_ = LineQpFollowMode::B;
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

bool CartesianTracker::line_arm_only(
    const Vec18& q0, int arm_begin, const Vec3& target, const Vec4* quat, bool ori_on,
    const std::string& frame,
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples) const {
  const int idle =
      arm_begin == kSliceLeft.begin ? kSliceRight.begin : kSliceLeft.begin;
  const Vec3 p0 = kin_.fk_pos(q0, frame);
  const double dist = (target - p0).norm();
  int n = 1;
  if (dist > kLineSampleM) {
    n = static_cast<int>(std::ceil(dist / kLineSampleM));
  }
  if (n > kLineMaxSamples) {
    return false;
  }
  Vec4 q_ori0(0.0, 0.0, 0.0, 1.0);
  Vec4 q_ori1 = q_ori0;
  if (ori_on && quat != nullptr) {
    q_ori0 = rot_to_quat_xyzw(kin_.fk(q0, frame).block<3, 3>(0, 0));
    q_ori1 = *quat;
  }
  samples.clear();
  Vec18 q = q0;
  samples.push_back(q);
  for (int i = 1; i <= n; ++i) {
    const double s = static_cast<double>(i) / static_cast<double>(n);
    const Vec3 p_des = (1.0 - s) * p0 + s * target;
    const Vec4 q_s = quat_slerp_xyzw(q_ori0, q_ori1, s);
    const Vec18 q_before = q;
    auto ik = [&](bool use_ori) {
      for (int it = 0; it < 60; ++it) {
        const Eigen::Matrix4d posed = kin_.fk(q, frame);
        const Vec3 e_p = p_des - posed.col(3).head<3>();
        Vec3 e_o = Vec3::Zero();
        if (use_ori) {
          e_o = ori_error_xyzw(q_s, rot_to_quat_xyzw(posed.block<3, 3>(0, 0)));
        }
        if (e_p.norm() < kLineSolveTolM && (!use_ori || e_o.norm() < 0.05)) {
          return true;
        }
        Mat6x18 jac = kin_.jacobian_spatial(q, frame);
        constexpr double kOri = 0.15;
        Eigen::Matrix<double, 6, 1> err;
        err.head<3>() = e_p;
        err.tail<3>() = Vec3::Zero();
        if (use_ori) {
          err.tail<3>() = kOri * e_o;
        }
        if (use_ori) {
          jac.bottomRows<3>() *= kOri;
        } else {
          jac.bottomRows<3>().setZero();
        }
        jac.leftCols<kNWaist>().setZero();
        jac.middleCols<kNArm>(idle).setZero();
        Mat18 gram = jac.transpose() * jac;
        Vec18 damp = Vec18::Constant(kWArm);
        damp.segment<2>(kSliceRight.begin + 5).setConstant(kWWrist);
        damp.segment<2>(kSliceLeft.begin + 5).setConstant(kWWrist);
        damp.head<kNWaist>().setConstant(1.0e3);
        damp.segment<kNArm>(idle).setConstant(1.0e3);
        gram.diagonal() += damp;
        const Eigen::LDLT<Mat18> ldlt(gram);
        if (ldlt.info() != Eigen::Success) {
          return false;
        }
        Vec18 dq = ldlt.solve(jac.transpose() * err);
        const double peak = dq.cwiseAbs().maxCoeff();
        if (peak > 0.08) {
          dq *= 0.08 / peak;
        }
        Vec18 qn = q + dq;
        for (int j = 0; j < kNDof; ++j) {
          const double lo = limits_.q_min[j] + q_limit_margin_;
          const double hi = limits_.q_max[j] - q_limit_margin_;
          if (hi > lo) {
            qn[j] = std::clamp(qn[j], lo, hi);
          }
        }
        qn.head<kNWaist>() = q0.head<kNWaist>();
        qn.segment<kNArm>(idle) = q0.segment<kNArm>(idle);
        if (!qn.allFinite()) {
          return false;
        }
        q = qn;
      }
      const auto pose = kin_.fk(q, frame);
      return (p_des - pose.col(3).head<3>()).norm() < kLineSolveTolM &&
          (!use_ori || ori_error_xyzw(q_s, rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm() < 0.05);
    };
    const bool step_ok = ik(ori_on);
    if (!step_ok) {
      return false;
    }
    if ((q - q_before).cwiseAbs().maxCoeff() > kLineJointJump) {
      return false;
    }
    samples.push_back(q);
  }
  return line_validate(samples, target, quat, ori_on, frame);
}

bool CartesianTracker::line_fit(
    const Vec18& q0, int arm_begin, const Vec3& target, const Vec4* quat, bool ori_on,
    bool lock_waist, const std::string& frame,
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples,
    bool& waist_locked, bool allow_relock, const Vec18* goal_seed,
    const Vec18* pose_pref, double sample_step_m) const {
  line_fit_straighten_axis_reverted_ = false;
  const Vec3 p0 = kin_.fk_pos(q0, frame);
  const double dist = (target - p0).norm();
  const double step =
      (sample_step_m > 1e-9) ? sample_step_m : kLineSampleM;
  int n = 1;
  if (dist > step) {
    n = static_cast<int>(std::ceil(dist / step));
  }
  if (n > kLineMaxSamples) {
    return false;
  }
  if (n < 1) {
    n = 1;
  }
  const double margin = q_limit_margin_;

  // Three pitch joints have different pivots. Keep their translation degrees
  // of freedom while constraining the sum; equal increments are not equivalent.
  auto pack = [&](const Vec18& q) {
    Red y;
    y.head<kNWaist>()=q.head<kNWaist>();
    y.template segment<kNArm>(kNWaist)=q.segment<kNArm>(arm_begin);
    return y;
  };
  auto expand = [&](const Red& y) {
    Vec18 q=q0;
    q.head<kNWaist>()=y.head<kNWaist>();
    q.segment<kNArm>(arm_begin)=y.template segment<kNArm>(kNWaist);
    return q;
  };
  auto clamp_y = [&](Red& y) {
    Red lo,hi;
    lo.head<kNWaist>()=limits_.q_min.head<kNWaist>().array()+margin;
    hi.head<kNWaist>()=limits_.q_max.head<kNWaist>().array()-margin;
    lo.template segment<kNArm>(kNWaist)=limits_.q_min.segment<kNArm>(arm_begin).array()+margin;
    hi.template segment<kNArm>(kNWaist)=limits_.q_max.segment<kNArm>(arm_begin).array()-margin;
    if(yaw_max_>0) {
      lo[kIdxYaw]=std::max(lo[kIdxYaw],q_yaw_pref_-yaw_max_);
      hi[kIdxYaw]=std::min(hi[kIdxYaw],q_yaw_pref_+yaw_max_);
    }
    y=y.cwiseMax(lo).cwiseMin(hi);
    if(torso_pitch_max_>0 && std::abs(y.head<3>().sum())>torso_pitch_max_) {
      const double desired=std::clamp(y.head<3>().sum(),-torso_pitch_max_,torso_pitch_max_);
      const auto before=y.head<3>().eval();
      double low=-10,high=10;
      for(int it=0;it<45;++it) {
        const double shift=(low+high)*.5;
        const double total=(before.array()+shift).max(lo.head<3>().array()).min(hi.head<3>().array()).sum();
        if(total<desired) low=shift;else high=shift;
      }
      y.head<3>()=(before.array()+(low+high)*.5).max(lo.head<3>().array()).min(hi.head<3>().array());
    }
    if(lock_waist) y.head<kNWaist>()=q0.head<kNWaist>();
  };
  auto weights = [&](const Red& y_prev,const Red& y) {
    const Vec18 qp=expand(y_prev),qc=expand(y);
    Red w;
    for(int k=0;k<kNWaist;++k) {
      const double base=k==kIdxYaw?kWYaw:kWPitch;
      w[k]=base*chan_gain(qc[k],limits_.q_min[k]+margin,limits_.q_max[k]-margin,qc[k]-qp[k]);
    }
    for(int i=0;i<kNArm;++i) {
      const int j=arm_begin+i;
      w[kNWaist+i]=(i>=5?kWWrist:kWArm)*chan_gain(qc[j],limits_.q_min[j]+margin,limits_.q_max[j]-margin,qc[j]-qp[j]);
    }
    return w;
  };
  Vec4 q_ori0(0.0, 0.0, 0.0, 1.0);
  Vec4 q_ori1 = q_ori0;
  if (ori_on && quat != nullptr) {
    const Eigen::Matrix4d posed = kin_.fk(q0, frame);
    q_ori0 = rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
    q_ori1 = *quat;
  }
  Aligned<Vec3> p_des(static_cast<std::size_t>(n + 1));
  Aligned<Vec4> quat_des(static_cast<std::size_t>(n + 1));
  for (int i = 0; i <= n; ++i) {
    const double s = static_cast<double>(i) / static_cast<double>(n);
    p_des[static_cast<std::size_t>(i)] = (1.0 - s) * p0 + s * target;
    quat_des[static_cast<std::size_t>(i)] =
        ori_on ? quat_slerp_xyzw(q_ori0, q_ori1, s) : q_ori0;
  }

  auto task = [&](const Red& y, int index, Eigen::Matrix<double, 6, kRed>& jac,
                  Eigen::Matrix<double, 6, 1>& err, double& ori_rad) {
    const Vec18 q = expand(y);
    const Mat6x18 full = kin_.jacobian_spatial(q, frame);
    jac.setZero();
    jac.leftCols<kNWaist>()=full.leftCols<kNWaist>();
    for (int a = 0; a < kNArm; ++a) {
      jac.col(kNWaist + a) = full.col(arm_begin + a);
    }
    err.setZero();
    err.head<3>() = p_des[static_cast<std::size_t>(index)] - kin_.fk_pos(q, frame);
    ori_rad = 0.0;
    if (ori_on) {
      constexpr double kOri = 0.15;
      const Eigen::Matrix4d posed = kin_.fk(q, frame);
      const Vec3 eo = ori_error_xyzw(
          quat_des[static_cast<std::size_t>(index)],
          rot_to_quat_xyzw(posed.block<3, 3>(0, 0)));
      ori_rad = eo.norm();
      err.tail<3>() = kOri * eo;
      jac.bottomRows<3>() *= kOri;
    }
  };
  auto worst_pos = [&](const Aligned<Red>& ys) {
    double worst = 0.0;
    for (int i = 0; i <= n; ++i) {
      const double e =
          (p_des[static_cast<std::size_t>(i)] - kin_.fk_pos(expand(ys[static_cast<std::size_t>(i)]), frame))
              .norm();
      worst = std::max(worst, e);
      if(ori_on) {
        const auto pose=kin_.fk(expand(ys[static_cast<std::size_t>(i)]),frame);
        worst=std::max(worst,0.03*ori_error_xyzw(quat_des[static_cast<std::size_t>(i)],
            rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm());
      }
    }
    return worst;
  };
  const Vec3 path_ab = target - p0;
  const double path_ab_norm = std::max(path_ab.norm(), 1e-9);
  auto cross_track = [&](const Red& y) {
    const Vec3 p = kin_.fk_pos(expand(y), frame);
    const double ab2 = std::max(path_ab.squaredNorm(), 1e-12);
    const double u = std::clamp((p - p0).dot(path_ab) / ab2, 0.0, 1.0);
    return (p - (p0 + u * path_ab)).norm();
  };
  auto along_m = [&](const Red& y) {
    return (kin_.fk_pos(expand(y), frame) - p0).dot(path_ab) / path_ab_norm;
  };
  auto keep_com = [&](Aligned<Red>& ys) {
    // Prefer a scaled waist step over an immediate full restore to the previous
    // sample. Blind restore creates local waist plateaus that look like "waist
    // never moves" even when free_fit was selected.
    for (int i = 1; i <= n; ++i) {
      Red& yi = ys[static_cast<std::size_t>(i)];
      const Red y_prev = ys[static_cast<std::size_t>(i - 1)];
      if (guards_.com_xy_inside(kin_.com(expand(yi)))) {
        continue;
      }
      bool fixed = false;
      for (double a : {0.75, 0.5, 0.25}) {
        Red trial = yi;
        trial.head<kNWaist>() =
            y_prev.head<kNWaist>() +
            a * (yi.head<kNWaist>() - y_prev.head<kNWaist>());
        if (guards_.com_xy_inside(kin_.com(expand(trial)))) {
          yi.head<kNWaist>() = trial.head<kNWaist>();
          fixed = true;
          break;
        }
      }
      if (!fixed) {
        yi.head<kNWaist>() = y_prev.head<kNWaist>();
      }
    }
  };

  auto limited_step = [&](const Red& y, const Red& dy) {
    Red next = y + dy;
    clamp_y(next);
    if (!guards_.com_xy_inside(kin_.com(expand(next)))) {
      next.head<kNWaist>()=y.head<kNWaist>();
    }
    Red applied = next - y;
    const double peak=applied.cwiseAbs().maxCoeff();
    if (peak > 0.12) {
      next = y + applied * (0.12 / peak);
      clamp_y(next);
      if (!guards_.com_xy_inside(kin_.com(expand(next)))) {
        next.head<kNWaist>()=y.head<kNWaist>();
      }
    }
    return next;
  };

  // Pitch-off-wall stays in the endpoint pose polish below. This pull is only
  // the optional shoulder/elbow preference, and only when that preference exists.
  auto refine_pose = [&](Red& y, int index) {
    if (pose_pref == nullptr) {
      return;
    }
    constexpr int kJ2 = 1;
    constexpr int kJ4 = 3;
    for (int it = 0; it < 24; ++it) {
      Eigen::Matrix<double, 6, kRed> jac;
      Eigen::Matrix<double, 6, 1> err;
      double ori_rad = 0.0;
      task(y, index, jac, err, ori_rad);
      if (err.head<3>().norm() > kLineSolveTolM || (ori_on && ori_rad > 0.08)) {
        break;
      }
      Red desire = Red::Zero();
      desire[kNWaist + kJ2] = (*pose_pref)[arm_begin + kJ2] - y[kNWaist + kJ2];
      desire[kNWaist + kJ4] = (*pose_pref)[arm_begin + kJ4] - y[kNWaist + kJ4];
      if (desire.norm() < 1e-4) {
        break;
      }
      Eigen::Matrix<double, 6, 6> jjt = jac * jac.transpose();
      jjt.diagonal().array() += 1e-6;
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(jjt);
      if (ldlt.info() != Eigen::Success) {
        break;
      }
      const Eigen::Matrix<double, kRed, 6> j_pinv =
          jac.transpose() * ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
      // Allow waist in the nullspace; acceptance still gates waist path budget.
      const Red dy =
          j_pinv * err + (MatR::Identity() - j_pinv * jac) * (line_pose_pref_w_ * desire);
      const Red next = limited_step(y, dy);
      Eigen::Matrix<double, 6, kRed> jac2;
      Eigen::Matrix<double, 6, 1> err2;
      double ori2 = 0.0;
      task(next, index, jac2, err2, ori2);
      if (err2.head<3>().norm() > kLineSolveTolM || (ori_on && ori2 > 0.08)) {
        break;
      }
      if ((next - y).norm() < 1e-5) {
        break;
      }
      y = next;
    }
  };

  Red y_goal = pack(goal_seed?*goal_seed:q0);
  clamp_y(y_goal);
  double w_scale = 1.0;
  double best_pos = 1e9;
  int stall = 0;
  for (int it = 0; it < 80; ++it) {
    Eigen::Matrix<double, 6, kRed> jac;
    Eigen::Matrix<double, 6, 1> err;
    double ori_rad = 0.0;
    task(y_goal, n, jac, err, ori_rad);
    const double pos = err.head<3>().norm();
    if (pos < kLineSolveTolM && (!ori_on || ori_rad < 0.05)) {
      break;
    }
    if (pos < best_pos - 0.001) {
      best_pos = pos;
      stall = 0;
    } else if (++stall > 12 && pos > kLineSampleTolM && w_scale > 0.05) {
      w_scale *= 0.2;
      stall = 0;
    }
    const Red w = weights(y_goal, y_goal) * w_scale;
    MatR gram = jac.transpose() * jac;
    gram.diagonal() += w;
    gram.diagonal().array() += 1e-8;
    const Eigen::LDLT<MatR> ldlt(gram);
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    const Red dy = ldlt.solve(jac.transpose() * err);
    const Red next = limited_step(y_goal, dy);
    if (!next.allFinite()) {
      break;
    }
    if ((next - y_goal).norm() < 1e-5) {
      if (err.head<3>().norm() > kLineSampleTolM && w_scale > 0.05) {
        w_scale *= 0.2;
        continue;
      }
      break;
    }
    y_goal = next;
  }
  refine_pose(y_goal, n);
  // Hand is already on the target. If the target is lower, keep moving the
  // waist in the hand's nullspace: drop the shoulder, don't arch forward, and
  // don't drive pitch onto the wall. High targets do not get this pull.
  // LoggedPrep once ended at foldx≈+61 mm while the shoulder dropped — keep
  // rules must not accept a step that makes sit-back worse.
  if (!lock_waist && target.z() < p0.z() - 0.03) {
    const double foldx_start =
        kin_.fk(expand(pack(q0)), "Leg_Y_Link2")(0, 3);
    for (int it = 0; it < 20; ++it) {
      const double z0 = kin_.fk(expand(y_goal), "Body_Link")(2, 3);
      const double x0 = kin_.fk(expand(y_goal), "Body_Link")(0, 3);
      const double f0 = kin_.fk(expand(y_goal), "Leg_Y_Link2")(0, 3);
      const double pitch = y_goal.head<3>().sum();
      const double foldx_now = f0 - foldx_start;
      const bool need_sit = foldx_now > -0.02;
      Red grad = Red::Zero();
      for (int k = 0; k < 3; ++k) {
        Red y2 = y_goal;
        y2[k] += 1e-4;
        clamp_y(y2);
        const double dz = (kin_.fk(expand(y2), "Body_Link")(2, 3) - z0) / 1e-4;
        const double dx = (kin_.fk(expand(y2), "Body_Link")(0, 3) - x0) / 1e-4;
        const double df = (kin_.fk(expand(y2), "Leg_Y_Link2")(0, 3) - f0) / 1e-4;
        // Drop the shoulder, penalize forward arch, and sit the fold back.
        // When sit-back is still short, weight foldx harder than height.
        const double w_df = need_sit ? 2.5 : 1.2;
        const double w_dz = need_sit ? 0.4 : 1.0;
        grad[k] = w_dz * dz + std::max(0.0, dx) + w_df * df;
      }
      const double comfort = 20.0 * kPi / 180.0;
      if (std::abs(pitch) > comfort) {
        const double sign = pitch > 0.0 ? 1.0 : -1.0;
        grad[0] += 0.2 * sign;
        grad[1] += 0.2 * sign;
        grad[2] += 0.2 * sign;
      }
      Eigen::Matrix<double, 6, kRed> jac;
      Eigen::Matrix<double, 6, 1> err;
      double ori_rad = 0.0;
      task(y_goal, n, jac, err, ori_rad);
      if (err.head<3>().norm() > kLineSampleTolM || (ori_on && ori_rad > 0.08)) {
        break;
      }
      Eigen::Matrix<double, 6, 6> jjt = jac * jac.transpose();
      jjt.diagonal().array() += 1e-6;
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(jjt);
      if (ldlt.info() != Eigen::Success) {
        break;
      }
      const Eigen::Matrix<double, kRed, 6> j_pinv =
          jac.transpose() * ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
      Red dir = -(MatR::Identity() - j_pinv * jac) * grad;
      const double nrm = dir.norm();
      if (!(nrm > 1e-6)) {
        break;
      }
      dir *= std::min(0.04, 0.08 / nrm);
      const Red next = limited_step(y_goal, dir);
      Eigen::Matrix<double, 6, kRed> jac2;
      Eigen::Matrix<double, 6, 1> err2;
      double ori2 = 0.0;
      task(next, n, jac2, err2, ori2);
      if (err2.head<3>().norm() > kLineSampleTolM || (ori_on && ori2 > 0.08)) {
        break;
      }
      const double z1 = kin_.fk(expand(next), "Body_Link")(2, 3);
      const double f1 = kin_.fk(expand(next), "Leg_Y_Link2")(0, 3);
      const double pitch1 = next.head<3>().sum();
      const bool dropped = z1 < z0 - 1e-4;
      const bool sat_back = f1 < f0 - 1e-4;
      const bool pitch_eased = std::abs(pitch1) + 1e-4 < std::abs(pitch);
      // Never accept a step that arches the fold further forward.
      if (f1 > f0 + 1e-4) {
        break;
      }
      if (need_sit) {
        if (!sat_back) {
          break;
        }
      } else if (!(dropped || sat_back || pitch_eased)) {
        break;
      }
      if (torso_pitch_max_ > 0.0 &&
          std::abs(pitch1) > torso_pitch_max_ - kPitchSquatEndMarginRad &&
          std::abs(pitch1) + 1e-6 >= std::abs(pitch)) {
        break;
      }
      y_goal = next;
    }
  }
  // 起点已经贴着俯仰硬限时，水平段也要把俯仰和拉回硬限内侧，否则候选要么被
  // 1.5° 余量拒掉，要么贴着墙交出去，下一步 QP 报 qp_status_1。
  if (!lock_waist && torso_pitch_max_ > 0.0 &&
      std::abs(pack(q0).head<3>().sum()) > torso_pitch_max_ &&
      std::abs(y_goal.head<3>().sum()) >
          torso_pitch_max_ - kPitchSquatEndMarginRad) {
    for (int it = 0; it < 16; ++it) {
      const double pitch = y_goal.head<3>().sum();
      if (std::abs(pitch) <= torso_pitch_max_ - kPitchSquatEndMarginRad) {
        break;
      }
      Eigen::Matrix<double, 6, kRed> jac;
      Eigen::Matrix<double, 6, 1> err;
      double ori_rad = 0.0;
      task(y_goal, n, jac, err, ori_rad);
      if (err.head<3>().norm() > kLineSampleTolM || (ori_on && ori_rad > 0.08)) {
        break;
      }
      Eigen::Matrix<double, 6, 6> jjt = jac * jac.transpose();
      jjt.diagonal().array() += 1e-6;
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(jjt);
      if (ldlt.info() != Eigen::Success) {
        break;
      }
      const Eigen::Matrix<double, kRed, 6> j_pinv =
          jac.transpose() * ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
      Red grad = Red::Zero();
      const double sign = pitch > 0.0 ? 1.0 : -1.0;
      grad[0] = sign;
      grad[1] = sign;
      grad[2] = sign;
      Red dir = -(MatR::Identity() - j_pinv * jac) * grad;
      const double nrm = dir.norm();
      if (!(nrm > 1e-6)) {
        break;
      }
      dir *= std::min(0.03, 0.06 / nrm);
      const Red next = limited_step(y_goal, dir);
      Eigen::Matrix<double, 6, kRed> jac2;
      Eigen::Matrix<double, 6, 1> err2;
      double ori2 = 0.0;
      task(next, n, jac2, err2, ori2);
      if (err2.head<3>().norm() > kLineSampleTolM || (ori_on && ori2 > 0.08)) {
        break;
      }
      if (!(std::abs(next.head<3>().sum()) + 1e-4 < std::abs(pitch))) {
        break;
      }
      y_goal = next;
    }
  }
  {
    const double goal_err =
        (p_des[static_cast<std::size_t>(n)] - kin_.fk_pos(expand(y_goal), frame)).norm();
    if (goal_err > 0.04) {
      return false;
    }
  }

  Aligned<Red> path(static_cast<std::size_t>(n + 1));
  path[0] = pack(q0);
  for (int i = 1; i <= n; ++i) {
    const double s = static_cast<double>(i) / static_cast<double>(n);
    path[static_cast<std::size_t>(i)] = (1.0 - s) * path[0] + s * y_goal;
    clamp_y(path[static_cast<std::size_t>(i)]);
  }
  keep_com(path);

  // 每个点从两端关节弦出发。中间点只把横向偏差收进软管，不把权重拆掉去换
  // 贴线：拆权重会让肩肘离开弦再回来。终点仍贴目标。沿程不得回头。
  double prev_along = 0.0;
  for (int i = 1; i <= n; ++i) {
    const Red ref = path[static_cast<std::size_t>(i)];
    const bool endpoint = i == n;
    double w_scale = 1.0;
    for (int it = 0; it < 24; ++it) {
      Eigen::Matrix<double, 6, kRed> jac;
      Eigen::Matrix<double, 6, 1> err;
      double ori_rad = 0.0;
      task(path[static_cast<std::size_t>(i)], i, jac, err, ori_rad);
      if (!endpoint) {
        // Reference is the hand's projection on the segment, not the knot
        // index. Pull across the line. Do not drag the arm forward or back
        // along the segment just to match the sample number.
        const Vec3 p = kin_.fk_pos(expand(path[static_cast<std::size_t>(i)]), frame);
        const double ab2 = std::max(path_ab.squaredNorm(), 1e-12);
        const double u = std::clamp((p - p0).dot(path_ab) / ab2, 0.0, 1.0);
        err.head<3>() = (p0 + u * path_ab) - p;
        if (ori_on) {
          const Eigen::Matrix4d posed =
              kin_.fk(expand(path[static_cast<std::size_t>(i)]), frame);
          const Vec3 eo = ori_error_xyzw(
              quat_slerp_xyzw(q_ori0, q_ori1, u),
              rot_to_quat_xyzw(posed.block<3, 3>(0, 0)));
          ori_rad = eo.norm();
          err.tail<3>() = 0.15 * eo;
        }
      }
      const bool ori_ok = !ori_on || ori_rad < 0.08;
      const bool line_ok = endpoint
                               ? err.head<3>().norm() < kLineSolveTolM
                               : cross_track(path[static_cast<std::size_t>(i)]) <=
                                     kLineTubeM;
      if (line_ok && ori_ok &&
          along_m(path[static_cast<std::size_t>(i)]) + 0.002 >= prev_along) {
        break;
      }
      Red w = weights(ref, path[static_cast<std::size_t>(i)]);
      if (!endpoint) {
        for (int a = 0; a < 5; ++a) {
          w[kNWaist + a] = std::max(w[kNWaist + a], kArmChordW);
        }
      } else {
        w *= w_scale;
      }
      MatR gram = jac.transpose() * jac;
      gram.diagonal() += w;
      gram.diagonal().array() += 1e-8;
      const Eigen::LDLT<MatR> ldlt(gram);
      if (ldlt.info() != Eigen::Success) {
        break;
      }
      const Red pull = ref - path[static_cast<std::size_t>(i)];
      const Red dy = ldlt.solve(jac.transpose() * err + w.cwiseProduct(pull));
      const Red next = limited_step(path[static_cast<std::size_t>(i)], dy);
      if (!next.allFinite()) {
        break;
      }
      if (!endpoint) {
        // A step that walks the shoulder/elbow off the chord, or sends the
        // hand back along the segment, is not a way to buy a tighter line.
        const double arm_off =
            (next.segment<5>(kNWaist) - ref.segment<5>(kNWaist)).norm();
        const double arm_was =
            (path[static_cast<std::size_t>(i)].segment<5>(kNWaist) -
             ref.segment<5>(kNWaist))
                .norm();
        if (arm_off > arm_was + 1e-4 || along_m(next) + kLineTubeM < prev_along) {
          break;
        }
      }
      if ((next - path[static_cast<std::size_t>(i)]).norm() < 1e-5) {
        if (endpoint && err.head<3>().norm() > kLineSolveTolM && w_scale > 0.05) {
          w_scale *= 0.2;
          continue;
        }
        break;
      }
      path[static_cast<std::size_t>(i)] = next;
    }
    refine_pose(path[static_cast<std::size_t>(i)], i);
    prev_along = std::max(prev_along, along_m(path[static_cast<std::size_t>(i)]));
  }
  keep_com(path);
  // Do not chase intermediate points back onto the line by weakening the chord
  // pull. Only the endpoint is tightened here.
  if (cross_track(path[static_cast<std::size_t>(n)]) > kLineSolveTolM ||
      (p_des[static_cast<std::size_t>(n)] -
       kin_.fk_pos(expand(path[static_cast<std::size_t>(n)]), frame))
              .norm() > kLineSolveTolM) {
      const Red ref = path[static_cast<std::size_t>(n)];
      double w_scale = 1.0;
      for (int it = 0; it < 24; ++it) {
        Eigen::Matrix<double, 6, kRed> jac;
        Eigen::Matrix<double, 6, 1> err;
        double ori_rad = 0.0;
        task(path[static_cast<std::size_t>(n)], n, jac, err, ori_rad);
        if (err.head<3>().norm() < kLineSolveTolM && (!ori_on || ori_rad < 0.08)) {
          break;
        }
        const Red w = weights(ref, path[static_cast<std::size_t>(n)]) * w_scale;
        MatR gram = jac.transpose() * jac;
        gram.diagonal() += w;
        gram.diagonal().array() += 1e-8;
        const Eigen::LDLT<MatR> ldlt(gram);
        if (ldlt.info() != Eigen::Success) {
          break;
        }
        const Red pull = ref - path[static_cast<std::size_t>(n)];
        const Red dy = ldlt.solve(jac.transpose() * err + w.cwiseProduct(pull));
        const Red next = limited_step(path[static_cast<std::size_t>(n)], dy);
        if (!next.allFinite()) {
          break;
        }
        if ((next - path[static_cast<std::size_t>(n)]).norm() < 1e-5) {
          if (err.head<3>().norm() > kLineSolveTolM && w_scale > 0.05) {
            w_scale *= 0.2;
            continue;
          }
          break;
        }
        path[static_cast<std::size_t>(n)] = next;
      }
      keep_com(path);
    }
  const Red y_end = path[static_cast<std::size_t>(n)];

  for (int pass = 0; pass < 2; ++pass) {
    if (worst_pos(path) < kLineTubeM) {
      break;
    }
    Aligned<MatR> diag(static_cast<std::size_t>(n));
    Aligned<MatR> upper(static_cast<std::size_t>(std::max(n - 1, 0)));
    Aligned<Red> rhs(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
      diag[static_cast<std::size_t>(k)].setZero();
      rhs[static_cast<std::size_t>(k)].setZero();
    }
    for (int k = 0; k < n - 1; ++k) {
      upper[static_cast<std::size_t>(k)].setZero();
    }
    for (int i = 1; i <= n; ++i) {
      const int k = i - 1;
      const Red w = weights(path[static_cast<std::size_t>(i - 1)], path[static_cast<std::size_t>(i)]);
      Red w_smooth = w;
      for (int a = 0; a < 5; ++a) {
        w_smooth[kNWaist + a] = std::max(w_smooth[kNWaist + a], kArmChordW);
      }
      const Red delta = path[static_cast<std::size_t>(i)] - path[static_cast<std::size_t>(i - 1)];
      const MatR weight = w_smooth.asDiagonal();
      diag[static_cast<std::size_t>(k)] += weight;
      rhs[static_cast<std::size_t>(k)] -= weight * delta;
      if (k > 0) {
        diag[static_cast<std::size_t>(k - 1)] += weight;
        upper[static_cast<std::size_t>(k - 1)] -= weight;
        rhs[static_cast<std::size_t>(k - 1)] += weight * delta;
      }
      Eigen::Matrix<double, 6, kRed> jac;
      Eigen::Matrix<double, 6, 1> err;
      double ori_rad = 0.0;
      task(path[static_cast<std::size_t>(i)], i, jac, err, ori_rad);
      diag[static_cast<std::size_t>(k)] += jac.transpose() * jac;
      rhs[static_cast<std::size_t>(k)] += jac.transpose() * err;
      diag[static_cast<std::size_t>(k)].diagonal().array() += 1e-8;
    }
    Aligned<Red> step;
    if (!solve_block(diag, upper, rhs, step)) {
      break;
    }
    double peak = 0.0;
    for (int k = 0; k < n; ++k) {
      peak=std::max(peak,step[static_cast<std::size_t>(k)].cwiseAbs().maxCoeff());
    }
    if (peak > 0.12) {
      const double scale = 0.12 / peak;
      for (Red& one : step) {
        one *= scale;
      }
    }
    const double base = worst_pos(path);
    Aligned<Red> best = path;
    double best_err = base;
    for (const double alpha : {1.0, 0.5, 0.25}) {
      Aligned<Red> trial = path;
      bool finite = true;
      for (int k = 0; k < n; ++k) {
        trial[static_cast<std::size_t>(k + 1)] += alpha * step[static_cast<std::size_t>(k)];
        if (!trial[static_cast<std::size_t>(k + 1)].allFinite()) {
          finite = false;
          break;
        }
        clamp_y(trial[static_cast<std::size_t>(k + 1)]);
      }
      if (!finite) {
        continue;
      }
      keep_com(trial);
      trial[static_cast<std::size_t>(n)] = y_end;
      const double err = worst_pos(trial);
      if (err < best_err) {
        best_err = err;
        best = std::move(trial);
      }
    }
    if (!(best_err < base - 1e-4)) {
      break;
    }
    path = std::move(best);
  }

  auto soft_line_ok = [&](const Aligned<Red>& ys) {
    if ((p_des[static_cast<std::size_t>(n)] -
         kin_.fk_pos(expand(ys[static_cast<std::size_t>(n)]), frame))
            .norm() > kLineSampleTolM) {
      return false;
    }
    double worst_cross = 0.0;
    for (int i = 1; i < n; ++i) {
      worst_cross = std::max(worst_cross, cross_track(ys[static_cast<std::size_t>(i)]));
    }
    if (worst_cross > kLineTubeHardM) {
      return false;
    }
    return true;
  };
  if (!soft_line_ok(path)) {
    return false;
  }
  for (int i = 1; i <= n; ++i) {
    const Vec18 jumped =
        expand(path[static_cast<std::size_t>(i)]) - expand(path[static_cast<std::size_t>(i - 1)]);
    if (jumped.cwiseAbs().maxCoeff() > kLineJointJump) {
      return false;
    }
  }

  // 俯仰和是冗余量；直线只约束手。端点不动，中间点在手零空间里拉回两端弦。
  // 架构约束：若拉直后任一腰俯仰轴弦折返超过约 1°（相对拉直前变差），整段
  // 退回拉直前路径——禁止用 J1/J2/J3 大折返去「买」总和单调（r19/r20 水平段
  // J3≈3.5°）。启动段总和回摆通常只需很小的单轴代价，仍可保留拉直结果。
  if (line_pitch_straighten_enable_ && !lock_waist && n > 1) {
    const auto pitch_of = [](const Red& y) { return y.head<3>().sum(); };
    const double p0 = pitch_of(path[0]);
    const double pN = pitch_of(path[static_cast<std::size_t>(n)]);
    // Endpoint squat is a real candidate (fold_10/15/20), not a silent rewrite.
    Aligned<Red> kept = path;
    for (int pass = 0; pass < 12; ++pass) {
      double worst = 0.0;
      for (int i = 1; i < n; ++i) {
        const double s = static_cast<double>(i) / static_cast<double>(n);
        const double want = (1.0 - s) * p0 + s * pN;
        worst = std::max(
            worst, std::abs(pitch_of(path[static_cast<std::size_t>(i)]) - want));
      }
      if (worst < 2e-4) {
        break;
      }
      bool moved = false;
      for (int i = 1; i < n; ++i) {
        Red& y = path[static_cast<std::size_t>(i)];
        const double s = static_cast<double>(i) / static_cast<double>(n);
        const double want = (1.0 - s) * p0 + s * pN;
        const double miss = want - pitch_of(y);
        if (std::abs(miss) < 2e-4) {
          continue;
        }
        Eigen::Matrix<double, 6, kRed> jac;
        Eigen::Matrix<double, 6, 1> err;
        double ori_rad = 0.0;
        task(y, i, jac, err, ori_rad);
        (void)ori_rad;
        Eigen::Matrix<double, 6, 6> jjt = jac * jac.transpose();
        jjt.diagonal().array() += 1e-6;
        const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(jjt);
        if (ldlt.info() != Eigen::Success) {
          continue;
        }
        const Eigen::Matrix<double, kRed, 6> j_pinv =
            jac.transpose() * ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
        const MatR nullspace = MatR::Identity() - j_pinv * jac;
        Eigen::Matrix<double, 1, kRed> prow = Eigen::Matrix<double, 1, kRed>::Zero();
        prow(0) = 1.0;
        prow(1) = 1.0;
        prow(2) = 1.0;
        const Red nvec = nullspace * prow.transpose();
        const double denom = (prow * nvec)(0, 0);
        if (!(std::abs(denom) > 1e-5)) {
          continue;
        }
        double alpha = miss / denom;
        constexpr double kMaxPitchStep = 0.05;
        if (std::abs(alpha * denom) > kMaxPitchStep) {
          alpha *= kMaxPitchStep / std::abs(alpha * denom);
        }
        const Red next = limited_step(y, nvec * alpha);
        double ori2 = 0.0;
        if (ori_on) {
          const Eigen::Matrix4d posed = kin_.fk(expand(next), frame);
          ori2 = ori_error_xyzw(
                     quat_des[static_cast<std::size_t>(i)],
                     rot_to_quat_xyzw(posed.block<3, 3>(0, 0)))
                     .norm();
        }
        if (cross_track(next) > kLineTubeM || (ori_on && ori2 > 0.08)) {
          continue;
        }
        if (!(std::abs(want - pitch_of(next)) < std::abs(miss) - 1e-5)) {
          continue;
        }
        y = next;
        moved = true;
      }
      if (!moved) {
        break;
      }
    }
    if (straighten_axis_turnback_regressed(kept, path,
                                           kWaistAxisTurnbackSumStageBlowRad)) {
      path = std::move(kept);
      line_fit_straighten_axis_reverted_ = true;
    } else {
      std::vector<Vec18, Eigen::aligned_allocator<Vec18>> trial;
      trial.reserve(static_cast<std::size_t>(n + 1));
      bool finite = true;
      for (int i = 0; i <= n; ++i) {
        const Vec18 q = expand(path[static_cast<std::size_t>(i)]);
        if (!q.allFinite()) {
          finite = false;
          break;
        }
        trial.push_back(q);
      }
      if (!finite || worst_pos(path) > kLineSampleTolM ||
          !line_validate(trial, target, quat, ori_on, frame)) {
        path = std::move(kept);
      } else {
        bool changed = false;
        for (int i = 1; i < n; ++i) {
          if ((path[static_cast<std::size_t>(i)] - kept[static_cast<std::size_t>(i)])
                  .cwiseAbs()
                  .maxCoeff() > 1e-4) {
            changed = true;
            break;
          }
        }
        if (changed) {
          ++line_pitch_straighten_hits_;
        }
      }
    }
  }

  // A forced-free cascade candidate must not silently be relocked.
  waist_locked = lock_waist;
  if (allow_relock) {
    const auto waist0=path[0].head<kNWaist>().eval();
    Aligned<Red> saved = path;
    for (int i = 1; i <= n; ++i) {
      path[static_cast<std::size_t>(i)].head<kNWaist>()=waist0;
    }
    if (worst_pos(path) <= kLineSampleTolM) {
      waist_locked = true;
    } else {
      path = std::move(saved);
      waist_locked = false;
    }
  }

  samples.clear();
  samples.reserve(static_cast<std::size_t>(n + 1));
  for (int i = 0; i <= n; ++i) {
    const Vec18 q = expand(path[static_cast<std::size_t>(i)]);
    if (!q.allFinite()) {
      return false;
    }
    samples.push_back(q);
  }
  if (!line_validate(samples, target, quat, ori_on, frame)) {
    return false;
  }
  return true;
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

bool CartesianTracker::line_validate(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples,
    const Vec3& target, const Vec4* quat, bool ori_on, const std::string& frame) const {
  if (samples.size() < 2) return false;
  const Vec3 start = kin_.fk_pos(samples.front(), frame);
  const Vec4 start_ori = rot_to_quat_xyzw(kin_.fk(samples.front(), frame).block<3,3>(0,0));
  const int n = static_cast<int>(samples.size()) - 1;
  // Soft-limit policy: the live start may already sit on / slightly past the
  // soft wall after tracking (common on retreat). Do not reject that. Still
  // enforce hard limits everywhere, and forbid soft excess from growing past
  // the start sample — retreat that moves away from the wall stays legal.
  const Vec18& q_start = samples.front();
  const double start_pitch_abs = std::abs(q_start.head<3>().sum());
  Vec18 start_soft_lo_ex = Vec18::Zero();
  Vec18 start_soft_hi_ex = Vec18::Zero();
  for (int k = 0; k < 18; ++k) {
    const double soft_lo = limits_.q_min[k] + q_limit_margin_;
    const double soft_hi = limits_.q_max[k] - q_limit_margin_;
    start_soft_lo_ex[k] = std::max(0.0, soft_lo - q_start[k]);
    start_soft_hi_ex[k] = std::max(0.0, q_start[k] - soft_hi);
  }
  auto joints_ok = [&](const Vec18& q) -> bool {
    if (!q.allFinite()) return false;
    for (int k = 0; k < 18; ++k) {
      if (q[k] < limits_.q_min[k] - 1e-9 || q[k] > limits_.q_max[k] + 1e-9) {
        return false;
      }
      const double soft_lo = limits_.q_min[k] + q_limit_margin_;
      const double soft_hi = limits_.q_max[k] - q_limit_margin_;
      const double lo_ex = std::max(0.0, soft_lo - q[k]);
      const double hi_ex = std::max(0.0, q[k] - soft_hi);
      if (lo_ex > start_soft_lo_ex[k] + 1e-9 ||
          hi_ex > start_soft_hi_ex[k] + 1e-9) {
        return false;
      }
    }
    return true;
  };
  for (int i = 0; i < n; ++i) {
    // Match joint-linear streaming / line_at, including 250 Hz interior ticks.
    for (int j = 0; j <= 5; ++j) {
      const double a = j / 5.0;
      const Vec18 q = (1-a)*samples[i] + a*samples[i+1];
      if (!joints_ok(q) || !guards_.com_xy_inside(kin_.com(q))) {
        return false;
      }
      // Pitch is part of the public validate contract. Forbid excess growth past
      // the path start (recovery envelope). Absolute ceiling for ordinary
      // start-inside uses hard + shared admit as IK/float residue only — the
      // 250 Hz filtered gate stays hard+round; QP projects into that hard box.
      if (torso_pitch_max_ > 0.0) {
        const double start_ex =
            std::max(0.0, start_pitch_abs - torso_pitch_max_);
        const double ex =
            std::max(0.0, std::abs(q.head<3>().sum()) - torso_pitch_max_);
        const double allow =
            std::max(start_ex, kStartPitchAdmitRad) + kPitchRoundEpsRad;
        if (ex > allow) {
          return false;
        }
      }
      const auto pose = kin_.fk(q, frame);
      const Vec3 p = pose.col(3).head<3>();
      const Vec3 ab = target - start;
      const double ab_norm = std::max(ab.norm(), 1e-9);
      const double along_m = (p - start).dot(ab) / ab_norm;
      const double u = std::clamp(along_m / ab_norm, 0.0, 1.0);
      const bool at_end = (i + 1 == n && j == 5);
      if (at_end) {
        if ((p - target).norm() > kLineSampleTolM) {
          return false;
        }
      } else if ((p - (start + u * ab)).norm() > kLineTubeHardM) {
        return false;
      }
      // 贴线是软管：端点姿态仍用到达容差，中间允许手离开直线时姿态一起偏开。
      const double ori_lim = at_end ? arrived_ori_rad_ : kLineOriHardRad;
      if (ori_on && quat) {
        const double oe = ori_error_xyzw(quat_slerp_xyzw(start_ori,*quat,u),
            rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm();
        if (oe > ori_lim) {
          return false;
        }
      }
    }
  }
  return true;
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
    const double joint_margin = std::min(
        (q-limits_.q_min-Vec18::Constant(q_limit_margin_)).minCoeff(),
        (limits_.q_max-q-Vec18::Constant(q_limit_margin_)).minCoeff());
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
    if (joint_margin < -1e-8 || !guards_.com_xy_inside(com) || pitch_margin < -1e-7) {
      result.ok = false;
      result.reason = joint_margin < -1e-8 ? "joint" :
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
  LineFoldContrast out;
  const bool ori_on = quat != nullptr && !quat_is_identity_xyzw(quat);
  const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
  const std::string frame = std::string(left ? kEeFrameLeft : kEeFrameRight);
  auto fill = [&](const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path,
                  std::vector<LinePoseKnot>* knots, double* drop, double* end_pitch) {
    knots->clear();
    if (path.size() < 2) {
      return;
    }
    const Vec3 p0 = kin_.fk_pos(path.front(), frame);
    const Vec3 ab = target - p0;
    const double z0 = kin_.fk(path.front(), "Body_Link")(2, 3);
    const int last = static_cast<int>(path.size()) - 1;
    const double fracs[] = {0.0, 0.10, 0.25, 0.50, 1.0};
    for (double frac : fracs) {
      const int idx = std::clamp(static_cast<int>(std::lround(frac * last)), 0, last);
      const Vec18& q = path[static_cast<std::size_t>(idx)];
      const Vec3 p = kin_.fk_pos(q, frame);
      const double along = (p - p0).dot(ab);
      const double t = std::clamp(along / std::max(ab.squaredNorm(), 1e-12), 0.0, 1.0);
      LinePoseKnot knot;
      knot.s = static_cast<double>(idx) / static_cast<double>(last);
      knot.hand_err_m = (p - (p0 + t * ab)).norm();
      knot.pitch_rad = q.head<3>().sum();
      knot.body_z_m = kin_.fk(q, "Body_Link")(2, 3);
      knot.body_x_m = kin_.fk(q, "Body_Link")(0, 3);
      knot.fold_x_m = kin_.fk(q, "Leg_Y_Link2")(0, 3);
      knot.waist_motion_rad = (q.head<3>() - q0.head<3>()).norm();
      knot.arm_motion_rad = (q.segment<kNArm>(arm) - q0.segment<kNArm>(arm)).norm();
      knots->push_back(knot);
    }
    *drop = z0 - knots->back().body_z_m;
    *end_pitch = knots->back().pitch_rad;
  };
  auto early = [](const std::vector<LinePoseKnot>& knots) {
    constexpr double kMove = 1.0 * kPi / 180.0;
    for (const auto& knot : knots) {
      if (knot.s > 1e-6 && knot.s <= 0.25 + 1e-9 &&
          knot.waist_motion_rad > kMove && knot.arm_motion_rad > kMove) {
        return true;
      }
    }
    return false;
  };
  auto certify = [&](const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path,
                     std::string* why) {
    if (!line_validate(path, target, quat, ori_on, frame)) {
      *why = "validate";
      return false;
    }
    const auto dir = check_path_direction_strict(path);
    if (!dir.ok) {
      *why = std::string("direction:") + dir.reason;
      return false;
    }
    if (successor != nullptr) {
      // Free waist+arm tip reachability (no lock-waist-only probe).
      std::vector<Vec18, Eigen::aligned_allocator<Vec18>> next;
      bool next_locked = false;
      const Vec4* next_quat = successor->ori_on ? &successor->quat_xyzw : nullptr;
      if (!line_fit(path.back(), arm, successor->xyz, next_quat, successor->ori_on,
                    /*lock_waist=*/false, frame, next, next_locked, false, nullptr,
                    nullptr, kLineSamplePoolM) ||
          next.size() < 2 || !check_path_direction_strict(next).ok) {
        *why = "successor";
        return false;
      }
    }
    *why = "hard_ok";
    return true;
  };

  std::vector<Vec18, Eigen::aligned_allocator<Vec18>> baseline;
  bool baseline_locked = false;
  const bool baseline_fit = line_fit(
      q0, arm, target, quat, ori_on, false, frame, baseline, baseline_locked,
      successor == nullptr, nullptr);
  if (!baseline_fit) {
    out.baseline_reason = "fit";
  } else if (certify(baseline, &out.baseline_reason)) {
    out.baseline_hard_ok = true;
    fill(baseline, &out.baseline_knots, &out.baseline_body_drop_m,
         &out.baseline_end_pitch_rad);
  }

  double best_drop = -1e9;
  bool best_early = false;
  std::string last_fail = "no_seed";
  for (const double deg : {10.0, 15.0, 20.0}) {
    const double a = deg * kPi / 180.0;
    Vec18 seed = q0;
    seed[0] += a;
    seed[1] -= 2.0 * a;
    seed[2] += a;
    for (int k = 0; k < 3; ++k) {
      seed[k] = std::clamp(seed[k], limits_.q_min[k] + q_limit_margin_,
                           limits_.q_max[k] - q_limit_margin_);
    }
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>> trial;
    bool locked = false;
    if (!line_fit(q0, arm, target, quat, ori_on, false, frame, trial, locked, false,
                  &seed)) {
      last_fail = "fit";
      continue;
    }
    std::string why;
    if (!certify(trial, &why)) {
      last_fail = why;
      continue;
    }
    std::vector<LinePoseKnot> knots;
    double drop = 0.0;
    double end_pitch = 0.0;
    fill(trial, &knots, &drop, &end_pitch);
    const bool overlap = early(knots);
    if (drop > best_drop + 1e-6 || (overlap && !best_early && drop > best_drop - 0.01)) {
      best_drop = drop;
      best_early = overlap;
      out.fold_hard_ok = true;
      out.fold_reason = "hard_ok";
      out.fold_body_drop_m = drop;
      out.fold_end_pitch_rad = end_pitch;
      out.fold_knots = std::move(knots);
      out.early_waist_arm_overlap = overlap;
    }
  }
  if (!out.fold_hard_ok) {
    out.fold_reason = last_fail;
  }
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
  // Tip reachability only during the candidate pool. Winner commits one tip
  // fit into successor_plan_ (see below) so the next segment can reuse it.
  auto probe = [&](const auto& path) {
    if (!line_tip_) return true;
    if (path.empty()) return false;
    const double reach = (kin_.fk_pos(path.back(), frame) - line_tip_->xyz).norm();
    if (reach > static_cast<double>(kLineMaxSamples) * kLineSampleM) {
      line_.reason = "cascade_unreachable";
      return false;
    }
    return true;
  };
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

  auto q_near=[&](const Vec18& a,const Vec18& b){
    return (a-b).cwiseAbs().maxCoeff()<=reverse_q_tol_;
  };
  auto reverse_intent=[&](){
    if(!reverse_last_line_enable_ || !last_line_cache_.valid || line_tip_.has_value()) {
      return false;
    }
    const auto& c=last_line_cache_;
    if(c.left!=left || c.frame!=frame || c.ori_on!=ori_on) return false;
    if((target-c.start_xyz).norm()>reverse_xyz_tol_) return false;
    if(ori_on && quat &&
       ori_error_xyzw(*quat,c.start_quat).norm()>arrived_ori_rad_) return false;
    return true;
  };
  bool ok=false;
  bool straighten_axis_reverted=false;
  bool have_accepted_dir=false;
  DirectionContract accepted_contract{};
  DirectionCheckResult accepted_dir{};
  // Rebuild waist as a linear interpolation q0→qN (monotone by construction on
  // each waist axis and on pitch sum), then re-IK the active arm to the
  // original Cartesian waypoints. Soft IK seeds may fold; hard contract still
  // decides acceptance after repair.
  auto repair_direction=[&](std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path)->bool {
    if(path.size()<2) return false;
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> repaired=path;
    const Vec18& q_start=path.front();
    const Vec18& q_end=path.back();
    const int n=static_cast<int>(repaired.size())-1;
    for(int i=1;i<=n;++i) {
      const double s=static_cast<double>(i)/static_cast<double>(n);
      repaired[static_cast<std::size_t>(i)].head<kNWaist>()=
          (1.0-s)*q_start.head<kNWaist>()+s*q_end.head<kNWaist>();
      for(int j=0;j<kNWaist;++j) {
        const double lo=limits_.q_min[j]+q_limit_margin_;
        const double hi=limits_.q_max[j]-q_limit_margin_;
        if(hi>lo) {
          repaired[static_cast<std::size_t>(i)][j]=
              std::clamp(repaired[static_cast<std::size_t>(i)][j],lo,hi);
        }
      }
    }
    const Vec3 p0=kin_.fk_pos(path.front(),frame);
    const int idle=arm==kSliceLeft.begin?kSliceRight.begin:kSliceLeft.begin;
    Vec4 q_ori0(0,0,0,1), q_ori1(0,0,0,1);
    if(ori_on && quat) {
      q_ori0=rot_to_quat_xyzw(kin_.fk(path.front(),frame).block<3,3>(0,0));
      q_ori1=*quat;
    }
    for(int i=1;i<=n;++i) {
      const double s=static_cast<double>(i)/static_cast<double>(n);
      const Vec3 p_des=(1.0-s)*p0+s*target;
      const Vec4 q_s=quat_slerp_xyzw(q_ori0,q_ori1,s);
      Vec18 q=repaired[static_cast<std::size_t>(i)];
      q.head<kNWaist>()=repaired[static_cast<std::size_t>(i)].head<kNWaist>();
      q.segment<kNArm>(idle)=path.front().segment<kNArm>(idle);
      bool ok_ik=false;
      for(int it=0;it<40;++it) {
        const auto posed=kin_.fk(q,frame);
        const Vec3 e_p=p_des-posed.col(3).head<3>();
        Vec3 e_o=ori_on?ori_error_xyzw(q_s,rot_to_quat_xyzw(posed.block<3,3>(0,0))):Vec3::Zero();
        if(e_p.norm()<kLineSolveTolM && (!ori_on || e_o.norm()<0.05)) { ok_ik=true; break; }
        Mat6x18 jac=kin_.jacobian_spatial(q,frame);
        Eigen::Matrix<double,6,1> err;
        err.head<3>()=e_p;
        err.tail<3>()=ori_on?(0.15*e_o).eval():Vec3::Zero();
        if(ori_on) jac.bottomRows<3>()*=0.15; else jac.bottomRows<3>().setZero();
        jac.leftCols<kNWaist>().setZero();
        jac.middleCols<kNArm>(idle).setZero();
        Mat18 gram=jac.transpose()*jac;
        Vec18 damp=Vec18::Constant(kWArm);
        damp.segment<2>(kSliceRight.begin+5).setConstant(kWWrist);
        damp.segment<2>(kSliceLeft.begin+5).setConstant(kWWrist);
        damp.head<kNWaist>().setConstant(1e3);
        damp.segment<kNArm>(idle).setConstant(1e3);
        gram.diagonal()+=damp;
        const Eigen::LDLT<Mat18> ldlt(gram);
        if(ldlt.info()!=Eigen::Success) break;
        Vec18 dq=ldlt.solve(jac.transpose()*err);
        const double peak=dq.cwiseAbs().maxCoeff();
        if(peak>0.08) dq*=0.08/peak;
        Vec18 qn=q+dq;
        for(int j=0;j<kNDof;++j) {
          const double lo=limits_.q_min[j]+q_limit_margin_;
          const double hi=limits_.q_max[j]-q_limit_margin_;
          if(hi>lo) qn[j]=std::clamp(qn[j],lo,hi);
        }
        qn.head<kNWaist>()=repaired[static_cast<std::size_t>(i)].head<kNWaist>();
        qn.segment<kNArm>(idle)=path.front().segment<kNArm>(idle);
        if(!qn.allFinite()) break;
        q=qn;
      }
      if(!ok_ik) return false;
      repaired[static_cast<std::size_t>(i)]=q;
    }
    if(!line_validate(repaired,target,quat,ori_on,frame)) return false;
    // Repair may commit only when the joint-linear stream itself is back on
    // the indexed line. Knot-only checks let a folded reverse through.
    {
      const Vec3 p_a = kin_.fk_pos(repaired.front(), frame);
      const Vec3 ab = target - p_a;
      const int n = static_cast<int>(repaired.size()) - 1;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= 5; ++j) {
          const double a = j / 5.0;
          const Vec18 q = (1.0 - a) * repaired[static_cast<std::size_t>(i)] +
                          a * repaired[static_cast<std::size_t>(i + 1)];
          const double s = (static_cast<double>(i) + a) / static_cast<double>(n);
          const Vec3 p = kin_.fk_pos(q, frame);
          if ((p - (p_a + s * ab)).norm() > kLineSampleTolM) return false;
        }
      }
    }
    if(torso_pitch_max_>0) {
      const double start_ex=std::max(0.0,std::abs(path.front().head<3>().sum())-torso_pitch_max_);
      for(const auto& q:repaired) {
        const double ex=std::max(0.0,std::abs(q.head<3>().sum())-torso_pitch_max_);
        if(ex>start_ex+1e-9 && ex>1e-9 &&
           std::abs(q.head<3>().sum())>torso_pitch_max_+1e-7) {
          // Allow shrinking excess; reject growth beyond start.
          if(ex>start_ex+1e-9) return false;
        }
      }
    }
    path=std::move(repaired);
    return true;
  };
  // Alternate endpoint: freeze near-zero waist axes / pitch sum at the start
  // so repair_direction can rebuild a monotone waist without keeping a tiny
  // opposing net that still forces mid-path fold under free_fit's q_end.
  auto freeze_near_zero_end=[&](std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path){
    if(path.size()<2) return;
    Vec18& q_end=path.back();
    const Vec18& q_start=path.front();
    constexpr double kFreezeNet=2e-3;  // ~0.115°
    for(int a=0;a<3;++a) {
      if(std::abs(q_end[a]-q_start[a])<=kFreezeNet) q_end[a]=q_start[a];
    }
    const double p0=q_start.head<3>().sum();
    const double pN=q_end.head<3>().sum();
    if(std::abs(pN-p0)<=kFreezeNet) {
      // Hold pitch sum at start by adjusting the least-moving waist axis.
      double need=p0-pN;
      int best=2;
      double best_abs=std::abs(q_end[2]-q_start[2]);
      for(int a=0;a<3;++a) {
        const double m=std::abs(q_end[a]-q_start[a]);
        if(m<=best_abs) { best_abs=m; best=a; }
      }
      q_end[best]+=need;
    }
  };
  auto accept_direction=[&](std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path,
                            DirectionContract* out_c, DirectionCheckResult* out_r,
                            bool allow_repair=true)->bool {
    if(!line_direction_strict_enable_) {
      if(out_c) *out_c=DirectionContract{};
      if(out_r) {
        *out_r=DirectionCheckResult{};
        std::snprintf(out_r->reason,sizeof(out_r->reason),"skipped_strict_off");
      }
      return true;
    }
    DirectionContract c=make_path_direction_contract(path);
    DirectionCheckResult chk=check_path_direction(path,c);
    if(!chk.ok && allow_repair) {
      if(repair_direction(path)) {
        c=make_path_direction_contract(path);
        chk=check_path_direction(path,c);
      } else {
        // Second try: freeze near-zero end axes then rebuild monotone waist.
        auto alt=path;
        freeze_near_zero_end(alt);
        if(repair_direction(alt)) {
          c=make_path_direction_contract(alt);
          chk=check_path_direction(alt,c);
          if(chk.ok) path=std::move(alt);
        }
      }
    }
    if(out_c) *out_c=c;
    if(out_r) *out_r=chk;
    return chk.ok;
  };

  // M03/P01: hard-gated candidates share one pool; no exclusive short-circuit.
  // Lexicographic pick: max pitch margin → max CoM margin → min path_cost.
  auto shadow_note=[&](const char* name, bool feasible, const char* detail){
    if(!line_diag_enable_ || !line_shadow_diag_enable_) return;
    char decision[48];
    const char* tag = feasible ? "ok" : "no";
    const char* why = (detail && detail[0] != '\0') ? detail : "-";
    std::snprintf(decision, sizeof(decision), "%.2s:%.40s", tag, why);
    line_diag_push(LineDiagLayer::Build, q0, Vec18::Zero(), 0.0, 0.0, name, decision);
  };

  struct LineCand {
    const char* name = "none";
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> samples;
    bool locked = true;
    bool forced = false;
    bool straighten_reverted = false;
    DirectionContract contract{};
    DirectionCheckResult dir{};
  };
  std::vector<LineCand> cands;
  auto push_cand=[&](const char* name,
                     std::vector<Vec18,Eigen::aligned_allocator<Vec18>> path,
                     bool path_locked, bool path_forced,
                     const DirectionContract& c, const DirectionCheckResult& d,
                     bool straighten_reverted=false){
    if(path.size()<2) return;
    LineCand cand;
    cand.name=name;
    cand.samples=std::move(path);
    cand.locked=path_locked;
    cand.forced=path_forced;
    cand.straighten_reverted=straighten_reverted;
    cand.contract=c;
    cand.dir=d;
    cands.push_back(std::move(cand));
  };

  auto path_cost=[&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path){
    if(path.empty()) return 1e9;
    double arm_tv=0.0, waist_tv=0.0, wrist_tv=0.0, smooth=0.0;
    for(std::size_t i=1;i<path.size();++i){
      arm_tv+=(path[i]-path[i-1]).segment(arm,kNArm).cwiseAbs().sum();
      waist_tv+=(path[i]-path[i-1]).head<kNWaist>().cwiseAbs().sum();
      wrist_tv+=(path[i]-path[i-1]).segment(arm+5,2).cwiseAbs().sum();
      if(i+1<path.size()){
        const Vec18 d2=path[i+1]-2.0*path[i]+path[i-1];
        smooth+=d2.segment(arm,kNArm).cwiseAbs().sum()+
                0.5*d2.head<kNWaist>().cwiseAbs().sum();
      }
    }
    const double elbow=std::abs(path.back()[arm+3]-line_q_pref_[arm+3]);
    const double pitch_abs=std::abs(path.back().head<3>().sum());
    // Stronger end-pitch pull so free_fit/arm_only prefer squat vs wall-hug.
    return arm_tv + 1.5*elbow + 0.8*wrist_tv + 0.2*waist_tv + 1.5*pitch_abs +
           0.35*smooth;
  };
  // Start is shared by every candidate; scoring uses end margin (squat/recover)
  // and the worst interior margin so a glued-to-wall start cannot dominate.
  auto path_pitch_margin=[&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path){
    if(path.empty() || torso_pitch_max_<=0) return 0.0;
    const double end_m=torso_pitch_max_-std::abs(path.back().head<3>().sum());
    double interior=end_m;
    for(std::size_t i=1;i+1<path.size();++i) {
      interior=std::min(interior,
                        torso_pitch_max_-std::abs(path[i].head<3>().sum()));
    }
    return 0.75*end_m + 0.25*interior;
  };
  auto path_com_margin=[&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path){
    if(path.empty()) return 0.0;
    double m=std::numeric_limits<double>::infinity();
    for(const auto& q:path) {
      const Vec3 com=kin_.com(q);
      m=std::min(m, std::min({
          com.x()-(com_xmin_+com_margin_),
          (com_xmax_-com_margin_)-com.x(),
          com.y()-(com_ymin_+com_margin_),
          (com_ymax_-com_margin_)-com.y()}));
    }
    return m;
  };
  // Lower is better. Pitch outside a 20° comfort band, useful shoulder drop on
  // a lower target, extra squat, forward arch, waist-only motion, travel, CoM.
  // A 1e-6 rad pitch edge cannot beat a clearly smoother path: callers use a
  // 0.02 deadband before looking at travel.
  auto pose_quality=[&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path){
    if(path.size()<2) return 1e9;
    const double comfort=20.0*kPi/180.0;
    const double deg=kPi/180.0;
    double pitch_cost=0.0;
    double waist_early=0.0;
    double arm_early=0.0;
    const int last=static_cast<int>(path.size())-1;
    for(int i=0;i<=last;++i){
      const auto& q=path[static_cast<std::size_t>(i)];
      pitch_cost+=std::max(0.0,std::abs(q.head<3>().sum())-comfort);
      const double s=static_cast<double>(i)/static_cast<double>(last);
      if(s>1e-6 && s<=0.25+1e-9){
        waist_early=std::max(waist_early,(q.head<3>()-path.front().head<3>()).norm());
        arm_early=std::max(arm_early,
            (q.segment(arm,kNArm)-path.front().segment(arm,kNArm)).norm());
      }
    }
    pitch_cost/=static_cast<double>(path.size());
    const Vec3 hand0=kin_.fk_pos(path.front(),frame);
    const bool lower_task=target.z()<hand0.z()-0.03;
    const double z0=kin_.fk(path.front(),"Body_Link")(2,3);
    const double x0=kin_.fk(path.front(),"Body_Link")(0,3);
    const double zN=kin_.fk(path.back(),"Body_Link")(2,3);
    const double xN=kin_.fk(path.back(),"Body_Link")(0,3);
    const double drop=z0-zN;
    const double hand_drop=std::max(0.0,hand0.z()-target.z());
    double height=0.0;
    double early_pen=0.0;
    if(lower_task){
      const double useful=std::min(std::max(0.0,drop),hand_drop);
      const double extra=std::max(0.0,drop-hand_drop-0.02);
      height=-0.8*useful+2.0*extra;
      // r39 tip: fold_15 won with max_arm_fold=0; soft 0.15 was too weak vs
      // body-drop reward. Waist-without-arm must lose to free_fit on same stage.
      if(waist_early>=deg && arm_early<deg) early_pen=2.0;
      else if(waist_early<deg || arm_early<deg) early_pen=0.15;
    }else{
      height=1.5*std::max(0.0,drop-0.01);
    }
    const double foldx = kin_.fk(path.back(), "Leg_Y_Link2")(0, 3) -
                         kin_.fk(path.front(), "Leg_Y_Link2")(0, 3);
    const double forward=1.2*std::max(0.0,(xN-x0)-0.02);
    const double sit = lower_task ? 2.0 * std::max(0.0, foldx + 0.04) : 0.0;
    const double com_pen=std::max(0.0,-path_com_margin(path));
    const double pitch_wall=std::max(0.0,-path_pitch_margin(path));
    return 8.0*pitch_cost+height+forward+sit+early_pen+0.05*path_cost(path)+
           4.0*com_pen+2.0*pitch_wall;
  };

  // Reuse a paired successor commitment when the real next goal matches the
  // cascade tip that was validated together with the preparation segment.
  if(successor_plan_.valid && !line_tip_.has_value() &&
     (target-successor_plan_.tip_xyz).norm()<=reverse_xyz_tol_ &&
     ori_on==successor_plan_.tip_ori_on &&
     (!ori_on || !quat ||
      ori_error_xyzw(*quat,successor_plan_.tip_quat).norm()<=arrived_ori_rad_) &&
     successor_plan_.tip_samples.size()>=2) {
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> tip=successor_plan_.tip_samples;
    DirectionContract tip_c;
    DirectionCheckResult tip_dir;
    auto take_succ=[&](std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path,
                       const DirectionContract& c,const DirectionCheckResult& d,
                       const char* detail){
      const bool path_locked=(path.back().head<kNWaist>()-path.front().head<kNWaist>())
                                 .cwiseAbs().maxCoeff()<=waist_lock_eps_;
      push_cand("successor_plan", std::move(path), path_locked, false, c, d);
      successor_plan_=SuccessorPlan{};
      shadow_note("successor_plan", true, detail);
    };
    if(!q_near(q0,tip.front())) tip.front()=q0;
    if(line_validate(tip,target,quat,ori_on,frame) &&
       accept_direction(tip,&tip_c,&tip_dir)) {
      take_succ(tip,tip_c,tip_dir,"accepted");
    } else {
      std::vector<Vec18,Eigen::aligned_allocator<Vec18>> seeded;
      bool seed_locked=false;
      const Vec18 tip_q=successor_plan_.tip_samples.back();
      if(!upper_.waist_locked &&
         line_fit(q0,arm,target,quat,ori_on,false,frame,seeded,seed_locked,false,&tip_q) &&
         accept_direction(seeded,&tip_c,&tip_dir) &&
         line_validate(seeded,target,quat,ori_on,frame)) {
        take_succ(seeded,tip_c,tip_dir,"reseeds_ok");
      } else {
        shadow_note("successor_plan", false, tip_dir.ok ? "validate" : tip_dir.reason);
      }
    }
  }

  bool reverse_hard_ok=false;
  if(reverse_intent()) {
    const auto& c=last_line_cache_;
    if(!q_near(q0,c.q_end)) {
      reverse_decision="rejected_qb_mismatch";
      clear_last_line_cache();
    } else if(c.samples.size()<2) {
      reverse_decision="rejected_empty_cache";
      clear_last_line_cache();
    } else {
      std::vector<Vec18,Eigen::aligned_allocator<Vec18>> rev;
      rev.reserve(c.samples.size());
      for(auto it=c.samples.rbegin(); it!=c.samples.rend(); ++it) rev.push_back(*it);
      rev.front()=q0;
      auto cache_still_on_line = [&]() {
        const Vec3 p_a = kin_.fk_pos(rev.front(), frame);
        const Vec3 ab = target - p_a;
        const int n = static_cast<int>(rev.size()) - 1;
        if (n < 1) return false;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j <= 5; ++j) {
            const double a = j / 5.0;
            const Vec18 qq = (1.0 - a) * rev[static_cast<std::size_t>(i)] +
                             a * rev[static_cast<std::size_t>(i + 1)];
            const double s = (static_cast<double>(i) + a) / static_cast<double>(n);
            if ((kin_.fk_pos(qq, frame) - (p_a + s * ab)).norm() > kLineSampleTolM) {
              return false;
            }
          }
        }
        return true;
      };
      if(!line_validate(rev,target,quat,ori_on,frame) || !probe(rev) ||
         !cache_still_on_line()) {
        reverse_decision="rejected_validate";
      } else {
        bool pitch_ok=true;
        if(torso_pitch_max_>0) {
          const double start_ex=std::max(
              0.0,std::abs(q0.head<3>().sum())-torso_pitch_max_);
          for(const auto& q:rev) {
            const double ex=std::max(
                0.0,std::abs(q.head<3>().sum())-torso_pitch_max_);
            if(ex>start_ex+1e-9) {
              pitch_ok=false;
              break;
            }
          }
        }
        if(!pitch_ok) {
          reverse_decision="rejected_pitch";
        } else {
          const bool waist_moves=(rev.back().head<kNWaist>()-rev.front().head<kNWaist>())
                                     .cwiseAbs().maxCoeff()>waist_lock_eps_;
          if(upper_.waist_locked && waist_moves) {
            reverse_decision="rejected_waist_locked";
          } else {
            DirectionContract rev_c;
            DirectionCheckResult rev_dir;
            if(!accept_direction(rev,&rev_c,&rev_dir)) {
              reverse_decision=std::string("rejected_direction;")+rev_dir.reason;
            } else {
              push_cand("reverse_cache", std::move(rev), !waist_moves, false, rev_c, rev_dir);
              reverse_decision="accepted";
              reverse_hard_ok=true;
              shadow_note("reverse_cache", true, "accepted");
            }
          }
        }
      }
      if(!reverse_hard_ok) {
        shadow_note("reverse_cache", false, reverse_decision.c_str());
        std::vector<Vec18,Eigen::aligned_allocator<Vec18>> seeded;
        bool seed_locked=false;
        DirectionContract seed_c;
        DirectionCheckResult seed_dir;
        if(!upper_.waist_locked &&
           line_fit(q0,arm,target,quat,ori_on,false,frame,seeded,seed_locked,false,&c.q_start) &&
           (!recovering_pitch || (!seed_locked && !seeded.empty() &&
             std::abs(seeded.back().head<3>().sum())<=
                 torso_pitch_max_-kPitchSquatEndMarginRad)) &&
           probe(seeded) && line_validate(seeded,target,quat,ori_on,frame) &&
           accept_direction(seeded,&seed_c,&seed_dir)) {
          push_cand("qa_seed", std::move(seeded), seed_locked, false, seed_c, seed_dir,
                    line_fit_straighten_axis_reverted_);
          reverse_decision="qa_seed_hard_ok;"+reverse_decision;
          shadow_note("qa_seed", true, "hard_ok");
        } else {
          reverse_decision=reverse_decision+";qa_seed_rejected";
          shadow_note("qa_seed", false, "rejected");
        }
      }
    }
  } else if (reverse_last_line_enable_) {
    reverse_decision="skipped_no_match";
  }

  bool forced=false;
  // Empty-pool refuse keeps a short per-candidate trace in line_.reason so a
  // later run can see which gate blocked each regular candidate.
  std::string pool_trace;
  auto pool_note=[&](const char* name, const char* why){
    if(name==nullptr || why==nullptr) return;
    if(!pool_trace.empty()) pool_trace.push_back(';');
    pool_trace.append(name);
    pool_trace.push_back('=');
    pool_trace.append(why);
    if(pool_trace.size()>280) pool_trace.resize(280);
  };
  const bool lower_task = target.z() < p0.z() - 0.03;
  auto fold_seed=[&](Vec18& s,double deg){
    const double a=deg*kPi/180.0;
    s[0]+=a; s[1]-=2.0*a; s[2]+=a;
    for(int k=0;k<3;++k){
      s[k]=std::clamp(s[k],limits_.q_min[k]+q_limit_margin_,
                      limits_.q_max[k]-q_limit_margin_);
    }
  };
  auto lower_pose_ok = [&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path) {
    if (!lower_task || path.size() < 2) return true;
    const double pitch = std::abs(path.back().head<3>().sum());
    const double arm_net =
        (path.back().segment(arm, kNArm) - path.front().segment(arm, kNArm)).norm();
    const double waist_net = (path.back().head<3>() - path.front().head<3>()).norm();
    if (torso_pitch_max_ > 0.0 &&
        pitch > torso_pitch_max_ - 4.0 * kPi / 180.0 &&
        arm_net < 12.0 * kPi / 180.0 && waist_net > 8.0 * kPi / 180.0) {
      line_.reason = "pose_wall_arm_frozen";
      return false;
    }
    const double dx = kin_.fk(path.back(), "Body_Link")(0, 3) -
                      kin_.fk(path.front(), "Body_Link")(0, 3);
    const double foldx = kin_.fk(path.back(), "Leg_Y_Link2")(0, 3) -
                         kin_.fk(path.front(), "Leg_Y_Link2")(0, 3);
    const double drop = kin_.fk(path.front(), "Body_Link")(2, 3) -
                        kin_.fk(path.back(), "Body_Link")(2, 3);
    // Hard floor ≈20 mm sit-back; +1 mm slack so a measured −19.6 mm fold_15
    // (LoggedPrep) is not rejected by float noise while +24 mm free_fit still is.
    if (drop > 0.04 && dx > 0.10 && foldx > -0.02 + 1e-3) {
      char buf[128];
      std::snprintf(buf, sizeof(buf),
                    "pose_forward_no_sit;drop_mm=%.1f;dx_mm=%.1f;foldx_mm=%.1f",
                    drop * 1000.0, dx * 1000.0, foldx * 1000.0);
      line_.reason = buf;
      return false;
    }
    return true;
  };

  // Planner never invents lock-waist / lock-arm candidates. Upstream
  // waist_locked is still respected below. fold_15 is only a squat seed;
  // free_fit always competes on the same stage.
  auto push_fitted=[&](const char* name, const Vec18* seed, bool lock_waist_fit){
    if(upper_.waist_locked && !lock_waist_fit) {
      pool_note(name,"skipped_waist_locked");
      return;
    }
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> trial;
    bool trial_locked=false;
    if(!line_fit(q0,arm,target,quat,ori_on,lock_waist_fit,frame,trial,trial_locked,false,seed,
                 nullptr,kLineSamplePoolM)){
      pool_note(name,"fit_fail");
      shadow_note(name,false,"fit_fail");
      return;
    }
    if(recovering_pitch && (trial_locked || trial.empty() ||
        std::abs(trial.back().head<3>().sum())>
            torso_pitch_max_-kPitchSquatEndMarginRad)){
      pool_note(name,"pitch_margin");
      shadow_note(name,false,"pitch_margin");
      return;
    }
    DirectionContract fit_c;
    DirectionCheckResult fit_d;
    if(!accept_direction(trial,&fit_c,&fit_d)){
      pool_note(name,fit_d.reason[0]!='\0'?fit_d.reason:"dir");
      shadow_note(name,false,fit_d.reason);
      return;
    }
    if(!probe(trial)){
      pool_note(name,"successor");
      shadow_note(name,false,"successor");
      return;
    }
    if(!lower_pose_ok(trial)){
      if(line_.reason.rfind("pose_",0)==0) {
        pool_note(name,line_.reason.c_str());
        shadow_note(name,false,line_.reason.c_str());
      } else {
        pool_note(name,"pose");
        shadow_note(name,false,"pose");
      }
      return;
    }
    push_cand(name,std::move(trial),trial_locked||lock_waist_fit,false,fit_c,fit_d,
              line_fit_straighten_axis_reverted_);
    shadow_note(name,true,"hard_ok");
  };

  const uint64_t straighten_hits_before=line_pitch_straighten_hits_;
  auto try_free_fit=[&](){
    // recovery_seed: prefer any hard-gated exclusive endpoint already in the pool.
    const Vec18* recovery_seed=nullptr;
    Vec18 recovery_seed_storage;
    if(recovering_pitch) {
      for(const auto& cand:cands) {
        if(cand.samples.empty()) continue;
        recovery_seed_storage=cand.samples.back();
        recovery_seed=&recovery_seed_storage;
        break;
      }
    }
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> free_samples;
    bool free_locked=false;
    const bool free_ok=!upper_.waist_locked && line_fit(q0,arm,target,quat,ori_on,false,
        frame,free_samples,free_locked,false,recovery_seed,nullptr,kLineSamplePoolM);
    DirectionContract free_c;
    DirectionCheckResult free_dir;
    const bool free_pitch_ok=!recovering_pitch ||
        (!free_locked && !free_samples.empty() &&
         std::abs(free_samples.back().head<3>().sum())<=
             torso_pitch_max_-kPitchSquatEndMarginRad);
    bool free_dir_ok=false;
    if(free_ok && free_pitch_ok && accept_direction(free_samples,&free_c,&free_dir)) {
      free_dir_ok=probe(free_samples);
      if(!free_dir_ok && line_tip_) {
        line_.reason="cascade_unreachable";
      }
    } else if(free_ok && free_pitch_ok && line_direction_strict_enable_ && !free_dir.ok) {
      line_.reason=std::string("candidate_direction_violation;")+free_dir.reason;
    }
    if(free_dir_ok && lower_pose_ok(free_samples)) {
      push_cand("free_fit", std::move(free_samples), free_locked, line_tip_.has_value(),
                free_c, free_dir, line_fit_straighten_axis_reverted_);
      shadow_note("free_fit", true, "hard_ok");
    } else {
      line_pitch_straighten_hits_=straighten_hits_before;
      const char* free_why = !free_ok ? "fit_fail"
                           : !free_pitch_ok ? "pitch_margin"
                           : !free_dir.ok ? "dir"
                           : !free_dir_ok ? "successor"
                           : (line_.reason.rfind("pose_wall_arm_frozen",0)==0 ? "pose_wall"
                              : line_.reason.rfind("pose_forward_no_sit",0)==0 ? "pose_nosit"
                              : "pose");
      if(free_why[0]=='p' && line_.reason.rfind("pose_",0)==0) {
        pool_note("free_fit", line_.reason.c_str());
        shadow_note("free_fit", false, line_.reason.c_str());
      } else {
        pool_note("free_fit", free_why);
        shadow_note("free_fit", false, free_why);
      }
    }
  };

  pool_note("arm_only","skipped_no_lock_candidates");
  shadow_note("arm_only", false, "skipped_no_lock_candidates");

  // fold_15 = squat seed (waist+arm free in the fit). Always compete with free_fit.
  if(!upper_.waist_locked && (lower_task || recovering_pitch)){
    Vec18 s15=q0;
    fold_seed(s15,15.0);
    push_fitted("fold_15",&s15,false);
  } else {
    pool_note("fold_15", lower_task || recovering_pitch ? "skipped_waist_locked"
                                                       : "skipped_not_lower");
  }
  try_free_fit();

  if(cands.empty()) {
    ok=false;
    line_.failure_category="pool_empty";
    // Do not bury tip-unreachable under pool_empty (cascade callers).
    if(line_.reason!="cascade_unreachable") {
      if(pool_trace.empty()) {
        line_.reason="pool_empty";
      } else {
        line_.reason=std::string("pool_empty;")+pool_trace;
      }
    }
    shadow_note("pool_empty", false, pool_trace.empty()?"none":pool_trace.c_str());
  } else {
    std::size_t best_i=0;
    double best_q=pose_quality(cands[0].samples);
    double best_travel=path_cost(cands[0].samples);
    for(std::size_t i=1;i<cands.size();++i){
      const double q=pose_quality(cands[i].samples);
      const double travel=path_cost(cands[i].samples);
      if(q+0.02<best_q || (std::abs(q-best_q)<=0.02 && travel+1e-9<best_travel)){
        best_i=i; best_q=q; best_travel=travel;
      }
    }
    auto& best=cands[best_i];
    const char* won=best.name;
    const bool fit_won=std::strcmp(won,"free_fit")==0 || std::strcmp(won,"qa_seed")==0 ||
        std::strcmp(won,"fold_10")==0 || std::strcmp(won,"fold_15")==0 ||
        std::strcmp(won,"fold_20")==0 || std::strcmp(won,"waist_lock")==0;
    if(!fit_won) line_pitch_straighten_hits_=straighten_hits_before;
    {
      const auto& path=best.samples;
      const double z0=kin_.fk(path.front(),"Body_Link")(2,3);
      const double zN=kin_.fk(path.back(),"Body_Link")(2,3);
      const double x0=kin_.fk(path.front(),"Body_Link")(0,3);
      const double xN=kin_.fk(path.back(),"Body_Link")(0,3);
      const double f0=kin_.fk(path.front(),"Leg_Y_Link2")(0,3);
      const double fN=kin_.fk(path.back(),"Leg_Y_Link2")(0,3);
      char buf[192];
      std::snprintf(buf,sizeof(buf),
          "src=%s;q=%.3f;drop_mm=%.1f;dx_mm=%.1f;foldx_mm=%.1f;pitch_deg=%.2f",
          won,best_q,(z0-zN)*1000.0,(xN-x0)*1000.0,(fN-f0)*1000.0,
          path.back().head<3>().sum()*180.0/kPi);
      pose_report=buf;
    }
    samples=std::move(best.samples);
    locked=best.locked;
    forced=best.forced;
    straighten_axis_reverted=best.straighten_reverted;
    candidate=best.name;
    accepted_contract=best.contract;
    accepted_dir=best.dir;
    have_accepted_dir=true;
    ok=true;
    if(std::strcmp(best.name,"reverse_cache")==0) {
      reverse_decision="accepted";
    } else if(std::strcmp(best.name,"qa_seed")==0) {
      if(reverse_decision.find("qa_seed_hard_ok")==std::string::npos) {
        reverse_decision="qa_seed_accepted;"+reverse_decision;
      } else {
        reverse_decision="qa_seed_accepted;"+reverse_decision;
      }
    }
    char pick[48];
    std::snprintf(pick,sizeof(pick),"win:%.40s",pose_report.c_str());
    shadow_note("pool_pick", true, pick);
  }

  // Densify without an unconditional second full line_fit: joint-lerp the pool
  // winner up to ~kLineSampleM, keep if hard gates pass; only then fall back to
  // a fine line_fit. Avoids the r37 ~160ms double-fit on long fold/free paths.
  if(ok && !samples.empty() &&
     (candidate=="free_fit" || candidate=="fold_15" || candidate=="qa_seed" ||
      candidate=="fold_10" || candidate=="fold_20" || candidate=="waist_lock")) {
    auto densify_ok=[&](std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path,
                        DirectionContract* c, DirectionCheckResult* d)->bool {
      return !path.empty() && accept_direction(path,c,d) && probe(path) &&
             lower_pose_ok(path) && line_validate(path,target,quat,ori_on,frame);
    };
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> fine;
    fine.reserve(samples.size()*2);
    fine.push_back(samples.front());
    for(std::size_t i=1;i<samples.size();++i) {
      const Vec18& a=samples[i-1];
      const Vec18& b=samples[i];
      const double hand_step=
          (kin_.fk_pos(b,frame)-kin_.fk_pos(a,frame)).norm();
      const int n_mid = hand_step > kLineSampleM
                            ? static_cast<int>(std::ceil(hand_step / kLineSampleM)) - 1
                            : 0;
      for(int k=1;k<=n_mid;++k) {
        const double u=static_cast<double>(k)/(n_mid+1);
        fine.push_back(((1.0-u)*a+u*b).eval());
      }
      fine.push_back(b);
      if(static_cast<int>(fine.size())>kLineMaxSamples) break;
    }
    if(static_cast<int>(fine.size())>kLineMaxSamples) fine.resize(kLineMaxSamples);
    DirectionContract fine_c;
    DirectionCheckResult fine_d;
    bool densified=false;
    if(densify_ok(fine,&fine_c,&fine_d)) {
      samples=std::move(fine);
      accepted_contract=fine_c;
      accepted_dir=fine_d;
      have_accepted_dir=true;
      densified=true;
    }
    if(!densified) {
      fine.clear();
      bool fine_locked=false;
      Vec18 densify_seed_storage;
      const Vec18* densify_seed=&samples.back();
      if(candidate=="fold_15" || candidate=="fold_10" || candidate=="fold_20") {
        densify_seed_storage=q0;
        const double deg = candidate=="fold_10" ? 10.0
                         : candidate=="fold_20" ? 20.0 : 15.0;
        fold_seed(densify_seed_storage, deg);
        densify_seed=&densify_seed_storage;
      } else if(candidate=="qa_seed") {
        densify_seed=&samples.back();
      }
      if(line_fit(q0,arm,target,quat,ori_on,locked,frame,fine,fine_locked,false,
                  densify_seed,nullptr,kLineSampleM) &&
         densify_ok(fine,&fine_c,&fine_d)) {
        samples=std::move(fine);
        locked=fine_locked;
        accepted_contract=fine_c;
        accepted_dir=fine_d;
        have_accepted_dir=true;
        straighten_axis_reverted=line_fit_straighten_axis_reverted_;
      }
    }
  }

  // One tip fit for the winner only — next segment reuses successor_plan_.
  bool cascade_full_body=false;
  if(ok && !samples.empty() && line_tip_.has_value()) {
    const Vec18& q_b=samples.back();
    const Vec4* tip_quat =
        line_tip_->ori_on ? &line_tip_->quat_xyzw : nullptr;
    const bool tip_ori=line_tip_->ori_on;
    const bool tip_lower =
        line_tip_->xyz.z() < kin_.fk_pos(q_b, frame).z() - 0.03;
    auto try_tip=[&](const Vec18* seed)->bool {
      std::vector<Vec18,Eigen::aligned_allocator<Vec18>> tip;
      bool tip_locked=false;
      DirectionContract tip_c;
      DirectionCheckResult tip_d;
      if(!line_fit(q_b,arm,line_tip_->xyz,tip_quat,tip_ori,false,frame,tip,
                   tip_locked,false,seed,nullptr,kLineSampleM)) {
        return false;
      }
      if(tip.size()<2) return false;
      if(!line_validate(tip,line_tip_->xyz,tip_quat,tip_ori,frame)) return false;
      if(!accept_direction(tip,&tip_c,&tip_d)) return false;
      successor_plan_=SuccessorPlan{};
      successor_plan_.valid=true;
      successor_plan_.tip_xyz=line_tip_->xyz;
      successor_plan_.tip_quat=tip_ori ? *tip_quat : Vec4(0,0,0,1);
      successor_plan_.tip_ori_on=tip_ori;
      successor_plan_.tip_samples=std::move(tip);
      successor_plan_.tip_contract=tip_c;
      successor_plan_.tip_check=tip_d;
      successor_plan_.generation=goal_generation_;
      successor_plan_.left=left;
      successor_plan_.frame=frame;
      successor_plan_.q_boundary=q_b;
      return true;
    };
    bool tip_ok=false;
    if(tip_lower && !upper_.waist_locked) {
      Vec18 s15=q_b;
      fold_seed(s15,15.0);
      tip_ok=try_tip(&s15);
    }
    if(!tip_ok) tip_ok=try_tip(&q_b);
    cascade_full_body=tip_ok;
    shadow_note("successor_commit", tip_ok, tip_ok ? "ok" : "fit_fail");
  }

  auto path_len=[&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path,int begin,int count){
    double s=0.0;
    for(std::size_t i=1;i<path.size();++i){
      s+=(path[i]-path[i-1]).segment(begin,count).cwiseAbs().sum();
    }
    return s;
  };
  auto elbow_err=[&](const std::vector<Vec18,Eigen::aligned_allocator<Vec18>>& path){
    if(path.empty()) return 0.0;
    return std::abs(path.back()[arm+3]-line_q_pref_[arm+3]);
  };
  // Optional posture candidate: only after a feasible baseline exists, waist may
  // move, and loaded q_pref is available. Default-off; never loosens hard checks.
  // Evidence on field exit fixtures: baseline often selects free_fit (arm-only
  // fails), still with large elbow deviation. Allow pose_pref on both.
  // Skip for reverse_cache / qa_seed — those already restore a known posture.
  const bool pose_eligible =
      candidate == "free_fit" || candidate == "fold_15";
  // Pitch-recovery segments may still accept free_fit; allow pose_pref with the
  // same end-pitch gate used by free_fit, without relaxing the pitch limit.
  if(ok && !samples.empty() && line_pose_pref_enable_ && !upper_.waist_locked &&
     pose_eligible) {
    const double base_elbow=elbow_err(samples);
    if(base_elbow < line_pose_pref_min_improve_) {
      pose_decision="skipped_near_pref";
    } else {
      std::vector<Vec18,Eigen::aligned_allocator<Vec18>> pref_samples;
      bool pref_locked=false;
      // Seed from the accepted baseline endpoint so soft posture pull starts from a
      // known feasible Cartesian solution instead of cold-starting near HOME.
      const Vec18 seed=samples.back();
      const bool fit_ok=line_fit(q0,arm,target,quat,ori_on,false,frame,pref_samples,
          pref_locked,false,&seed,&line_q_pref_);
      if(!fit_ok || pref_samples.empty()) {
        pose_decision="rejected_fit";
      } else if(pref_locked) {
        pose_decision="rejected_relock";
      } else if(recovering_pitch &&
                std::abs(pref_samples.back().head<3>().sum())>
                    torso_pitch_max_-kPitchSquatEndMarginRad) {
        pose_decision="rejected_pitch";
      } else if(!probe(pref_samples)) {
        pose_decision="rejected_cascade";
      } else if(!line_validate(pref_samples,target,quat,ori_on,frame)) {
        pose_decision="rejected_validate";
      } else {
        const double improve=base_elbow-elbow_err(pref_samples);
        const double waist_extra=path_len(pref_samples,0,kNWaist)-path_len(samples,0,kNWaist);
        const double wrist_extra=path_len(pref_samples,arm+5,2)-path_len(samples,arm+5,2);
        char metric[160];
        std::snprintf(metric,sizeof(metric),
            "improve_deg=%.3f;waist_extra_deg=%.3f;wrist_extra_deg=%.3f;elbow_end_err_deg=%.3f",
            improve*180.0/kPi,waist_extra*180.0/kPi,wrist_extra*180.0/kPi,
            elbow_err(pref_samples)*180.0/kPi);
        DirectionContract pref_c;
        DirectionCheckResult pref_dir;
        if(improve < line_pose_pref_min_improve_) {
          pose_decision=std::string("rejected_no_improve;")+metric;
        } else if(waist_extra > line_pose_pref_waist_budget_) {
          pose_decision=std::string("rejected_waist_budget;")+metric;
        } else if(wrist_extra > line_pose_pref_wrist_budget_) {
          pose_decision=std::string("rejected_wrist_budget;")+metric;
        } else if(!accept_direction(pref_samples,&pref_c,&pref_dir)) {
          pose_decision=std::string("rejected_direction;")+pref_dir.reason+";"+metric;
        } else {
          straighten_axis_reverted=line_fit_straighten_axis_reverted_;
          samples=std::move(pref_samples);
          locked=pref_locked;
          forced=line_tip_.has_value();
          candidate="pose_pref";
          pose_decision=std::string("accepted;")+metric;
          accepted_contract=pref_c; accepted_dir=pref_dir; have_accepted_dir=true;
        }
      }
    }
  } else if(!line_pose_pref_enable_) {
    pose_decision="disabled";
  } else if(ok && !pose_eligible) {
    pose_decision="skipped_non_baseline";
  } else if(upper_.waist_locked) {
    pose_decision="skipped_waist_locked";
  } else {
    pose_decision="skipped";
  }

  // Strict mode: absolute waist/pitch direction is the final gate for every
  // candidate. Old sum-only straighten + axis-reverted exemption is baseline-only
  // when strict is off; never auto-relax the hard mask.
  constexpr double kPitchTurnbackMax=0.05*kPi/180.0;
  if (ok && !samples.empty()) {
    if(line_direction_strict_enable_) {
      if(!have_accepted_dir || !accepted_dir.ok) {
        if(!accept_direction(samples,&accepted_contract,&accepted_dir)) {
          ok=false;
          have_accepted_dir=true;
          line_.reason=std::string("direction_search_exhausted;")+accepted_dir.reason;
        } else {
          have_accepted_dir=true;
        }
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
      } else if (cross > kLineTubeHardM) {
        if (why) {
          *why = "tube";
        }
        return false;
      }
      if (ori_on && quat &&
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
      // Also require direction contract on Hermite-sampled dense knots.
      bool dir_ok = true;
      if (line_direction_strict_enable_ && accepted_dir.ok) {
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
  if (line_rate_ > 1e-4 && !line_path_qp(q_from,line_.prog,step_lim,next)) {
    line_.prog=old_prog;
    line_commit_pending_=false;
    line_diag_push(LineDiagLayer::FollowQp,q_from,q_from,line_.prog,0.0,
                   line_.candidate.c_str(),"qp_fail");
    refuse_line(line_.reason.empty()?"path_qp_or_tube":line_.reason); out.dq.setZero(); return false;
  }
  line_diag_push(LineDiagLayer::FollowQp,next,q_from,line_.prog,0.0,
                 line_.candidate.c_str(),"qp");
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
