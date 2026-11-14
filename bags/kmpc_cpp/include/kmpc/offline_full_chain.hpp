#pragma once
// Offline A/B full-chain runner: planner → unified time law → QP → 50/250 Hz
// stream → production final safety gate. Shared helpers; no ROS/hardware.
// Feedback is driven by the actual final filtered command (ideal/lag/stale).
// Fault localization: first anomalous layer / time / joint / value / reason.
// Insufficient evidence → "无法定位" (unknown), never guess.
//
// Consumers must add bags/kmpc_cpp/ros2 to the include path (for line_stream.hpp).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/StdVector>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/line_async_handoff.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/line_final_gate.hpp"
#include "kmpc/path_layer_metrics.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/types.hpp"
#include "kmpc/whole_body_guards.hpp"
#include "line_stream.hpp"

namespace kmpc {

enum class ChainLayer : uint8_t {
  Plan = 0,
  TimeLaw,
  Qp,
  Commit,
  InterpRaw,
  InterpFilt,
  FinalGate,
  Unknown,
};

inline const char* chain_layer_name(ChainLayer layer) {
  switch (layer) {
    case ChainLayer::Plan: return "plan";
    case ChainLayer::TimeLaw: return "time_law";
    case ChainLayer::Qp: return "qp";
    case ChainLayer::Commit: return "commit";
    case ChainLayer::InterpRaw: return "interp_raw";
    case ChainLayer::InterpFilt: return "interp_filt";
    case ChainLayer::FinalGate: return "final_gate";
    case ChainLayer::Unknown: return "unknown";
  }
  return "unknown";
}

enum class FeedbackMode : uint8_t {
  IdealFinal = 0,  // q_meas follows final (default, proves gate+drain)
  LagFinal = 1,    // feedback is final from N stream ticks ago (see lag_ticks)
  StaleHold = 2,   // freeze feedback after first commit (stale path)
};

struct ChainFault {
  bool found = false;
  bool unknown = false;  // 无法定位
  ChainLayer layer = ChainLayer::Unknown;
  double t_s = 0.0;
  int joint = -1;
  double value = 0.0;
  double limit = 0.0;
  char reason[128]{"ok"};
};

struct ChainTickRecord {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t tick_id = 0;
  uint64_t generation = 0;
  double t_s = 0.0;
  double dt_s = 0.0;
  double prog = 0.0;
  double s_dot = 0.0;
  double commit_scale = 1.0;
  double qp_dx = 0.0;
  double qp_proj = 0.0;
  uint8_t qp_mode = 0;
  bool plan_ran = false;
  bool qp_ran = false;
  bool commit_ran = false;
  bool interp_ran = false;
  bool final_ran = false;
  bool missing = false;
  char note[96]{};
  Vec18 q_plan = Vec18::Zero();
  Vec18 q_qp = Vec18::Zero();
  Vec18 q_commit = Vec18::Zero();
  Vec18 q_raw = Vec18::Zero();
  Vec18 q_filt = Vec18::Zero();
  Vec18 q_final = Vec18::Zero();
  Vec18 q_feedback = Vec18::Zero();
};

struct ChainStreamSample {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t tick_id = 0;
  uint64_t stream_i = 0;
  double t_s = 0.0;
  double dt_s = 0.0;
  Vec18 q_raw = Vec18::Zero();
  Vec18 q_filt = Vec18::Zero();
  Vec18 q_final = Vec18::Zero();
  Vec18 q_feedback = Vec18::Zero();
  Vec18 dq_final = Vec18::Zero();
  Vec18 ddq_final = Vec18::Zero();
};

struct FirstAxisAnomaly {
  bool found = false;
  ChainLayer layer = ChainLayer::Unknown;
  double t_s = 0.0;
  int joint = -1;
  double value = 0.0;
  double limit = 0.0;
  char source[64]{"none"};
};

struct ChainRunResult {
  std::string profile_name;
  bool completed = false;
  bool cmd_arrived = false;       // final command within pose tol
  bool feedback_arrived = false;  // feedback near final command
  bool refused = false;
  std::string refuse_reason;
  double final_pos_err_m = 0.0;
  double final_ori_err_rad = 0.0;
  double max_waist_speed = 0.0;
  double max_pitch_speed = 0.0;
  double max_joint_speed = 0.0;
  double max_joint_accel = 0.0;
  double max_qp_dx = 0.0;
  int commit_scale_events = 0;
  int reverse_events = 0;
  double max_reverse_deg = 0.0;
  PathSmoothReport smooth{};
  PathCornerMetrics corners{};
  ChainFault first_fault{};
  FirstAxisAnomaly first_axis{};
  std::vector<ChainTickRecord> ticks;
  std::vector<ChainStreamSample> stream;
};

struct OfflineChainConfig {
  double control_dt = 0.02;   // 50 Hz
  double stream_dt = 0.004;   // 250 Hz
  double line_dq_scale = 0.60;
  double speed_scale = 0.35;
  double accel_s = 0.3;
  int max_control_ticks = 400;
  bool simulate_stream_advance = true;
  double stream_advance_frac = 0.25;
  FeedbackMode feedback = FeedbackMode::IdealFinal;
  // LagFinal: feedback trails final by this many stream samples (>=1).
  // Parameterized for later field calibration; not a device claim.
  int feedback_lag_ticks = 1;
  // StaleHold: after this many control ticks from t=0, resume IdealFinal
  // (negative = never resume). Used to test freeze then recover.
  int feedback_resume_control_ticks = -1;
  int drain_extra_ticks = 40;  // continue after hold until feedback catches
  double arrive_pos_tol_m = 0.01;
  double arrive_ori_tol_rad = 0.08;
  double feedback_joint_tol_rad = 0.02;
  // When true, skip reset/profile/rearm/rebuild and follow an already-adopted
  // active line (worker→main Hermite handoff tests).
  bool reuse_active_line = false;
};

// Resolve a report slot under a shared parent so A/B and Hermite (or any two
// writers) never contend for the same summary.md when KMPC_CHAIN_REPORT_DIR is
// fixed for a whole suite run.
inline std::string chain_report_slot_dir(const std::string& parent,
                                         const char* slot) {
  if (!slot || !slot[0]) {
    return parent;
  }
  return parent + "/" + slot;
}

inline ChainFault make_fault(ChainLayer layer, double t, int joint, double value,
                             double limit, const char* reason,
                             bool unknown = false) {
  ChainFault f;
  f.found = true;
  f.unknown = unknown;
  f.layer = layer;
  f.t_s = t;
  f.joint = joint;
  f.value = value;
  f.limit = limit;
  std::snprintf(f.reason, sizeof(f.reason), "%s", reason ? reason : "unknown");
  return f;
}

inline ChainLayer refuse_layer_from_reason(const std::string& reason) {
  if (reason.rfind("qp_", 0) == 0 || reason.find("qp_") != std::string::npos) {
    return ChainLayer::Qp;
  }
  if (reason.find("plan") != std::string::npos ||
      reason.find("build") != std::string::npos ||
      reason.find("ik_") != std::string::npos ||
      reason.find("sample") != std::string::npos ||
      reason.find("cascade") != std::string::npos ||
      reason.find("direction") != std::string::npos ||
      reason.find("pitch") != std::string::npos ||
      reason.find("tube") != std::string::npos) {
    // Plan-time rejects share many tags; if not qp_* treat as plan unless
    // clearly stream/final.
    if (reason.find("stream_") == 0 || reason.find("final_") == 0) {
      return ChainLayer::FinalGate;
    }
    return ChainLayer::Plan;
  }
  if (reason.find("stream_") == 0 || reason.find("final_") != std::string::npos) {
    return ChainLayer::FinalGate;
  }
  return ChainLayer::Unknown;
}

inline void write_vec18_csv(std::ostream& os, const Vec18& q) {
  for (int i = 0; i < kNDof; ++i) {
    if (i) {
      os << ',';
    }
    os << q[i];
  }
}

inline ChainRunResult run_offline_full_chain(
    CartesianTracker& tracker, Kinematics& kin, const Config& cfg,
    const Vec18& q0, const Vec3& goal, const Vec4& quat, bool left,
    CartesianTracker::LineChainProfile profile,
    const OfflineChainConfig& opt = {}) {
  ChainRunResult out;
  out.profile_name = CartesianTracker::line_chain_profile_name(profile);
  if (!opt.reuse_active_line) {
    tracker.reset();
    tracker.set_line_chain_profile(profile);
    tracker.rearm_line(1);
  }
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(opt.speed_scale, opt.accel_s);
  tracker.set_line_diag_enable(true);
  const std::string frame =
      left ? std::string(kEeFrameLeft) : std::string(kEeFrameRight);
  if (opt.reuse_active_line && !tracker.line_active()) {
    out.first_fault =
        make_fault(ChainLayer::Plan, 0.0, -1, 0.0, 0.0,
                   "reuse_active_line_but_inactive", /*unknown=*/true);
    return out;
  }

  const LineFinalGateLimits gate_lim = make_final_gate_limits(cfg);
  const WholeBodyGuards& guards = tracker.guards();

  Vec18 q_meas = q0;
  Vec18 stream_from = q0;
  Vec18 stream_to = q0;
  Vec18 stream_raw = q0;
  Vec18 stream_pending_to = q0;
  Vec18 q_final = q0;
  Vec18 q_feedback = q0;
  Vec18 prev_final = q0;
  Vec18 prev_dq = Vec18::Zero();
  double stream_u = 1.0;
  bool stream_active = false;
  bool stream_pending = false;
  bool geo_ready = false;
  LineFinalGateGeo geo{};
  ros2io::LineSmoother smoother;
  smoother.reset(q0);
  DirectionEnvelope dir_env{};
  bool dir_seeded = false;
  double t = 0.0;
  bool hold_seen = false;
  int drain_left = 0;
  int control_tick_i = 0;
  bool stale_frozen = false;
  bool stale_resumed = false;
  const int lag_n = std::max(1, opt.feedback_lag_ticks);
  std::deque<Vec18, Eigen::aligned_allocator<Vec18>> lag_buf;
  for (int i = 0; i < lag_n; ++i) {
    lag_buf.push_back(q0);
  }

  auto note_fault_once = [&](const ChainFault& f) {
    if (!out.first_fault.found) {
      out.first_fault = f;
    }
  };
  auto note_axis_once = [&](ChainLayer layer, double ts, int j, double v,
                            double lim, const char* src) {
    if (out.first_axis.found) {
      return;
    }
    out.first_axis.found = true;
    out.first_axis.layer = layer;
    out.first_axis.t_s = ts;
    out.first_axis.joint = j;
    out.first_axis.value = v;
    out.first_axis.limit = lim;
    std::snprintf(out.first_axis.source, sizeof(out.first_axis.source), "%s",
                  src ? src : "unknown");
  };

  auto sync_geo = [&]() {
    if (!tracker.line_active() || tracker.line_samples().empty()) {
      return;
    }
    geo.frame = tracker.line_ee_frame();
    geo.target = tracker.line_target_xyz();
    geo.quat1 = tracker.line_target_quat();
    geo.ori_on = tracker.line_ori_on();
    const Vec18& q_start = tracker.line_samples().front();
    const auto pose0 = kin.fk(q_start, geo.frame);
    geo.start = pose0.col(3).head<3>();
    geo.quat0 = rot_to_quat_xyzw(pose0.block<3, 3>(0, 0));
    if (!geo_ready) {
      geo.previous_along = 0.0;
      geo_ready = true;
    }
  };

  // Must assign feedback from the *previous* finals before pushing current.
  auto update_feedback = [&](const Vec18& final_cmd) {
    const bool resume =
        opt.feedback_resume_control_ticks >= 0 &&
        control_tick_i >= opt.feedback_resume_control_ticks;
    if (resume) {
      stale_resumed = true;
      stale_frozen = false;
    }
    if (opt.feedback == FeedbackMode::StaleHold && stale_frozen && !resume) {
      return;  // keep frozen feedback
    }
    if (opt.feedback == FeedbackMode::LagFinal && !resume) {
      q_feedback = lag_buf.front();
      lag_buf.pop_front();
      lag_buf.push_back(final_cmd);
      return;
    }
    if (opt.feedback == FeedbackMode::StaleHold && !stale_frozen &&
        !stream_active) {
      q_feedback = final_cmd;
      return;
    }
    // IdealFinal, or StaleHold resumed / pre-commit.
    q_feedback = final_cmd;
  };

  for (int tick = 0; tick < opt.max_control_ticks; ++tick) {
    control_tick_i = tick;
    ChainTickRecord rec;
    rec.tick_id = static_cast<uint64_t>(tick + 1);
    rec.generation = tracker.goal_generation();
    rec.t_s = t;
    rec.dt_s = opt.control_dt;
    tracker.set_control_tick_id(rec.tick_id);

    if (opt.simulate_stream_advance && stream_active && stream_u < 1.0) {
      stream_u = std::min(1.0, stream_u + opt.stream_advance_frac);
      stream_raw = (1.0 - stream_u) * stream_from + stream_u * stream_to;
    }
    if (stream_active && stream_pending && stream_u >= 1.0 - 1e-15) {
      stream_from = stream_to;
      stream_to = stream_pending_to;
      stream_u = 0.0;
      stream_raw = stream_from;
      stream_pending = false;
    }

    // Feedback for next plan: driven by final (not raw).
    q_meas = q_feedback;
    const Vec18 line_ref = stream_pending ? stream_pending_to
                           : stream_active ? stream_to
                                           : q_meas;
    tracker.set_line_reference(line_ref);
    const Vec3* L = left ? &goal : nullptr;
    const Vec3* R = left ? nullptr : &goal;
    const Vec4* qL = left ? &quat : nullptr;
    const Vec4* qR = left ? nullptr : &quat;
    // Depth-1 queue full: do not propose a replacement endpoint mid-chord.
    const bool wait_chord_queue =
        stream_pending && stream_u < 1.0 - 1e-9;
    if (wait_chord_queue) {
      rec.plan_ran = tracker.line_active();
      rec.prog = tracker.line_progress();
      rec.s_dot = tracker.line_rate();
      std::snprintf(rec.note, sizeof(rec.note), "wait_chord_queue");
    } else {
      (void)tracker.compute(q_meas, cfg.q_pref, L, R, qL, qR);
      rec.plan_ran = tracker.line_active() || tracker.line_refused();
      rec.prog = tracker.line_progress();
      rec.s_dot = tracker.line_rate();
      rec.qp_mode = static_cast<uint8_t>(tracker.line_qp_follow_mode());
      rec.qp_dx = tracker.last_line_qp_dx_norm();
      rec.qp_proj = tracker.last_line_qp_proj_max();
      out.max_qp_dx = std::max(out.max_qp_dx, rec.qp_dx);
      out.smooth = tracker.last_path_smooth_report();
      out.corners = tracker.last_path_corner_metrics();
      sync_geo();

      if (tracker.line_refused()) {
        out.refused = true;
        out.refuse_reason = tracker.line_reason();
        const ChainLayer layer = refuse_layer_from_reason(out.refuse_reason);
        rec.qp_ran = (layer == ChainLayer::Qp);
        rec.plan_ran = true;
        std::snprintf(rec.note, sizeof(rec.note), "refused:%.80s",
                      out.refuse_reason.c_str());
        note_fault_once(make_fault(
            layer == ChainLayer::Unknown ? ChainLayer::Unknown : layer, t, -1,
            0.0, 0.0, out.refuse_reason.c_str(),
            /*unknown=*/layer == ChainLayer::Unknown));
        out.ticks.push_back(rec);
        break;
      }

      if (!tracker.line_active()) {
        std::snprintf(rec.note, sizeof(rec.note), "inactive");
        out.ticks.push_back(rec);
        t += opt.control_dt;
        continue;
      }

      if (tracker.line_holding() || tracker.last_arrived()) {
        hold_seen = true;
        if (drain_left <= 0) {
          drain_left = opt.drain_extra_ticks;
        }
        // Measure error on final command (what would be sent).
        const auto pose = kin.fk(q_final, frame);
        out.final_pos_err_m = (goal - pose.col(3).head<3>()).norm();
        out.final_ori_err_rad =
            ori_error_xyzw(quat, rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                .norm();
        const bool cmd_ok =
            out.final_pos_err_m <= opt.arrive_pos_tol_m &&
            out.final_ori_err_rad <= opt.arrive_ori_tol_rad;
        out.cmd_arrived = out.cmd_arrived || cmd_ok;
        const double fb_err = (q_feedback - q_final).cwiseAbs().maxCoeff();
        const bool fb_ok = fb_err < opt.feedback_joint_tol_rad;
        out.feedback_arrived = out.feedback_arrived || (cmd_ok && fb_ok);
        // Completed only when BOTH command and feedback are at goal.
        if (cmd_ok && fb_ok) {
          out.completed = true;
          std::snprintf(rec.note, sizeof(rec.note), "hold_arrived");
          out.ticks.push_back(rec);
          break;
        }
        if (--drain_left <= 0) {
          // Drain timeout must NOT ignore feedback lag/freeze.
          if (cmd_ok && fb_ok) {
            out.completed = true;
            std::snprintf(rec.note, sizeof(rec.note), "hold_drain_done");
          } else if (cmd_ok && !fb_ok) {
            note_fault_once(make_fault(ChainLayer::FinalGate, t, -1, fb_err,
                                       opt.feedback_joint_tol_rad,
                                       "hold_feedback_not_caught_up"));
            std::snprintf(rec.note, sizeof(rec.note),
                          "hold_feedback_not_caught_up");
          } else {
            note_fault_once(make_fault(ChainLayer::FinalGate, t, -1,
                                       out.final_pos_err_m, opt.arrive_pos_tol_m,
                                       "hold_not_arrived"));
            std::snprintf(rec.note, sizeof(rec.note), "hold_not_arrived");
          }
          out.ticks.push_back(rec);
          break;
        }
        std::snprintf(rec.note, sizeof(rec.note), "hold_drain");
        // Keep streaming last segment to drain smoother/feedback.
      }

      rec.q_plan = tracker.last_line_qp_nominal().allFinite()
                       ? tracker.last_line_qp_nominal()
                       : tracker.line_cmd();
      rec.q_qp = tracker.line_cmd();
      rec.qp_ran = true;

      // Tail-feedback wait / pace ramp: no new propose, but keep draining the
      // 250 Hz smoother so IdealFinal feedback can catch the last stream_to.
      if (!hold_seen && !tracker.line_commit_pending()) {
        std::snprintf(rec.note, sizeof(rec.note),
                      stream_active ? "tail_wait_drain" : "no_propose_wait");
        if (!stream_active) {
          out.ticks.push_back(rec);
          t += opt.control_dt;
          continue;
        }
        // fall through to stream loop with stream_u already at/near 1
      }
    }

    if (!wait_chord_queue && tracker.line_commit_pending()) {
      // Mirror production: commit geometry from the last accepted endpoint, not
      // a mid-chord sample. Mid-chord rebase produced the exact 5/6 scale stutter.
      const Vec18 commit_from = stream_active ? stream_to : stream_raw;
      const bool queue_until_end =
          stream_active && stream_u < 1.0 - 1e-9;
      const Vec18 proposed = tracker.line_cmd();
      const Vec18 cap =
          tracker.effective_dq_max() * opt.control_dt * opt.line_dq_scale;
      const auto bounded =
          ros2io::bound_line_stream_step(commit_from, proposed, cap);
      if (!bounded) {
        tracker.abort_line_stream_commit();
        note_fault_once(make_fault(ChainLayer::Commit, t, -1, 0.0, 0.0,
                                   "stream_nonfinite_or_invalid_cap"));
        std::snprintf(rec.note, sizeof(rec.note), "commit_reject_cap");
        out.ticks.push_back(rec);
        break;
      }
      rec.commit_scale = bounded->scale;
      if (bounded->scale < 1.0 - 1e-9) {
        ++out.commit_scale_events;
      }
      const auto check =
          tracker.line_stream_segment_check(commit_from, bounded->target);
      if (!check.ok) {
        tracker.abort_line_stream_commit();
        note_fault_once(make_fault(ChainLayer::Commit, t, -1,
                                   check.tube_margin_m, 0.0, check.reason));
        std::snprintf(rec.note, sizeof(rec.note), "commit_reject_%.70s",
                      check.reason);
        out.ticks.push_back(rec);
        break;
      }
      if (!tracker.confirm_line_stream_commit(bounded->target, bounded->scale)) {
        note_fault_once(make_fault(
            ChainLayer::Commit, t, -1,
            tracker.last_progress_recover().residual_rad, 0.015,
            tracker.last_progress_recover().reason,
            /*unknown=*/tracker.last_progress_recover().unknown));
        std::snprintf(rec.note, sizeof(rec.note), "progress_%.80s",
                      tracker.last_progress_recover().reason);
        out.ticks.push_back(rec);
        break;
      }
      rec.commit_ran = true;
      rec.q_commit = bounded->target;
      rec.prog = tracker.line_progress();
      if (queue_until_end) {
        stream_pending = true;
        stream_pending_to = bounded->target;
        std::snprintf(rec.note, sizeof(rec.note), "commit_queued");
      } else {
        if (stream_active) {
          stream_from = stream_to;
        } else {
          stream_from = commit_from;
        }
        stream_to = bounded->target;
        stream_u = 0.0;
        stream_raw = stream_from;
        stream_active = true;
        stream_pending = false;
      }
      if (opt.feedback == FeedbackMode::StaleHold && !stale_resumed &&
          !stale_frozen) {
        q_feedback = stream_from;  // freeze once at first commit
        stale_frozen = true;
      }
      if (!dir_seeded && tracker.line_direction_strict_enabled()) {
        direction_envelope_seed(&dir_env, tracker.line_direction_contract(),
                                stream_from);
        dir_seeded = true;
      }
    }

    const int n_stream = std::max(
        1, static_cast<int>(std::lround(opt.control_dt / opt.stream_dt)));
    bool stream_fault = false;
    for (int s = 0; s < n_stream; ++s) {
      if (stream_active && stream_pending && stream_u >= 1.0 - 1e-15) {
        stream_from = stream_to;
        stream_to = stream_pending_to;
        stream_u = 0.0;
        stream_raw = stream_from;
        stream_pending = false;
      }
      if (stream_active) {
        stream_u = std::min(1.0, stream_u + opt.stream_dt / opt.control_dt);
        stream_raw = (1.0 - stream_u) * stream_from + stream_u * stream_to;
      }
      const Vec18 filt = smoother.push(stream_raw, opt.stream_dt);
      rec.interp_ran = true;
      rec.q_raw = stream_raw;
      rec.q_filt = filt;

      // Production final gate (joint/pitch/CoM/FK tube/progress/ori).
      LineFinalGateGeo gate_geo = geo;
      const char* fault = line_final_filtered_fault(
          filt, /*pitch_extra=*/0.0, gate_lim, kin, guards,
          geo_ready ? &gate_geo : nullptr);
      if (fault != nullptr) {
        note_fault_once(make_fault(ChainLayer::FinalGate,
                                   t + s * opt.stream_dt, -1, 0.0, 0.0, fault));
        std::snprintf(rec.note, sizeof(rec.note), "final_%.80s", fault);
        out.refused = true;
        out.refuse_reason = fault;
        out.ticks.push_back(rec);
        stream_fault = true;
        break;
      }
      if (geo_ready) {
        geo.previous_along = gate_geo.previous_along;
      }
      if (dir_seeded) {
        DirectionEnvelope probe = dir_env;
        const auto dir = direction_envelope_accept(&probe, filt);
        if (!dir.ok) {
          note_fault_once(make_fault(ChainLayer::FinalGate,
                                     t + s * opt.stream_dt, dir.violated_axis,
                                     0.0, 0.0, dir.reason));
          std::snprintf(rec.note, sizeof(rec.note), "final_dir_%.70s",
                        dir.reason);
          out.refused = true;
          out.refuse_reason = dir.reason;
          out.ticks.push_back(rec);
          stream_fault = true;
          break;
        }
        dir_env = probe;
      }

      const Vec18 dq = (filt - prev_final) / opt.stream_dt;
      const Vec18 ddq = (dq - prev_dq) / opt.stream_dt;
      for (int a = 0; a < 3; ++a) {
        out.max_waist_speed =
            std::max(out.max_waist_speed, std::abs(dq[a]));
      }
      out.max_joint_speed =
          std::max(out.max_joint_speed, dq.cwiseAbs().maxCoeff());
      out.max_joint_accel =
          std::max(out.max_joint_accel, ddq.cwiseAbs().maxCoeff());
      const double dp =
          (filt.head<3>().sum() - prev_final.head<3>().sum()) / opt.stream_dt;
      out.max_pitch_speed = std::max(out.max_pitch_speed, std::abs(dp));

      // Axis anomaly vs effective dq_max (unverified device a_max → speed only).
      const Vec18 dq_lim = tracker.effective_dq_max();
      for (int j = 0; j < kNDof; ++j) {
        if (dq_lim[j] > 1e-9 && std::abs(dq[j]) > dq_lim[j] * 1.05) {
          note_axis_once(ChainLayer::FinalGate, t + s * opt.stream_dt, j,
                         dq[j], dq_lim[j], "final_speed");
        }
      }

      if (dir_seeded) {
        for (int a = 0; a < 3; ++a) {
          const int sgn = tracker.line_direction_contract().sign[a];
          if (sgn != 0 &&
              (filt[a] - prev_final[a]) * static_cast<double>(sgn) < -1e-6) {
            ++out.reverse_events;
            out.max_reverse_deg = std::max(
                out.max_reverse_deg,
                std::abs(filt[a] - prev_final[a]) * 180.0 / kPi);
            note_fault_once(make_fault(ChainLayer::FinalGate,
                                       t + s * opt.stream_dt, a,
                                       filt[a] - prev_final[a], 0.0,
                                       "unexpected_reverse"));
          }
        }
      }

      prev_dq = dq;
      prev_final = filt;
      q_final = filt;
      rec.final_ran = true;
      rec.q_final = filt;
      update_feedback(filt);
      rec.q_feedback = q_feedback;

      ChainStreamSample ss;
      ss.tick_id = rec.tick_id;
      ss.stream_i = static_cast<uint64_t>(s);
      ss.t_s = t + s * opt.stream_dt;
      ss.dt_s = opt.stream_dt;
      ss.q_raw = stream_raw;
      ss.q_filt = filt;
      ss.q_final = filt;
      ss.q_feedback = q_feedback;
      ss.dq_final = dq;
      ss.ddq_final = ddq;
      out.stream.push_back(ss);
    }
    if (stream_fault) {
      return out;
    }

    if (!hold_seen) {
      std::snprintf(rec.note, sizeof(rec.note), "ok_scale=%.3f",
                    rec.commit_scale);
    }
    out.ticks.push_back(rec);
    t += opt.control_dt;
  }

  if (!out.completed && !out.refused && !out.first_fault.found) {
    out.first_fault = make_fault(ChainLayer::Unknown, t, -1, 0.0, 0.0,
                                 "timeout_or_incomplete", /*unknown=*/true);
  }
  return out;
}

inline bool write_chain_compare_report(const std::string& out_dir,
                                      const ChainRunResult& legacy,
                                      const ChainRunResult& neu,
                                      const char* git_head) {
  namespace fs = std::filesystem;
  std::error_code ec;
  // Never silently overwrite an existing evidence summary unless forced.
  const fs::path summary_path = fs::path(out_dir) / "summary.md";
  if (fs::exists(summary_path, ec)) {
    const char* overwrite = std::getenv("KMPC_CHAIN_REPORT_OVERWRITE");
    if (!(overwrite && overwrite[0] == '1')) {
      std::fprintf(stderr,
                   "write_chain_compare_report: refuse overwrite of %s "
                   "(set KMPC_CHAIN_REPORT_OVERWRITE=1 to force)\n",
                   summary_path.string().c_str());
      return false;
    }
  }
  fs::create_directories(out_dir, ec);
  if (ec) {
    return false;
  }
  auto write_metrics = [&](const ChainRunResult& r, const char* name) {
    const std::string path = out_dir + "/" + name + "_metrics.csv";
    std::ofstream csv(path);
    if (!csv) {
      return false;
    }
    csv << "tick_id,generation,t_s,dt_s,prog,s_dot,commit_scale,qp_dx,qp_proj,"
           "qp_mode,plan_ran,qp_ran,commit_ran,interp_ran,final_ran,missing,"
           "note";
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_plan_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_qp_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_commit_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_raw_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_filt_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_final_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_fb_" << i;
    }
    csv << '\n';
    for (const auto& tick : r.ticks) {
      csv << tick.tick_id << "," << tick.generation << "," << tick.t_s << ","
          << tick.dt_s << "," << tick.prog << "," << tick.s_dot << ","
          << tick.commit_scale << "," << tick.qp_dx << "," << tick.qp_proj
          << "," << static_cast<int>(tick.qp_mode) << "," << tick.plan_ran
          << "," << tick.qp_ran << "," << tick.commit_ran << ","
          << tick.interp_ran << "," << tick.final_ran << "," << tick.missing
          << ",\"" << tick.note << "\",";
      write_vec18_csv(csv, tick.q_plan);
      csv << ',';
      write_vec18_csv(csv, tick.q_qp);
      csv << ',';
      write_vec18_csv(csv, tick.q_commit);
      csv << ',';
      write_vec18_csv(csv, tick.q_raw);
      csv << ',';
      write_vec18_csv(csv, tick.q_filt);
      csv << ',';
      write_vec18_csv(csv, tick.q_final);
      csv << ',';
      write_vec18_csv(csv, tick.q_feedback);
      csv << '\n';
    }
    return true;
  };
  auto write_stream = [&](const ChainRunResult& r, const char* name) {
    const std::string path = out_dir + "/" + name + "_stream250.csv";
    std::ofstream csv(path);
    if (!csv) {
      return false;
    }
    csv << "tick_id,stream_i,t_s,dt_s";
    for (int i = 0; i < kNDof; ++i) {
      csv << ",q_final_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",dq_" << i;
    }
    for (int i = 0; i < kNDof; ++i) {
      csv << ",ddq_" << i;
    }
    csv << ",q_fb_0\n";
    for (const auto& s : r.stream) {
      csv << s.tick_id << "," << s.stream_i << "," << s.t_s << "," << s.dt_s
          << ",";
      write_vec18_csv(csv, s.q_final);
      csv << ',';
      write_vec18_csv(csv, s.dq_final);
      csv << ',';
      write_vec18_csv(csv, s.ddq_final);
      csv << ',' << s.q_feedback[0] << '\n';
    }
    return true;
  };
  if (!write_metrics(legacy, "legacy") || !write_metrics(neu, "new") ||
      !write_stream(legacy, "legacy") || !write_stream(neu, "new")) {
    return false;
  }

  auto fault_line = [](const ChainFault& f) -> std::string {
    if (!f.found) {
      return "none";
    }
    char buf[256];
    if (f.unknown) {
      std::snprintf(buf, sizeof(buf),
                    "无法定位 layer=%s t=%.4f joint=%d value=%.6g reason=%s",
                    chain_layer_name(f.layer), f.t_s, f.joint, f.value,
                    f.reason);
    } else {
      std::snprintf(buf, sizeof(buf),
                    "layer=%s t=%.4fs joint=%d value=%.6g limit=%.6g reason=%s",
                    chain_layer_name(f.layer), f.t_s, f.joint, f.value, f.limit,
                    f.reason);
    }
    return buf;
  };

  std::ofstream md(out_dir + "/summary.md");
  if (!md) {
    return false;
  }
  md << "# Offline full-chain A/B report\n\n";
  md << "- source_id: `" << (git_head ? git_head : "unknown") << "`\n";
  md << "- default_chain_switched: **no** (Legacy remains default)\n";
  md << "- final_gate: production `line_final_filtered_fault` + direction\n";
  md << "- feedback: driven by final filtered command (IdealFinal default)\n";
  md << "- note: 1559 report directory was previously overwritten by a rerun; "
        "do not treat it as untouched original evidence\n\n";
  md << "## Results\n\n";
  md << "| profile | completed | cmd_arr | fb_arr | refused | pos_err_m | "
        "ori_err_rad | max_v | max_a | max_qp_dx | scale_events | reverse | "
        "first_fault |\n";
  md << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- |\n";
  auto row = [&](const ChainRunResult& r) {
    md << "| " << r.profile_name << " | " << (r.completed ? "yes" : "no")
       << " | " << (r.cmd_arrived ? "yes" : "no") << " | "
       << (r.feedback_arrived ? "yes" : "no") << " | "
       << (r.refused ? r.refuse_reason : "-") << " | " << r.final_pos_err_m
       << " | " << r.final_ori_err_rad << " | " << r.max_joint_speed << " | "
       << r.max_joint_accel << " | " << r.max_qp_dx << " | "
       << r.commit_scale_events << " | " << r.reverse_events << " (max "
       << r.max_reverse_deg << " deg) | " << fault_line(r.first_fault)
       << " |\n";
  };
  row(legacy);
  row(neu);
  md << "\n## Smooth / corners (new chain)\n\n";
  md << "- smooth_reason: `" << neu.smooth.reason << "`\n";
  md << "- hermite_verified: " << (neu.smooth.hermite_verified ? "yes" : "no")
     << "\n";
  md << "- hermite_applied: " << (neu.smooth.applied ? "yes" : "no") << "\n";
  md << "- geometry_c1_claim: " << (neu.smooth.geometry_c1 ? "approx" : "no")
     << "\n";
  md << "- corner_max_jump_rad: " << neu.corners.max_tangent_jump_rad
     << " worst_knot=" << neu.corners.worst_knot
     << " jumps_n=" << neu.corners.tangent_jumps.size() << "\n";
  md << "\n## First axis anomaly\n\n";
  auto axis_line = [](const FirstAxisAnomaly& a) {
    if (!a.found) {
      return std::string("none");
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "layer=%s t=%.4f joint=%d value=%.6g limit=%.6g src=%s",
                  chain_layer_name(a.layer), a.t_s, a.joint, a.value, a.limit,
                  a.source);
    return std::string(buf);
  };
  md << "- legacy: " << axis_line(legacy.first_axis) << "\n";
  md << "- new: " << axis_line(neu.first_axis) << "\n";
  md << "\n## Raw metrics\n\n";
  md << "- `legacy_metrics.csv` / `new_metrics.csv` (50 Hz, full q vectors)\n";
  md << "- `legacy_stream250.csv` / `new_stream250.csv` (250 Hz dq/ddq)\n";
  md << "\n## Notes\n\n";
  md << "- Feedback follows final filtered command (not raw).\n";
  md << "- Device a_max not separately configured — accel uses step_cap/dt^2.\n";
  md << "- Dynamic tip-over is **not** proven (no IMU).\n";
  md << "- Failures that cannot be attributed are marked 无法定位.\n";
  md << "- Production ROS `line_chain_profile` default remains `legacy`.\n";
  return true;
}

}  // namespace kmpc
