#include "kmpc/trajectory_validator.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr double kLegacyHardTubeM = 0.20;

void note(const char** fault, const char* why) {
  if (*fault == nullptr) {
    *fault = why;
  }
}

double pitch_abs(const Vec18& q) { return std::abs(q[0] + q[1] + q[2]); }

// Distance from p to segment ab; returns distance and the clamped parameter.
double segment_distance(const Vec3& p, const Vec3& a, const Vec3& b) {
  const Vec3 ab = b - a;
  const double ab2 = ab.squaredNorm();
  if (ab2 < 1e-16) {
    return (p - b).norm();
  }
  const double t = std::clamp((p - a).dot(ab) / ab2, 0.0, 1.0);
  return (p - (a + t * ab)).norm();
}

constexpr double kJ4StageMonoTolRad = 5e-4;

bool j4_stage_monotone_ok(
    const JointCurve& samples, std::size_t nstage,
    const std::vector<StageBoundary>* boundaries, std::string* detail) {
  if (nstage == 0 || samples.size() < 2) {
    return true;
  }
  auto range_for_stage = [&](std::size_t s,
                             std::size_t* i0,
                             std::size_t* i1) -> bool {
    if (boundaries != nullptr && s < boundaries->size()) {
      *i1 = std::min((*boundaries)[s].sample_index, samples.size() - 1);
      *i0 = (s == 0) ? 0
                     : std::min((*boundaries)[s - 1].sample_index + 1,
                                samples.size() - 1);
      return *i0 <= *i1;
    }
    return false;
  };
  for (std::size_t s = 0; s < nstage; ++s) {
    std::size_t i0 = 0;
    std::size_t i1 = samples.size() - 1;
    if (boundaries != nullptr && !boundaries->empty()) {
      if (!range_for_stage(s, &i0, &i1)) {
        continue;
      }
    }
    const double q_start = samples[i0][kIdxYaw];
    const double q_end = samples[i1][kIdxYaw];
    const double net = q_end - q_start;
    int sign = 0;
    if (std::abs(net) > kJ4StageMonoTolRad) {
      sign = net > 0.0 ? 1 : -1;
    }
    double q_run = q_start;
    for (std::size_t i = i0; i <= i1; ++i) {
      const double qy = samples[i][kIdxYaw];
      if (sign == 0) {
        if (std::abs(qy - q_start) > kJ4StageMonoTolRad) {
          if (detail != nullptr) {
            std::ostringstream os;
            os << "stage=" << s << " sample=" << i << " flat_violation";
            *detail = os.str();
          }
          return false;
        }
      } else if (sign > 0) {
        if (qy + kJ4StageMonoTolRad < q_run) {
          if (detail != nullptr) {
            std::ostringstream os;
            os << "stage=" << s << " sample=" << i
               << " reverse_amt_rad=" << (q_run - qy);
            *detail = os.str();
          }
          return false;
        }
        q_run = std::max(q_run, qy);
      } else {
        if (qy - kJ4StageMonoTolRad > q_run) {
          if (detail != nullptr) {
            std::ostringstream os;
            os << "stage=" << s << " sample=" << i
               << " reverse_amt_rad=" << (qy - q_run);
            *detail = os.str();
          }
          return false;
        }
        q_run = std::min(q_run, qy);
      }
    }
  }
  return true;
}

void note_tcp_cross_track_diag(TrajectoryCertificate* out, std::size_t stage,
                               double cross_m, double t_s, double s,
                               const char* source) {
  if (out->tcp_fault_stage < 0 || cross_m >= out->worst_tcp_cross_track_m) {
    out->tcp_fault_stage = static_cast<int>(stage);
    out->tcp_fault_t_s = t_s;
    out->tcp_fault_s = s;
    out->tcp_fault_source = source;
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(3);
    os << "stage=" << stage << " worst_mm=" << (cross_m * 1000.0)
       << " t_s=" << t_s << " s=" << s << " source=" << source;
    out->fault_detail = os.str();
  }
}

}  // namespace

std::size_t stage_for_sample_point(
    std::size_t sample_i, std::size_t nstage,
    const std::vector<StageBoundary>* boundaries) {
  if (nstage == 0) {
    return 0;
  }
  if (boundaries == nullptr || boundaries->empty()) {
    return std::min(sample_i, nstage - 1);
  }
  for (std::size_t s = 0; s < boundaries->size() && s < nstage; ++s) {
    if (sample_i <= (*boundaries)[s].sample_index) {
      return s;
    }
  }
  return nstage - 1;
}

std::size_t stage_for_sample_span(std::size_t span_left_i, std::size_t nstage,
                                  const std::vector<StageBoundary>* boundaries) {
  if (nstage == 0) {
    return 0;
  }
  return stage_for_sample_point(span_left_i + 1, nstage, boundaries);
}

std::size_t stage_for_curve_progress(
    double s, std::size_t n_samples, std::size_t nstage,
    const std::vector<StageBoundary>* boundaries) {
  if (n_samples < 2 || nstage == 0) {
    return 0;
  }
  const double prog =
      std::clamp(s, 0.0, 1.0) * static_cast<double>(n_samples - 1);
  const std::size_t sample_i = static_cast<std::size_t>(std::floor(prog + 1e-12));
  const std::size_t sample_right =
      std::min(sample_i + 1, n_samples - 1);
  const double frac = prog - static_cast<double>(sample_i);
  if (frac > 1e-9 && sample_right > sample_i) {
    return stage_for_sample_span(sample_i, nstage, boundaries);
  }
  return stage_for_sample_point(sample_i, nstage, boundaries);
}

TrajectoryCertificate validate_path(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards,
    const std::vector<StageBoundary>* stage_boundaries) {
  TrajectoryCertificate out;
  out.coverage = PathCoverage::DiscreteSamples;
  out.continuous_certified = false;
  out.com_model = "not_checked";
  out.min_joint_margin_rad = 0.0;
  out.min_pitch_margin_rad = 0.0;
  const char* fault = nullptr;

  if (samples.size() < 2) {
    out.first_fault = "too_few_samples";
    return out;
  }
  if (task.stages.empty()) {
    out.first_fault = "stage_count_not_supported";
    return out;
  }
  const std::string_view frame = task.left ? kEeFrameLeft : kEeFrameRight;
  const std::size_t nstage = task.stages.size();
  // Polyline vertices: certified start TCP, then one endpoint per stage.
  std::vector<Vec3, Eigen::aligned_allocator<Vec3>> poly;
  poly.reserve(nstage + 1);
  poly.push_back(kin.fk_pos(samples.front(), frame));
  for (const TaskStage& stage : task.stages) {
    if (stage.geometry == TaskGeometry::FollowCurve) {
      out.first_fault = "follow_curve_not_implemented";
      return out;
    }
    if (!(stage.position_tol_m > 0.0) ||
        !std::isfinite(stage.position_tol_m)) {
      out.first_fault = "bad_task_tol";
      return out;
    }
    // LIN uses the task tolerance only. The 0.20 m hard tube is retired.
    if (stage.geometry == TaskGeometry::Lin &&
        stage.position_tol_m >= kLegacyHardTubeM) {
      out.first_fault = "lin_tol_is_legacy_hard_tube";
      return out;
    }
    if (!stage.position.allFinite()) {
      out.first_fault = "bad_task_tol";
      return out;
    }
    poly.push_back(stage.position);
  }

  // Joint hard box + pitch + CoM + TCP in one pass so FK/CoM cache hits the
  // same q (coverage unchanged).
  out.hard_box_ok = true;
  out.work_box_ok = true;
  out.pitch_ok = true;
  const double start_abs = pitch_abs(samples.front());
  const double task_tcp_dz_m =
      task_stages_tcp_dz_m(kin.fk_pos(samples.front(), frame).z(), task.stages);
  const double plan_cap = pitch_planning_cap_rad(policy.torso_pitch_max_rad,
                                                 start_abs, task_tcp_dz_m);
  const double pitch_bound = pitch_abs_bound(plan_cap, start_abs, false);
  out.min_joint_margin_rad = joint_hard_margin_rad(
      samples.front(), policy.q_min, policy.q_max);
  out.min_work_margin_rad = joint_work_margin_rad(
      samples.front(), samples.front(), policy.q_min, policy.q_max,
      policy.q_limit_margin_rad);
  out.min_pitch_margin_rad = pitch_bound - start_abs;
  out.com_checked = guards != nullptr;
  out.com_ok = guards != nullptr;
  out.com_model = guards == nullptr ? "not_checked" : "xy_work_rectangle";
  if (guards == nullptr) {
    note(&fault, "com_not_checked");
  }
  out.task_path_ok = true;
  std::vector<Vec3, Eigen::aligned_allocator<Vec3>> tcps;
  tcps.reserve(samples.size());
  for (const Vec18& q : samples) {
    const double joint_margin =
        joint_hard_margin_rad(q, policy.q_min, policy.q_max);
    out.min_joint_margin_rad = std::min(out.min_joint_margin_rad, joint_margin);
    if (joint_violates_hard_box(q, policy.q_min, policy.q_max)) {
      out.hard_box_ok = false;
      note(&fault, "joint");
    }
    const double work_margin = joint_work_margin_rad(
        q, samples.front(), policy.q_min, policy.q_max,
        policy.q_limit_margin_rad);
    out.min_work_margin_rad = std::min(out.min_work_margin_rad, work_margin);
    if (joint_violates_effective_work_box(q, samples.front(), policy)) {
      out.work_box_ok = false;
      note(&fault, "work_box");
    }
    if (joint_violates_yaw_task_box(q, policy)) {
      out.yaw_task_ok = false;
      note(&fault, "yaw_task");
      if (out.fault_detail.empty()) {
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(4);
        os << "yaw_rad=" << q[kIdxYaw] << " limit="
           << policy.yaw_abs_max_rad;
        out.fault_detail = os.str();
      }
    }
    const double pa = pitch_abs(q);
    const double pitch_margin = pitch_bound - pa;
    out.min_pitch_margin_rad = std::min(out.min_pitch_margin_rad, pitch_margin);
    if (!(pitch_margin >= -kPitchRoundEpsRad) || !q.head<3>().allFinite()) {
      out.pitch_ok = false;
      note(&fault, "pitch");
    }
    if (guards != nullptr) {
      if (!q.allFinite() || !guards->com_xy_inside(kin.com(q))) {
        out.com_ok = false;
        note(&fault, "com");
      }
    }
    tcps.push_back(kin.fk_pos(q, frame));
  }
  const bool use_boundaries =
      stage_boundaries != nullptr && !stage_boundaries->empty();
  for (std::size_t i = 0; i < tcps.size(); ++i) {
    const Vec3& p = tcps[i];
    std::size_t seg = 0;
    double best = 1e30;
    if (use_boundaries) {
      seg = stage_for_sample_point(i, nstage, stage_boundaries);
      best = segment_distance(p, poly[seg], poly[seg + 1]);
    } else {
      for (std::size_t s = 0; s < nstage; ++s) {
        const double d = segment_distance(p, poly[s], poly[s + 1]);
        if (d < best) {
          best = d;
          seg = s;
        }
      }
    }
    const TaskStage& stage = task.stages[seg];
    // PTP cross-track is diagnostic only — do not pollute the geometry metric.
    if (stage.geometry == TaskGeometry::Lin) {
      out.worst_tcp_cross_track_m =
          std::max(out.worst_tcp_cross_track_m, best);
    }
    if (stage.geometry == TaskGeometry::Lin &&
        best > stage.position_tol_m) {
      const double s_frac = samples.size() <= 1
                                ? 0.0
                                : static_cast<double>(i) /
                                      static_cast<double>(samples.size() - 1);
      note_tcp_cross_track_diag(&out, seg, best, /*t_s=*/-1.0, s_frac,
                                "knot_samples");
    }
  }
  for (std::size_t s = 0; s + 1 < nstage; ++s) {
    double reached = 1e30;
    if (use_boundaries && s < stage_boundaries->size()) {
      const std::size_t bi = (*stage_boundaries)[s].sample_index;
      if (bi < tcps.size()) {
        reached = (tcps[bi] - poly[s + 1]).norm();
      }
    } else {
      for (const Vec3& p : tcps) {
        reached = std::min(reached, (p - poly[s + 1]).norm());
      }
    }
    if (reached > task.stages[s].position_tol_m) {
      out.task_path_ok = false;
      note(&fault, "tcp_stage_endpoint");
    }
  }
  out.worst_tcp_endpoint_m = (tcps.back() - poly.back()).norm();
  if (out.worst_tcp_endpoint_m > task.stages.back().position_tol_m) {
    out.task_path_ok = false;
    note(&fault, "tcp_endpoint");
  }

  // Orientation, only where the stage asks for it, at the stage endpoint.
  for (std::size_t s = 0; s < nstage; ++s) {
    const TaskStage& stage = task.stages[s];
    if (!stage.has_orientation) {
      continue;
    }
    std::size_t best_i = tcps.size() - 1;
    if (use_boundaries && s < stage_boundaries->size()) {
      best_i = std::min((*stage_boundaries)[s].sample_index, tcps.size() - 1);
    } else if (s + 1 < nstage) {
      double reached = 1e30;
      for (std::size_t i = 0; i < tcps.size(); ++i) {
        const double d = (tcps[i] - poly[s + 1]).norm();
        if (d < reached) {
          reached = d;
          best_i = i;
        }
      }
    }
    const auto pose = kin.fk(samples[best_i], frame);
    const double oe = ori_error_xyzw(
        stage.quat, rot_to_quat_xyzw(pose.block<3, 3>(0, 0))).norm();
    if (!(oe <= stage.orientation_tol_rad)) {
      out.task_path_ok = false;
      note(&fault, "tcp_orientation");
    }
  }

  // J4 may reverse within a stage (human-like torso yaw). Foldback gates
  // remain on arm/wrist via quality cost and other cert rows.
  out.j4_stage_mono_ok = true;
  (void)j4_stage_monotone_ok;

  out.passed = fault == nullptr && out.hard_box_ok && out.work_box_ok &&
               out.yaw_task_ok && out.j4_stage_mono_ok && out.pitch_ok &&
               out.com_checked && out.com_ok && out.task_path_ok;
  out.first_fault = fault == nullptr ? "ok" : fault;
  return out;
}

TrajectoryCertificate validate_discrete_trajectory(
    const MotionTask& task, const JointCurve& samples,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards,
    const std::vector<StageBoundary>* stage_boundaries) {
  if (task.stages.empty()) {
    TrajectoryCertificate out;
    out.first_fault = "stage_count_not_supported";
    return out;
  }
  return validate_path(task, samples, policy, kin, guards, stage_boundaries);
}

TrajectoryCertificate validate_actual_q_of_t(
    const TrajectoryPackage& package, const MotionTask& task,
    const MotionConstraintPolicy& policy, const Kinematics& kin,
    const WholeBodyGuards* guards) {
  TrajectoryCertificate out =
      validate_path(task, package.samples, policy, kin, guards,
                    &package.stage_boundaries);
  const char* fault = nullptr;
  if (out.first_fault != nullptr && out.first_fault[0] != '\0' &&
      std::string(out.first_fault) != "ok") {
    fault = out.first_fault;
  }
  // The executed interpolation follows s(t): it must be monotone in both.
  const TimeLawTable& law = package.time_law;
  bool law_ok = true;
  if (law.s.size() != law.t_s.size()) {
    law_ok = false;
    note(&fault, "time_law_size");
  } else if (!law.s.empty()) {
    if (law.s.front() < -1e-9 || law.s.back() > 1.0 + 1e-9) {
      law_ok = false;
      note(&fault, "time_law_range");
    }
    for (std::size_t i = 1; i < law.s.size(); ++i) {
      if (law.s[i] < law.s[i - 1] - 1e-12) {
        law_ok = false;
        note(&fault, "time_law_s_not_monotone");
        break;
      }
    }
    for (std::size_t i = 1; i < law.t_s.size(); ++i) {
      if (law.t_s[i] < law.t_s[i - 1] - 1e-9) {
        law_ok = false;
        note(&fault, "time_law_t_not_monotone");
        break;
      }
    }
  }
  if (!(law.duration_s >= 0.0) || !std::isfinite(law.duration_s)) {
    law_ok = false;
    note(&fault, "time_law_duration");
  }

  // Sample the *actual* executable Hermite q(t) at the 250 Hz send period.
  // Offline sampling + online supervisor guard ≠ continuous certification.
  bool exec_ok = true;
  constexpr double kSendDt = 0.004;
  if (law_ok && package.samples.size() >= 2 && law.duration_s > 0.0) {
    const std::string_view frame = task.left ? kEeFrameLeft : kEeFrameRight;
    const std::size_t nstage = task.stages.size();
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> poly;
    poly.reserve(nstage + 1);
    poly.push_back(kin.fk_pos(package.samples.front(), frame));
    for (const TaskStage& stage : task.stages) {
      poly.push_back(stage.position);
    }

    // Sample Hermite q(t) on the real 250 Hz send grid. Derivatives come from
    // that send sequence (finite differences), not from the h=1e-3 probe inside
    // sample_trajectory_at — that probe is too noisy for hard accel/jerk gates.
    const int n_check =
        std::max(1, static_cast<int>(std::ceil(law.duration_s / kSendDt)));
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>> qs;
    std::vector<double> ts;
    qs.reserve(static_cast<std::size_t>(n_check) + 2);
    ts.reserve(static_cast<std::size_t>(n_check) + 2);
    auto push_sample = [&](double t, double send_t) -> bool {
      if (!ts.empty() && send_t <= ts.back() + 1e-12) {
        return true;  // skip duplicate endpoint / stalled clock
      }
      TrajectoryTimeSample samp;
      // q-only: dq/ddq for exec gates come from the send-period finite
      // differences below — avoid 3× Hermite probes per tick.
      if (!sample_trajectory_at(package, t, &samp, /*need_derivatives=*/false) ||
          !samp.ok) {
        note(&fault, "exec_sample_fail");
        return false;
      }
      if (joint_violates_hard_box(samp.q, policy.q_min, policy.q_max)) {
        note(&fault, "exec_hard_box");
        return false;
      }
      if (joint_violates_effective_work_box(samp.q, package.samples.front(),
                                            policy)) {
        note(&fault, "exec_work_box");
        return false;
      }
      if (joint_violates_yaw_task_box(samp.q, policy)) {
        note(&fault, "exec_yaw_task");
        return false;
      }
      if (policy.torso_pitch_max_rad > 0.0) {
        const double start_abs = pitch_abs(package.samples.front());
        const double task_tcp_dz_m = task_stages_tcp_dz_m(
            kin.fk_pos(package.samples.front(), frame).z(), task.stages);
        const double plan_cap = pitch_planning_cap_rad(
            policy.torso_pitch_max_rad, start_abs, task_tcp_dz_m);
        const double bound =
            pitch_abs_bound(plan_cap, start_abs, /*for_gate=*/true);
        const double pitch = samp.q[0] + samp.q[1] + samp.q[2];
        if (!pitch_abs_ok(std::abs(pitch), plan_cap, start_abs,
                          /*for_gate=*/true)) {
          note(&fault, "exec_pitch");
          if (out.fault_detail.empty()) {
            std::ostringstream os;
            os.setf(std::ios::fixed);
            os.precision(4);
            os << "t_s=" << samp.t_s << " s=" << samp.s << " pitch_rad="
               << pitch << " bound_rad=" << bound
               << " plan_cap=" << plan_cap << " tcp_dz_m=" << task_tcp_dz_m
               << " start_abs=" << start_abs;
            out.fault_detail = os.str();
          }
          return false;
        }
      }
      // Reuse this FK for CoM + TCP so the Hermite tick is evaluated once.
      const Vec3 tcp = kin.fk_pos(samp.q, frame);
      if (guards != nullptr && !guards->com_xy_inside(kin.com(samp.q))) {
        note(&fault, "exec_com");
        return false;
      }
      if (nstage > 0 && !poly.empty()) {
        const std::size_t seg = stage_for_curve_progress(
            samp.s, package.samples.size(), nstage,
            &package.stage_boundaries);
        const double cross =
            segment_distance(tcp, poly[seg], poly[seg + 1]);
        const TaskStage& stage = task.stages[seg];
        if (stage.geometry == TaskGeometry::Lin) {
          out.worst_tcp_cross_track_m =
              std::max(out.worst_tcp_cross_track_m, cross);
          if (cross > stage.position_tol_m) {
            note_tcp_cross_track_diag(&out, seg, cross, samp.t_s, samp.s,
                                      "hermite_exec");
            // Keep scanning for worst deviation; velocity gates still run.
          }
        }
      }
      qs.push_back(samp.q);
      ts.push_back(send_t);
      return true;
    };
    // Shared with the time parameterizer: internal STOP/ACK hold and
    // restart have the same derivative contract as package start/end.
    for (const auto& probe : trajectory_send_probe_grid(package, kSendDt)) {
      if (!push_sample(probe.curve_s, probe.send_s)) {
        exec_ok = false;
        break;
      }
    }
    // Match SafetySupervisor 250 Hz boxes exactly (incl. software_limited
    // effective accel/jerk). Never use a looser offline margin than online.
    if (exec_ok) {
      for (std::size_t i = 1; i < qs.size(); ++i) {
        const double dt = std::max(ts[i] - ts[i - 1], 1e-4);
        for (int j = 0; j < kNDof; ++j) {
          const double dq_max = std::max(policy.dq_max[j], 1e-6);
          const double step = qs[i][j] - qs[i - 1][j];
          if (std::abs(step) > dq_max * dt + 1e-9) {
            exec_ok = false;
            note(&fault, "exec_step");
            break;
          }
        }
        if (!exec_ok) {
          break;
        }
      }
    }
    if (exec_ok) {
      for (std::size_t i = 2; i < qs.size(); ++i) {
        const double dt1 = std::max(ts[i] - ts[i - 1], 1e-4);
        const double dt0 = std::max(ts[i - 1] - ts[i - 2], 1e-4);
        for (int j = 0; j < kNDof; ++j) {
          const double dq_max = std::max(policy.dq_max[j], 1e-6);
          const double ddq_pol = std::max(policy.ddq_max[j], 1e-6);
          const double ddq_soft =
              policy.a_max_calibrated ? ddq_pol
                                      : std::max(ddq_pol, dq_max / 0.02);
          const double v1 = (qs[i][j] - qs[i - 1][j]) / dt1;
          const double v0 = (qs[i - 1][j] - qs[i - 2][j]) / dt0;
          // Same gate as SafetySupervisor: |dv| <= 4 * ddq_soft * dt.
          if (std::abs(v1 - v0) > 4.0 * ddq_soft * dt1 + 1e-6) {
            exec_ok = false;
            note(&fault, "exec_ddq");
            break;
          }
        }
        if (!exec_ok) {
          break;
        }
      }
    }
    if (exec_ok) {
      for (std::size_t i = 3; i < qs.size(); ++i) {
        const double dt2 = std::max(ts[i] - ts[i - 1], 1e-4);
        const double dt1 = std::max(ts[i - 1] - ts[i - 2], 1e-4);
        const double dt0 = std::max(ts[i - 2] - ts[i - 3], 1e-4);
        for (int j = 0; j < kNDof; ++j) {
          const double dq_max = std::max(policy.dq_max[j], 1e-6);
          const double ddq_pol = std::max(policy.ddq_max[j], 1e-6);
          const double ddq_soft =
              policy.a_max_calibrated ? ddq_pol
                                      : std::max(ddq_pol, dq_max / 0.02);
          const double j_pol = std::max(policy.dddq_max[j], 1e-6);
          const double j_soft =
              policy.j_max_calibrated ? j_pol
                                      : std::max(j_pol, ddq_soft / 0.02);
          const double v2 = (qs[i][j] - qs[i - 1][j]) / dt2;
          const double v1 = (qs[i - 1][j] - qs[i - 2][j]) / dt1;
          const double v0 = (qs[i - 2][j] - qs[i - 3][j]) / dt0;
          const double a1 = (v2 - v1) / dt2;
          const double a0 = (v1 - v0) / dt1;
          // Same gate as SafetySupervisor: |da| <= 16 * j_soft * dt.
          if (std::abs(a1 - a0) > 16.0 * j_soft * dt2 + 1e-3) {
            exec_ok = false;
            note(&fault, "exec_jerk");
            break;
          }
        }
        if (!exec_ok) {
          break;
        }
      }
    }
    // Sampling proves discrete send feasibility under online guard only.
    out.coverage = PathCoverage::AdaptiveSampledWithOnlineGuard;
    out.coverage_note = "sampled+online_guard;hermite_q_of_t_at_250hz";
    out.continuous_certified = false;
    if (!policy.j_max_calibrated) {
      out.coverage_note += ";jerk_source=" + law.jerk_source;
    }
    if (law.software_limited) {
      out.coverage_note += ";software_limited";
    }
    if (!out.fault_detail.empty()) {
      out.coverage_note += ";" + out.fault_detail;
    }
  }

  // fault may be set by Hermite TCP after a green densify path — always
  // recompute passed from the live fault pointer.
  out.passed = fault == nullptr && law_ok && exec_ok && out.hard_box_ok &&
               out.work_box_ok && out.pitch_ok && out.com_checked &&
               out.com_ok && out.task_path_ok;
  out.first_fault = fault == nullptr ? "ok" : fault;
  return out;
}

}  // namespace kmpc
