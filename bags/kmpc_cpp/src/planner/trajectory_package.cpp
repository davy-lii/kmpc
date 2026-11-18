#include "kmpc/trajectory_package.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "kmpc/motion_task.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
// v3: planner work-box (q_limit_margin) is a planning/cert recovery contract.
constexpr uint64_t kDigestVersion = 3;

uint64_t mix(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

uint64_t mix_double(uint64_t hash, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double must be 8 bytes");
  std::memcpy(&bits, &value, sizeof(bits));
  return mix(hash, bits);
}

uint64_t mix_vec(uint64_t hash, const Vec18& q) {
  for (int i = 0; i < kNDof; ++i) {
    hash = mix_double(hash, q[i]);
  }
  return hash;
}

uint64_t mix_str(uint64_t hash, const std::string& s) {
  hash = mix(hash, static_cast<uint64_t>(s.size()));
  for (char c : s) {
    hash = mix(hash, static_cast<uint64_t>(static_cast<unsigned char>(c)));
  }
  return hash;
}

}  // namespace

uint64_t package_digest(const TrajectoryPackage& package) {
  uint64_t hash = kFnvOffset;
  hash = mix(hash, kDigestVersion);
  hash = mix(hash, package.generation);
  hash = mix(hash, package.task_digest);
  hash = mix(hash, package.model_digest);
  hash = mix(hash, package.limits_digest);
  hash = mix(hash, package.path_id);
  hash = mix(hash, package.left ? 1u : 0u);
  hash = mix(hash, package.start_dq_known ? 1u : 0u);
  hash = mix_vec(hash, package.start_q);
  hash = mix_vec(hash, package.start_dq);
  hash = mix(hash, static_cast<uint64_t>(package.samples.size()));
  for (const Vec18& q : package.samples) {
    hash = mix_vec(hash, q);
  }
  hash = mix(hash, static_cast<uint64_t>(package.stage_boundaries.size()));
  for (const StageBoundary& b : package.stage_boundaries) {
    hash = mix(hash, static_cast<uint64_t>(b.sample_index));
    hash = mix(hash, b.stage_id);
    hash = mix(hash, b.stop_required ? 1u : 0u);
    hash = mix(hash, b.wait_for_external_ack ? 1u : 0u);
    hash = mix_double(hash, b.position_tol_m);
    hash = mix_double(hash, b.orientation_tol_rad);
    hash = mix(hash, b.has_orientation ? 1u : 0u);
  }
  const TimeLawTable& law = package.time_law;
  hash = mix(hash, static_cast<uint64_t>(law.s.size()));
  for (double v : law.s) {
    hash = mix_double(hash, v);
  }
  hash = mix(hash, static_cast<uint64_t>(law.t_s.size()));
  for (double v : law.t_s) {
    hash = mix_double(hash, v);
  }
  hash = mix_double(hash, law.duration_s);
  hash = mix(hash, law.software_limited ? 1u : 0u);
  hash = mix_str(hash, law.accel_source);
  hash = mix_str(hash, law.jerk_source);
  hash = mix_str(hash, package.planner);
  return hash;
}

uint64_t limits_digest_of(const MotionConstraintPolicy& policy) {
  uint64_t hash = kFnvOffset;
  hash = mix(hash, kDigestVersion);
  hash = mix_vec(hash, policy.q_min);
  hash = mix_vec(hash, policy.q_max);
  hash = mix_vec(hash, policy.dq_max);
  hash = mix_vec(hash, policy.ddq_max);
  hash = mix_vec(hash, policy.dddq_max);
  hash = mix_double(hash, policy.torso_pitch_max_rad);
  hash = mix_double(hash, policy.q_limit_margin_rad);
  hash = mix_str(hash, "work_box_v1");
  hash = mix_double(hash, policy.yaw_abs_max_rad);
  hash = mix(hash, policy.yaw_limited ? 1u : 0u);
  hash = mix(hash, policy.a_max_calibrated ? 1u : 0u);
  hash = mix(hash, policy.j_max_calibrated ? 1u : 0u);
  return hash;
}

namespace {

uint64_t fnv1a_bytes(const std::string& bytes) {
  uint64_t hash = kFnvOffset;
  for (unsigned char c : bytes) {
    hash = mix(hash, static_cast<uint64_t>(c));
  }
  return hash;
}

std::string load_urdf_bytes(const std::string& urdf, bool is_xml) {
  if (is_xml) {
    return urdf;
  }
  std::ifstream in(urdf, std::ios::binary);
  if (!in) {
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string hex_u64(uint64_t v) {
  std::ostringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(16) << v;
  return ss.str();
}

}  // namespace

std::string model_identity_string(const std::string& urdf, bool is_xml,
                                  uint64_t limits_digest) {
  const std::string bytes = load_urdf_bytes(urdf, is_xml);
  const uint64_t content = fnv1a_bytes(bytes);
  return "urdf:" + hex_u64(content) + ":limits:" + hex_u64(limits_digest);
}

uint64_t seal_motion_task_identity(MotionTask* task, const std::string& urdf,
                                   bool is_xml,
                                   const MotionConstraintPolicy& policy) {
  if (task == nullptr) {
    return 0;
  }
  const uint64_t lim = limits_digest_of(policy);
  task->model_digest = model_identity_string(urdf, is_xml, lim);
  return motion_task_digest(*task);
}

double time_at_sample_index(const TrajectoryPackage& package,
                            std::size_t sample_index) {
  const auto& law = package.time_law;
  if (law.t_s.empty()) {
    return 0.0;
  }
  if (law.t_s.size() == package.samples.size() &&
      sample_index < law.t_s.size()) {
    return law.t_s[sample_index];
  }
  if (package.samples.size() < 2 || law.s.size() != law.t_s.size()) {
    return std::min(law.duration_s, law.t_s.back());
  }
  const double s_tgt = std::clamp(
      static_cast<double>(sample_index) /
          static_cast<double>(package.samples.size() - 1),
      0.0, 1.0);
  for (std::size_t i = 1; i < law.s.size(); ++i) {
    if (law.s[i] + 1e-12 >= s_tgt) {
      const double s0 = law.s[i - 1];
      const double s1 = law.s[i];
      const double t0 = law.t_s[i - 1];
      const double t1 = law.t_s[i];
      if (s1 - s0 <= 1e-12) {
        return t1;
      }
      if (law.ds_dt.size() == law.s.size()) {
        // Sparse time tables need the inverse of the executed cubic s(t),
        // not a linear inverse that would stop at the wrong joint sample.
        double lo = t0, hi = t1;
        for (int k = 0; k < 48; ++k) {
          const double mid = 0.5 * (lo + hi);
          TrajectoryTimeSample sample;
          if (!sample_trajectory_at(package, mid, &sample, false) || !sample.ok)
            return t1;
          if (sample.s < s_tgt) lo = mid;
          else hi = mid;
        }
        return 0.5 * (lo + hi);
      }
      const double a = (s_tgt - s0) / (s1 - s0);
      return t0 + a * (t1 - t0);
    }
  }
  return law.t_s.back();
}

std::vector<TrajectorySendProbe> trajectory_send_probe_grid(
    const TrajectoryPackage& package, double send_dt) {
  std::vector<TrajectorySendProbe> out;
  const double duration = package.time_law.duration_s;
  if (!(send_dt > 0.0) || !std::isfinite(duration) || duration < 0.0) return out;
  std::vector<double> stops;
  for (const auto& b : package.stage_boundaries) {
    if ((b.stop_required || b.wait_for_external_ack) &&
        b.sample_index > 0 && b.sample_index < package.samples.size()) {
      stops.push_back(time_at_sample_index(package, b.sample_index));
    }
  }
  stops.push_back(duration);
  std::sort(stops.begin(), stops.end());
  stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
  double sending = -3.0 * send_dt;
  auto push = [&](double curve) {
    sending += send_dt;
    out.push_back({sending, curve});
  };
  push(0.0); push(0.0); push(0.0);
  double start = 0.0;
  for (double end : stops) {
    end = std::clamp(end, start, duration);
    const int ticks = static_cast<int>(std::ceil((end - start) / send_dt));
    for (int i = 1; i <= ticks; ++i)
      push(std::min(end, start + i * send_dt));
    push(end); push(end);
    start = end;
  }
  return out;
}

AdmitDecision admit_package_start(
    const TrajectoryPackage& package, const Vec18& q, const Vec18& dq,
    bool dq_known, double q_tol_rad, double dq_tol) {
  AdmitDecision decision;
  if (!q.allFinite() || (dq_known && !dq.allFinite())) {
    decision.reason = "start_nonfinite";
    return decision;
  }
  if ((package.start_q - q).cwiseAbs().maxCoeff() > q_tol_rad) {
    decision.reason = "start_q_mismatch";
    return decision;
  }
  if (!package.samples.empty() &&
      (package.samples.front() - package.start_q).cwiseAbs().maxCoeff() >
          q_tol_rad) {
    decision.reason = "curve_start_rewritten";
    return decision;
  }
  if (package.start_dq_known != dq_known) {
    decision.reason = "start_dq_unknown";
    return decision;
  }
  if (package.start_dq_known &&
      (package.start_dq - dq).cwiseAbs().maxCoeff() > dq_tol) {
    decision.reason = "start_dq_mismatch";
    return decision;
  }
  decision.ok = true;
  decision.reason = "ok";
  return decision;
}

FeedbackStabilityReport evaluate_feedback_stability(
    const FeedbackRing18& ring, double now_s, double max_age_s,
    double window_s, int min_samples, double max_range_rad) {
  FeedbackStabilityReport out;
  if (!(std::isfinite(now_s) && max_age_s > 0.0 && window_s > 0.0 &&
        min_samples > 0 && max_range_rad > 0.0)) {
    out.reason = "bad_stability_args";
    return out;
  }
  if (ring.count <= 0) {
    out.reason = "feedback_incomplete";
    return out;
  }

  Vec18 lo = Vec18::Constant(std::numeric_limits<double>::infinity());
  Vec18 hi = Vec18::Constant(-std::numeric_limits<double>::infinity());
  int used = 0;
  double oldest_age = 0.0;
  double newest_age = std::numeric_limits<double>::infinity();
  bool saw_gap = false;
  double prev_stamp = std::numeric_limits<double>::quiet_NaN();

  // Walk newest→oldest within the stability window.
  for (int i = 0; i < ring.count; ++i) {
    const int idx = (ring.head - 1 - i + FeedbackRing18::kCap * 4) %
                    FeedbackRing18::kCap;
    const FeedbackSample18& s = ring.buf[idx];
    if (!s.valid || !s.q.allFinite() || !std::isfinite(s.recv_stamp_s)) {
      continue;
    }
    const double age = now_s - s.recv_stamp_s;
    if (age < -1e-3) {
      // Future stamp is not trusted as fresh feedback.
      out.reason = "feedback_stamp_skew";
      return out;
    }
    if (age > window_s + 1e-9) {
      break;
    }
    if (age > max_age_s && used == 0) {
      out.reason = "feedback_stale";
      out.newest_age_s = age;
      return out;
    }
    if (std::isfinite(prev_stamp)) {
      // Hole larger than max_age inside the window → not continuously covered.
      if (prev_stamp - s.recv_stamp_s > max_age_s + 1e-9) {
        saw_gap = true;
      }
    }
    prev_stamp = s.recv_stamp_s;
    for (int j = 0; j < kNDof; ++j) {
      lo[j] = std::min(lo[j], s.q[j]);
      hi[j] = std::max(hi[j], s.q[j]);
    }
    oldest_age = std::max(oldest_age, age);
    newest_age = std::min(newest_age, age);
    ++used;
  }

  out.sample_count = used;
  out.oldest_age_s = oldest_age;
  out.newest_age_s = (used > 0) ? newest_age : 0.0;
  if (used < min_samples) {
    out.reason = "feedback_incomplete";
    return out;
  }
  if (saw_gap) {
    out.reason = "feedback_sample_gap";
    return out;
  }
  // Require the retained samples to cover most of the stability window.
  if (oldest_age + 1e-9 < 0.8 * window_s) {
    out.reason = "feedback_incomplete";
    return out;
  }

  int worst = -1;
  double worst_range = 0.0;
  for (int j = 0; j < kNDof; ++j) {
    // Range on unwrapped samples in the short window; wrap-aware via delta
    // from first sample if plain hi-lo is large (near ±pi).
    double range = hi[j] - lo[j];
    if (range > kPi) {
      range = 0.0;
      // Recompute peak |wrap delta| from newest sample as reference.
      const int newest_idx =
          (ring.head - 1 + FeedbackRing18::kCap) % FeedbackRing18::kCap;
      const double ref = ring.buf[newest_idx].q[j];
      for (int i = 0; i < used; ++i) {
        const int idx = (ring.head - 1 - i + FeedbackRing18::kCap * 4) %
                        FeedbackRing18::kCap;
        range = std::max(range,
                         std::abs(wrap_angle_delta(ring.buf[idx].q[j], ref)));
      }
    }
    if (range > worst_range) {
      worst_range = range;
      worst = j;
    }
  }
  out.worst_joint = worst;
  out.worst_range_rad = worst_range;
  if (worst_range > max_range_rad) {
    out.reason = "start_not_stable";
    return out;
  }
  out.ok = true;
  out.reason = "ok";
  return out;
}

FastAdoptDecision evaluate_fast_adopt(
    const TrajectoryPackage& package, const PlanStartSnapshot& start,
    const Vec18& q_meas_now, bool meas_valid, double meas_stamp_s,
    const Vec18& q_cmd_now, bool has_cmd_now, const FeedbackRing18& ring,
    double now_s, double track_err_max_rad, const Vec18& dq_max,
    double dt_step) {
  FastAdoptDecision d;
  std::ostringstream detail;
  detail << std::fixed << std::setprecision(5);

  if (!package.certificate.passed || package.samples.size() < 2) {
    d.reason = "uncertified_package";
    d.detail = "uncertified_package";
    return d;
  }
  if (!meas_valid || !q_meas_now.allFinite()) {
    d.wait_for_stable = true;
    d.reason = "feedback_incomplete";
    d.detail = "meas_invalid";
    return d;
  }
  if (!std::isfinite(meas_stamp_s) || (now_s - meas_stamp_s) > kFeedbackMaxAgeS) {
    d.wait_for_stable = true;
    d.reason = "feedback_stale";
    detail << "meas_age=" << (now_s - meas_stamp_s);
    d.detail = detail.str();
    return d;
  }

  const FeedbackStabilityReport stab =
      evaluate_feedback_stability(ring, now_s);
  if (!stab.ok) {
    d.wait_for_stable = true;
    d.reason = stab.reason;
    detail << "stab=" << stab.reason << " n=" << stab.sample_count
           << " worst_j=" << stab.worst_joint
           << " range=" << stab.worst_range_rad
           << " age0=" << stab.oldest_age_s << " age1=" << stab.newest_age_s;
    d.detail = detail.str();
    return d;
  }

  if (!start.has_meas_start || !start.q_meas_start.allFinite()) {
    d.reason = "missing_meas_start";
    d.detail = "plan start lacked valid q_meas_start";
    return d;
  }

  // Planning-window feedback change: q_meas_now - q_meas_start (NOT vs plan_q).
  int worst_j = -1;
  double worst_d = 0.0;
  for (int j = 0; j < kNDof; ++j) {
    const double dj =
        std::abs(wrap_angle_delta(q_meas_now[j], start.q_meas_start[j]));
    if (dj > worst_d) {
      worst_d = dj;
      worst_j = j;
    }
  }
  d.worst_plan_delta_joint = worst_j;
  d.worst_plan_delta_rad = worst_d;

  double meas_vs_plan = 0.0;
  for (int j = 0; j < kNDof; ++j) {
    meas_vs_plan = std::max(
        meas_vs_plan,
        std::abs(wrap_angle_delta(q_meas_now[j], package.start_q[j])));
  }
  d.meas_vs_plan_start_rad = meas_vs_plan;

  (void)q_cmd_now;
  (void)has_cmd_now;
  (void)track_err_max_rad;
  (void)dq_max;
  (void)dt_step;

  detail << "plan_dJ=" << worst_j << " plan_d=" << worst_d
         << " meas_vs_plan=" << d.meas_vs_plan_start_rad
         << " stab_n=" << stab.sample_count
         << " stab_range=" << stab.worst_range_rad;

  if (worst_d > kPlanMeasDeltaMaxRad) {
    d.request_replan = true;
    d.reason = "plan_meas_moved";
    d.detail = detail.str();
    return d;
  }

  if (d.meas_vs_plan_start_rad > kAdoptStartQTolRad + 1e-12) {
    d.request_replan = true;
    d.reason = "start_q_mismatch";
    d.detail = detail.str();
    return d;
  }

  const AdmitDecision admit = admit_package_start(
      package, q_meas_now, Vec18::Zero(), false, kAdoptStartQTolRad, 0.5);
  if (!admit.ok) {
    d.request_replan = true;
    d.reason = admit.reason;
    d.detail = detail.str();
    return d;
  }

  d.ok = true;
  d.fast_path = true;
  d.reason = "fast_adopt";
  d.q_continuity = package.start_q;
  d.detail = detail.str();
  return d;
}


std::string format_vec18_compact(const Vec18& q, int precision) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision);
  for (int j = 0; j < kNDof; ++j) {
    if (j) {
      os << ',';
    }
    os << 'j' << j << '=' << q[j];
  }
  return os.str();
}

FeedbackWindowStats summarize_feedback_windows(
    const FeedbackRing18& ring, double submit_recv_s, double adopt_recv_s,
    const Vec18& q_meas_start, bool has_meas_start, double pre_window_s) {
  FeedbackWindowStats out;
  if (!(std::isfinite(submit_recv_s) && std::isfinite(adopt_recv_s) &&
        pre_window_s > 0.0)) {
    out.pre_reason = "bad_window_args";
    out.during_reason = "bad_window_args";
    return out;
  }
  if (ring.count <= 0) {
    out.pre_reason = "cache_empty";
    out.during_reason = "cache_empty";
    return out;
  }

  Vec18 pre_lo = Vec18::Constant(std::numeric_limits<double>::infinity());
  Vec18 pre_hi = Vec18::Constant(-std::numeric_limits<double>::infinity());
  Vec18 during_peak = Vec18::Zero();
  int pre_n = 0;
  int during_n = 0;
  double pre_oldest = 0.0;
  double pre_newest = std::numeric_limits<double>::infinity();

  for (int i = 0; i < ring.count; ++i) {
    const int idx = (ring.head - 1 - i + FeedbackRing18::kCap * 4) %
                    FeedbackRing18::kCap;
    const FeedbackSample18& s = ring.buf[idx];
    if (!s.valid || !s.q.allFinite() || !std::isfinite(s.recv_stamp_s)) {
      continue;
    }
    // Pre-submit window on receive clock: (submit-pre_window, submit].
    if (s.recv_stamp_s <= submit_recv_s + 1e-9 &&
        s.recv_stamp_s >= submit_recv_s - pre_window_s - 1e-9) {
      for (int j = 0; j < kNDof; ++j) {
        pre_lo[j] = std::min(pre_lo[j], s.q[j]);
        pre_hi[j] = std::max(pre_hi[j], s.q[j]);
      }
      const double age = submit_recv_s - s.recv_stamp_s;
      pre_oldest = std::max(pre_oldest, age);
      pre_newest = std::min(pre_newest, age);
      ++pre_n;
    }
    // During planning: (submit, adopt] vs q_meas_start.
    if (has_meas_start && s.recv_stamp_s > submit_recv_s - 1e-9 &&
        s.recv_stamp_s <= adopt_recv_s + 1e-9) {
      for (int j = 0; j < kNDof; ++j) {
        during_peak[j] = std::max(
            during_peak[j],
            std::abs(wrap_angle_delta(s.q[j], q_meas_start[j])));
      }
      ++during_n;
    }
  }

  out.pre_samples = pre_n;
  out.during_samples = during_n;
  if (pre_n < kFeedbackMinSamples || pre_oldest + 1e-9 < 0.8 * pre_window_s) {
    out.pre_reason = "data_insufficient";
    out.pre_ok = false;
  } else {
    int worst = -1;
    double worst_r = 0.0;
    for (int j = 0; j < kNDof; ++j) {
      double range = pre_hi[j] - pre_lo[j];
      if (range > kPi) {
        range = 0.0;
        const double ref = 0.5 * (pre_lo[j] + pre_hi[j]);
        range = std::max(std::abs(wrap_angle_delta(pre_hi[j], ref)),
                         std::abs(wrap_angle_delta(pre_lo[j], ref)));
      }
      out.pre_range_rad[j] = range;
      if (range > worst_r) {
        worst_r = range;
        worst = j;
      }
    }
    out.pre_worst_joint = worst;
    out.pre_worst_range_rad = worst_r;
    out.pre_ok = true;
    out.pre_reason = "ok";
  }

  if (!has_meas_start) {
    out.during_reason = "missing_meas_start";
    out.during_ok = false;
  } else if (during_n < 1) {
    out.during_reason = "data_insufficient";
    out.during_ok = false;
  } else {
    int worst = -1;
    double worst_d = 0.0;
    for (int j = 0; j < kNDof; ++j) {
      out.during_delta_rad[j] = during_peak[j];
      if (during_peak[j] > worst_d) {
        worst_d = during_peak[j];
        worst = j;
      }
    }
    out.during_worst_joint = worst;
    out.during_worst_delta_rad = worst_d;
    out.during_ok = true;
    out.during_reason = "ok";
  }
  return out;
}

std::string format_plan_adopt_diag(
    const PlanStartSnapshot& start, double adopt_ros_s,
    double adopt_meas_source_s, bool has_adopt_meas_source,
    double adopt_meas_recv_s, bool has_adopt_meas_recv,
    const Vec18& q_meas_adopt, bool has_meas_adopt, const Vec18& q_cmd_now,
    bool has_cmd_now, const FeedbackWindowStats& windows, const char* result,
    const char* reason, double elapsed_s) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(5);
  auto stamp_or_unknown = [](bool ok, double v) -> std::string {
    if (!ok || !std::isfinite(v)) {
      return "unknown";
    }
    std::ostringstream s;
    s << std::fixed << std::setprecision(5) << v;
    return s.str();
  };

  os << "result=" << (result ? result : "unset")
     << " why=" << (reason ? reason : "unset")
     << " elapsed_s=" << elapsed_s
     << " submit_ros=" << stamp_or_unknown(std::isfinite(start.submit_ros_s),
                                           start.submit_ros_s)
     << " submit_clock=ros_time"
     << " plan_meas_src="
     << stamp_or_unknown(start.has_meas_source, start.meas_source_s)
     << " plan_meas_src_clock="
     << (start.has_meas_source ? "joint_states.header" : "unknown")
     << " plan_meas_recv="
     << stamp_or_unknown(start.has_meas_recv, start.meas_recv_s)
     << " plan_meas_recv_clock=ros_time"
     << " adopt_ros=" << stamp_or_unknown(std::isfinite(adopt_ros_s), adopt_ros_s)
     << " adopt_clock=ros_time"
     << " adopt_meas_src="
     << stamp_or_unknown(has_adopt_meas_source, adopt_meas_source_s)
     << " adopt_meas_src_clock="
     << (has_adopt_meas_source ? "joint_states.header" : "unknown")
     << " adopt_meas_recv="
     << stamp_or_unknown(has_adopt_meas_recv, adopt_meas_recv_s)
     << " adopt_meas_recv_clock=ros_time";

  os << " q_plan_start=[" << format_vec18_compact(start.q_plan_start) << "]";
  if (start.has_meas_start) {
    os << " q_meas_plan=[" << format_vec18_compact(start.q_meas_start) << "]";
  } else {
    os << " q_meas_plan=unknown";
  }
  if (has_meas_adopt) {
    os << " q_meas_adopt=[" << format_vec18_compact(q_meas_adopt) << "]";
    if (start.has_meas_start) {
      Vec18 d = Vec18::Zero();
      for (int j = 0; j < kNDof; ++j) {
        d[j] = wrap_angle_delta(q_meas_adopt[j], start.q_meas_start[j]);
      }
      os << " d_meas_plan_to_adopt=[" << format_vec18_compact(d) << "]";
    }
  } else {
    os << " q_meas_adopt=unknown";
  }
  if (has_cmd_now) {
    os << " q_cmd_now=[" << format_vec18_compact(q_cmd_now) << "]";
    if (has_meas_adopt) {
      Vec18 d = Vec18::Zero();
      for (int j = 0; j < kNDof; ++j) {
        d[j] = wrap_angle_delta(q_meas_adopt[j], q_cmd_now[j]);
      }
      os << " d_meas_vs_cmd=[" << format_vec18_compact(d) << "]";
    }
  } else {
    os << " q_cmd_now=none";
  }

  os << " pre200_ok=" << (windows.pre_ok ? 1 : 0)
     << " pre200_why=" << windows.pre_reason
     << " pre200_n=" << windows.pre_samples
     << " pre200_worst_j=" << windows.pre_worst_joint
     << " pre200_worst_range=" << windows.pre_worst_range_rad;
  if (windows.pre_ok) {
    os << " pre200_range=[" << format_vec18_compact(windows.pre_range_rad)
       << "]";
  }
  os << " during_ok=" << (windows.during_ok ? 1 : 0)
     << " during_why=" << windows.during_reason
     << " during_n=" << windows.during_samples
     << " during_worst_j=" << windows.during_worst_joint
     << " during_worst_d=" << windows.during_worst_delta_rad;
  if (windows.during_ok) {
    os << " during_delta=[" << format_vec18_compact(windows.during_delta_rad)
       << "]";
  }
  return os.str();
}

bool sample_trajectory_at(const TrajectoryPackage& package, double t_s,
                          TrajectoryTimeSample* out, bool need_derivatives) {
  if (out == nullptr) {
    return false;
  }
  *out = TrajectoryTimeSample{};
  const JointCurve& samples = package.samples;
  const TimeLawTable& law = package.time_law;
  if (samples.empty()) {
    out->reason = "empty_samples";
    return false;
  }
  if (samples.size() == 1) {
    out->q = samples.front();
    out->dq.setZero();
    out->ddq.setZero();
    out->s = 0.0;
    out->t_s = 0.0;
    out->ok = true;
    out->reason = "ok";
    return true;
  }
  if (law.t_s.size() < 2 || law.s.size() != law.t_s.size()) {
    out->reason = "bad_time_law";
    return false;
  }
  const double t_end = std::max(law.duration_s, law.t_s.back());
  const double t = std::clamp(t_s, 0.0, t_end);
  out->t_s = t;

  auto s_at_t = [&](double tt) -> double {
    std::size_t i = 0;
    while (i + 1 < law.t_s.size() && law.t_s[i + 1] < tt - 1e-15) {
      ++i;
    }
    const std::size_t i1 = std::min(i + 1, law.t_s.size() - 1);
    const double t0 = law.t_s[i];
    const double t1 = law.t_s[i1];
    const double s0 = law.s[i];
    const double s1 = law.s[i1];
    if (t1 - t0 <= 1e-12) {
      return s1;
    }
    const double h = t1 - t0;
    const double a = std::clamp((tt - t0) / h, 0.0, 1.0);
    if (law.ds_dt.size() <= i1) return s0 + a * (s1 - s0);
    // Honour planned endpoint/STOP/ACK rates. Linear interpolation ignored
    // ds_dt=0 and turned every stop into an abrupt velocity discontinuity.
    const double secant = (s1 - s0) / h;
    if (!(secant > 0.0)) return s0;
    double m0 = std::clamp(law.ds_dt[i], 0.0, 3.0 * secant);
    double m1 = std::clamp(law.ds_dt[i1], 0.0, 3.0 * secant);
    const double norm = std::hypot(m0 / secant, m1 / secant);
    if (norm > 3.0) { m0 *= 3.0 / norm; m1 *= 3.0 / norm; }
    const double a2 = a * a, a3 = a2 * a;
    return std::clamp((2*a3 - 3*a2 + 1)*s0 + (a3 - 2*a2 + a)*h*m0 +
                      (-2*a3 + 3*a2)*s1 + (a3 - a2)*h*m1, s0, s1);
  };

  const double s = s_at_t(t);
  out->s = s;
  const double prog = s * static_cast<double>(samples.size() - 1);
  out->q = hermite_joint_at(samples, prog);

  if (!need_derivatives) {
    out->dq.setZero();
    out->ddq.setZero();
    if (!out->q.allFinite()) {
      out->reason = "nonfinite_sample";
      out->ok = false;
      return false;
    }
    out->ok = true;
    out->reason = "ok";
    return true;
  }

  // Time derivatives via central differences on the executed Hermite curve.
  const double h = 1e-3;
  const double s_m = s_at_t(std::max(0.0, t - h));
  const double s_p = s_at_t(std::min(t_end, t + h));
  const Vec18 q_m =
      hermite_joint_at(samples, s_m * static_cast<double>(samples.size() - 1));
  const Vec18 q_p =
      hermite_joint_at(samples, s_p * static_cast<double>(samples.size() - 1));
  const double denom = std::max(2.0 * h, 1e-9);
  out->dq = (q_p - q_m) / denom;
  // Second derivative: (q_p - 2 q + q_m) / h^2
  out->ddq = (q_p - 2.0 * out->q + q_m) / (h * h);
  if (!out->q.allFinite() || !out->dq.allFinite() || !out->ddq.allFinite()) {
    out->reason = "nonfinite_sample";
    out->ok = false;
    return false;
  }
  out->ok = true;
  out->reason = "ok";
  return true;
}

}  // namespace kmpc
