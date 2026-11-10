// 单臂够点：整段直线一次求完，之后只沿这条路径推进度。
// 变量是四个独立腰关节加这条臂。另一条臂保持出发角度。
// 步长用直线自己的速度比例。单臂解不出就停住，不进跟踪。双臂、标定仍走 QP。

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/path_progress.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
constexpr double kLineSolveTolM = 0.0015;
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

void CartesianTracker::line_clear() {
  line_ = LineRun{};
  line_q_.setZero();
  line_rate_ = 0.0;
  line_rate_acc_ = 0.0;
  line_tip_.reset();
  line_reference_.reset();
  last_line_cache_committed_ = false;
  reverse_decision_ = reverse_last_line_enable_ ? "skipped" : "disabled";
  line_seg_start_s_ = -1.0;
  line_tail_enter_brake_s_ = -1.0;
  line_tail_enter_2cm_s_ = -1.0;
  line_tail_hold_s_ = -1.0;
  last_line_qp_fail_ = LineQpFailSnapshot{};
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
  ev.t_steady_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
  ev.prog = prog;
  ev.build_ms = build_ms;
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
  if (!line_diag_enable_ || !q_meas.allFinite()) return;
  if (source_stamp_s >= 0.0 && std::isfinite(source_stamp_s)) {
    line_diag_source_ns_ = static_cast<int64_t>(std::llround(source_stamp_s * 1e9));
  } else {
    line_diag_source_ns_ = -1;
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
    const Vec18* pose_pref) const {
  line_fit_straighten_axis_reverted_ = false;
  const Vec3 p0 = kin_.fk_pos(q0, frame);
  const double dist = (target - p0).norm();
  int n = 1;
  if (dist > kLineSampleM) {
    n = static_cast<int>(std::ceil(dist / kLineSampleM));
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
  auto keep_com = [&](Aligned<Red>& ys) {
    for (int i = 1; i <= n; ++i) {
      const Vec18 q = expand(ys[static_cast<std::size_t>(i)]);
      if (!guards_.com_xy_inside(kin_.com(q))) {
        ys[static_cast<std::size_t>(i)].head<kNWaist>()=ys[static_cast<std::size_t>(i-1)].head<kNWaist>();
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

  // Task-priority nullspace pull toward shoulder J2 / elbow J4 preferences.
  // Accept a step only when Cartesian residual stays within the solve tolerance.
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

  // 每个点从两端连线出发，收到自己的直线位置。连线是整段转角最少的那条。
  for (int i = 1; i <= n; ++i) {
    const Red ref = path[static_cast<std::size_t>(i)];
    double w_scale = 1.0;
    for (int it = 0; it < 24; ++it) {
      Eigen::Matrix<double, 6, kRed> jac;
      Eigen::Matrix<double, 6, 1> err;
      double ori_rad = 0.0;
      task(path[static_cast<std::size_t>(i)], i, jac, err, ori_rad);
      if (err.head<3>().norm() < kLineSolveTolM && (!ori_on || ori_rad < 0.08)) {
        break;
      }
      const Red w = weights(ref, path[static_cast<std::size_t>(i)]) * w_scale;
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
      if ((next - path[static_cast<std::size_t>(i)]).norm() < 1e-5) {
        if (err.head<3>().norm() > kLineSolveTolM && w_scale > 0.05) {
          w_scale *= 0.2;
          continue;
        }
        break;
      }
      path[static_cast<std::size_t>(i)] = next;
    }
    refine_pose(path[static_cast<std::size_t>(i)], i);
  }
  keep_com(path);
  if (worst_pos(path) > kLineSampleTolM) {
    for (int i = 1; i <= n; ++i) {
      const Red ref = path[static_cast<std::size_t>(i)];
      for (int it = 0; it < 16; ++it) {
        Eigen::Matrix<double, 6, kRed> jac;
        Eigen::Matrix<double, 6, 1> err;
        double ori_rad = 0.0;
        task(path[static_cast<std::size_t>(i)], i, jac, err, ori_rad);
        if (err.head<3>().norm() < kLineSolveTolM) {
          break;
        }
        const Red w = weights(ref, path[static_cast<std::size_t>(i)]) * 0.2;
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
        if (!next.allFinite() ||
            (next - path[static_cast<std::size_t>(i)]).norm() < 1e-5) {
          break;
        }
        path[static_cast<std::size_t>(i)] = next;
      }
    }
    keep_com(path);
  }

  for (int pass = 0; pass < 8; ++pass) {
    if (worst_pos(path) < kLineSolveTolM) {
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
      const Red delta = path[static_cast<std::size_t>(i)] - path[static_cast<std::size_t>(i - 1)];
      const MatR weight = w.asDiagonal();
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

  if (worst_pos(path) > kLineSampleTolM) {
    return false;
  }
  for (int i = 1; i <= n; ++i) {
    const Vec18 jumped =
        expand(path[static_cast<std::size_t>(i)]) - expand(path[static_cast<std::size_t>(i - 1)]);
    if (jumped.cwiseAbs().maxCoeff() > kLineJointJump) {
      return false;
    }
  }

  // 俯仰和是一个冗余量，直线只约束手。端点不动，中间点只在手的零空间里拉回两端连线。
  // 拉完仍要过 3 mm 管子；拉不动就留在原来的可行解上，不把整段判失败。
  // 总和本来就已经接近单调、拉直却挤出单轴折返时，退回拉直前的路径。
  if (line_pitch_straighten_enable_ && !lock_waist && n > 1) {
    ++line_pitch_straighten_hits_;
    const auto pitch_of = [](const Red& y) { return y.head<3>().sum(); };
    const double p0 = pitch_of(path[0]);
    const double pN = pitch_of(path[static_cast<std::size_t>(n)]);
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
        const Red next = limited_step(y, nvec * alpha + j_pinv * err);
        Eigen::Matrix<double, 6, kRed> jac2;
        Eigen::Matrix<double, 6, 1> err2;
        double ori2 = 0.0;
        task(next, i, jac2, err2, ori2);
        if (err2.head<3>().norm() > kLineSampleTolM || (ori_on && ori2 > 0.08)) {
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
    if (straighten_axis_turnback_regressed(kept, path)) {
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
  return line_validate(samples, target, quat, ori_on, frame);
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
  line_.on = false;
  line_.hold = false;
  line_.refused = true;
  line_.reason = reason;
  last_arrived_ = false;
  line_rate_ = 0.0;
  line_rate_acc_ = 0.0;
  clear_last_line_cache();
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
  for (int i = 0; i < n; ++i) {
    // Match joint-linear streaming, including all four 250 Hz interior ticks.
    for (int j = 0; j <= 5; ++j) {
      const double a = j / 5.0;
      const Vec18 q = (1-a)*samples[i] + a*samples[i+1];
      if (!q.allFinite() ||
          (q.array() < (limits_.q_min.array()+q_limit_margin_-1e-9)).any() ||
          (q.array() > (limits_.q_max.array()-q_limit_margin_+1e-9)).any() ||
          !guards_.com_xy_inside(kin_.com(q))) return false;
      const double t = (i+a)/n;
      const auto pose = kin_.fk(q, frame);
      if ((pose.col(3).head<3>() - ((1-t)*start+t*target)).norm() > kLineSampleTolM) return false;
      if (ori_on && quat && ori_error_xyzw(quat_slerp_xyzw(start_ori,*quat,t),
          rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm() > arrived_ori_rad_) return false;
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
  const double pitch_bound=std::max(torso_pitch_max_,std::abs(line_.samples.front().head<3>().sum()));
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
    const double tube_margin=kLineSampleTolM-(position-(start+t*ab)).norm();
    // Convert the historical dot-product progress tolerance to meters.
    const double progress_margin=(along-previous+1e-6*std::max(ab.norm(),1e-6)) /
        std::max(ab.norm(),1e-6);
    result.tube_margin_m=std::min(result.tube_margin_m,tube_margin);
    result.progress_margin_m=std::min(result.progress_margin_m,progress_margin);
    if(tube_margin<0 || progress_margin<0) {
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
  auto probe = [&](const auto& path) {
    if (!line_tip_) return true;
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> next;
    const auto& tip=*line_tip_;
    return line_arm_only(path.back(),arm,tip.xyz,tip.ori_on?&tip.quat_xyzw:nullptr,
        tip.ori_on,frame,next);
  };
  std::vector<Vec18,Eigen::aligned_allocator<Vec18>> samples;
  bool locked=true;
  std::string candidate="none";
  std::string pose_decision="disabled";
  std::string reverse_decision=reverse_last_line_enable_ ? "skipped" : "disabled";
  const bool recovering_pitch=torso_pitch_max_>0 && std::abs(q0.head<3>().sum())>torso_pitch_max_+1e-6;
  if (recovering_pitch && upper_.waist_locked) { line_.reason="waist_locked_outside"; return false; }

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
      // Re-anchor endpoints to the live q0 / Cartesian target while keeping the
      // interior nominal samples (equal Cartesian progress, not servo timing).
      rev.front()=q0;
      if(!line_validate(rev,target,quat,ori_on,frame) || !probe(rev)) {
        reverse_decision="rejected_validate";
      } else {
        // reverse_cache may start slightly outside (servo overshoot). Do not
        // require the cached end back in the hard box here — that rejected field
        // A/B and fought QP's bounded recovery. Only forbid worsening excess
        // vs live q0; free_fit/qa_seed still keep their end-in-box gates.
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
            samples=std::move(rev); locked=!waist_moves; ok=true;
            candidate="reverse_cache"; reverse_decision="accepted";
          }
        }
      }
      if(!ok) {
        // Second candidate: re-solve B→A with terminal seed = cached qA.
        std::vector<Vec18,Eigen::aligned_allocator<Vec18>> seeded;
        bool seed_locked=false;
        if(!upper_.waist_locked &&
           line_fit(q0,arm,target,quat,ori_on,false,frame,seeded,seed_locked,true,&c.q_start) &&
           (!recovering_pitch || (!seed_locked && !seeded.empty() &&
             std::abs(seeded.back().head<3>().sum())<=torso_pitch_max_+1e-7)) &&
           probe(seeded) && line_validate(seeded,target,quat,ori_on,frame)) {
          straighten_axis_reverted=line_fit_straighten_axis_reverted_;
          samples=std::move(seeded); locked=seed_locked; ok=true;
          candidate="qa_seed";
          reverse_decision="qa_seed_accepted;reverse_rejected_validate";
        } else {
          reverse_decision=reverse_decision+";qa_seed_rejected";
        }
      }
    }
  } else if (reverse_last_line_enable_) {
    reverse_decision="skipped_no_match";
  }

  if(!ok) ok=line_arm_only(q0,arm,target,quat,ori_on,frame,samples);
  constexpr double fold_ok=16.0*kPi/180.0;
  bool forced=false;
  if(candidate!="reverse_cache" && candidate!="qa_seed") {
    const double arm_fold=ok?max_arm_fold(samples,arm):0.0;
    const bool locked_pass=!recovering_pitch && ok && arm_fold<fold_ok && probe(samples);
    if (locked_pass) {
      candidate="arm_only";
    } else {
      std::vector<Vec18,Eigen::aligned_allocator<Vec18>> free_samples;
      bool free_locked=false;
      const Vec18* recovery_seed=(recovering_pitch && ok && !samples.empty()) ? &samples.back() : nullptr;
      const bool free_ok=!upper_.waist_locked && line_fit(q0,arm,target,quat,ori_on,false,
          frame,free_samples,free_locked,!line_tip_.has_value() && !recovering_pitch,recovery_seed);
      if (free_ok && (!recovering_pitch || (!free_locked && std::abs(free_samples.back().head<3>().sum())<=torso_pitch_max_+1e-7)) && probe(free_samples)) {
        straighten_axis_reverted=line_fit_straighten_axis_reverted_;
        samples=std::move(free_samples); locked=free_locked; ok=true;
        forced=line_tip_.has_value();
        candidate="free_fit";
      } else {
        ok=false;
        if (line_tip_) line_.reason="cascade_unreachable";
      }
    }
  }
  // A bounded second seed is derived from the real next pose: solve the tip,
  // reverse its locked-waist path to the preparation pose, then fit the current
  // segment to that endpoint. No guessed tip, relaxed orientation or cost changes.
  if(!ok && line_tip_ && !upper_.waist_locked) {
    const auto& tip=*line_tip_;
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> tip_path,back_path,trial;
    bool dummy=false,trial_locked=false;
    if(line_fit(q0,arm,tip.xyz,tip.ori_on?&tip.quat_xyzw:nullptr,tip.ori_on,false,
        frame,tip_path,dummy,false) &&
       line_arm_only(tip_path.back(),arm,target,quat,ori_on,frame,back_path)) {
      const Vec18 seed=back_path.back();
      if(line_fit(q0,arm,target,quat,ori_on,false,frame,trial,trial_locked,false,&seed) &&
         (!recovering_pitch || (!trial_locked && !trial.empty() && std::abs(trial.back().head<3>().sum())<=torso_pitch_max_+1e-7)) && probe(trial)) {
        straighten_axis_reverted=line_fit_straighten_axis_reverted_;
        samples=std::move(trial);locked=trial_locked;forced=true;ok=true;
        candidate="tip_seed";
      }
    }
  }
  // Prefer the existing fixed-waist successor candidates. Only when all of
  // them fail, prove the next straight segment with the same constrained
  // whole-body solver that will execute it. Arm-only failure is not evidence
  // that a task permitting waist motion is unreachable.
  bool cascade_full_body=false;
  if(!ok && line_tip_ && !upper_.waist_locked) {
    const auto& tip=*line_tip_;
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> trial,next;
    bool trial_locked=false,next_locked=false;
    if(line_fit(q0,arm,target,quat,ori_on,false,frame,trial,trial_locked,false) &&
       (!recovering_pitch || std::abs(trial.back().head<3>().sum())<=torso_pitch_max_+1e-7)) {
      const bool trial_reverted=line_fit_straighten_axis_reverted_;
      if(line_fit(trial.back(),arm,tip.xyz,tip.ori_on?&tip.quat_xyzw:nullptr,
                  tip.ori_on,false,frame,next,next_locked)) {
      straighten_axis_reverted=trial_reverted;
      samples=std::move(trial);locked=trial_locked;forced=true;ok=true;
      cascade_full_body=true;
      candidate="cascade_full_body";
      }
    }
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
      candidate == "arm_only" || candidate == "free_fit";
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
                std::abs(pref_samples.back().head<3>().sum())>torso_pitch_max_+1e-7) {
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
        if(improve < line_pose_pref_min_improve_) {
          pose_decision=std::string("rejected_no_improve;")+metric;
        } else if(waist_extra > line_pose_pref_waist_budget_) {
          pose_decision=std::string("rejected_waist_budget;")+metric;
        } else if(wrist_extra > line_pose_pref_wrist_budget_) {
          pose_decision=std::string("rejected_wrist_budget;")+metric;
        } else {
          straighten_axis_reverted=line_fit_straighten_axis_reverted_;
          samples=std::move(pref_samples);
          locked=pref_locked;
          forced=line_tip_.has_value();
          candidate="pose_pref";
          pose_decision=std::string("accepted;")+metric;
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

  // Sum-only straighten that worsens a waist-axis chord turnback is reverted
  // inside line_fit. That pre-straighten path is accepted, including residual
  // sum turnback. A straightened path that still reverses the sum is refused.
  constexpr double kPitchTurnbackMax=0.05*kPi/180.0;
  if (ok && line_pitch_straighten_enable_ && !straighten_axis_reverted &&
      line_pitch_turnback(samples)>kPitchTurnbackMax) {
    ok=false;
    line_.reason="pitch_turnback_unresolved";
  }

  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  reverse_decision_=reverse_decision;
  if (!ok || samples.empty()) {
    if(recovering_pitch) line_.reason="pitch_recovery_infeasible";
    line_.build_ms=ms;
    line_.reverse_decision=reverse_decision;
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
  built.pose_pref_decision=pose_decision;
  built.reverse_decision=reverse_decision;
  built.max_arm_fold=max_arm_fold(built.samples,arm);
  built.frozen=dist<=kLineFreezeM;
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
  const double want = speed * brake;
  last_line_path_remain_m_ = path_remain;
  last_line_brake_ = brake;
  last_line_want_ = want;
  // 加减速只乘在路径进度这一个标量上，再去采样已经求好的折线。不能按关节各做一条 S 曲线，
  // 否则手会离开直线，预备点/抓取点上会重新出现走过再折返。
  if (speed <= 1e-4) {
    line_rate_ = 0.0;
    line_rate_acc_ = 0.0;
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
    if(remain>kLineSampleTolM || !ori_ok) {
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
    line_diag_push(LineDiagLayer::FollowQp,q_from,q_from,line_.prog,0.0,
                   line_.candidate.c_str(),"qp_fail");
    refuse_line(line_.reason.empty()?"path_qp_or_tube":line_.reason); out.dq.setZero(); return false;
  }
  line_diag_push(LineDiagLayer::FollowQp,next,q_from,line_.prog,0.0,
                 line_.candidate.c_str(),"qp");
  const Vec18 dq=next-q_from;
  line_q_=next;
  if((line_.target-kin_.fk_pos(next,line_.frame)).norm()<=kLineFreezeM) line_.frozen=true;
  last_arrived_ = false;
  line_diag_push(LineDiagLayer::FollowCommit,next,q0,line_.prog,0.0,
                 line_.candidate.c_str(),"commit");
  out.dq = dq;
  return true;
}

}  // namespace kmpc
