// 单臂够点：整段直线一次求完，之后只沿这条路径推进度。
// 变量是四个独立腰关节加这条臂。另一条臂保持出发角度。
// 步长用直线自己的速度比例。单臂解不出就停住，不进跟踪。双臂、标定仍走 QP。

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/path_progress.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>

#include <Eigen/Cholesky>

#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr double kLineSampleM = 0.005;
constexpr double kLineReplanM = 0.008;
constexpr double kLineBrakeM = 0.08;
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

void CartesianTracker::line_clear() {
  line_ = LineRun{};
  line_q_.setZero();
  line_rate_ = 0.0;
  line_tip_.reset();
  line_reference_.reset();
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
    bool& waist_locked, bool allow_relock, const Vec18* goal_seed) const {
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

bool CartesianTracker::line_stream_segment_ok(const Vec18& from,const Vec18& to) const {
  if(!line_.on || line_.samples.empty() || !from.allFinite() || !to.allFinite()) return false;
  const Vec3 start=kin_.fk_pos(line_.samples.front(),line_.frame);
  const Vec3 ab=line_.target-start;
  double previous=(kin_.fk_pos(from,line_.frame)-start).dot(ab);
  const double pitch_bound=std::max(torso_pitch_max_,std::abs(line_.samples.front().head<3>().sum()));
  for(int i=1;i<=5;++i) {
    const Vec18 q=from+(i/5.0)*(to-from);
    if((q.array()<(limits_.q_min.array()+q_limit_margin_-1e-8)).any() ||
       (q.array()>(limits_.q_max.array()-q_limit_margin_+1e-8)).any() ||
       !guards_.com_xy_inside(kin_.com(q)) ||
       (torso_pitch_max_>0 && std::abs(q.head<3>().sum())>pitch_bound+1e-7)) return false;
    const Vec3 position=kin_.fk_pos(q,line_.frame);
    const double along=(position-start).dot(ab);
    const double t=std::clamp(along/std::max(ab.squaredNorm(),1e-12),0.0,1.0);
    if((position-(start+t*ab)).norm()>kLineSampleTolM || along<previous-1e-6*std::max(ab.norm(),1e-6)) return false;
    previous=along;
  }
  return true;
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
  if(torso_pitch_max_>0 && std::abs(q0.head<3>().sum())>torso_pitch_max_+0.001) {
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
  const bool recovering_pitch=torso_pitch_max_>0 && std::abs(q0.head<3>().sum())>torso_pitch_max_+1e-6;
  if (recovering_pitch && upper_.waist_locked) { line_.reason="waist_locked_outside"; return false; }
  bool ok=line_arm_only(q0,arm,target,quat,ori_on,frame,samples);
  constexpr double fold_ok=16.0*kPi/180.0;
  const bool locked_pass=!recovering_pitch && ok && max_arm_fold(samples,arm)<fold_ok && probe(samples);
  bool forced=false;
  if (!locked_pass) {
    std::vector<Vec18,Eigen::aligned_allocator<Vec18>> free_samples;
    bool free_locked=false;
    const Vec18* recovery_seed=(recovering_pitch && ok && !samples.empty()) ? &samples.back() : nullptr;
    const bool free_ok=!upper_.waist_locked && line_fit(q0,arm,target,quat,ori_on,false,
        frame,free_samples,free_locked,!line_tip_.has_value() && !recovering_pitch,recovery_seed);
    if (free_ok && (!recovering_pitch || (!free_locked && std::abs(free_samples.back().head<3>().sum())<=torso_pitch_max_+1e-7)) && probe(free_samples)) {
      samples=std::move(free_samples); locked=free_locked; ok=true;
      forced=line_tip_.has_value();
    } else {
      ok=false;
      if (line_tip_) line_.reason="cascade_unreachable";
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
        samples=std::move(trial);locked=trial_locked;forced=true;ok=true;
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
       (!recovering_pitch || std::abs(trial.back().head<3>().sum())<=torso_pitch_max_+1e-7) &&
       line_fit(trial.back(),arm,tip.xyz,tip.ori_on?&tip.quat_xyzw:nullptr,
                tip.ori_on,false,frame,next,next_locked)) {
      samples=std::move(trial);locked=trial_locked;forced=true;ok=true;
      cascade_full_body=true;
    }
  }
  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  if (!ok || samples.empty()) { if(recovering_pitch) line_.reason="pitch_recovery_infeasible"; line_.build_ms=ms; return false; }
  LineRun built;
  built.on=true; built.left=left; built.ori_on=ori_on; built.waist_locked=locked;
  built.target=target; built.quat=ori_on?*quat:Vec4(0,0,0,1); built.frame=frame;
  built.length_m=dist; built.samples=std::move(samples); built.build_ms=ms;
  built.cascade_checked=line_tip_.has_value(); built.cascade_passed=line_tip_.has_value();
  built.cascade_forced_waist=forced;
  built.cascade_full_body=cascade_full_body;
  built.frozen=dist<=kLineBrakeM;
  line_=std::move(built); line_q_=q0; line_rate_=0.0; ++line_gen_;
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
      (line_.target - kin_.fk_pos(line_q_,line_.frame)).norm() <= kLineBrakeM) line_.frozen=true;
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
    out.dq.setZero();
    return true;
  }

  const double nseg =
      line_.samples.size() <= 1
          ? 0.0
          : static_cast<double>(line_.samples.size() - 1);
  // 步长看路径自身的增量，不看 q0。q0 偏了也不把方向拉去追它。
  double brake = 1.0;
  const double path_remain = line_.length_m * (1.0-line_.prog/std::max(nseg,1.0));
  if (path_remain < kLineBrakeM) brake = std::max(0.05,path_remain/kLineBrakeM);
  const double speed = line_speed_scale_;
  const double want = speed * brake;
  if (speed <= 1e-4) {
    line_rate_ = 0.0;
  } else {
    double du = 1.0;
    if (line_accel_s_ > 1e-3) {
      du = constraints_.dt() / line_accel_s_;
    }
    if (line_rate_ < want) {
      line_rate_ = std::min(want, line_rate_ + du);
    } else {
      line_rate_ = std::max(want, line_rate_ - du);
    }
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
  // Once the path ends inside the accepted pose tolerance, do not spend more
  // ticks reducing a harmless residual through redundant joint motion.
  if(at_end && remain<=kLineSampleTolM && ori_ok &&
      line_rate_<=0.1*line_speed_scale_+1e-9) {
    line_.hold=true;line_q_=q_from;line_rate_=0;last_arrived_=true;
    out.dq.setZero();return true;
  }
  Vec18 next = q_from;
  if (line_rate_ > 1e-4 && !line_path_qp(q_from,line_.prog,step_lim,next)) {
    line_.prog=old_prog; refuse_line(line_.reason.empty()?"path_qp_or_tube":line_.reason); out.dq.setZero(); return false;
  }
  const Vec18 dq=next-q_from;
  line_q_=next;
  if((line_.target-kin_.fk_pos(next,line_.frame)).norm()<=kLineBrakeM) line_.frozen=true;
  last_arrived_ = false;
  out.dq = dq;
  return true;
}

}  // namespace kmpc
