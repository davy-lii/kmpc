#pragma once
// 对应 Python cartesian_tracker 顶层辅助函数。对照 09-04 §3。
// damped LS / ClampMag / 相对四元数；禁止 6 行一张 SVD（J-PARSE 在 jparse.hpp）。

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>

#include "kmpc/types.hpp"

namespace kmpc {

inline constexpr double kTargetJumpM = 0.02;
inline constexpr double kTargetJumpOriRad = 0.17;
inline constexpr double kIdentityQuatEps = 1e-6;
inline constexpr double kDeepExitM = 0.02;
inline constexpr double kPitchWallEpsDeg = 0.5;
// Shared build/QP/250Hz pitch-recovery admit: allow measured start slightly
// outside ±torso_pitch_max, never permanently widen the hard box.
// 20261110: command sat on -22°, measured about -22.129°, and 0.002 rad
// (~0.115°, gate near -22.115°) still refused. 0.005 rad (~0.286°) covers
// that edge.
inline constexpr double kStartPitchAdmitRad = 0.005;
// Straighten pulls only the pitch sum. If the sum was already within 0.15° of
// monotonic and the pass then reverses a waist pitch axis by more than 0.05°,
// drop it and keep the pre-straighten path. A real sum repair (startup pitch)
// is kept even when one axis moves.
inline constexpr double kWaistAxisTurnbackRevertRad = 0.05 * kPi / 180.0;
inline constexpr double kWaistAxisTurnbackWorseRad = 0.02 * kPi / 180.0;
inline constexpr double kPitchSumGainWorthKeepingRad = 0.15 * kPi / 180.0;

template <class SampleAt>
inline double chord_turnback(int n, SampleAt sample_at) {
  if (n < 2) {
    return 0.0;
  }
  const double front = sample_at(0);
  const double back = sample_at(n - 1);
  const double net = back - front;
  if (std::abs(net) < 1e-4) {
    double lo = front;
    double hi = front;
    for (int i = 1; i < n; ++i) {
      const double v = sample_at(i);
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    return hi - lo;
  }
  const double dir = net > 0.0 ? 1.0 : -1.0;
  double best = dir * front;
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double p = dir * sample_at(i);
    best = std::max(best, p);
    worst = std::max(worst, best - p);
  }
  return worst;
}

// True when a nearly monotonic pitch sum was redistributed into a waist-axis
// chord turnback. Callers then keep the pre-straighten samples.
template <class Path>
inline bool straighten_axis_turnback_regressed(const Path& before, const Path& after) {
  const int n = static_cast<int>(before.size());
  if (n < 2 || static_cast<int>(after.size()) != n) {
    return true;
  }
  double worst_increase = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double pre = chord_turnback(n, [&](int i) {
      return before[static_cast<std::size_t>(i)][axis];
    });
    const double post = chord_turnback(n, [&](int i) {
      return after[static_cast<std::size_t>(i)][axis];
    });
    if (post > kWaistAxisTurnbackRevertRad &&
        post > pre + kWaistAxisTurnbackWorseRad) {
      worst_increase = std::max(worst_increase, post - pre);
    }
  }
  if (!(worst_increase > kWaistAxisTurnbackRevertRad)) {
    return false;
  }
  const auto sum_at = [&](const Path& path, int i) {
    const auto& q = path[static_cast<std::size_t>(i)];
    return q[0] + q[1] + q[2];
  };
  const double sum_gain = chord_turnback(n, [&](int i) { return sum_at(before, i); }) -
                          chord_turnback(n, [&](int i) { return sum_at(after, i); });
  return sum_gain < kPitchSumGainWorthKeepingRad;
}
inline constexpr double kStallWallEzM = 0.05;
inline constexpr double kStallWallE0M = 0.20;
inline constexpr double kStallJ2SlackDeg = 8.0;

inline Vec3 clamp_mag(const Vec3& w, double d) {
  if (d <= 0.0) {
    return Vec3::Zero();
  }
  const double nrm = w.norm();
  if (nrm <= d) {
    return w;
  }
  return (d / nrm) * w;
}

inline double clamp_axis(double val, double cap) {
  if (cap <= 0.0) {
    return val;
  }
  if (val > cap) {
    return cap;
  }
  if (val < -cap) {
    return -cap;
  }
  return val;
}

inline Vec3 clamp_arm_residual(const Vec3& raw, double d_xz, double d_y) {
  Vec3 out = raw;
  out[0] = clamp_axis(out[0], d_xz);
  out[1] = clamp_axis(out[1], d_y);
  out[2] = clamp_axis(out[2], d_xz);
  return out;
}

template <int Cols>
Eigen::Matrix<double, Cols, 1> damped_joint_ls(
    const Eigen::Matrix<double, 3, Cols>& jac, const Vec3& err, double lam) {
  const double damp = lam < 0.0 ? 0.0 : lam;
  Eigen::Matrix<double, Cols, Cols> gram = jac.transpose() * jac;
  gram.diagonal().array() += damp * damp;
  const Eigen::Matrix<double, Cols, 1> rhs = jac.transpose() * err;
  const Eigen::LDLT<Eigen::Matrix<double, Cols, Cols>> ldlt(gram);
  if (ldlt.info() != Eigen::Success) {
    return Eigen::Matrix<double, Cols, 1>::Zero();
  }
  return ldlt.solve(rhs);
}

inline Vec3 damped_pitch_ls(const Eigen::Matrix<double, 3, 3>& j_pitch,
                            const Vec3& err, double lam) {
  return damped_joint_ls<3>(j_pitch, err, lam);
}

inline bool quat_is_identity_xyzw(const Vec4* quat) {
  if (quat == nullptr) {
    return true;
  }
  const double x = (*quat)[0];
  const double y = (*quat)[1];
  const double z = (*quat)[2];
  const double w = (*quat)[3];
  const double nrm = std::sqrt(w * w + x * x + y * y + z * z);
  if (nrm < 1e-12) {
    return true;
  }
  return std::sqrt(x * x + y * y + z * z) / nrm < kIdentityQuatEps;
}

// 工具系 → 法兰系的固定旋转。20261015 华成实机数据实锤（双臂×抓放 4 次，
// 两两差 ≤3.4°）：华成到达的法兰姿态 = 下发欧拉角 ⊗ Rz(+90°)。
// 旧常量（180° 绕 (−s,+s,0)）与真值差 Ry(180°)，腕 J5 被迫拧满 ±177° 撞限位。
// 旧注释「法兰 −Y = 工具 +X」两个常量都满足，仅靠它定不死映射，以实测为准。
// 单位四元数原样返回（位置模式）。
inline Vec4 tool_to_flange_quat_xyzw(const Vec4& tool) {
  if (quat_is_identity_xyzw(&tool)) {
    return tool;
  }
  constexpr double s = 0.7071067811865476;
  constexpr double dx = 0.0;
  constexpr double dy = 0.0;
  constexpr double dz = s;
  constexpr double dw = s;
  const double tx = tool[0];
  const double ty = tool[1];
  const double tz = tool[2];
  const double tw = tool[3];
  return Vec4(
      tw * dx + tx * dw + ty * dz - tz * dy,
      tw * dy - tx * dz + ty * dw + tz * dx,
      tw * dz + tx * dy - ty * dx + tz * dw,
      tw * dw - tx * dx - ty * dy - tz * dz);
}

inline Vec4 rot_to_quat_xyzw(const Mat3& r) {
  const double t = r(0, 0) + r(1, 1) + r(2, 2);
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (t > 0.0) {
    const double s = std::sqrt(t + 1.0) * 2.0;
    w = 0.25 * s;
    x = (r(2, 1) - r(1, 2)) / s;
    y = (r(0, 2) - r(2, 0)) / s;
    z = (r(1, 0) - r(0, 1)) / s;
  } else if (r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2)) {
    const double s = std::sqrt(1.0 + r(0, 0) - r(1, 1) - r(2, 2)) * 2.0;
    w = (r(2, 1) - r(1, 2)) / s;
    x = 0.25 * s;
    y = (r(0, 1) + r(1, 0)) / s;
    z = (r(0, 2) + r(2, 0)) / s;
  } else if (r(1, 1) > r(2, 2)) {
    const double s = std::sqrt(1.0 + r(1, 1) - r(0, 0) - r(2, 2)) * 2.0;
    w = (r(0, 2) - r(2, 0)) / s;
    x = (r(0, 1) + r(1, 0)) / s;
    y = 0.25 * s;
    z = (r(1, 2) + r(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + r(2, 2) - r(0, 0) - r(1, 1)) * 2.0;
    w = (r(1, 0) - r(0, 1)) / s;
    x = (r(0, 2) + r(2, 0)) / s;
    y = (r(1, 2) + r(2, 1)) / s;
    z = 0.25 * s;
  }
  const double nrm = std::sqrt(w * w + x * x + y * y + z * z);
  if (nrm < 1e-12) {
    return Vec4(0.0, 0.0, 0.0, 1.0);
  }
  return Vec4(x / nrm, y / nrm, z / nrm, w / nrm);
}

inline void wxyz_from_xyzw(const Vec4& quat, double& w, double& x, double& y,
                           double& z) {
  x = quat[0];
  y = quat[1];
  z = quat[2];
  w = quat[3];
  const double nrm = std::sqrt(w * w + x * x + y * y + z * z);
  if (nrm < 1e-12) {
    w = 1.0;
    x = 0.0;
    y = 0.0;
    z = 0.0;
    return;
  }
  w /= nrm;
  x /= nrm;
  y /= nrm;
  z /= nrm;
}

// 同一进度上的姿态。点积为负时先翻到短弧，避免规划中途腕拧一圈。
inline Vec4 quat_slerp_xyzw(const Vec4& a_in, const Vec4& b_in, double t) {
  Vec4 a = a_in;
  Vec4 b = b_in;
  auto normalize = [](Vec4& q) {
    const double n = q.norm();
    if (n < 1e-12) {
      q = Vec4(0.0, 0.0, 0.0, 1.0);
    } else {
      q /= n;
    }
  };
  normalize(a);
  normalize(b);
  if (t < 0.0) {
    t = 0.0;
  } else if (t > 1.0) {
    t = 1.0;
  }
  double dot = a.dot(b);
  if (dot < 0.0) {
    b = -b;
    dot = -dot;
  }
  if (dot > 0.9995) {
    Vec4 q = a + t * (b - a);
    normalize(q);
    return q;
  }
  const double theta = std::acos(dot > 1.0 ? 1.0 : dot);
  const double s = std::sin(theta);
  Vec4 q = (std::sin((1.0 - t) * theta) / s) * a +
           (std::sin(t * theta) / s) * b;
  normalize(q);
  return q;
}

inline Vec3 ori_error_xyzw(const Vec4& quat_des, const Vec4& quat_curr) {
  double wd, xd, yd, zd;
  double wc, xc, yc, zc;
  wxyz_from_xyzw(quat_des, wd, xd, yd, zd);
  wxyz_from_xyzw(quat_curr, wc, xc, yc, zc);
  const double w = wd * wc + xd * xc + yd * yc + zd * zc;
  const double x = xd * wc - wd * xc - yd * zc + zd * yc;
  const double y = yd * wc + xd * zc - wd * yc - zd * xc;
  const double z = zd * wc - xd * yc + yd * xc - wd * zc;
  const double sgn = w >= 0.0 ? 1.0 : -1.0;
  return Vec3(2.0 * sgn * x, 2.0 * sgn * y, 2.0 * sgn * z);
}

}  // namespace kmpc
