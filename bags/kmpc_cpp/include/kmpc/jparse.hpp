#pragma once
// 对应 Python cartesian_tracker.jparse_scale_task / pad_tracking_jacobian。
// 对照 09-04：J-PARSE 两块各垫（Jv、Jω 各一张 SVD）。禁止 6 行合成一张。

#include <Eigen/SVD>

#include "kmpc/types.hpp"

namespace kmpc {

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> pad_tracking_jacobian(
    const Eigen::Matrix<double, Rows, Cols>& jac, double gamma) {
  if (gamma <= 0.0) {
    return jac;
  }
  Eigen::JacobiSVD<Eigen::Matrix<double, Rows, Cols>> svd(
      jac, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& s = svd.singularValues();
  if (s.size() == 0 || s[0] <= 1e-12) {
    return jac;
  }
  const double floor = gamma * s[0];
  const int ns = static_cast<int>(s.size());
  Eigen::Matrix<double, Rows, Rows> sigma =
      Eigen::Matrix<double, Rows, Rows>::Zero();
  for (int i = 0; i < ns && i < Rows; ++i) {
    sigma(i, i) = s[i] < floor ? floor : s[i];
  }
  return (svd.matrixU() * sigma) * svd.matrixV().leftCols(ns).transpose();
}

inline Vec3 jparse_scale_task(const Mat3x18& jac, const Vec3& err, double gamma) {
  if (gamma <= 0.0) {
    return err;
  }
  Eigen::JacobiSVD<Eigen::Matrix<double, 3, 18>> svd(
      jac, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::Vector3d s = svd.singularValues();
  if (s.size() == 0 || s[0] <= 1e-12) {
    return err;
  }
  const double floor = gamma * s[0];
  Eigen::Vector3d scale = Eigen::Vector3d::Ones();
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] < floor) {
      scale[i] = s[i] / floor;
    }
  }
  const Eigen::Matrix<double, 3, 3> U = svd.matrixU();
  return U * scale.asDiagonal() * (U.transpose() * err);
}

inline Mat3x18 pad_waist_columns(const Mat3x18& jac, double gamma) {
  Mat3x18 out = jac;
  const Eigen::Matrix<double, 3, kNWaist> waist = jac.leftCols<kNWaist>();
  out.leftCols<kNWaist>() = pad_tracking_jacobian(waist, gamma);
  return out;
}

}  // namespace kmpc
