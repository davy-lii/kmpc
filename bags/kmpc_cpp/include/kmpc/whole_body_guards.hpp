#pragma once
// 全身硬盒：俯仰和、CoM 工作墙、yaw。笛卡尔与关节空间共用 18 维行。
// 笛卡尔装配：俯仰/yaw/CoM 都打 y0 与 y_{N-1}。CoM 用工作墙 + stay-feasible。
// 肘下墙只给关节空间；盒外用 min(...,0) 只许返回，禁止一步拉回。

#include "kmpc/config.hpp"
#include "kmpc/types.hpp"

#include <Eigen/Core>

namespace kmpc {

struct QDesFlags {
  bool pitch_clip = false;
  bool elbow_clip = false;
  bool com_reject = false;
  bool projected = false;
};

class WholeBodyGuards {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit WholeBodyGuards(const KmpcParams& params, double yaw_pref = 0.0);

  void set_yaw_pref(double yaw_pref) { q_yaw_pref_ = yaw_pref; }

  static double pitch(const Vec18& q) { return q[0] + q[1] + q[2]; }
  double torso_pitch_max() const { return torso_pitch_max_; }
  double yaw_max() const { return yaw_max_; }
  double com_work_xmin() const;
  double com_work_xmax() const;
  double com_work_ymin() const;
  double com_work_ymax() const;

  void pitch_sum_bounds(const Vec18& q0, double& lo, double& hi) const;
  // 机型 ±15° ∩ 朝 θ* 的可行段。盒外仍 min(.,0)/max(.,0)。
  void pitch_toward_bounds(
      const Vec18& q0, double theta_star, double& lo, double& hi) const;
  void com_work_xy(Eigen::Vector2d& lo, Eigen::Vector2d& hi) const;
  void com_step_bounds(
      const Vec3& p_com, Eigen::Vector2d& dlo, Eigen::Vector2d& dhi) const;
  bool com_at_work_wall(const Vec3& p_com) const;

  // 18 维 Ax >= b。n 为已有行数，函数往上追加。
  void append_pitch(const Vec18& q0, MatIneq18& A, VecIneq& b, int& n) const;
  void append_pitch_toward(
      const Vec18& q0, double theta_star, MatIneq18& A, VecIneq& b,
      int& n) const;
  void append_yaw(const Vec18& q0, MatIneq18& A, VecIneq& b, int& n) const;
  void append_com(
      const Vec3& p_com, const Mat3x18& j_com, MatIneq18& A, VecIneq& b,
      int& n) const;
  void append_elbow(const Vec18& q0, MatIneq18& A, VecIneq& b, int& n) const;

  // 笛卡尔 36 维：俯仰/yaw/CoM 两端。
  void stamp_pitch_both_ends(
      const Vec18& q0, MatIneq& A, VecIneq& b, int& n) const;
  void stamp_yaw_both_ends(
      const Vec18& q0, MatIneq& A, VecIneq& b, int& n) const;
  void stamp_com_both_ends(
      const Vec3& p_com, const Mat3x18& j_com, MatIneq& A, VecIneq& b,
      int& n) const;

  bool com_xy_inside(const Vec3& p_com) const;
  Vec18 project_q_des(const Vec18& q_raw, QDesFlags& flags) const;

 private:
  static void add_row36(
      MatIneq& A, VecIneq& b, int& n, const Vec36& row, double rhs);
  static void add_row18(
      MatIneq18& A, VecIneq& b, int& n, const Vec18& a, double rhs);
  void stamp_both_ends(
      MatIneq& A, VecIneq& b, int& n, const Vec18& a, double lo,
      double hi) const;

  Vec18 j_torso_ = Vec18::Zero();
  double torso_pitch_max_ = 0.0;
  double yaw_max_ = 0.0;
  double q_yaw_pref_ = 0.0;
  double com_margin_ = 0.025;
  double com_xmin_ = -0.322;
  double com_xmax_ = 0.138;
  double com_ymin_ = -0.235;
  double com_ymax_ = 0.235;
  double com_work_m_ = 0.06;
};

}  // namespace kmpc
