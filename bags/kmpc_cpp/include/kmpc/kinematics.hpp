#pragma once
// 对应 Python kmpc/urdf_kinematics.py；对照 09-04 §0 / §2.3。
// Pinocchio 做 FK / 几何 J；CoM 与 J_com 用与 Python 同式的子树加权（不要用
// pinocchio::centerOfMass，那个 API 的 data.com[0] 与现码 p_com 不是同一个量）。

#include <memory>
#include <string>
#include <string_view>

#include "kmpc/types.hpp"

namespace kmpc {

struct InertialQuantities {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 p_com = Vec3::Zero();
  Mat3x18 J_com = Mat3x18::Zero();
  Vec18 mdiag = Vec18::Zero();
};

class Kinematics {
 public:
  enum class UrdfInput { Path, Xml };

  explicit Kinematics(
      const std::string& src, UrdfInput input = UrdfInput::Path);
  Kinematics(const Kinematics&) = delete;
  Kinematics& operator=(const Kinematics&) = delete;
  Kinematics(Kinematics&&) noexcept;
  Kinematics& operator=(Kinematics&&) noexcept;
  ~Kinematics();

  double total_mass() const;

  Eigen::Matrix4d fk(const Vec18& q18, std::string_view link) const;
  Vec3 fk_pos(const Vec18& q18, std::string_view link) const;
  Vec3 fk_point(const Vec18& q18, std::string_view link, const Vec3& p_local) const;

  // 6×18，[Jv; Jω]，base/world（world_to_base 是单位阵）。p_local 默认连杆原点。
  Mat6x18 jacobian_spatial(const Vec18& q18, std::string_view link) const;
  Mat6x18 jacobian_spatial(
      const Vec18& q18, std::string_view link, const Vec3& p_local) const;
  Mat3x18 jacobian_pos(const Vec18& q18, std::string_view link) const;

  InertialQuantities inertial_quantities(const Vec18& q18) const;
  Vec3 com(const Vec18& q18) const;
  Mat3x18 jacobian_com(const Vec18& q18) const;
  Vec18 mass_matrix_diag(const Vec18& q18) const;

  // λ 使腕对角均值 * λ ≈ r_arm。只在构造 HOME 上标定，Tracker 里不要热改。
  double inertia_lambda(const Vec18& q18, double r_arm) const;

 private:
  void finish_bind();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kmpc
