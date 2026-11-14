// 与 Python UrdfKinematics 同 URDF、同 18 维映射。对照 09-04 §0。

#include "kmpc/kinematics.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "kmpc/config.hpp"

namespace kmpc {
namespace {

std::string to_string(std::string_view v) { return std::string(v); }

}  // namespace

struct Kinematics::Impl {
  pinocchio::Model model;
  mutable pinocchio::Data data;
  mutable Eigen::VectorXd q_pin;
  mutable Eigen::Matrix<double, 6, Eigen::Dynamic> J6;
  mutable Eigen::Matrix<double, 6, Eigen::Dynamic> Jjoint;
  std::array<pinocchio::JointIndex, kNDof> jid{};
  std::array<int, kNDof> idx_q{};
  std::array<int, kNDof> idx_v{};
  double total_mass = 0.0;
  // level: 0 empty, 1 FK placements, 2 joint Jacobians. Exact q match only.
  mutable int level = 0;
  mutable Vec18 cached_q = Vec18::Zero();
  mutable bool com_ready = false;
  mutable bool jcom_ready = false;
  mutable bool mdiag_ready = false;
  mutable Vec3 cached_com = Vec3::Zero();
  mutable Mat3x18 cached_jcom = Mat3x18::Zero();
  mutable Vec18 cached_mdiag = Vec18::Zero();
  mutable uint64_t n_fk = 0;
  mutable uint64_t n_jac = 0;
  mutable uint64_t n_crba = 0;

  void apply_q(const Vec18& q18) const {
    for (int i = 0; i < kNDof; ++i) {
      q_pin[idx_q[static_cast<std::size_t>(i)]] = q18[i];
    }
  }

  pinocchio::FrameIndex frame_id(std::string_view link) const {
    const std::string name = to_string(link);
    if (!model.existFrame(name)) {
      throw ConfigError("URDF has no frame " + name);
    }
    return model.getFrameId(name);
  }

  bool same_q(const Vec18& q18) const {
    return level > 0 && (cached_q - q18).squaredNorm() == 0.0;
  }

  void forget_derived() const {
    com_ready = false;
    jcom_ready = false;
    mdiag_ready = false;
  }

  void ensure_fk(const Vec18& q18) const {
    if (same_q(q18) && level >= 1) {
      return;
    }
    apply_q(q18);
    pinocchio::forwardKinematics(model, data, q_pin);
    pinocchio::updateFramePlacements(model, data);
    cached_q = q18;
    level = 1;
    forget_derived();
    ++n_fk;
  }

  void ensure_jac(const Vec18& q18) const {
    if (same_q(q18) && level >= 2) {
      return;
    }
    apply_q(q18);
    pinocchio::computeJointJacobians(model, data, q_pin);
    pinocchio::updateFramePlacements(model, data);
    cached_q = q18;
    level = 2;
    forget_derived();
    ++n_jac;
  }

  void update_fk(const Vec18& q18) const { ensure_fk(q18); }

  void update_fk_jac(const Vec18& q18) const { ensure_jac(q18); }

  Vec3 com_cached() const {
    if (com_ready) {
      return cached_com;
    }
    Vec3 s = Vec3::Zero();
    for (pinocchio::JointIndex j = 0; j < model.inertias.size(); ++j) {
      const double m = model.inertias[j].mass();
      if (m <= 0.0) {
        continue;
      }
      s += m * body_com_world(j);
    }
    cached_com.setZero();
    if (total_mass > 1e-12) {
      cached_com = s / total_mass;
    }
    com_ready = true;
    return cached_com;
  }

  Mat3x18 jcom_cached() const {
    if (jcom_ready) {
      return cached_jcom;
    }
    cached_jcom.setZero();
    const double M = total_mass;
    for (int i = 0; i < kNDof; ++i) {
      const pinocchio::JointIndex joint = jid[static_cast<std::size_t>(i)];
      const int iv = idx_v[static_cast<std::size_t>(i)];
      double m_c = 0.0;
      Vec3 p_c = Vec3::Zero();
      subtree_mass_com(joint, &m_c, &p_c);
      Jjoint.setZero();
      pinocchio::getJointJacobian(
          model, data, joint, pinocchio::LOCAL_WORLD_ALIGNED, Jjoint);
      const Vec3 z = Jjoint.block<3, 1>(3, iv);
      const Vec3 p_j = data.oMi[joint].translation();
      if (m_c > 1e-12 && M > 1e-12) {
        cached_jcom.col(i) = (m_c / M) * z.cross(p_c - p_j);
      }
    }
    jcom_ready = true;
    return cached_jcom;
  }

  void ensure_mdiag(const Vec18& q18) const {
    ensure_fk(q18);
    if (mdiag_ready) {
      return;
    }
    // LOCAL CRBA writes M. It does not replace oMi. data.J is left as the last
    // computeJointJacobians wrote it; jacobian_pos() after this must still match.
    pinocchio::crba(model, data, q_pin);
    for (int i = 0; i < kNDof; ++i) {
      const int iv = idx_v[static_cast<std::size_t>(i)];
      cached_mdiag[i] = std::max(data.M(iv, iv), 0.0);
    }
    mdiag_ready = true;
    ++n_crba;
  }

  Mat6x18 extract_spatial(pinocchio::FrameIndex fid, const Vec3* p_local) const {
    J6.setZero();
    pinocchio::getFrameJacobian(
        model, data, fid, pinocchio::LOCAL_WORLD_ALIGNED, J6);
    Mat6x18 out = Mat6x18::Zero();
    for (int i = 0; i < kNDof; ++i) {
      out.col(i) = J6.col(idx_v[static_cast<std::size_t>(i)]);
    }
    if (p_local != nullptr && p_local->squaredNorm() > 0.0) {
      const Vec3 p_world = data.oMf[fid].rotation() * (*p_local);
      for (int i = 0; i < kNDof; ++i) {
        const Vec3 w = out.block<3, 1>(3, i);
        out.block<3, 1>(0, i) += w.cross(p_world);
      }
    }
    return out;
  }

  Vec3 body_com_world(pinocchio::JointIndex j) const {
    return data.oMi[j].act(model.inertias[j].lever());
  }

  void subtree_mass_com(
      pinocchio::JointIndex jid, double* mass, Vec3* com) const {
    double m = 0.0;
    Vec3 s = Vec3::Zero();
    for (pinocchio::JointIndex k : model.subtrees[jid]) {
      const double mk = model.inertias[k].mass();
      if (mk <= 0.0) {
        continue;
      }
      m += mk;
      s += mk * body_com_world(k);
    }
    *mass = m;
    if (m > 1e-12) {
      *com = s / m;
    } else {
      *com = Vec3::Zero();
    }
  }
};

Kinematics::Kinematics(const std::string& src, UrdfInput input)
    : impl_(std::make_unique<Impl>()) {
  if (input == UrdfInput::Xml) {
    pinocchio::urdf::buildModelFromXML(src, impl_->model);
  } else {
    pinocchio::urdf::buildModel(src, impl_->model);
  }
  finish_bind();
}

void Kinematics::finish_bind() {
  impl_->data = pinocchio::Data(impl_->model);
  impl_->q_pin = pinocchio::neutral(impl_->model);
  impl_->J6.setZero(6, impl_->model.nv);
  impl_->Jjoint.setZero(6, impl_->model.nv);
  for (int i = 0; i < kNDof; ++i) {
    const std::string name(kJointNames[static_cast<std::size_t>(i)]);
    if (!impl_->model.existJointName(name)) {
      throw ConfigError("URDF missing MPC joint " + name);
    }
    const pinocchio::JointIndex jid = impl_->model.getJointId(name);
    impl_->jid[static_cast<std::size_t>(i)] = jid;
    impl_->idx_q[static_cast<std::size_t>(i)] = impl_->model.idx_qs[jid];
    impl_->idx_v[static_cast<std::size_t>(i)] = impl_->model.idx_vs[jid];
  }
  for (const pinocchio::Inertia& in : impl_->model.inertias) {
    impl_->total_mass += in.mass();
  }
  if (!impl_->model.existFrame(std::string(kPalmFrameLeft)) ||
      !impl_->model.existFrame(std::string(kPalmFrameRight))) {
    throw ConfigError("URDF missing l_hand_link / r_hand_link");
  }
}

Kinematics::Kinematics(Kinematics&&) noexcept = default;
Kinematics& Kinematics::operator=(Kinematics&&) noexcept = default;
Kinematics::~Kinematics() = default;

double Kinematics::total_mass() const { return impl_->total_mass; }

Eigen::Matrix4d Kinematics::fk(const Vec18& q18, std::string_view link) const {
  impl_->update_fk(q18);
  const pinocchio::SE3& oMf = impl_->data.oMf[impl_->frame_id(link)];
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner<3, 3>() = oMf.rotation();
  T.col(3).head<3>() = oMf.translation();
  return T;
}

Vec3 Kinematics::fk_pos(const Vec18& q18, std::string_view link) const {
  return fk(q18, link).col(3).head<3>();
}

Vec3 Kinematics::fk_point(
    const Vec18& q18, std::string_view link, const Vec3& p_local) const {
  impl_->update_fk(q18);
  const pinocchio::SE3& oMf = impl_->data.oMf[impl_->frame_id(link)];
  return oMf.translation() + oMf.rotation() * p_local;
}

Mat6x18 Kinematics::jacobian_spatial(
    const Vec18& q18, std::string_view link) const {
  impl_->update_fk_jac(q18);
  return impl_->extract_spatial(impl_->frame_id(link), nullptr);
}

Mat6x18 Kinematics::jacobian_spatial(
    const Vec18& q18, std::string_view link, const Vec3& p_local) const {
  impl_->update_fk_jac(q18);
  return impl_->extract_spatial(impl_->frame_id(link), &p_local);
}

Mat3x18 Kinematics::jacobian_pos(const Vec18& q18, std::string_view link) const {
  return jacobian_spatial(q18, link).topRows<3>();
}

InertialQuantities Kinematics::inertial_quantities(const Vec18& q18) const {
  Impl& im = *impl_;
  im.ensure_jac(q18);
  InertialQuantities out;
  out.p_com = im.com_cached();
  out.J_com = im.jcom_cached();
  im.ensure_mdiag(q18);
  out.mdiag = im.cached_mdiag;
  return out;
}

Vec3 Kinematics::com(const Vec18& q18) const {
  impl_->ensure_fk(q18);
  return impl_->com_cached();
}

Mat3x18 Kinematics::jacobian_com(const Vec18& q18) const {
  impl_->ensure_jac(q18);
  return impl_->jcom_cached();
}

Vec18 Kinematics::mass_matrix_diag(const Vec18& q18) const {
  impl_->ensure_mdiag(q18);
  return impl_->cached_mdiag;
}

KinEvalStats Kinematics::kin_eval_stats() const {
  KinEvalStats stats;
  stats.fk_updates = impl_->n_fk;
  stats.jac_updates = impl_->n_jac;
  stats.crba_updates = impl_->n_crba;
  return stats;
}

void Kinematics::reset_kin_eval_stats() const {
  impl_->n_fk = 0;
  impl_->n_jac = 0;
  impl_->n_crba = 0;
}

double Kinematics::inertia_lambda(const Vec18& q18, double r_arm) const {
  const Vec18 mdiag = mass_matrix_diag(q18);
  double mean = 0.0;
  for (int idx : kWristIndices) {
    mean += mdiag[idx];
  }
  mean /= static_cast<double>(kWristIndices.size());
  if (mean <= 1e-12) {
    return 1.0;
  }
  return r_arm / mean;
}

}  // namespace kmpc
