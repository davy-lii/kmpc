#include "kmpc/trajectory_follower.hpp"

#include <cmath>
#include <string_view>
#include <vector>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "kmpc/constraint_policy.hpp"
#include "kmpc/tracker_math.hpp"

namespace kmpc {
namespace {

constexpr double kExcessEps = 1e-9;
constexpr double kNumericEps = 1e-12;
constexpr int kRecoveryTimeoutTicks = 500;  // 10 s at 50 Hz
constexpr int kBrakeTicks = 5;

double pitch_of(const Vec18& q) { return q[0] + q[1] + q[2]; }

std::string_view ee_frame(bool left) {
  return left ? kEeFrameLeft : kEeFrameRight;
}

}  // namespace

TrajectoryFollower::TrajectoryFollower(const Kinematics& kin,
                                       const WholeBodyGuards* guards,
                                       MotionConstraintPolicy policy)
    : kin_(kin), guards_(guards), policy_(std::move(policy)) {}

void TrajectoryFollower::reset() {
  has_package_ = false;
  package_ = TrajectoryPackage{};
  state_ = FollowerState::Hold;
  has_cmd_ = false;
  q_cmd_.setZero();
  q_prev_cmd_.setZero();
  recovery_excess0_ = 0.0;
  recovery_ticks_ = 0;
  brake_ticks_ = 0;
  last_ = FollowerTick{};
}

void TrajectoryFollower::hold_at(const Vec18& q) {
  reset();
  q_cmd_ = q;
  q_prev_cmd_ = q;
  has_cmd_ = q.allFinite();
  last_ = make_hold(q, "hold");
}

bool TrajectoryFollower::begin_pitch_recovery(const Vec18& q_meas) {
  if (!q_meas.allFinite()) {
    return false;
  }
  has_package_ = false;
  package_ = TrajectoryPackage{};
  q_cmd_ = q_meas;
  q_prev_cmd_ = q_meas;
  has_cmd_ = true;
  recovery_excess0_ = pitch_excess_of(q_meas);
  recovery_ticks_ = 0;
  brake_ticks_ = 0;
  if (recovery_excess0_ > kExcessEps) {
    state_ = FollowerState::Recovering;
  } else {
    state_ = FollowerState::Hold;
  }
  last_ = make_hold(q_meas, state_ == FollowerState::Recovering
                                ? "recovering"
                                : "hold_inside");
  last_.state = state_;
  last_.pitch_excess = recovery_excess0_;
  return true;
}

bool TrajectoryFollower::adopt(const TrajectoryPackage& package,
                               const Vec18& q_meas, const Vec18& dq_meas,
                               bool dq_known, double q_tol_rad, double dq_tol,
                               std::string* why) {
  const AdmitDecision admit =
      admit_package_start(package, q_meas, dq_meas, dq_known, q_tol_rad, dq_tol);
  if (!admit.ok) {
    if (why != nullptr) {
      *why = admit.reason;
    }
    state_ = FollowerState::Hold;
    has_package_ = false;
    last_ = make_hold(has_cmd_ ? q_cmd_ : q_meas, admit.reason);
    return false;
  }
  if (!package.certificate.passed || package.samples.size() < 2) {
    if (why != nullptr) {
      *why = "uncertified_package";
    }
    state_ = FollowerState::Hold;
    has_package_ = false;
    last_ = make_hold(has_cmd_ ? q_cmd_ : q_meas, "uncertified_package");
    return false;
  }

  const double ex = pitch_excess_of(q_meas);
  package_ = package;
  has_package_ = true;
  q_cmd_ = q_meas;
  q_prev_cmd_ = q_meas;
  has_cmd_ = true;
  recovery_ticks_ = 0;
  brake_ticks_ = 0;
  recovery_excess0_ = ex;
  if (ex > kExcessEps) {
    // Measured start outside hard pitch: recover first; do not treat the
    // ordinary certified package as executable until inside.
    state_ = FollowerState::Recovering;
  } else {
    state_ = FollowerState::Active;
  }
  if (why != nullptr) {
    *why = "ok";
  }
  last_ = make_hold(q_meas, state_ == FollowerState::Recovering
                                ? "adopt_recovering"
                                : "adopt_active");
  last_.state = state_;
  last_.generation = package.generation;
  last_.path_id = package.path_id;
  last_.pitch_excess = ex;
  return true;
}

double TrajectoryFollower::pitch_excess_of(const Vec18& q) const {
  const double hard = policy_.torso_pitch_max_rad;
  if (!(hard > 0.0) || !q.allFinite()) {
    return 0.0;
  }
  return std::max(0.0, std::abs(pitch_of(q)) - hard);
}

Vec18 TrajectoryFollower::step_cap(double dt) const {
  Vec18 cap = Vec18::Ones() * 1e9;
  const double d = std::max(dt, 1e-4);
  for (int j = 0; j < kNDof; ++j) {
    const double dq = std::max(policy_.dq_max[j], 1e-6);
    cap[j] = dq * d;
  }
  return cap;
}

FollowerTick TrajectoryFollower::make_hold(const Vec18& q,
                                           const char* reason) const {
  FollowerTick tick;
  tick.q_plan = q;
  tick.q_ref = q;
  tick.q_follow = q;
  tick.state = FollowerState::Hold;
  tick.action = FollowerAction::Continue;
  tick.reason = reason;
  tick.pitch_excess = pitch_excess_of(q);
  if (has_package_) {
    tick.generation = package_.generation;
    tick.path_id = package_.path_id;
  }
  return tick;
}

Vec18 TrajectoryFollower::interpolate_plan(double t_s, double* s_out) const {
  const JointCurve& samples = package_.samples;
  const TimeLawTable& law = package_.time_law;
  if (samples.empty()) {
    if (s_out != nullptr) {
      *s_out = 0.0;
    }
    return has_cmd_ ? q_cmd_ : Vec18::Zero();
  }
  if (samples.size() == 1 || law.t_s.size() < 2 ||
      law.s.size() != law.t_s.size()) {
    if (s_out != nullptr) {
      *s_out = 0.0;
    }
    return samples.front();
  }
  const double t_end = law.t_s.back();
  const double t = std::clamp(t_s, 0.0, t_end);
  std::size_t i = 0;
  while (i + 1 < law.t_s.size() && law.t_s[i + 1] < t - kNumericEps) {
    ++i;
  }
  const double t0 = law.t_s[i];
  const double t1 = law.t_s[std::min(i + 1, law.t_s.size() - 1)];
  const double alpha =
      (t1 - t0) > kNumericEps ? (t - t0) / (t1 - t0) : 0.0;
  const double s0 = law.s[i];
  const double s1 = law.s[std::min(i + 1, law.s.size() - 1)];
  const double s = s0 + alpha * (s1 - s0);
  if (s_out != nullptr) {
    *s_out = s;
  }
  const double u = s * static_cast<double>(samples.size() - 1);
  const std::size_t j =
      std::min(static_cast<std::size_t>(std::floor(u)), samples.size() - 1);
  const std::size_t j1 = std::min(j + 1, samples.size() - 1);
  const double frac = u - static_cast<double>(j);
  return samples[j] + frac * (samples[j1] - samples[j]);
}

bool TrajectoryFollower::command_feasible(
    const Vec18& q_cmd, const Vec18& q_prev, double dt,
    double pitch_excess_ceiling, const char** fail_tag) const {
  if (!q_cmd.allFinite()) {
    if (fail_tag != nullptr) {
      *fail_tag = "nonfinite";
    }
    return false;
  }
  if (joint_violates_hard_box(q_cmd, policy_.q_min, policy_.q_max)) {
    if (fail_tag != nullptr) {
      *fail_tag = "hard_box";
    }
    return false;
  }
  const Vec18 cap = step_cap(dt);
  for (int j = 0; j < kNDof; ++j) {
    if (std::abs(q_cmd[j] - q_prev[j]) > cap[j] + 1e-9) {
      if (fail_tag != nullptr) {
        *fail_tag = "step_limit";
      }
      return false;
    }
  }
  const double hard = policy_.torso_pitch_max_rad;
  if (hard > 0.0) {
    const double ex = std::max(0.0, std::abs(pitch_of(q_cmd)) - hard);
    if (ex > pitch_excess_ceiling + kPitchRoundEpsRad) {
      if (fail_tag != nullptr) {
        *fail_tag = "pitch_excess_growth";
      }
      return false;
    }
  }
  if (guards_ != nullptr && !guards_->com_xy_inside(kin_.com(q_cmd))) {
    if (fail_tag != nullptr) {
      *fail_tag = "com";
    }
    return false;
  }
  return true;
}

bool TrajectoryFollower::try_qp_correction(
    const Vec18& q_nom, const Vec18& q_prev, const Vec18& q_meas, double dt,
    Vec18* q_out, const char** fail_tag) const {
  // Small δq about the certified nominal. No posture reselection: trust region
  // is a few milliradians and the objective prefers staying on q_nom.
  constexpr int n = kNDof;
  const Vec18 cap = step_cap(dt);
  // Stay on the certified nominal: min ||δq||² (+ tiny prev smoothness).
  (void)q_meas;
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) * 200.0;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
  for (int j = 0; j < n; ++j) {
    H(j, j) += 0.2;
  }
  Eigen::VectorXd lb(n);
  Eigen::VectorXd ub(n);
  for (int j = 0; j < n; ++j) {
    const double box_lo = policy_.q_min[j] - q_nom[j];
    const double box_hi = policy_.q_max[j] - q_nom[j];
    const double step_lo = (q_prev[j] - cap[j]) - q_nom[j];
    const double step_hi = (q_prev[j] + cap[j]) - q_nom[j];
    lb[j] = std::max(box_lo, step_lo);
    ub[j] = std::min(box_hi, step_hi);
    // Tiny trust so QP cannot mint a new waist/arm division.
    lb[j] = std::max(lb[j], -0.015);
    ub[j] = std::min(ub[j], 0.015);
    if (lb[j] > ub[j] + 1e-12) {
      if (fail_tag != nullptr) {
        *fail_tag = "qp_box_empty";
      }
      return false;
    }
  }

  // Pitch / CoM linear inequalities on δq.
  std::vector<Eigen::Matrix<double, 1, kNDof>> Arows;
  std::vector<double> blo;
  std::vector<double> bup;
  const double hard = policy_.torso_pitch_max_rad;
  if (hard > 0.0) {
    Eigen::Matrix<double, 1, kNDof> pitch = Eigen::Matrix<double, 1, kNDof>::Zero();
    pitch.head<3>().setOnes();
    const double p0 = pitch_of(q_nom);
    Arows.push_back(pitch);
    blo.push_back(-hard - p0);
    bup.push_back(hard - p0);
  }
  if (guards_ != nullptr) {
    const Vec3 com = kin_.com(q_nom);
    const Mat3x18 Jc = kin_.jacobian_com(q_nom);
    // Stay-feasible half-spaces (same as historical QP CoM rows).
    Eigen::Vector2d dlo;
    Eigen::Vector2d dhi;
    guards_->com_step_bounds(com, dlo, dhi);
    for (int a = 0; a < 2; ++a) {
      Eigen::Matrix<double, 1, kNDof> row = Jc.row(a);
      Arows.push_back(row);
      blo.push_back(dlo[a]);
      bup.push_back(dhi[a]);
    }
  }
  const int nin = static_cast<int>(Arows.size());
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nin, n);
  Eigen::VectorXd cl = Eigen::VectorXd::Zero(nin);
  Eigen::VectorXd cu = Eigen::VectorXd::Zero(nin);
  for (int i = 0; i < nin; ++i) {
    C.row(i) = Arows[static_cast<std::size_t>(i)];
    cl[i] = blo[static_cast<std::size_t>(i)];
    cu[i] = bup[static_cast<std::size_t>(i)];
  }

  proxsuite::proxqp::dense::QP<double> qp(n, 0, nin, true);
  qp.settings.eps_abs = 1e-6;
  qp.settings.max_iter = 100;
  qp.settings.verbose = false;
  qp.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  try {
    using proxsuite::nullopt;
    qp.init(H, g, nullopt, nullopt, C, cl, cu, lb, ub, true);
    qp.solve();
  } catch (...) {
    if (fail_tag != nullptr) {
      *fail_tag = "qp_exception";
    }
    return false;
  }
  const auto st = qp.results.info.status;
  if (st != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED &&
      st != proxsuite::proxqp::QPSolverOutput::
                 PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE) {
    if (fail_tag != nullptr) {
      *fail_tag = "qp_infeasible";
    }
    return false;
  }
  Vec18 dq = Vec18::Zero();
  for (int j = 0; j < n; ++j) {
    dq[j] = qp.results.x[j];
  }
  const Vec18 q_cand = q_nom + dq;
  if (!command_feasible(q_cand, q_prev, dt, 0.0, fail_tag)) {
    return false;
  }
  // TCP tube vs certified nominal.
  if (has_package_) {
    const auto frame = ee_frame(package_.left);
    const Vec3 p_nom = kin_.fk_pos(q_nom, frame);
    const Vec3 p_cand = kin_.fk_pos(q_cand, frame);
    if ((p_cand - p_nom).norm() > tcp_tube_m_ + 1e-9) {
      if (fail_tag != nullptr) {
        *fail_tag = "tcp_tube";
      }
      return false;
    }
  }
  *q_out = q_cand;
  return true;
}

FollowerTick TrajectoryFollower::recover_tick(const Vec18& q_meas, double dt) {
  FollowerTick tick;
  tick.state = FollowerState::Recovering;
  tick.generation = has_package_ ? package_.generation : 0;
  tick.path_id = has_package_ ? package_.path_id : 0;
  ++recovery_ticks_;

  const double hard = policy_.torso_pitch_max_rad;
  const double ex_meas = pitch_excess_of(q_meas);
  tick.pitch_excess = ex_meas;
  tick.q_plan = q_meas;
  if (!(hard > 0.0)) {
    state_ = has_package_ ? FollowerState::Active : FollowerState::Hold;
    tick.state = state_;
    tick.q_ref = q_meas;
    tick.q_follow = q_meas;
    tick.action = FollowerAction::Continue;
    tick.reason = "recover_no_pitch_limit";
    q_prev_cmd_ = q_cmd_;
    q_cmd_ = q_meas;
    return tick;
  }
  if (ex_meas <= kExcessEps) {
    // Back inside hard box: leave recovery. Ordinary package only resumes if
    // present and still admissible; otherwise HOLD and ask for replan.
    if (has_package_) {
      state_ = FollowerState::Active;
      tick.state = FollowerState::Active;
      tick.action = FollowerAction::Continue;
      tick.reason = "recovered_active";
    } else {
      state_ = FollowerState::Hold;
      tick.state = FollowerState::Hold;
      tick.action = FollowerAction::NeedReplan;
      tick.reason = "recovered_need_replan";
    }
    tick.q_ref = q_meas;
    tick.q_follow = q_meas;
    q_prev_cmd_ = q_cmd_;
    q_cmd_ = q_meas;
    return tick;
  }

  if (recovery_ticks_ > kRecoveryTimeoutTicks) {
    state_ = FollowerState::Hold;
    tick.state = FollowerState::Hold;
    tick.action = FollowerAction::ProtectStop;
    tick.reason = "pitch_recovery_timeout";
    tick.q_ref = q_cmd_;
    tick.q_follow = q_cmd_;
    return tick;
  }

  // §7.4: excess(t+dt) <= excess(t) + eps. Reduce with available waist step
  // room without widening joint caps or J1 58°.
  const Vec18 cap = step_cap(dt);
  const double pitch = pitch_of(q_meas);
  const double sign = pitch > 0.0 ? 1.0 : -1.0;
  Vec18 q_cmd = q_meas;
  double remaining = ex_meas;
  // Prefer J2 then J3 then J1 for recovery pull (class-human sit), but never
  // violate hard box or per-tick cap. Do not assume one axis can take all.
  const int order[3] = {1, 2, 0};
  for (int k = 0; k < 3 && remaining > kExcessEps; ++k) {
    const int a = order[k];
    const double room_box = sign > 0.0 ? (q_cmd[a] - policy_.q_min[a])
                                       : (policy_.q_max[a] - q_cmd[a]);
    const double room_step = cap[a];
    const double room = std::max(0.0, std::min(room_box, room_step));
    const double take = std::min(remaining, room);
    q_cmd[a] -= sign * take;
    remaining -= take;
  }
  // Clamp only to the true hard box (no soft margin widen).
  for (int j = 0; j < kNDof; ++j) {
    q_cmd[j] = std::clamp(q_cmd[j], policy_.q_min[j], policy_.q_max[j]);
  }

  const double ex_cmd = pitch_excess_of(q_cmd);
  const double ceiling = ex_meas + kPitchRoundEpsRad;
  const char* fail = nullptr;
  if (ex_cmd > ceiling + kExcessEps ||
      !command_feasible(q_cmd, q_cmd_, dt, ceiling, &fail)) {
    // Cannot safely reduce or even hold excess → protect stop.
    state_ = FollowerState::Hold;
    tick.state = FollowerState::Hold;
    tick.action = FollowerAction::ProtectStop;
    tick.reason = fail != nullptr ? fail : "pitch_recovery_infeasible";
    tick.q_ref = q_cmd_;
    tick.q_follow = q_cmd_;
    return tick;
  }
  if (!(ex_cmd < ex_meas - 1e-12) && remaining > kExcessEps &&
      recovery_ticks_ > 1) {
    // No progress possible under caps: still OK if excess did not grow; keep
    // trying until timeout. If literally zero waist room → protect.
    double waist_room = 0.0;
    for (int a = 0; a < 3; ++a) {
      const double room_box = sign > 0.0 ? (q_meas[a] - policy_.q_min[a])
                                         : (policy_.q_max[a] - q_meas[a]);
      waist_room += std::max(0.0, std::min(room_box, cap[a]));
    }
    if (waist_room < 1e-12) {
      state_ = FollowerState::Hold;
      tick.state = FollowerState::Hold;
      tick.action = FollowerAction::ProtectStop;
      tick.reason = "pitch_recovery_no_room";
      tick.q_ref = q_cmd_;
      tick.q_follow = q_cmd_;
      return tick;
    }
  }

  tick.q_ref = q_cmd;
  tick.q_follow = q_cmd;
  tick.action = FollowerAction::Continue;
  tick.reason = "recovering";
  tick.pitch_excess = ex_cmd;
  q_prev_cmd_ = q_cmd_;
  q_cmd_ = q_cmd;
  return tick;
}

FollowerTick TrajectoryFollower::brake_tick(const Vec18& q_meas, double dt) {
  FollowerTick tick;
  tick.state = FollowerState::Braking;
  tick.generation = has_package_ ? package_.generation : 0;
  tick.path_id = has_package_ ? package_.path_id : 0;
  tick.pitch_excess = pitch_excess_of(q_meas);
  ++brake_ticks_;
  const Vec18 cap = step_cap(dt);
  Vec18 q_cmd = q_cmd_;
  // Decelerate toward measured/hold: shrink step each brake tick.
  const double scale =
      std::max(0.0, 1.0 - static_cast<double>(brake_ticks_) /
                              static_cast<double>(kBrakeTicks));
  for (int j = 0; j < kNDof; ++j) {
    const double target = q_meas[j];
    const double step =
        std::clamp(target - q_cmd[j], -cap[j] * scale, cap[j] * scale);
    q_cmd[j] += step;
    q_cmd[j] = std::clamp(q_cmd[j], policy_.q_min[j], policy_.q_max[j]);
  }
  tick.q_plan = q_cmd;
  tick.q_ref = q_cmd;
  tick.q_follow = q_cmd;
  q_prev_cmd_ = q_cmd_;
  q_cmd_ = q_cmd;
  if (brake_ticks_ >= kBrakeTicks) {
    state_ = FollowerState::Hold;
    has_package_ = false;
    tick.state = FollowerState::Hold;
    tick.action = FollowerAction::NeedReplan;
    tick.reason = "braked_need_replan";
  } else {
    tick.action = FollowerAction::Brake;
    tick.reason = "braking";
  }
  return tick;
}

FollowerTick TrajectoryFollower::step(const Vec18& q_meas, double t_s,
                                      double dt) {
  if (!q_meas.allFinite()) {
    state_ = FollowerState::Hold;
    last_ = make_hold(has_cmd_ ? q_cmd_ : Vec18::Zero(), "meas_nonfinite");
    last_.action = FollowerAction::ProtectStop;
    return last_;
  }
  if (!has_cmd_) {
    q_cmd_ = q_meas;
    q_prev_cmd_ = q_meas;
    has_cmd_ = true;
  }

  // Auto-enter recovery if pitch leaves the hard box while active.
  if (state_ == FollowerState::Active &&
      pitch_excess_of(q_meas) > kExcessEps) {
    state_ = FollowerState::Recovering;
    recovery_excess0_ = pitch_excess_of(q_meas);
    recovery_ticks_ = 0;
  }

  if (state_ == FollowerState::Recovering) {
    last_ = recover_tick(q_meas, dt);
    return last_;
  }
  if (state_ == FollowerState::Braking) {
    last_ = brake_tick(q_meas, dt);
    return last_;
  }
  if (state_ == FollowerState::Hold || !has_package_) {
    last_ = make_hold(q_cmd_, "hold");
    return last_;
  }

  // ACTIVE: follow certified plan.
  FollowerTick tick;
  tick.state = FollowerState::Active;
  tick.generation = package_.generation;
  tick.path_id = package_.path_id;
  double s = 0.0;
  const Vec18 q_plan = interpolate_plan(t_s, &s);
  tick.q_plan = q_plan;
  tick.s = s;
  tick.t_s = t_s;
  tick.pitch_excess = pitch_excess_of(q_meas);

  const double dur = package_.time_law.duration_s;
  if (dur > 0.0 && t_s >= dur - 1e-9 &&
      (q_meas - package_.samples.back()).cwiseAbs().maxCoeff() < 0.02) {
    tick.q_ref = package_.samples.back();
    tick.q_follow = tick.q_ref;
    tick.action = FollowerAction::Arrived;
    tick.reason = "arrived";
    state_ = FollowerState::Hold;
    q_prev_cmd_ = q_cmd_;
    q_cmd_ = tick.q_ref;
    last_ = tick;
    return last_;
  }

  Vec18 q_ref = q_plan;
  const char* fail = nullptr;
  bool qp_used = false;
  if (enable_qp_) {
    const Vec3 p_plan =
        kin_.fk_pos(q_plan, ee_frame(package_.left));
    const Vec3 p_meas =
        kin_.fk_pos(q_meas, ee_frame(package_.left));
    const double err = (p_meas - p_plan).norm();
    if (err > 1e-4 && err <= track_err_m_) {
      Vec18 q_corr;
      if (try_qp_correction(q_plan, q_cmd_, q_meas, dt, &q_corr, &fail)) {
        q_ref = q_corr;
        qp_used = true;
      } else {
        // Infeasible small correction → brake / need replan. Do not reshape.
        state_ = FollowerState::Braking;
        brake_ticks_ = 0;
        tick.state = FollowerState::Braking;
        tick.action = FollowerAction::NeedReplan;
        tick.reason = fail != nullptr ? fail : "qp_need_replan";
        tick.q_ref = q_cmd_;
        tick.q_follow = q_cmd_;
        tick.qp_used = false;
        last_ = tick;
        return last_;
      }
    }
  }

  if (!command_feasible(q_ref, q_cmd_, dt, 0.0, &fail)) {
    state_ = FollowerState::Braking;
    brake_ticks_ = 0;
    tick.state = FollowerState::Braking;
    tick.action = FollowerAction::Brake;
    tick.reason = fail != nullptr ? fail : "follow_infeasible";
    tick.q_ref = q_cmd_;
    tick.q_follow = q_cmd_;
    last_ = tick;
    return last_;
  }

  tick.q_ref = q_ref;
  tick.q_follow = q_ref;
  tick.action = FollowerAction::Continue;
  tick.reason = qp_used ? "follow_qp" : "follow_nominal";
  tick.qp_used = qp_used;
  q_prev_cmd_ = q_cmd_;
  q_cmd_ = q_ref;
  last_ = tick;
  return last_;
}

}  // namespace kmpc
