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
constexpr int kRecoveryTimeoutTicks = 500;
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
  wait_stage_id_ = 0;
  wait_ack_elapsed_s_ = 0.0;
  stage_cursor_ = 0;
  brake_requested_ = false;
  exec_t_ = 0.0;
  timeline_frozen_ = false;
  freeze_exec_t_ = 0.0;
  freeze_q_.setZero();
  last_ = FollowerTick{};
}

void TrajectoryFollower::hold_at(const Vec18& q) {
  reset();
  q_cmd_ = q;
  q_prev_cmd_ = q;
  has_cmd_ = q.allFinite();
  last_ = make_hold(q, "hold");
}

void TrajectoryFollower::request_brake() {
  if (state_ == FollowerState::Hold) {
    return;
  }
  brake_requested_ = true;
  if (state_ != FollowerState::Braking) {
    state_ = FollowerState::Braking;
    brake_ticks_ = 0;
  }
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
  last_.pitch_excess_ceiling = recovery_excess0_;
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
  stage_cursor_ = 0;
  wait_stage_id_ = 0;
  wait_ack_elapsed_s_ = 0.0;
  brake_requested_ = false;
  exec_t_ = 0.0;
  timeline_frozen_ = false;
  freeze_exec_t_ = 0.0;
  freeze_q_ = q_meas;
  if (ex > kExcessEps) {
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
  last_.pitch_excess_ceiling = ex;
  return true;
}

bool TrajectoryFollower::on_stage_ack(uint64_t generation, uint32_t stage_id,
                                      bool success, const Vec18& q_meas,
                                      std::string* why) {
  if (state_ != FollowerState::WaitAck) {
    if (why) *why = "not_waiting_ack";
    return false;
  }
  if (!has_package_ || package_.generation != generation ||
      wait_stage_id_ != stage_id) {
    if (why) *why = "ack_mismatch";
    return false;
  }
  if (!success) {
    state_ = FollowerState::Hold;
    has_package_ = false;
    if (why) *why = "ack_failed";
    last_ = make_hold(q_meas.allFinite() ? q_meas : q_cmd_, "ack_failed");
    last_.action = FollowerAction::NeedReplan;
    return false;
  }
  // Must still be near the stage endpoint sample.
  const StageBoundary* b = nullptr;
  for (const auto& sb : package_.stage_boundaries) {
    if (sb.stage_id == stage_id) {
      b = &sb;
      break;
    }
  }
  if (b == nullptr || b->sample_index >= package_.samples.size()) {
    if (why) *why = "ack_stage_missing";
    return false;
  }
  const Vec18& q_stage = package_.samples[b->sample_index];
  if (!q_meas.allFinite() ||
      (q_meas - q_stage).cwiseAbs().maxCoeff() > 0.05) {
    state_ = FollowerState::Hold;
    has_package_ = false;
    if (why) *why = "ack_pose_drift";
    last_ = make_hold(q_meas.allFinite() ? q_meas : q_cmd_, "ack_pose_drift");
    last_.action = FollowerAction::NeedReplan;
    return false;
  }
  wait_ack_elapsed_s_ = 0.0;
  wait_stage_id_ = 0;
  ++stage_cursor_;
  // Resume from the frozen stage time — never jump to wall-clock future.
  exec_t_ = freeze_exec_t_;
  timeline_frozen_ = false;
  state_ = FollowerState::Active;
  if (why) *why = "ok";
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

const StageBoundary* TrajectoryFollower::stage_at_progress(double s) const {
  if (!has_package_ || package_.samples.size() < 2) {
    return nullptr;
  }
  const double u = s * static_cast<double>(package_.samples.size() - 1);
  for (std::size_t i = stage_cursor_; i < package_.stage_boundaries.size();
       ++i) {
    const StageBoundary& b = package_.stage_boundaries[i];
    if (b.sample_index == 0) {
      continue;
    }
    if (static_cast<double>(b.sample_index) <= u + 1e-9) {
      return &b;
    }
  }
  return nullptr;
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
  constexpr int n = kNDof;
  const Vec18 cap = step_cap(dt);
  // Track measured TCP/joint error toward certified nominal — not zero gradient.
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) * 40.0;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
  for (int j = 0; j < n; ++j) {
    // Prefer δq that moves q_nom+δq toward reducing (q_meas - q_nom) error
    // while staying near q_nom (trust).
    const double e = q_meas[j] - q_nom[j];
    g[j] = -2.0 * 8.0 * e;  // soft pull of corrected cmd toward meas-consistent
    H(j, j) += 80.0;        // stay near certified q_nom
    H(j, j) += 0.5;         // tiny prev smoothness
    (void)q_prev;
  }
  if (has_package_) {
    const auto frame = ee_frame(package_.left);
    const Vec3 p_nom = kin_.fk_pos(q_nom, frame);
    const Vec3 p_meas = kin_.fk_pos(q_meas, frame);
    const Mat3x18 J = kin_.jacobian_pos(q_nom, frame);
    const Vec3 e_tcp = p_meas - p_nom;
    // min ||J δq - (-e_tcp * scale)|| roughly → g += 2 J^T (J*0 - (-e))
    const double w_tcp = 200.0;
    g -= (2.0 * w_tcp) * (J.transpose() * (-e_tcp));
    H += (2.0 * w_tcp) * (J.transpose() * J);
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
    lb[j] = std::max(lb[j], -0.015);
    ub[j] = std::min(ub[j], 0.015);
    if (lb[j] > ub[j] + 1e-12) {
      if (fail_tag != nullptr) {
        *fail_tag = "qp_box_empty";
      }
      return false;
    }
  }

  std::vector<Eigen::Matrix<double, 1, kNDof>> Arows;
  std::vector<double> blo;
  std::vector<double> bup;
  const double hard = policy_.torso_pitch_max_rad;
  if (hard > 0.0) {
    Eigen::Matrix<double, 1, kNDof> pitch =
        Eigen::Matrix<double, 1, kNDof>::Zero();
    pitch.head<3>().setOnes();
    const double p0 = pitch_of(q_nom);
    Arows.push_back(pitch);
    blo.push_back(-hard - p0);
    bup.push_back(hard - p0);
  }
  if (guards_ != nullptr) {
    const Vec3 com = kin_.com(q_nom);
    const Mat3x18 Jc = kin_.jacobian_com(q_nom);
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
  tick.pitch_excess_ceiling = std::max(recovery_excess0_, ex_meas);
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

  const Vec18 cap = step_cap(dt);
  const double pitch = pitch_of(q_meas);
  const double sign = pitch > 0.0 ? 1.0 : -1.0;
  Vec18 q_cmd = q_meas;
  double remaining = ex_meas;
  const int order[3] = {1, 2, 0};
  for (int k = 0; k < 3 && remaining > kExcessEps; ++k) {
    const int a = order[k];
    const double room_box = sign > 0.0 ? (q_cmd[a] - policy_.q_min[a])
                                       : (policy_.q_max[a] - q_cmd[a]);
    const double room = std::max(0.0, std::min(room_box, cap[a]));
    const double take = std::min(remaining, room);
    q_cmd[a] -= sign * take;
    remaining -= take;
  }
  for (int j = 0; j < kNDof; ++j) {
    q_cmd[j] = std::clamp(q_cmd[j], policy_.q_min[j], policy_.q_max[j]);
  }

  const double ex_cmd = pitch_excess_of(q_cmd);
  const double ceiling = ex_meas + kPitchRoundEpsRad;
  const char* fail = nullptr;
  if (ex_cmd > ceiling + kExcessEps ||
      !command_feasible(q_cmd, q_cmd_, dt, ceiling, &fail)) {
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
    brake_requested_ = false;
    tick.state = FollowerState::Hold;
    tick.action = FollowerAction::NeedReplan;
    tick.reason = "braked_need_replan";
  } else {
    tick.action = FollowerAction::Brake;
    tick.reason = "braking";
  }
  return tick;
}

FollowerTick TrajectoryFollower::step(const Vec18& q_meas, double dt) {
  const double dtp = std::max(dt, 1e-4);
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

  if (brake_requested_ && state_ != FollowerState::Braking &&
      state_ != FollowerState::Recovering) {
    state_ = FollowerState::Braking;
    brake_ticks_ = 0;
  }

  if (state_ == FollowerState::Active &&
      pitch_excess_of(q_meas) > kExcessEps) {
    state_ = FollowerState::Recovering;
    recovery_excess0_ = pitch_excess_of(q_meas);
    recovery_ticks_ = 0;
  }

  if (state_ == FollowerState::Recovering) {
    last_ = recover_tick(q_meas, dtp);
    last_.t_s = exec_t_;
    last_.exec_t_s = exec_t_;
    last_.timeline_frozen = timeline_frozen_;
    last_.freeze_q = freeze_q_;
    return last_;
  }
  if (state_ == FollowerState::Braking) {
    last_ = brake_tick(q_meas, dtp);
    last_.t_s = exec_t_;
    last_.exec_t_s = exec_t_;
    last_.timeline_frozen = timeline_frozen_;
    last_.freeze_q = freeze_q_;
    return last_;
  }
  if (state_ == FollowerState::WaitAck) {
    // Execution clock stays frozen; only wall-elapsed ACK timeout advances.
    timeline_frozen_ = true;
    wait_ack_elapsed_s_ += dtp;
    FollowerTick tick = make_hold(freeze_q_.allFinite() ? freeze_q_ : q_cmd_,
                                  "wait_external_ack");
    tick.state = FollowerState::WaitAck;
    tick.action = FollowerAction::WaitExternalAck;
    tick.wait_stage_id = wait_stage_id_;
    tick.generation = package_.generation;
    tick.path_id = package_.path_id;
    tick.t_s = exec_t_;
    tick.exec_t_s = exec_t_;
    tick.timeline_frozen = true;
    tick.freeze_q = freeze_q_;
    tick.q_plan = freeze_q_;
    tick.q_ref = freeze_q_;
    tick.q_follow = freeze_q_;
    if (ack_timeout_s_ > 0.0 && wait_ack_elapsed_s_ > ack_timeout_s_) {
      state_ = FollowerState::Hold;
      has_package_ = false;
      timeline_frozen_ = false;
      tick.state = FollowerState::Hold;
      tick.action = FollowerAction::NeedReplan;
      tick.reason = "ack_timeout";
      tick.timeline_frozen = false;
    }
    last_ = tick;
    return last_;
  }
  if (state_ == FollowerState::Hold || !has_package_) {
    last_ = make_hold(q_cmd_, "hold");
    last_.t_s = exec_t_;
    last_.exec_t_s = exec_t_;
    last_.timeline_frozen = timeline_frozen_;
    return last_;
  }

  // Advance execution time only while the timeline is not frozen at a barrier.
  if (!timeline_frozen_) {
    exec_t_ += dtp;
  }

  FollowerTick tick;
  tick.state = FollowerState::Active;
  tick.generation = package_.generation;
  tick.path_id = package_.path_id;
  tick.t_s = exec_t_;
  tick.exec_t_s = exec_t_;
  tick.timeline_frozen = timeline_frozen_;
  tick.freeze_q = freeze_q_;
  tick.pitch_excess = pitch_excess_of(q_meas);

  TrajectoryTimeSample samp;
  if (!sample_trajectory_at(package_, exec_t_, &samp) || !samp.ok) {
    state_ = FollowerState::Braking;
    brake_ticks_ = 0;
    tick.state = FollowerState::Braking;
    tick.action = FollowerAction::Brake;
    tick.reason = samp.reason != nullptr ? samp.reason : "sample_fail";
    tick.q_ref = q_cmd_;
    tick.q_follow = q_cmd_;
    last_ = tick;
    return last_;
  }
  const Vec18 q_plan = samp.q;
  tick.q_plan = q_plan;
  tick.s = samp.s;

  // Stage stop / external ACK gates — freeze exec timeline at the barrier.
  if (const StageBoundary* b = stage_at_progress(samp.s)) {
    const bool at_stop = b->stop_required || b->wait_for_external_ack;
    if (at_stop && b->sample_index < package_.samples.size()) {
      const Vec18& q_stage = package_.samples[b->sample_index];
      const double barrier_t = time_at_sample_index(package_, b->sample_index);
      if (exec_t_ > barrier_t) {
        exec_t_ = barrier_t;
      }
      timeline_frozen_ = true;
      freeze_exec_t_ = barrier_t;
      freeze_q_ = q_stage;
      tick.t_s = exec_t_;
      tick.exec_t_s = exec_t_;
      tick.timeline_frozen = true;
      tick.freeze_q = q_stage;
      const bool stopped =
          (q_meas - q_stage).cwiseAbs().maxCoeff() < 0.03 &&
          (q_cmd_ - q_stage).cwiseAbs().maxCoeff() < 0.03;
      if (b->wait_for_external_ack) {
        if (!stopped) {
          tick.q_ref = q_stage;
          tick.q_follow = q_stage;
          tick.q_plan = q_stage;
          tick.action = FollowerAction::Continue;
          tick.reason = "decel_to_stage_stop";
          q_prev_cmd_ = q_cmd_;
          q_cmd_ = q_stage;
          last_ = tick;
          return last_;
        }
        state_ = FollowerState::WaitAck;
        wait_stage_id_ = b->stage_id;
        wait_ack_elapsed_s_ = 0.0;
        tick.state = FollowerState::WaitAck;
        tick.action = FollowerAction::WaitExternalAck;
        tick.wait_stage_id = b->stage_id;
        tick.q_ref = q_stage;
        tick.q_follow = q_stage;
        tick.q_plan = q_stage;
        tick.reason = "wait_external_ack";
        q_cmd_ = q_stage;
        last_ = tick;
        return last_;
      }
      if (b->stop_required && !stopped) {
        tick.q_ref = q_stage;
        tick.q_follow = q_stage;
        tick.q_plan = q_stage;
        tick.action = FollowerAction::Continue;
        tick.reason = "stop_required_decel";
        q_cmd_ = q_stage;
        last_ = tick;
        return last_;
      }
      if (b->stop_required && stopped) {
        // Actually stopped: release the barrier and continue the next stage.
        ++stage_cursor_;
        timeline_frozen_ = false;
        tick.timeline_frozen = false;
      }
    }
  }

  const double dur = package_.time_law.duration_s;
  const bool time_done = dur > 0.0 && exec_t_ >= dur - 1e-9;
  tick.time_done = time_done;
  if (time_done) {
    tick.q_ref = package_.samples.back();
    tick.q_follow = tick.q_ref;
    tick.q_offset.setZero();
    tick.action = FollowerAction::Arrived;
    tick.reason = "time_done_claim";
    timeline_frozen_ = true;
    freeze_exec_t_ = exec_t_;
    freeze_q_ = tick.q_ref;
    tick.timeline_frozen = true;
    tick.freeze_q = tick.q_ref;
    q_prev_cmd_ = q_cmd_;
    q_cmd_ = tick.q_ref;
    last_ = tick;
    return last_;
  }

  Vec18 q_ref = q_plan;
  const char* fail = nullptr;
  bool qp_used = false;
  if (enable_qp_) {
    const Vec3 p_plan = kin_.fk_pos(q_plan, ee_frame(package_.left));
    const Vec3 p_meas = kin_.fk_pos(q_meas, ee_frame(package_.left));
    const double err = (p_meas - p_plan).norm();
    if (err > 1e-4 && err <= track_err_m_) {
      Vec18 q_corr;
      if (try_qp_correction(q_plan, q_cmd_, q_meas, dtp, &q_corr, &fail)) {
        q_ref = q_corr;
        qp_used = true;
      } else {
        state_ = FollowerState::Braking;
        brake_ticks_ = 0;
        tick.state = FollowerState::Braking;
        tick.action = FollowerAction::NeedReplan;
        tick.reason = fail != nullptr ? fail : "qp_need_replan";
        tick.q_ref = q_cmd_;
        tick.q_follow = q_cmd_;
        last_ = tick;
        return last_;
      }
    }
  }

  if (!command_feasible(q_ref, q_cmd_, dtp, 0.0, &fail)) {
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
  tick.q_offset = q_ref - q_plan;
  tick.action = FollowerAction::Continue;
  tick.reason = qp_used ? "follow_qp" : "follow_nominal";
  tick.qp_used = qp_used;
  q_prev_cmd_ = q_cmd_;
  q_cmd_ = q_ref;
  last_ = tick;
  return last_;
}

}  // namespace kmpc
