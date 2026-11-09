#pragma once
// 对应 Python kmpc/cartesian_tracker.py；对照 09-04 §1 / §3。
// compute 只编排：FK/J → 装配 P,g,A,b → ProxQP → ARRIVED。Safety / Planner 不进这里。
// 腰锁只覆盖 dq_max。z_home / λ / q4_pref 钉构造期 HOME，不随运行 q_calib 热改。

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/StdVector>

#include "kmpc/config.hpp"
#include "kmpc/constraints.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/qp_solver.hpp"
#include "kmpc/types.hpp"
#include "kmpc/upper_command.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct TrackerStep {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 dq = Vec18::Zero();
  Vec6 ee_error = Vec6::Zero();
};

struct LineTarget {
  Vec3 xyz = Vec3::Zero();
  Vec4 quat_xyzw = Vec4(0, 0, 0, 1);
  bool ori_on = false;
  std::string frame = std::string(kBaseFrame);
};

class CartesianTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CartesianTracker(Kinematics& kin, const Config& cfg);

  void set_upper_commands(const UpperCommand& cmd);
  void reset();
  void reset_warm();
  void clear_arrived() { last_arrived_ = false; }

  TrackerStep compute(
      const Vec18& q0,
      const Vec18& q_pref,
      const Vec3* target_left,
      const Vec3* target_right,
      const Vec4* quat_left = nullptr,
      const Vec4* quat_right = nullptr,
      std::string_view frame_left = kEeFrameLeft,
      std::string_view frame_right = kEeFrameRight);

  Vec18 inertia_r(const Vec18& q0) const;
  Vec18 offset_weights(int n_active, bool left_active, bool calibrating) const;
  Vec18 idle_offset_grad(
      const Vec18& q0, const Vec18& q_pref, const Vec18& w_o, int n_active,
      bool left_active, bool calibrating) const;

  const Mat36& last_P() const { return last_P_; }
  const Vec36& last_g() const { return last_g_; }
  double last_e0() const { return last_e0_; }
  Vec6 last_e_ori() const { return last_e_ori_; }
  bool last_arrived() const { return last_arrived_; }
  bool last_ori_on() const { return last_ori_on_; }
  Vec3 last_com() const { return last_com_; }
  double last_pitch() const { return last_pitch_; }
  double last_theta_des() const { return last_theta_des_; }
  double last_c_des_x() const { return last_c_des_x_; }
  bool last_com_hold() const { return last_com_hold_; }
  bool last_com_work_wall() const { return last_com_work_wall_; }
  double last_com_work_xmin() const { return last_com_work_xmin_; }
  double last_com_work_xmax() const { return last_com_work_xmax_; }
  double last_com_slack_x() const { return last_com_slack_x_; }
  bool last_rise_hold() const { return torso_rise_hold_; }
  double z_home() const { return z_home_; }
  double r_lambda() const { return r_lambda_; }
  QpStatus last_qp_status() const { return last_qp_status_; }
  int last_n_ineq() const { return last_n_ineq_; }
  bool stall_blocked_by_wall_slide(
      const Vec18& q0, double e_z, double e0) const;
  // 与 QP 同拍的盒，含腰锁 ε。Safety 出口应喂同一份。
  Vec18 effective_dq_max() const;
  // 单臂够点走事先直线。双臂、标定仍走原来的 QP。单臂这条线解不出就停住。
  bool line_active() const { return line_.on; }
  bool line_holding() const { return line_.hold; }
  // Refusal remains latched until an explicit segment rearm/reset.
  bool line_refused() const { return line_.refused; }
  // 本拍要发的路径点。进度只沿路径加，不看实测把方向拉开。
  Vec18 line_cmd() const { return line_q_; }
  int line_gen() const { return line_gen_; }
  // Path-rate update only. Does not change planning, freeze, or safety.
  // Experiment default is r10b first-order; r11 S-curve remains the same-build baseline.
  enum class LinePaceMode : uint8_t {
    R11Scurve = 0,
    R10bFirstOrder = 1,
  };
  static constexpr LinePaceMode kDefaultLinePaceMode = LinePaceMode::R10bFirstOrder;
  static const char* line_pace_mode_name(LinePaceMode mode) {
    return mode == LinePaceMode::R10bFirstOrder ? "r10b_first_order" : "r11_scurve";
  }
  static bool parse_line_pace_mode(std::string_view text, LinePaceMode* out) {
    if (!out) {
      return false;
    }
    if (text == "r11_scurve") {
      *out = LinePaceMode::R11Scurve;
      return true;
    }
    if (text == "r10b_first_order") {
      *out = LinePaceMode::R10bFirstOrder;
      return true;
    }
    return false;
  }

  // speed_scale 是直线自己的比例，0..1，不跟 QP 的 dq_scale。accel_s<=0 表示这一拍直接用满，不爬坡。
  void set_line_pace(double speed_scale, double accel_s);
  // 和插补器同一个 jerk_s：比例从 0 收到巡航要这么多秒的加速度爬坡。只作用在路径进度这一个标量上。
  void set_line_jerk_s(double jerk_s);
  void set_line_pace_mode(LinePaceMode mode) { line_pace_mode_ = mode; }
  LinePaceMode line_pace_mode() const { return line_pace_mode_; }
  // Brake scale only. Freeze distance stays at the compile-time 0.08 m constant.
  // Returns false on non-finite or non-positive values; caller must refuse startup.
  bool set_line_brake_distance_m(double meters);
  double line_brake_distance_m() const { return line_brake_distance_m_; }
  double line_rate() const { return line_rate_; }
  // S-curve acceleration state. First-order mode keeps this at 0 (not applicable).
  double line_rate_acc() const { return line_rate_acc_; }
  double last_line_path_remain_m() const { return last_line_path_remain_m_; }
  double last_line_brake() const { return last_line_brake_; }
  double last_line_want() const { return last_line_want_; }
  // Explicit segment boundary. Target updates alone never clear a refusal.
  bool rearm_line(uint64_t generation, const std::optional<LineTarget>& tip = std::nullopt);
  void set_line_reference(const Vec18& q) { line_reference_ = q; }
  void refuse_line(const std::string& reason);
  uint64_t goal_generation() const { return goal_generation_; }
  bool line_frozen() const { return line_.frozen; }
  bool line_abort_latched() const { return line_.abort_latched; }
  bool line_cascade_checked() const { return line_.cascade_checked; }
  bool line_cascade_passed() const { return line_.cascade_passed; }
  bool line_cascade_forced_waist() const { return line_.cascade_forced_waist; }
  bool line_cascade_full_body() const { return line_.cascade_full_body; }
  bool line_waist_locked() const { return line_.waist_locked; }
  const std::string& line_reason() const { return line_.reason; }
  const std::string& line_candidate() const { return line_.candidate; }
  const std::string& line_pose_pref_decision() const { return line_.pose_pref_decision; }
  double line_max_arm_fold() const { return line_.max_arm_fold; }
  double line_build_ms() const { return line_.build_ms; }
  double line_progress() const { return line_.prog; }
  double line_length() const { return line_.length_m; }
  size_t line_sample_count() const { return line_.samples.size(); }
  const auto& line_samples() const { return line_.samples; }
  int line_qp_solves() const { return line_qp_solves_; }
  // Default-off posture candidate for straight-line builds. Uses loaded q_pref.
  // Does not change baseline selection when enable=false.
  void set_line_pose_pref(bool enable) { line_pose_pref_enable_ = enable; }
  bool line_pose_pref_enabled() const { return line_pose_pref_enable_; }
  void set_line_pose_pref_q(const Vec18& q_pref) { line_q_pref_ = q_pref; }

  // Immediate return reuses the last successful nominal samples in reverse.
  // Not a servo-time-series reverse. There is no off switch.
  const std::string& reverse_line_decision() const { return reverse_decision_; }
  void clear_last_line_cache();
  // Snapshot/import so async prepare_line workers can see the main-tracker cache.
  struct LastLineCache {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool valid = false;
    bool left = true;
    bool ori_on = false;
    std::string frame;
    Vec3 start_xyz = Vec3::Zero();
    Vec3 end_xyz = Vec3::Zero();
    Vec4 start_quat = Vec4(0, 0, 0, 1);
    Vec4 end_quat = Vec4(0, 0, 0, 1);
    Vec18 q_start = Vec18::Zero();
    Vec18 q_end = Vec18::Zero();
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>> samples;
    uint64_t generation = 0;
  };
  LastLineCache last_line_cache() const { return last_line_cache_; }
  void import_last_line_cache(const LastLineCache& cache) { last_line_cache_ = cache; }
  // Call once when the active line first reaches measured/command hold.
  void maybe_commit_last_line_cache(const Vec18& q_meas);

  // Default-off layered diagnostics. RT path only writes a preallocated ring.
  enum class LineDiagLayer : uint8_t {
    Build = 0,
    FollowQp,
    FollowCommit,
    Feedback,
  };
  struct LineDiagEvent {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineDiagLayer layer = LineDiagLayer::Build;
    uint64_t goal_generation = 0;
    int line_gen = 0;
    double t_steady_s = 0.0;
    double prog = 0.0;
    double build_ms = 0.0;
    char candidate[32]{};
    char decision[48]{};
    Vec18 q = Vec18::Zero();
    Vec18 q_aux = Vec18::Zero();  // QP from / feedback, depending on layer
    // steady_clock 纪元纳秒。source_stamp_ns 只在有消息头时填写，否则 -1。
    uint64_t mono_ns = 0;
    int64_t source_stamp_ns = -1;
    uint64_t recv_mono_ns = 0;
    uint64_t run_id = 0;
    uint64_t seq = 0;
  };
  void set_line_diag_enable(bool enable);
  void set_line_diag_run_id(uint64_t run_id) { line_diag_run_id_ = run_id; }
  bool line_diag_enabled() const { return line_diag_enable_; }
  size_t line_diag_overflows() const { return line_diag_overflows_; }
  size_t line_diag_size() const;
  // Oldest→newest copy; safe to call off the RT thread when diag is paused/off.
  std::vector<LineDiagEvent> line_diag_snapshot() const;
  std::vector<LineDiagEvent> line_diag_take();
  void line_diag_splice(const std::vector<LineDiagEvent>& events);
  void line_diag_clear();
  void line_diag_push_feedback(const Vec18& q_meas, double source_stamp_s);

  bool line_needs_build(bool left, const Vec3& target, const Vec4* quat) const;
  bool line_measured_arrived(const Vec18& q) const;
  bool line_stream_segment_ok(const Vec18& from, const Vec18& to) const;



 private:
  struct JacTerm {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool left = true;
    Mat3x18 jac_v = Mat3x18::Zero();
    Mat3x18 jac_v_pad = Mat3x18::Zero();
    Vec3 e = Vec3::Zero();
    Mat3x18 jac_w = Mat3x18::Zero();
    Mat3x18 jac_w_pad = Mat3x18::Zero();
    Vec3 e_ori = Vec3::Zero();
    bool ori_on = false;
  };

  void on_tracking_state_changed();
  void clear_hold_warm();

  void assemble_vo(
      const Mat3x18& jac_f, const Vec3& err, const Vec3* pitch_lo,
      const Vec3* pitch_hi, double e_z, double pitch_sum_lo,
      double pitch_sum_hi, const Mat3x18* jac_omega, const Vec3* e_ori,
      bool ori_on, Mat18& h_vo, Vec18& g_vo);

  void com_terminal(
      const Vec3& p_com, const Mat3x18& j_com, double e_z, Mat18& h_com,
      Vec18& g_com, MatIneq& A, VecIneq& b, int& n_ineq);

  void torso_synergy_terminal(
      const Vec18& q0, double e_z, Mat18& h, Vec18& g);
  // 腕状态屏障（20261015）：在 assemble_endpoints 之后对 last_P_/last_g_
  // 两个端点对角块加项；弹簧打在状态 q0+y 上往警戒区边界回拉。
  void apply_joint_barrier(const Vec18& q0, const Vec18& r);

  static double working_ez(const JacTerm* terms, int n_terms);
  void update_pitch_pin(const Vec18& q0, double e_z);
  void pitch_sum_bounds(const Vec18& q0, double& lo, double& hi) const;
  void pitch_ineq(const Vec18& q0, MatIneq& A, VecIneq& b, int& n_ineq) const;
  void yaw_ineq(const Vec18& q0, MatIneq& A, VecIneq& b, int& n_ineq) const;

  Vec18 solve_horizon(
      const Vec18& q0, const Vec18& w_diag, const Vec18& w_o, const Vec18& g_wo,
      const Vec18& r, const JacTerm* terms, int n_terms, const Vec3& p_com,
      const Mat3x18& j_com, int idle_begin);

  double active_norm(const Vec6& err, bool has_left, bool has_right) const;
  bool targets_jumped(const Vec3* left, const Vec3* right);
  bool ori_jumped(const Vec3& e_left, const Vec3& e_right);
  bool ori_arrived_ok(const JacTerm* terms, int n_terms) const;
  static bool one_target_jumped(const Vec3* cur, bool has_prev, const Vec3& prev);
  Vec18 apply_arrived(
      const Vec18& dq, double e0, bool jumped, const Vec18& q0, double e_z,
      bool ori_ok);

  bool is_deep_drop(double e_z);
  void refresh_deep_drop(double e_z, double z_star);
  bool pitch_at_lower_wall(double pitch0) const;
  Vec3 damped_waist_yhat(
      const Eigen::Matrix<double, 3, 3>& j_pitch, const Vec3& e_cap,
      double e_z);
  Vec3 clip_yhat_pitch_sum(
      const Vec3& y_hat, double lo, double hi, bool keep_j3_zero,
      bool dump_j3) const;
  Vec3 waist_yhat_metric(double e_z);
  double arm_residual_cap(double e_w_z, double e0_z, double e_z);

  void e0_hist_clear();
  void e0_hist_append(double e0);
  bool e0_hist_stalled() const;

  // 20261029：一条直线一次求完。腰 4 轴加这条臂 7 轴，另一条臂不动。
  // Waist joints keep their actual pivots; pitch sum and CoM stay constrained.
  // Replan drift outside 8 cm, freeze the endpoint inside 8 cm; refuse is latched.
 public:
  struct LineRun {
    bool on = false;
    bool hold = false;
    bool refused = false;
    bool left = true;
    bool ori_on = false;
    bool waist_locked = false;
    bool frozen = false;
    bool abort_latched = false;
    bool cascade_checked = false;
    bool cascade_passed = false;
    bool cascade_forced_waist = false;
    bool cascade_full_body = false;
    // arm_only | free_fit | tip_seed | cascade_full_body | pose_pref |
    // reverse_cache | qa_seed | none
    std::string candidate = "none";
    // skipped | accepted | rejected_<reason> | disabled
    std::string pose_pref_decision = "disabled";
    // skipped | accepted | rejected_<reason>
    std::string reverse_decision = "skipped";
    double max_arm_fold = 0.0;
    std::string reason;
    // 异步建线线程的诊断副本。adopt 后拆走，不参与跟踪计算。
    std::vector<LineDiagEvent> diag_events;
    double build_ms = 0.0;
    Vec3 target = Vec3::Zero();
    Vec4 quat = Vec4(0.0, 0.0, 0.0, 1.0);
    std::string frame;
    std::vector<Vec18, Eigen::aligned_allocator<Vec18>> samples;
    double prog = 0.0;
    double length_m = 0.0;
  };

  LineRun prepare_line(const Vec18& q, bool left, const Vec3& target, const Vec4* quat) {
    if (!line_build(q,left,target,quat,std::string(left?kEeFrameLeft:kEeFrameRight)))
      refuse_line(line_.reason);
    return line_;
  }
  void adopt_line(LineRun plan) {
    if (!plan.diag_events.empty()) {
      line_diag_splice(plan.diag_events);
      plan.diag_events.clear();
    }
    reverse_decision_ = plan.reverse_decision.empty() ? reverse_decision_ : plan.reverse_decision;
    line_ = std::move(plan);
    if (!line_.samples.empty()) line_q_=line_.samples.front();
    line_rate_=0; line_rate_acc_=0; ++line_gen_; last_arrived_=false;
    last_line_cache_committed_ = false;
  }
 private:
  void line_clear();
  bool follow_line(
      const Vec18& q0, const Vec3* left, const Vec3* right, const Vec4* quat_left,
      const Vec4* quat_right, std::string_view frame_left,
      std::string_view frame_right, TrackerStep& out);
  bool line_build(
      const Vec18& q0, bool left, const Vec3& target, const Vec4* quat,
      const std::string& frame);
  bool line_arm_only(
      const Vec18& q0, int arm_begin, const Vec3& target, const Vec4* quat,
      bool ori_on, const std::string& frame,
      std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples) const;
  bool line_fit(
      const Vec18& q0, int arm_begin, const Vec3& target, const Vec4* quat,
      bool ori_on, bool lock_waist, const std::string& frame,
      std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples,
      bool& waist_locked, bool allow_relock = true, const Vec18* goal_seed = nullptr,
      const Vec18* pose_pref = nullptr) const;
  bool line_validate(const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& samples,
      const Vec3& target, const Vec4* quat, bool ori_on, const std::string& frame) const;
  bool line_path_qp(const Vec18& from, double next_prog, const Vec18& cap, Vec18& to);

  Vec18 line_at(double prog) const;
  bool line_target_far(const Vec3& target, const Vec4* quat) const;

  LineRun line_;
  uint64_t goal_generation_ = 0;
  std::optional<LineTarget> line_tip_;
  std::optional<Vec18> line_reference_;
  int line_qp_solves_ = 0;

  Vec18 line_q_ = Vec18::Zero();
  int line_gen_ = 0;
  double line_speed_scale_ = 1.0;
  double line_accel_s_ = 0.0;
  double line_jerk_s_ = 0.08;
  LinePaceMode line_pace_mode_ = kDefaultLinePaceMode;
  double line_brake_distance_m_ = 0.08;
  double line_rate_ = 0.0;
  double line_rate_acc_ = 0.0;
  double last_line_path_remain_m_ = 0.0;
  double last_line_brake_ = 1.0;
  double last_line_want_ = 0.0;
  bool line_pose_pref_enable_ = false;
  Vec18 line_q_pref_ = Vec18::Zero();
  // Soft posture pull and acceptance budgets (radians unless noted).
  double line_pose_pref_w_ = 0.35;
  double line_pose_pref_min_improve_ = 5.0 * kPi / 180.0;
  // Calibrated from offline left/right place-exit fixtures (20261105): pose_pref
  // trials improved elbow ~14° with ~26–31° extra waist path; keep margin.
  double line_pose_pref_waist_budget_ = 35.0 * kPi / 180.0;
  double line_pose_pref_wrist_budget_ = 25.0 * kPi / 180.0;

  std::string reverse_decision_ = "skipped";
  LastLineCache last_line_cache_{};
  bool last_line_cache_committed_ = false;
  // Joint-space match for reverse start (~2°/axis) and Cartesian end match (3 mm).
  double reverse_q_tol_ = 2.0 * kPi / 180.0;
  double reverse_xyz_tol_ = 0.003;

  bool line_diag_enable_ = false;
  uint64_t line_diag_run_id_ = 0;
  uint64_t line_diag_seq_ = 0;
  int64_t line_diag_source_ns_ = -1;
  static constexpr size_t kLineDiagCap = 4096;
  std::vector<LineDiagEvent> line_diag_ring_;
  size_t line_diag_begin_ = 0;
  size_t line_diag_count_ = 0;
  size_t line_diag_overflows_ = 0;
  void line_diag_push(LineDiagLayer layer, const Vec18& q, const Vec18& q_aux,
                      double prog, double build_ms, const char* candidate,
                      const char* decision);

  Kinematics& kin_;
  JointVelocityLimits limits_;
  ConstraintBuilder constraints_;
  KmpcParams params_;
  WholeBodyGuards guards_;
  QpSolver qp_;

  double q_pos_ = 400.0;
  double q_offset_ = 400.0;
  double q_offset_y_ = 150.0;
  double w_idle_arm_ = 40.0;
  double w_o_arm_ = 1.0;
  double track_gamma_ = 0.1;
  double vo_clamp_m_ = 0.12;
  double vo_clamp_y_m_ = 0.40;
  double vo_arm_gamma_ = 0.3;
  double waist_gamma_ = 0.3;
  int n_horizon_ = 15;
  double q_limit_margin_ = 0.005236;
  double track_alpha_ = 0.20;
  double arrived_err_ = 0.01;
  double arrived_exit_ = 0.02;
  double q_com_ = 40.0;
  double com_margin_ = 0.025;
  double com_xmin_ = -0.322;
  double com_xmax_ = 0.138;
  double com_ymin_ = -0.235;
  double com_ymax_ = 0.235;
  Eigen::Vector2d c_sup_ = Eigen::Vector2d::Zero();
  double q_torso_ = 250.0;
  double q_syn_ = 80.0;
  double alpha_syn_ = 1.75;
  double q_yaw_ = 80.0;
  double torso_pitch_max_ = 0.0;
  double torso_drop_m_ = 0.40;
  double torso_k_per_m_ = 0.0;
  bool torso_rise_hold_ = true;
  bool torso_reach_pitch_follow_ = true;
  std::vector<int> barrier_joints_{8, 15};
  double barrier_warn_rad_ = 0.0;
  double barrier_floor_rad_ = 0.0;
  double barrier_alpha_ = 0.0;
  double barrier_w_max_ratio_ = 50.0;
  double com_hold_ez_m_ = 0.03;
  double yaw_max_ = 0.0;
  double arrived_stall_m_ = 0.0;
  int arrived_stall_ticks_ = 25;
  double waist_lock_eps_ = 1e-5;
  double waist_pred_lambda_ = 0.05;
  double vo_arm_floor_m_ = 0.03;
  double q_ori_ = 20.0;
  double vo_clamp_ori_rad_ = 0.35;
  double arrived_ori_rad_ = 0.087;
  double s_joint_ = 0.5;
  double w_joint_ = 8.0;
  double r_arm_ = 0.05;
  double r_lambda_ = 1.0;
  double q_yaw_pref_ = 0.0;
  double z_home_ = 0.0;
  double z_star_ = 0.0;
  bool deep_on_ = false;
  double last_e0_ = 0.0;
  Vec6 last_e_ori_ = Vec6::Zero();
  bool last_ori_on_ = false;
  bool last_arrived_ = false;
  Vec3 last_com_ = Vec3::Zero();
  double last_pitch_ = 0.0;
  double last_theta_des_ = 0.0;
  double last_c_des_x_ = 0.0;
  bool last_com_hold_ = false;
  bool last_com_work_wall_ = false;
  double last_com_work_xmin_ = 0.0;
  double last_com_work_xmax_ = 0.0;
  double last_com_slack_x_ = 0.0;
  QpStatus last_qp_status_ = QpStatus::Success;
  int last_n_ineq_ = 0;

  Mat3 qo_ = Mat3::Zero();
  Vec18 j_torso_ = Vec18::Zero();
  Vec18 h_syn_ = Vec18::Zero();
  Vec18 dq_prev_ = Vec18::Zero();
  Vec36 y_prev_ = Vec36::Zero();
  Mat36 last_P_ = Mat36::Zero();
  Vec36 last_g_ = Vec36::Zero();

  bool has_prev_left_ = false;
  bool has_prev_right_ = false;
  Vec3 prev_left_ = Vec3::Zero();
  Vec3 prev_right_ = Vec3::Zero();
  Vec3 e_ori_prev_left_ = Vec3::Zero();
  Vec3 e_ori_prev_right_ = Vec3::Zero();
  bool has_pitch_pin_ = false;
  double pitch_pin_ = 0.0;

  UpperCommand upper_{};
  TrackingState tracking_state_prev_ = TrackingState::Tracking;

  static constexpr int kE0HistCap = 64;
  std::array<double, kE0HistCap> e0_hist_{};
  int e0_hist_n_ = 0;
  int e0_hist_i_ = 0;
  int e0_hist_cap_ = 25;
  double e0_prev_ = 0.0;
};

}  // namespace kmpc
