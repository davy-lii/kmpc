#pragma once
// 对应 Python kmpc/limits.py + kmpc_params.yaml。
// yaml 只在 Config 读一次；运行期只读。对照 09-04 数值表与开发方案 §3。

#include <stdexcept>
#include <string>
#include <vector>

#include "kmpc/types.hpp"

namespace kmpc {

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// 对应 Python JointLimits。QP 盒与 Safety 必须拿同一份，禁止各自 YAML::LoadFile。
struct JointVelocityLimits {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q_min = Vec18::Zero();
  Vec18 q_max = Vec18::Zero();
  Vec18 dq_max = Vec18::Zero();
};

// 键名与 Python kmpc_params.yaml / declare_parameter 相同，不发明第二套。
struct KmpcParams {
  double dt = 0.02;
  double control_rate = 50.0;
  int n_horizon = 15;
  double q_limit_margin = 0.005236;
  double w_joint = 8.0;
  double r_arm = 0.05;
  double track_alpha = 0.20;
  double arrived_err = 0.01;
  double arrived_exit = 0.02;
  double arrived_stall_mm = 0.5;
  int arrived_stall_ticks = 25;
  double s_joint = 0.5;
  double q_pos = 400.0;
  double q_offset = 400.0;
  double q_offset_y = 150.0;
  double q_ori = 20.0;
  double vo_clamp_ori_rad = 0.35;
  double arrived_ori_rad = 0.087;
  double q_com = 40.0;
  double com_margin = 0.025;
  double com_xmin = -0.322;
  double com_xmax = 0.138;
  double com_ymin = -0.235;
  double com_ymax = 0.235;
  // 相对「外墙±com_margin」再缩的工作墙。0 = 墙位回到现码内墙。
  double com_work_m = 0.06;
  double q_torso = 250.0;
  // 20261015 华成实测俯仰和 -20.2°，15° 盒会削掉安全姿势。
  // 20261110 现场指令贴 -22°、实测约 -22.129° 被拒；硬盒放到 23°。
  double torso_pitch_max_deg = 23.0;
  double torso_hold_deg = 2.0;
  double torso_step_deg = 0.08;
  double torso_drop_m = 0.40;
  double torso_k_per_m = 0.327;
  // 起身 θ_des=clip(θ0)，不再 max(κ e_z)。false = 09-04 换边。
  bool torso_rise_hold = true;
  // 20261015：下降方向也放开 κ e_z（θ_des 跟高度误差走，不再钉死）。
  // 深下降 is_deep_drop 分支不受影响。false = 旧行为（下降也钉死）。
  bool torso_reach_pitch_follow = true;
  // 腕状态屏障（20261015 方案第一刀）：靠限位变贵，弹簧打在状态上往回放。
  // 默认双臂 J5（index 8/15）。barrier_alpha=0 完全关闭（回滚）。
  std::vector<int> barrier_joints = {8, 15};
  double barrier_warn_deg = 20.0;
  double barrier_floor_deg = 5.0;
  double barrier_alpha = 2.0;
  double barrier_w_max_ratio = 50.0;
  // |e_z|≥此值时 V_com 的 x 靶跟当前 p_com.x；0 = 关闭。
  double com_hold_ez_m = 0.03;
  double q_syn = 80.0;
  double alpha_syn = 1.75;
  double q_yaw = 80.0;
  double yaw_max_deg = 35.0;
  double w_idle_arm = 40.0;
  double w_o_arm = 1.0;
  double track_gamma = 0.1;
  double vo_clamp_m = 0.12;
  double vo_clamp_y_m = 0.40;
  double vo_arm_floor_m = 0.03;
  double vo_arm_gamma = 0.3;
  double waist_gamma = 0.3;
  double waist_pred_lambda = 0.05;
  double waist_lock_eps = 1.0e-5;
  double calib_tol_rad = 0.02;
  double cmd_reset_rad = 0.15;
};

struct Config {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JointVelocityLimits limits;
  Vec18 q_pref = Vec18::Zero();
  std::string q_pref_station = "default";
  KmpcParams params;

  static Config load(
      const std::string& joint_limits_yaml,
      const std::string& q_pref_yaml,
      const std::string& kmpc_params_yaml,
      const std::string& station = "default");
};

// 腰锁只覆盖副本，禁止写 P / R，禁止改 joint_limits.yaml。
inline Vec18 effective_dq_max(
    const JointVelocityLimits& limits,
    bool waist_locked,
    double waist_lock_eps) {
  Vec18 dq = limits.dq_max;
  if (waist_locked) {
    dq.segment<kNWaist>(kSliceWaist.begin).setConstant(waist_lock_eps);
  }
  return dq;
}

}  // namespace kmpc
