// yaml 只在这里读。运行期 Config 只读，热路径禁止再 LoadFile。

#include "kmpc/config.hpp"

#include <cmath>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace kmpc {
namespace {

void append_doubles(const YAML::Node& node, std::vector<double>* out) {
  if (!node) {
    return;
  }
  if (node.IsSequence()) {
    for (const auto& item : node) {
      append_doubles(item, out);
    }
    return;
  }
  out->push_back(node.as<double>());
}

Vec18 read_vec18(const YAML::Node& node, const char* field) {
  std::vector<double> tmp;
  append_doubles(node, &tmp);
  if (static_cast<int>(tmp.size()) != kNDof) {
    throw ConfigError(
        std::string(field) + " must have 18 entries, got " +
        std::to_string(tmp.size()));
  }
  Vec18 out;
  for (int i = 0; i < kNDof; ++i) {
    out[i] = tmp[static_cast<std::size_t>(i)];
  }
  return out;
}

double req_double(const YAML::Node& p, const char* key, double fallback) {
  if (!p[key]) {
    return fallback;
  }
  return p[key].as<double>();
}

bool req_bool(const YAML::Node& p, const char* key, bool fallback) {
  if (!p[key]) {
    return fallback;
  }
  return p[key].as<bool>();
}

int req_int(const YAML::Node& p, const char* key, int fallback) {
  if (!p[key]) {
    return fallback;
  }
  return p[key].as<int>();
}

std::vector<int> req_int_list(
    const YAML::Node& p, const char* key, const std::vector<int>& fallback) {
  if (!p[key]) {
    return fallback;
  }
  if (!p[key].IsSequence()) {
    throw ConfigError(std::string(key) + " must be a sequence of int");
  }
  std::vector<int> out;
  for (const auto& item : p[key]) {
    out.push_back(item.as<int>());
  }
  return out;
}

JointVelocityLimits load_limits(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  if (!root["names"] || !root["names"].IsSequence()) {
    throw ConfigError("joint_limits.yaml missing names");
  }
  if (static_cast<int>(root["names"].size()) != kNDof) {
    throw ConfigError("joint_limits names size != 18");
  }
  for (int i = 0; i < kNDof; ++i) {
    const std::string name = root["names"][i].as<std::string>();
    if (name != kJointNames[static_cast<std::size_t>(i)]) {
      throw ConfigError("joint_limits names mismatch at " + std::to_string(i));
    }
  }
  JointVelocityLimits lim;
  lim.q_min = read_vec18(root["q_min"], "q_min");
  lim.q_max = read_vec18(root["q_max"], "q_max");
  lim.dq_max = read_vec18(root["dq_max"], "dq_max");
  if ((lim.q_min.array() >= lim.q_max.array()).any()) {
    throw ConfigError("q_min must be < q_max");
  }
  if ((lim.dq_max.array() <= 0.0).any()) {
    throw ConfigError("dq_max must be > 0");
  }
  return lim;
}

KmpcParams load_params(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  YAML::Node p = root["kmpc_node"]["ros__parameters"];
  if (!p) {
    p = root;
  }
  KmpcParams out;
  out.dt = req_double(p, "dt", out.dt);
  out.control_rate = req_double(p, "control_rate", out.control_rate);
  out.n_horizon = req_int(p, "n_horizon", out.n_horizon);
  out.q_limit_margin = req_double(p, "q_limit_margin", out.q_limit_margin);
  out.w_joint = req_double(p, "w_joint", out.w_joint);
  out.r_arm = req_double(p, "r_arm", out.r_arm);
  out.track_alpha = req_double(p, "track_alpha", out.track_alpha);
  out.arrived_err = req_double(p, "arrived_err", out.arrived_err);
  out.arrived_exit = req_double(p, "arrived_exit", out.arrived_exit);
  out.arrived_stall_mm = req_double(p, "arrived_stall_mm", out.arrived_stall_mm);
  out.arrived_stall_ticks = req_int(p, "arrived_stall_ticks", out.arrived_stall_ticks);
  out.s_joint = req_double(p, "s_joint", out.s_joint);
  out.q_pos = req_double(p, "q_pos", out.q_pos);
  out.q_offset = req_double(p, "q_offset", out.q_offset);
  out.q_offset_y = req_double(p, "q_offset_y", out.q_offset_y);
  out.q_ori = req_double(p, "q_ori", out.q_ori);
  out.vo_clamp_ori_rad = req_double(p, "vo_clamp_ori_rad", out.vo_clamp_ori_rad);
  out.arrived_ori_rad = req_double(p, "arrived_ori_rad", out.arrived_ori_rad);
  out.q_com = req_double(p, "q_com", out.q_com);
  out.com_margin = req_double(p, "com_margin", out.com_margin);
  out.com_xmin = req_double(p, "com_xmin", out.com_xmin);
  out.com_xmax = req_double(p, "com_xmax", out.com_xmax);
  out.com_ymin = req_double(p, "com_ymin", out.com_ymin);
  out.com_ymax = req_double(p, "com_ymax", out.com_ymax);
  out.com_work_m = req_double(p, "com_work_m", out.com_work_m);
  out.q_torso = req_double(p, "q_torso", out.q_torso);
  out.torso_pitch_max_deg =
      req_double(p, "torso_pitch_max_deg", out.torso_pitch_max_deg);
  out.torso_hold_deg = req_double(p, "torso_hold_deg", out.torso_hold_deg);
  out.torso_step_deg = req_double(p, "torso_step_deg", out.torso_step_deg);
  out.torso_drop_m = req_double(p, "torso_drop_m", out.torso_drop_m);
  out.torso_k_per_m = req_double(p, "torso_k_per_m", out.torso_k_per_m);
  out.torso_rise_hold = req_bool(p, "torso_rise_hold", out.torso_rise_hold);
  out.torso_reach_pitch_follow = req_bool(
      p, "torso_reach_pitch_follow", out.torso_reach_pitch_follow);
  out.barrier_joints = req_int_list(p, "barrier_joints", out.barrier_joints);
  out.barrier_warn_deg = req_double(p, "barrier_warn_deg", out.barrier_warn_deg);
  out.barrier_floor_deg =
      req_double(p, "barrier_floor_deg", out.barrier_floor_deg);
  out.barrier_alpha = req_double(p, "barrier_alpha", out.barrier_alpha);
  out.barrier_w_max_ratio =
      req_double(p, "barrier_w_max_ratio", out.barrier_w_max_ratio);
  out.com_hold_ez_m = req_double(p, "com_hold_ez_m", out.com_hold_ez_m);
  out.q_syn = req_double(p, "q_syn", out.q_syn);
  out.alpha_syn = req_double(p, "alpha_syn", out.alpha_syn);
  out.q_yaw = req_double(p, "q_yaw", out.q_yaw);
  out.yaw_max_deg = req_double(p, "yaw_max_deg", out.yaw_max_deg);
  out.w_idle_arm = req_double(p, "w_idle_arm", out.w_idle_arm);
  out.w_o_arm = req_double(p, "w_o_arm", out.w_o_arm);
  out.track_gamma = req_double(p, "track_gamma", out.track_gamma);
  out.vo_clamp_m = req_double(p, "vo_clamp_m", out.vo_clamp_m);
  out.vo_clamp_y_m = req_double(p, "vo_clamp_y_m", out.vo_clamp_y_m);
  out.vo_arm_floor_m = req_double(p, "vo_arm_floor_m", out.vo_arm_floor_m);
  out.vo_arm_gamma = req_double(p, "vo_arm_gamma", out.vo_arm_gamma);
  out.waist_gamma = req_double(p, "waist_gamma", out.waist_gamma);
  out.waist_pred_lambda = req_double(p, "waist_pred_lambda", out.waist_pred_lambda);
  out.waist_lock_eps = req_double(p, "waist_lock_eps", out.waist_lock_eps);
  out.calib_tol_rad = req_double(p, "calib_tol_rad", out.calib_tol_rad);
  out.cmd_reset_rad = req_double(p, "cmd_reset_rad", out.cmd_reset_rad);
  if (out.dt <= 0.0 || out.n_horizon < 1) {
    throw ConfigError("dt must be > 0 and n_horizon >= 1");
  }
  if (out.waist_lock_eps <= 0.0) {
    throw ConfigError("waist_lock_eps must be > 0");
  }
  if (!std::isfinite(out.q_pos) || out.q_pos < 0.0) {
    throw ConfigError("q_pos must be finite and >= 0");
  }
  if (out.barrier_alpha < 0.0 || out.barrier_w_max_ratio <= 0.0) {
    throw ConfigError("barrier_alpha must be >= 0 and barrier_w_max_ratio > 0");
  }
  if (out.barrier_alpha > 0.0 && out.barrier_warn_deg <= out.barrier_floor_deg) {
    throw ConfigError("barrier_warn_deg must be > barrier_floor_deg");
  }
  return out;
}

}  // namespace

Config Config::load(const std::string& joint_limits_yaml,
                    const std::string& kmpc_params_yaml) {
  Config cfg;
  cfg.limits = load_limits(joint_limits_yaml);
  cfg.params = load_params(kmpc_params_yaml);
  return cfg;
}

}  // namespace kmpc
