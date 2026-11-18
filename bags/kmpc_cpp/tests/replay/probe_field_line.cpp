// Offline field-fixture line_build probe. No ROS / hardware.
// Usage:
//   probe_field_line fixture.json out.json [pose_pref=0|1] [dq_scale] [accel_s]
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/tracker_math.hpp"

using namespace kmpc;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool parse_array(const std::string& s, const std::string& key, std::vector<double>* out) {
  const std::string needle = "\"" + key + "\"";
  auto pos = s.find(needle);
  if (pos == std::string::npos) return false;
  pos = s.find('[', pos);
  auto end = s.find(']', pos);
  if (pos == std::string::npos || end == std::string::npos) return false;
  out->clear();
  std::stringstream ss(s.substr(pos + 1, end - pos - 1));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.find_first_not_of(" \t\n\r") == std::string::npos) continue;
    out->push_back(std::stod(tok));
  }
  return true;
}

bool parse_boolish(const std::string& s, const std::string& key, bool* out) {
  const std::string needle = "\"" + key + "\"";
  auto pos = s.find(needle);
  if (pos == std::string::npos) return false;
  pos = s.find(':', pos);
  if (pos == std::string::npos) return false;
  auto rest = s.substr(pos + 1, 16);
  if (rest.find("true") != std::string::npos) {
    *out = true;
    return true;
  }
  if (rest.find("false") != std::string::npos) {
    *out = false;
    return true;
  }
  return false;
}

bool parse_string(const std::string& s, const std::string& key, std::string* out) {
  const std::string needle = "\"" + key + "\"";
  auto pos = s.find(needle);
  if (pos == std::string::npos) return false;
  pos = s.find('"', pos + needle.size());
  if (pos == std::string::npos) return false;
  auto end = s.find('"', pos + 1);
  if (end == std::string::npos) return false;
  *out = s.substr(pos + 1, end - pos - 1);
  return true;
}

double path_len(const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path,
                int begin, int count) {
  double s = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    s += (path[i] - path[i - 1]).segment(begin, count).cwiseAbs().sum();
  }
  return s;
}

int count_reversals(const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& path,
                    int idx, double thr) {
  if (path.size() < 3) return 0;
  int n = 0;
  double prev = path[1][idx] - path[0][idx];
  for (std::size_t i = 2; i < path.size(); ++i) {
    const double d = path[i][idx] - path[i - 1][idx];
    if (std::abs(prev) > thr && std::abs(d) > thr && prev * d < 0.0) ++n;
    if (std::abs(d) > thr) prev = d;
  }
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "probe_field_line fixture.json out.json [pose_pref] [dq] [accel]\n";
    return 2;
  }
  const bool pose_pref = argc > 3 && std::string(argv[3]) != "0";
  const double dq = argc > 4 ? std::stod(argv[4]) : 0.3;
  const double accel = argc > 5 ? std::stod(argv[5]) : 0.8;

  const std::string raw = read_file(argv[1]);
  std::vector<double> qv, xyzv, quatv, tipv, tquatv;
  if (!parse_array(raw, "q", &qv) || qv.size() != 18) return 3;
  if (!parse_array(raw, "xyz", &xyzv) || xyzv.size() != 3) return 4;
  if (!parse_array(raw, "quat", &quatv) || quatv.size() != 4) return 5;
  bool left = true;
  parse_boolish(raw, "left", &left);
  std::string name = "case";
  parse_string(raw, "name", &name);
  std::string gen;
  parse_string(raw, "generation", &gen);
  const bool has_tip = parse_array(raw, "tip_xyz", &tipv) && tipv.size() == 3 &&
                       parse_array(raw, "tip_quat", &tquatv) && tquatv.size() == 4;

  Config cfg = Config::load(std::string(KMPC_CONFIG_DIR) + "/joint_limits.yaml", std::string(KMPC_CONFIG_DIR) + "/kmpc_params.yaml");
  Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  CartesianTracker tracker(kin, cfg);
  tracker.set_line_pace(dq, accel);
  tracker.set_line_pose_pref(pose_pref);

  Vec18 q0;
  for (int i = 0; i < 18; ++i) q0[i] = qv[static_cast<std::size_t>(i)];
  Vec3 xyz(xyzv[0], xyzv[1], xyzv[2]);
  // Notes log tool quat; tracker expects flange quat for compute path.
  Vec4 quat = tool_to_flange_quat_xyzw(Vec4(quatv[0], quatv[1], quatv[2], quatv[3]));

  std::optional<LineTarget> tip;
  if (has_tip) {
    LineTarget t;
    t.xyz = Vec3(tipv[0], tipv[1], tipv[2]);
    t.quat_xyzw = tool_to_flange_quat_xyzw(Vec4(tquatv[0], tquatv[1], tquatv[2], tquatv[3]));
    t.ori_on = true;
    tip = t;
  }
  tracker.rearm_line(1, tip);
  tracker.set_line_reference(q0);

  const auto plan = tracker.prepare_line(q0, left, xyz, &quat);
  const int arm = left ? kSliceLeft.begin : kSliceRight.begin;
  const int j4 = arm + 3;

  std::ofstream out(argv[2]);
  out.setf(std::ios::fixed);
  out.precision(10);
  out << "{\n";
  out << "  \"name\":\"" << name << "\",\n";
  out << "  \"generation\":\"" << gen << "\",\n";
  out << "  \"approximate_replay\":true,\n";
  out << "  \"pose_pref_enable\":" << (pose_pref ? "true" : "false") << ",\n";
  out << "  \"dq_scale\":" << dq << ",\n";
  out << "  \"accel_s\":" << accel << ",\n";
  out << "  \"ok\":" << (plan.on ? "true" : "false") << ",\n";
  out << "  \"refused\":" << (tracker.line_refused() ? "true" : "false") << ",\n";
  out << "  \"reason\":\"" << (plan.reason.empty() ? "none" : plan.reason) << "\",\n";
  out << "  \"candidate\":\"" << plan.candidate << "\",\n";
  out << "  \"pose_pref_decision\":\"" << plan.pose_pref_decision << "\",\n";
  out << "  \"line_waist_locked\":" << (plan.waist_locked ? "true" : "false") << ",\n";
  out << "  \"cascade_forced_waist\":" << (plan.cascade_forced_waist ? "true" : "false") << ",\n";
  out << "  \"cascade_full_body\":" << (plan.cascade_full_body ? "true" : "false") << ",\n";
  out << "  \"max_arm_fold_deg\":" << rad_to_deg(plan.max_arm_fold) << ",\n";
  out << "  \"samples\":" << plan.samples.size() << ",\n";
  out << "  \"length_m\":" << plan.length_m << ",\n";
  out << "  \"build_ms\":" << plan.build_ms << ",\n";
  if (!plan.samples.empty()) {
    const auto& a = plan.samples.front();
    const auto& b = plan.samples.back();
    out << "  \"elbow_start_deg\":" << rad_to_deg(a[j4]) << ",\n";
    out << "  \"elbow_end_deg\":" << rad_to_deg(b[j4]) << ",\n";
    out << "  \"elbow_net_deg\":" << rad_to_deg(b[j4] - a[j4]) << ",\n";
    out << "  \"elbow_pref_end_err_deg\":"
        << rad_to_deg(std::abs(b[j4] - kmpc::test_fixture_q0()[j4])) << ",\n";
    double peak = 0.0, travel = 0.0;
    for (std::size_t i = 0; i < plan.samples.size(); ++i) {
      peak = std::max(peak, std::abs(plan.samples[i][j4] - kmpc::test_fixture_q0()[j4]));
      if (i) travel += std::abs(plan.samples[i][j4] - plan.samples[i - 1][j4]);
    }
    out << "  \"elbow_peak_pref_err_deg\":" << rad_to_deg(peak) << ",\n";
    out << "  \"elbow_path_deg\":" << rad_to_deg(travel) << ",\n";
    out << "  \"waist_path_deg\":" << rad_to_deg(path_len(plan.samples, 0, 4)) << ",\n";
    out << "  \"wrist_path_deg\":" << rad_to_deg(path_len(plan.samples, arm + 5, 2)) << ",\n";
    out << "  \"waist_j1_reversals\":" << count_reversals(plan.samples, 0, 1e-5) << ",\n";
    out << "  \"waist_j2_reversals\":" << count_reversals(plan.samples, 1, 1e-5) << ",\n";
    const Vec3 p_end = kin.fk_pos(b, left ? kEeFrameLeft : kEeFrameRight);
    out << "  \"end_mm\":" << (p_end - xyz).norm() * 1000.0 << ",\n";
    out << "  \"samples_q\":[";
    for (std::size_t i = 0; i < plan.samples.size(); ++i) {
      if (i) out << ",";
      out << "[";
      for (int j = 0; j < 18; ++j) {
        if (j) out << ",";
        out << plan.samples[i][j];
      }
      out << "]";
    }
    out << "]\n";
  } else {
    out << "  \"samples_q\":[]\n";
  }
  out << "}\n";
  std::cout << name << " candidate=" << plan.candidate
            << " waist_locked=" << plan.waist_locked
            << " pose=" << plan.pose_pref_decision
            << " fold_deg=" << rad_to_deg(plan.max_arm_fold) << "\n";
  return plan.on ? 0 : 1;
}
