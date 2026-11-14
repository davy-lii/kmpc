// x86 control-only bench for certified planning latency.
// Numbers here are NOT Jetson proof. Fill Jetson results in the issue draft note.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/unified_multistage_planner.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) {
    return 0.0;
  }
  std::sort(xs.begin(), xs.end());
  const double idx = p * static_cast<double>(xs.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(idx);
  const std::size_t hi = std::min(lo + 1, xs.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return xs[lo] * (1.0 - frac) + xs[hi] * frac;
}

kmpc::MotionTask make_reach(const kmpc::Vec18& q0, bool left,
                            const std::vector<kmpc::Vec3>& goals) {
  kmpc::MotionTask task;
  task.generation = 100;
  task.q = q0;
  task.dq_known = false;
  task.left = left;
  task.frame =
      left ? std::string(kmpc::kEeFrameLeft) : std::string(kmpc::kEeFrameRight);
  for (std::size_t i = 0; i < goals.size(); ++i) {
    kmpc::TaskStage st;
    st.geometry = kmpc::TaskGeometry::Lin;
    st.position = goals[i];
    st.position_tol_m = 0.003;
    st.stage_id = static_cast<uint32_t>(i + 1);
    st.stop = (i + 1 == goals.size()) ? kmpc::StageStop::StopRequired
                                      : kmpc::StageStop::Continuous;
    task.stages.push_back(st);
  }
  return task;
}

struct Case {
  const char* name;
  int stages;
  double reach_m;
};

}  // namespace

int main() {
  const auto config = kmpc::Config::load(
      std::string(KMPC_CONFIG_DIR) + "/joint_limits.yaml",
      std::string(KMPC_CONFIG_DIR) + "/q_pref.yaml",
      std::string(KMPC_CONFIG_DIR) + "/kmpc_params.yaml");
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  const auto weights = kmpc::default_cost_weights(config.limits.dq_max);
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);

  std::cout << "bench_certified_planning host=x86_control_only "
               "(NOT jetson proof)\n";

  const Case cases[] = {
      {"reach_5cm_1stage", 1, 0.05},
      {"reach_15cm_1stage", 1, 0.15},
      {"reach_24cm_1stage", 1, 0.24},
      {"reach_15cm_2stage", 2, 0.15},
      {"reach_15cm_3stage", 3, 0.15},
  };

  const int warmup = 2;
  const int reps = 25;
  int certified_before_10 = 0;
  int total_ok = 0;

  for (const auto& c : cases) {
    std::vector<kmpc::Vec3> goals;
    if (c.stages == 1) {
      goals.push_back(p0 + kmpc::Vec3(c.reach_m, 0.0, 0.0));
    } else if (c.stages == 2) {
      goals.push_back(p0 + kmpc::Vec3(c.reach_m * 0.6, 0.0, 0.0));
      goals.push_back(p0 + kmpc::Vec3(c.reach_m, 0.0, -0.04));
    } else {
      goals.push_back(p0 + kmpc::Vec3(c.reach_m * 0.4, 0.0, 0.0));
      goals.push_back(p0 + kmpc::Vec3(c.reach_m * 0.7, 0.0, -0.05));
      goals.push_back(p0 + kmpc::Vec3(c.reach_m, 0.0, 0.02));
    }
    const auto task = make_reach(q0, true, goals);
    std::vector<double> cpu_ms;
    cpu_ms.reserve(reps);
    int ok_n = 0;
    int under_10 = 0;
    for (int i = 0; i < warmup + reps; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto res =
          kmpc::solve_to_verified_package(kin, guards, task, policy, weights);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (i < warmup) {
        continue;
      }
      cpu_ms.push_back(ms);
      if (res.ok) {
        ++ok_n;
        ++total_ok;
        if (ms <= 10.0) {
          ++under_10;
          ++certified_before_10;
        }
      }
    }
    const double rate =
        cpu_ms.empty() ? 0.0
                       : 100.0 * static_cast<double>(under_10) /
                             static_cast<double>(cpu_ms.size());
    std::cout << "case=" << c.name << " stages=" << c.stages
              << " reach_m=" << c.reach_m << " ok=" << ok_n << "/" << reps
              << " planning_cpu_ms_p50=" << percentile(cpu_ms, 0.50)
              << " p95=" << percentile(cpu_ms, 0.95)
              << " p99=" << percentile(cpu_ms, 0.99)
              << " certified_before_10ms_rate=" << rate << "%\n";
  }

  const double overall = total_ok == 0
                             ? 0.0
                             : 100.0 * static_cast<double>(certified_before_10) /
                                   static_cast<double>(total_ok);
  std::cout << "overall_certified_before_10ms_rate=" << overall
            << "% (x86_control_only; jetson TBD)\n";
  return 0;
}
