// Certified planning latency bench. Label host by actual machine arch so Jetson
// aarch64 runs are not mis-tagged as x86_control_only.
// Default hard budget is production kPlannerBudgetMs (1s). The 10 ms figure is
// only a soft statistic (certified_before_10ms_*). Use --hard_deadline_ms to
// intentionally hard-abort the optimizer early for isolated experiments.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

#include "kmpc/config.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/trajectory_package.hpp"
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

struct PhaseAcc {
  std::vector<double> total;
  std::vector<double> seed;
  std::vector<double> sqp;
  std::vector<double> densify;
  std::vector<double> time_law;
  std::vector<double> certify;
  std::vector<double> report;
};

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
    st.orientation_tol_rad = 0.05;
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

int main(int argc, char** argv) {
  int warmup = 3;
  int reps = 60;
  // Soft 10 ms statistic only — never the default optimizer abort.
  double soft_stat_ms = kmpc::kPlannerPerfStatMs;
  // Production hard budget unless --hard_deadline_ms is set explicitly.
  double hard_deadline_ms = kmpc::kPlannerBudgetMs;
  bool hard_deadline_explicit = false;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--warmup") {
      warmup = std::max(0, std::atoi(argv[++i]));
    } else if (a == "--reps") {
      reps = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--deadline_ms") {
      // Backward-compatible alias for the soft 10 ms statistic only.
      soft_stat_ms = std::atof(argv[++i]);
    } else if (a == "--hard_deadline_ms") {
      hard_deadline_ms = std::atof(argv[++i]);
      hard_deadline_explicit = true;
    } else if (a == "--soft_stat_ms") {
      soft_stat_ms = std::atof(argv[++i]);
    }
  }
  (void)hard_deadline_explicit;

  const auto config = kmpc::Config::load(
      std::string(KMPC_CONFIG_DIR) + "/joint_limits.yaml",
      std::string(KMPC_CONFIG_DIR) + "/q_pref.yaml",
      std::string(KMPC_CONFIG_DIR) + "/kmpc_params.yaml");
  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::Kinematics kin(urdf);
  kmpc::WholeBodyGuards guards(config.params, config.q_pref[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);
  std::string weight_why;
  auto weights = kmpc::load_trajectory_quality_weights(
      std::string(KMPC_CONFIG_DIR) + "/trajectory_quality.yaml",
      config.limits.dq_max, &weight_why);
  for (int j = 0; j < kmpc::kNDof; ++j) {
    const double dq = std::max(config.limits.dq_max[j], 1e-3);
    weights.motion_scale[j] = 1.0 / dq;
  }
  const kmpc::Vec18 q0 = config.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);
  kmpc::FastTrajectoryOptimizer optimizer(kin, &guards);

  std::string host_tag = "unknown_host";
#if defined(__linux__)
  utsname uts{};
  if (uname(&uts) == 0) {
    const std::string mach = uts.machine;
    if (mach.find("aarch64") != std::string::npos ||
        mach.find("arm64") != std::string::npos) {
      host_tag = "jetson_offline_aarch64";
    } else if (mach.find("x86_64") != std::string::npos ||
               mach.find("amd64") != std::string::npos) {
      host_tag = "x86_control_only";
    } else {
      host_tag = mach;
    }
  }
#endif
  std::cout << "bench_certified_planning host=" << host_tag;
  if (host_tag == "x86_control_only") {
    std::cout << " (NOT jetson proof)";
  }
  std::cout << " warmup=" << warmup << " reps=" << reps
            << " hard_deadline_ms=" << hard_deadline_ms
            << " soft_stat_ms=" << soft_stat_ms
            << " weights=" << (weight_why.empty() ? "production_yaml" : weight_why)
            << "\n";
  std::cout << "note=synthetic_reaches_are_not_field_success_proxy\n";

  const Case cases[] = {
      {"reach_5cm_1stage", 1, 0.05},
      {"reach_15cm_1stage", 1, 0.15},
      {"reach_24cm_1stage", 1, 0.24},
      {"reach_15cm_2stage", 2, 0.15},
      {"reach_15cm_3stage", 3, 0.15},
  };

  int certified_before_soft = 0;
  int total_ok = 0;
  int total_deadline_reject = 0;
  int total_reps_all = 0;

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

    PhaseAcc acc;
    int ok_n = 0;
    int under_soft = 0;
    int deadline_reject = 0;
    int cert_fail = 0;

    for (int i = 0; i < warmup + reps; ++i) {
      kmpc::MotionTask task = make_reach(q0, true, goals);
      kmpc::seal_motion_task_identity(&task, urdf, /*is_xml=*/false, policy);
      kmpc::RobotState state;
      state.q = task.q;
      state.dq_known = false;
      state.fresh = true;

      const auto t0 = std::chrono::steady_clock::now();
      const auto res =
          optimizer.plan(task, state, policy, weights, hard_deadline_ms);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (i < warmup) {
        continue;
      }
      ++total_reps_all;
      acc.total.push_back(ms);
      acc.seed.push_back(res.seed_ik_ms);
      acc.sqp.push_back(res.sqp_ms);
      acc.densify.push_back(res.densify_ms);
      acc.time_law.push_back(res.time_law_ms);
      acc.certify.push_back(res.certify_ms);
      acc.report.push_back(res.report_ms);

      const bool deadline_rej =
          res.deadline_hit || res.reason == "deadline" ||
          (!res.ok && res.reason.find("deadline") != std::string::npos);
      if (deadline_rej) {
        ++deadline_reject;
        ++total_deadline_reject;
      }
      if (res.ok) {
        ++ok_n;
        ++total_ok;
        if (ms <= soft_stat_ms) {
          ++under_soft;
          ++certified_before_soft;
        }
      } else if (!deadline_rej) {
        ++cert_fail;
      }
    }

    const double success_rate =
        acc.total.empty()
            ? 0.0
            : 100.0 * static_cast<double>(ok_n) /
                  static_cast<double>(acc.total.size());
    const double soft_rate =
        acc.total.empty()
            ? 0.0
            : 100.0 * static_cast<double>(under_soft) /
                  static_cast<double>(acc.total.size());
    const double deadline_rate =
        acc.total.empty()
            ? 0.0
            : 100.0 * static_cast<double>(deadline_reject) /
                  static_cast<double>(acc.total.size());
    std::cout << "case=" << c.name << " stages=" << c.stages
              << " reach_m=" << c.reach_m << " ok=" << ok_n << "/" << reps
              << " success_rate=" << success_rate << "%"
              << " planning_cpu_ms_p50=" << percentile(acc.total, 0.50)
              << " p95=" << percentile(acc.total, 0.95)
              << " p99=" << percentile(acc.total, 0.99)
              << " certified_before_" << soft_stat_ms
              << "ms_rate=" << soft_rate << "%"
              << " budget_timeout_rate=" << deadline_rate << "%"
              << " cert_fail=" << cert_fail << "\n";
    std::cout << "  phases_p50 seed=" << percentile(acc.seed, 0.50)
              << " sqp=" << percentile(acc.sqp, 0.50)
              << " densify=" << percentile(acc.densify, 0.50)
              << " time=" << percentile(acc.time_law, 0.50)
              << " certify=" << percentile(acc.certify, 0.50)
              << " report=" << percentile(acc.report, 0.50) << "\n";
    std::cout << "  phases_p95 seed=" << percentile(acc.seed, 0.95)
              << " sqp=" << percentile(acc.sqp, 0.95)
              << " densify=" << percentile(acc.densify, 0.95)
              << " time=" << percentile(acc.time_law, 0.95)
              << " certify=" << percentile(acc.certify, 0.95)
              << " report=" << percentile(acc.report, 0.95) << "\n";
    std::cout << "  phases_p99 seed=" << percentile(acc.seed, 0.99)
              << " sqp=" << percentile(acc.sqp, 0.99)
              << " densify=" << percentile(acc.densify, 0.99)
              << " time=" << percentile(acc.time_law, 0.99)
              << " certify=" << percentile(acc.certify, 0.99)
              << " report=" << percentile(acc.report, 0.99) << "\n";
  }

  const double overall_success =
      total_reps_all == 0
          ? 0.0
          : 100.0 * static_cast<double>(total_ok) /
                static_cast<double>(total_reps_all);
  const double overall_soft =
      total_ok == 0 ? 0.0
                    : 100.0 * static_cast<double>(certified_before_soft) /
                          static_cast<double>(total_ok);
  const double deadline_overall =
      total_reps_all == 0
          ? 0.0
          : 100.0 * static_cast<double>(total_deadline_reject) /
                static_cast<double>(total_reps_all);
  std::cout << "overall_success_rate=" << overall_success << "% ("
            << host_tag << ") overall_certified_before_" << soft_stat_ms
            << "ms_rate=" << overall_soft
            << "% overall_budget_timeout_rate=" << deadline_overall << "%\n";
  return 0;
}
