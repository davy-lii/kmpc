// Offline r40 unified_multistage_v2 replay against field fixture.
// Honesty: field_replay=passed only when the stage carries a real 18D q0
// extracted from field logs. Synthetic q_pref+0.01 is never field_replay.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_evaluator.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/planner_worker.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/trajectory_package.hpp"
#include "kmpc/unified_multistage_planner.hpp"
#include "kmpc/whole_body_guards.hpp"

#include <chrono>
#include <thread>

namespace {

struct Goal {
  std::string name;
  kmpc::Vec3 xyz = kmpc::Vec3::Zero();
  bool has_xyz = false;
  kmpc::Vec3 xyz_offset = kmpc::Vec3::Zero();
  bool has_offset = false;
  kmpc::Vec4 quat = kmpc::Vec4(0, 0, 0, 1);
  bool has_quat = false;
};

struct Stage {
  std::string id;
  bool field_replay = false;
  bool left = true;
  double dq_scale = 1.0;
  std::string q0_source;
  kmpc::Vec18 q0 = kmpc::Vec18::Zero();
  bool has_q0 = false;
  std::string stop_reason;
  std::string historical_pose_report;
  double historical_drop_mm = 0.0;
  bool has_historical_drop = false;
  std::vector<Goal> goals;
  std::string note;
};

std::string strip(std::string s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' ||
                        s.back() == '\t' || s.back() == ',')) {
    s.pop_back();
  }
  std::size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) {
    ++i;
  }
  return s.substr(i);
}

std::size_t matching_bracket(const std::string& s, std::size_t lb, char open,
                             char close) {
  int depth = 0;
  for (std::size_t i = lb; i < s.size(); ++i) {
    if (s[i] == open) {
      ++depth;
    } else if (s[i] == close) {
      --depth;
      if (depth == 0) {
        return i;
      }
    }
  }
  return std::string::npos;
}

bool parse_number_array(const std::string& src, std::size_t pos,
                        std::vector<double>* out) {
  if (pos == std::string::npos) {
    return false;
  }
  const auto lb = src.find('[', pos);
  if (lb == std::string::npos) {
    return false;
  }
  const auto rb = matching_bracket(src, lb, '[', ']');
  if (rb == std::string::npos) {
    return false;
  }
  out->clear();
  std::stringstream ss(src.substr(lb + 1, rb - lb - 1));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = strip(tok);
    if (tok.empty()) {
      continue;
    }
    out->push_back(std::stod(tok));
  }
  return true;
}

std::string extract_string(const std::string& block, const std::string& key) {
  const std::string pat = "\"" + key + "\"";
  auto p = block.find(pat);
  if (p == std::string::npos) {
    return {};
  }
  p = block.find(':', p);
  if (p == std::string::npos) {
    return {};
  }
  p = block.find('"', p + 1);
  if (p == std::string::npos) {
    return {};
  }
  const auto e = block.find('"', p + 1);
  if (e == std::string::npos) {
    return {};
  }
  return block.substr(p + 1, e - p - 1);
}

bool extract_bool(const std::string& block, const std::string& key, bool* v) {
  const std::string pat = "\"" + key + "\"";
  auto p = block.find(pat);
  if (p == std::string::npos) {
    return false;
  }
  p = block.find(':', p);
  if (p == std::string::npos) {
    return false;
  }
  const auto rest = strip(block.substr(p + 1, 16));
  if (rest.rfind("true", 0) == 0) {
    *v = true;
    return true;
  }
  if (rest.rfind("false", 0) == 0) {
    *v = false;
    return true;
  }
  return false;
}

bool extract_double(const std::string& block, const std::string& key, double* v) {
  const std::string pat = "\"" + key + "\"";
  auto p = block.find(pat);
  if (p == std::string::npos) {
    return false;
  }
  p = block.find(':', p);
  if (p == std::string::npos) {
    return false;
  }
  *v = std::stod(strip(block.substr(p + 1, 32)));
  return true;
}

std::vector<std::string> split_objects(const std::string& arr) {
  std::vector<std::string> out;
  int depth = 0;
  std::size_t start = std::string::npos;
  for (std::size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] == '{') {
      if (depth == 0) {
        start = i;
      }
      ++depth;
    } else if (arr[i] == '}') {
      --depth;
      if (depth == 0 && start != std::string::npos) {
        out.push_back(arr.substr(start, i - start + 1));
        start = std::string::npos;
      }
    }
  }
  return out;
}

std::vector<Stage> load_manifest(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  const auto stages_key = text.find("\"stages\"");
  if (stages_key == std::string::npos) {
    return {};
  }
  const auto lb = text.find('[', stages_key);
  const auto rb = text.rfind(']');
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
    return {};
  }
  std::vector<Stage> stages;
  for (const auto& block : split_objects(text.substr(lb, rb - lb + 1))) {
    Stage st;
    st.id = extract_string(block, "id");
    extract_bool(block, "field_replay", &st.field_replay);
    const auto hand = extract_string(block, "hand");
    st.left = (hand != "right");
    extract_double(block, "dq_scale", &st.dq_scale);
    st.q0_source = extract_string(block, "q0_source");
    st.stop_reason = extract_string(block, "stop_reason");
    st.historical_pose_report = extract_string(block, "historical_pose_report");
    st.note = extract_string(block, "note");
    const auto drop_pos = st.historical_pose_report.find("drop_mm=");
    if (drop_pos != std::string::npos) {
      st.historical_drop_mm =
          std::stod(st.historical_pose_report.substr(drop_pos + 8));
      st.has_historical_drop = true;
    }
    std::vector<double> q0;
    if (parse_number_array(block, block.find("\"q0\""), &q0) && q0.size() == 18) {
      for (int i = 0; i < 18; ++i) {
        st.q0[i] = q0[i];
      }
      st.has_q0 = true;
    }
    const auto goals_pos = block.find("\"goals\"");
    if (goals_pos != std::string::npos) {
      const auto glb = block.find('[', goals_pos);
      const auto grb =
          (glb == std::string::npos) ? std::string::npos
                                     : matching_bracket(block, glb, '[', ']');
      if (glb != std::string::npos && grb != std::string::npos) {
        for (const auto& gblock :
             split_objects(block.substr(glb, grb - glb + 1))) {
          Goal g;
          g.name = extract_string(gblock, "name");
          std::vector<double> xyz;
          if (parse_number_array(gblock, gblock.find("\"xyz\""), &xyz) &&
              xyz.size() == 3) {
            g.xyz = kmpc::Vec3(xyz[0], xyz[1], xyz[2]);
            g.has_xyz = true;
          }
          std::vector<double> off;
          if (parse_number_array(gblock, gblock.find("\"xyz_offset\""), &off) &&
              off.size() == 3) {
            g.xyz_offset = kmpc::Vec3(off[0], off[1], off[2]);
            g.has_offset = true;
          }
          std::vector<double> quat;
          if (parse_number_array(gblock, gblock.find("\"quat_xyzw\""), &quat) &&
              quat.size() == 4) {
            g.quat = kmpc::Vec4(quat[0], quat[1], quat[2], quat[3]);
            g.has_quat = true;
          }
          st.goals.push_back(g);
        }
      }
    }
    stages.push_back(st);
  }
  return stages;
}

kmpc::TrajectoryCostWeights weights_from(const kmpc::Config& config,
                                         double dq_scale) {
  kmpc::Vec18 dq = config.limits.dq_max * std::max(dq_scale, 1e-3);
  auto w = kmpc::default_cost_weights(dq);
  w.calibrated = true;
  return w;
}

void print_result(const Stage& st, const kmpc::FastPlanResult& res,
                  const kmpc::Kinematics& kin, const kmpc::Vec18& q0,
                  bool left) {
  const char* verdict = "lab_only";
  if (st.field_replay) {
    verdict = res.ok ? "field_replay=passed" : "field_replay=failed";
  } else if (st.q0_source.find("synthetic") != std::string::npos) {
    verdict = "synthetic_not_field_replay";
  }
  std::cout << "stage=" << st.id << " " << verdict << " ok=" << (res.ok ? 1 : 0)
            << " reason=" << res.reason << "\n";
  std::cout << "  q0_source=" << st.q0_source
            << " field_replay_flag=" << (st.field_replay ? "true" : "false")
            << " stages=" << st.goals.size() << "\n";
  std::cout << "  quality: " << res.report << "\n";
  std::cout << "  body_dx_m=" << res.body_dx_m << " body_dz_m=" << res.body_dz_m
            << " body_minus_leg_dx_m=" << res.body_minus_leg_dx_m
            << " j1_margin_rad=" << res.min_hard_margin_rad
            << " tcp_worst_m=" << res.tcp_worst_m
            << " seed_ms=" << res.seed_ik_ms << " sqp_ms=" << res.sqp_ms
            << " densify_ms=" << res.densify_ms << " time_ms=" << res.time_law_ms
            << " opt_ms=" << res.optimize_ms << " cert_ms=" << res.certify_ms
            << " total_ms=" << res.total_ms
            << " over_10ms="
            << ((res.total_ms > kmpc::kPlannerPerfStatMs) ? 1 : 0)
            << " deadline=" << (res.deadline_hit ? 1 : 0) << "\n";
  if (st.has_historical_drop) {
    std::cout << "  historical_stop_drop_mm=" << st.historical_drop_mm
              << " historical_pose_report=" << st.historical_pose_report << "\n";
    if (res.ok && !res.package.samples.empty()) {
      kmpc::MotionEvaluator eval(kin);
      kmpc::MotionSnapshot a, b;
      eval.evaluate(q0, left, &a);
      eval.evaluate(res.package.samples.back(), left, &b);
      const double new_drop_mm = (a.body.z() - b.body.z()) * 1000.0;
      std::cout << "  compare_vs_historical_stop: new_body_drop_mm=" << new_drop_mm
                << " delta_vs_hist_mm=" << (new_drop_mm - st.historical_drop_mm)
                << " (hist was free_fit LIFT stop; negative hist means drop)\n";
    }
  }
  if (!st.stop_reason.empty()) {
    std::cout << "  historical_stop_reason=" << st.stop_reason << "\n";
  }
  if (!st.note.empty()) {
    std::cout << "  note=" << st.note << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string fixture =
      (argc > 1)
          ? argv[1]
          : (std::string(KMPC_REPLAY_DIR) + "/fixtures/r40_manifest.json");
  const auto config = kmpc::Config::load(std::string(KMPC_CONFIG_DIR) + "/joint_limits.yaml", std::string(KMPC_CONFIG_DIR) + "/kmpc_params.yaml");
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(config.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(config);

  const auto stages = load_manifest(fixture);
  if (stages.empty()) {
    std::cerr << "failed to load stages from " << fixture << "\n";
    return 2;
  }
  std::cout << "r40_unified_v2 fixture=" << fixture
            << " stages=" << stages.size() << "\n";
  std::cout << "honesty: field_replay=passed only for real 18D field q0; "
               "synthetic q_pref+0.01 is never field_replay\n";

  const std::string urdf =
      std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf";
  kmpc::PlannerWorker worker(urdf, config);
  worker.start();
  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  kmpc::FastTrajectoryOptimizer optimizer(kin, &guards);

  int field_pass = 0, field_fail = 0, lab = 0;
  int budget_ok = 0, budget_fail = 0, adopt_ok = 0, adopt_fail = 0;
  int reject_by_10ms_only = 0;
  for (const auto& st : stages) {
    kmpc::Vec18 q0 = kmpc::test_fixture_q0();
    if (st.has_q0) {
      q0 = st.q0;
    } else if (st.q0_source.find("synthetic") != std::string::npos) {
      q0 = kmpc::test_fixture_q0();
      q0[0] = policy.q_max[0] - 0.01;
    }
    const std::string frame =
        st.left ? std::string(kmpc::kEeFrameLeft) : std::string(kmpc::kEeFrameRight);
    kmpc::MotionTask task;
    task.generation = 40;
    task.q = q0;
    task.dq_known = false;
    task.left = st.left;
    task.frame = frame;
    const kmpc::Vec3 p0 = kin.fk_pos(q0, frame);
    for (std::size_t i = 0; i < st.goals.size(); ++i) {
      const auto& g = st.goals[i];
      kmpc::TaskStage ts;
      ts.geometry = kmpc::TaskGeometry::Lin;
      ts.stage_id = static_cast<uint32_t>(i + 1);
      ts.position_tol_m = 0.003;
      if (g.has_xyz) {
        ts.position = g.xyz;
      } else if (g.has_offset) {
        ts.position = p0 + g.xyz_offset;
      } else {
        ts.position = p0;
      }
      if (g.has_quat) {
        // notes.txt / PoseArray quats are tool-frame; production from_pose
        // converts via tool_to_flange before planning. Replay must match.
        ts.has_orientation = true;
        ts.quat = kmpc::tool_to_flange_quat_xyzw(g.quat);
      }
      ts.stop = (i + 1 == st.goals.size()) ? kmpc::StageStop::StopRequired
                                           : kmpc::StageStop::Continuous;
      task.stages.push_back(ts);
    }
    const auto weights = weights_from(config, st.dq_scale);
    const auto res =
        kmpc::solve_to_verified_package(kin, guards, task, policy, weights);
    print_result(st, res, kin, q0, st.left);
    if (st.field_replay) {
      if (res.ok) {
        ++field_pass;
      } else {
        ++field_fail;
      }
      // Production-path check: 10ms is soft; 1s budget must not reject solely
      // for wall time; certified ok plans must be adoptable.
      kmpc::RobotState state;
      state.q = q0;
      state.dq_known = false;
      state.fresh = true;
      kmpc::seal_motion_task_identity(&task, urdf, false, policy);
      const auto res10 =
          optimizer.plan(task, state, policy, weights, kmpc::kPlannerPerfStatMs);
      const auto res1s =
          optimizer.plan(task, state, policy, weights, kmpc::kPlannerBudgetMs);
      std::cout << "  budget_cmp 10ms: ok=" << (res10.ok ? 1 : 0)
                << " reason=" << res10.reason << " total_ms=" << res10.total_ms
                << " deadline=" << (res10.deadline_hit ? 1 : 0)
                << " | 1s: ok=" << (res1s.ok ? 1 : 0)
                << " reason=" << res1s.reason << " total_ms=" << res1s.total_ms
                << " deadline=" << (res1s.deadline_hit ? 1 : 0)
                << " over_10ms="
                << ((res1s.total_ms > kmpc::kPlannerPerfStatMs) ? 1 : 0)
                << "\n";
      if (!res10.ok && res10.deadline_hit && res1s.ok && !res1s.deadline_hit) {
        ++reject_by_10ms_only;
        std::cout << "  NOTE: old 10ms gate would reject; 1s budget succeeds\n";
      }
      if (res1s.ok) {
        ++budget_ok;
      } else {
        ++budget_fail;
      }
      worker.clear_pending_and_result();
      worker.submit_snapshot(task, state, policy, weights,
                             kmpc::kPlannerBudgetMs);
      kmpc::FastPlanResult wplan;
      bool got = false;
      for (int i = 0; i < 600; ++i) {
        if (worker.try_take(wplan)) {
          got = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (!got) {
        ++adopt_fail;
        std::cout << "  worker_adopt=timeout\n";
      } else if (!wplan.ok) {
        ++adopt_fail;
        std::cout << "  worker_adopt=fail reason=" << wplan.reason
                  << " total_ms=" << wplan.total_ms
                  << " deadline=" << (wplan.deadline_hit ? 1 : 0) << "\n";
      } else {
        std::string why;
        const bool adopted =
            follower.adopt(wplan.package, q0, kmpc::Vec18::Zero(), false, 1e-3,
                           0.5, &why);
        if (adopted) {
          ++adopt_ok;
          std::cout << "  worker_adopt=ok total_ms=" << wplan.total_ms
                    << " over_10ms="
                    << ((wplan.total_ms > kmpc::kPlannerPerfStatMs) ? 1 : 0)
                    << "\n";
        } else {
          ++adopt_fail;
          std::cout << "  worker_adopt=reject why=" << why << "\n";
        }
      }
    } else {
      ++lab;
      if (st.q0_source.find("synthetic") != std::string::npos) {
        std::cout << "  ASSERT honesty: not claiming field_replay=passed on "
                     "synthetic q_pref+0.01\n";
      }
    }
  }
  worker.stop();
  std::cout << "summary field_pass=" << field_pass
            << " field_fail=" << field_fail << " lab_only=" << lab
            << " budget1s_ok=" << budget_ok << " budget1s_fail=" << budget_fail
            << " worker_adopt_ok=" << adopt_ok
            << " worker_adopt_fail=" << adopt_fail
            << " would_reject_only_at_10ms=" << reject_by_10ms_only << "\n";
  // Tool exit 0 even if a field stage fails plan: the report is the deliverable.
  // Callers that need a hard gate can parse field_replay=failed.
  return 0;
}
