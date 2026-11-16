#pragma once
// Resident planner thread for unified_multistage_v2.
// Latest-generation-wins queue. Owns Kinematics + FastTrajectoryOptimizer
// workspace. submit_snapshot / try_take never block 50/250 Hz callers.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "kmpc/config.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/trajectory_cost.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace kmpc {

struct PlannerJob {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MotionTask task;
  RobotState state;
  MotionConstraintPolicy policy;
  TrajectoryCostWeights weights;
  double deadline_ms = kPlannerBudgetMs;
  uint64_t seq = 0;
};

class PlannerWorker {
 public:
  // urdf: worker-owned Kinematics (not shared with the control thread).
  // urdf_is_xml: true when `urdf` is the robot_description XML string.
  PlannerWorker(std::string urdf, Config config, bool urdf_is_xml = false);
  ~PlannerWorker();

  PlannerWorker(const PlannerWorker&) = delete;
  PlannerWorker& operator=(const PlannerWorker&) = delete;

  void start();
  void stop();  // join; safe to call multiple times

  // Non-blocking. Replaces any pending job (latest generation wins).
  void submit_snapshot(const MotionTask& task, const RobotState& state);
  void submit_snapshot(const MotionTask& task, const RobotState& state,
                       const MotionConstraintPolicy& policy,
                       const TrajectoryCostWeights& weights,
                       double deadline_ms);

  // Non-blocking. Returns true and fills out when a newer result is ready.
  bool try_take(FastPlanResult& out);

  // Drop pending job and any unconsumed result (task replace / cancel).
  void clear_pending_and_result();

  bool running() const { return running_.load(std::memory_order_acquire); }
  uint64_t submitted_seq() const {
    return submitted_seq_.load(std::memory_order_relaxed);
  }
  uint64_t completed_seq() const {
    return completed_seq_.load(std::memory_order_relaxed);
  }
  const TrajectoryCostWeights& default_weights() const;
  const std::string& quality_load_note() const;

 private:
  void loop();

  std::string urdf_path_;
  bool urdf_is_xml_ = false;
  Config config_;
  MotionConstraintPolicy default_policy_;
  TrajectoryCostWeights default_weights_;
  std::string quality_load_note_;

  std::unique_ptr<Kinematics> kin_;
  std::unique_ptr<WholeBodyGuards> guards_;
  std::unique_ptr<FastTrajectoryOptimizer> optimizer_;

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_requested_ = false;
  bool has_pending_ = false;
  PlannerJob pending_;
  bool has_result_ = false;
  FastPlanResult result_;

  std::atomic<bool> running_{false};
  std::atomic<uint64_t> submitted_seq_{0};
  std::atomic<uint64_t> completed_seq_{0};
  std::thread thread_;
};

}  // namespace kmpc
