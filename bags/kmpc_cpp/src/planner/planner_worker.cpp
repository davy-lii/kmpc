#include "kmpc/planner_worker.hpp"

namespace kmpc {

PlannerWorker::PlannerWorker(std::string urdf, Config config, bool urdf_is_xml)
    : urdf_path_(std::move(urdf)),
      urdf_is_xml_(urdf_is_xml),
      config_(std::move(config)) {
  default_policy_ = policy_from_config(config_);
  default_weights_.motion_scale = Vec18::Ones();
  for (int j = 0; j < kNDof; ++j) {
    const double dq = std::max(config_.limits.dq_max[j], 1e-3);
    default_weights_.motion_scale[j] = 1.0 / dq;
  }
  default_weights_.calibrated = true;
}

PlannerWorker::~PlannerWorker() { stop(); }

void PlannerWorker::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_requested_ = false;
    has_pending_ = false;
    has_result_ = false;
  }
  if (urdf_is_xml_) {
    kin_ = std::make_unique<Kinematics>(urdf_path_,
                                        Kinematics::UrdfInput::Xml);
  } else {
    kin_ = std::make_unique<Kinematics>(urdf_path_);
  }
  guards_ = std::make_unique<WholeBodyGuards>(config_.params,
                                              config_.q_pref[kIdxYaw]);
  optimizer_ = std::make_unique<FastTrajectoryOptimizer>(*kin_, guards_.get());
  thread_ = std::thread([this] { loop(); });
}

void PlannerWorker::stop() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_requested_ = true;
    has_pending_ = false;
  }
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  running_.store(false, std::memory_order_release);
  optimizer_.reset();
  guards_.reset();
  kin_.reset();
}

void PlannerWorker::submit_snapshot(const MotionTask& task,
                                    const RobotState& state) {
  submit_snapshot(task, state, default_policy_, default_weights_, 10.0);
}

void PlannerWorker::submit_snapshot(const MotionTask& task,
                                    const RobotState& state,
                                    const MotionConstraintPolicy& policy,
                                    const TrajectoryCostWeights& weights,
                                    double deadline_ms) {
  PlannerJob job;
  job.task = task;
  job.state = state;
  job.policy = policy;
  job.weights = weights;
  job.deadline_ms = deadline_ms;
  job.seq = submitted_seq_.fetch_add(1, std::memory_order_relaxed) + 1;
  {
    std::lock_guard<std::mutex> lock(mu_);
    // Latest-generation-wins: overwrite any unconsumed pending job.
    pending_ = std::move(job);
    has_pending_ = true;
  }
  cv_.notify_one();
}

bool PlannerWorker::try_take(FastPlanResult& out) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!has_result_) {
    return false;
  }
  out = std::move(result_);
  has_result_ = false;
  return true;
}

void PlannerWorker::loop() {
  while (true) {
    PlannerJob job;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return stop_requested_ || has_pending_; });
      if (stop_requested_ && !has_pending_) {
        break;
      }
      job = std::move(pending_);
      has_pending_ = false;
    }
    FastPlanResult result;
    if (optimizer_ != nullptr) {
      // Align task start with the measured snapshot (caller may leave task.q
      // stale). Generation is already on the task.
      MotionTask task = job.task;
      task.q = job.state.q;
      task.dq = job.state.dq;
      task.dq_known = job.state.dq_known;
      result = optimizer_->plan(task, job.state, job.policy, job.weights,
                                job.deadline_ms);
    } else {
      result.ok = false;
      result.reason = "worker_not_ready";
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      // Drop stale results if a newer job was already submitted.
      if (job.seq >= submitted_seq_.load(std::memory_order_relaxed) ||
          !has_pending_) {
        // Keep result if it is still the latest completed seq, or if nothing
        // newer is pending. Always publish; try_take consumers check generation.
        result_ = std::move(result);
        has_result_ = true;
        completed_seq_.store(job.seq, std::memory_order_relaxed);
      }
      if (stop_requested_ && !has_pending_) {
        break;
      }
    }
  }
}

}  // namespace kmpc
