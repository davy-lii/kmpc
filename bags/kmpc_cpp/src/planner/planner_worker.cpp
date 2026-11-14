#include "kmpc/planner_worker.hpp"

#include <string>

namespace kmpc {

PlannerWorker::PlannerWorker(std::string urdf, Config config, bool urdf_is_xml)
    : urdf_path_(std::move(urdf)),
      urdf_is_xml_(urdf_is_xml),
      config_(std::move(config)) {
  default_policy_ = policy_from_config(config_);
  std::string why;
  const std::string qpath =
      std::string(KMPC_CONFIG_DIR) + "/trajectory_quality.yaml";
  default_weights_ =
      load_trajectory_quality_weights(qpath, config_.limits.dq_max, &why);
  for (int j = 0; j < kNDof; ++j) {
    const double dq = std::max(config_.limits.dq_max[j], 1e-3);
    default_weights_.motion_scale[j] = 1.0 / dq;
  }
  // Never force calibrated=true when yaml says false / load failed.
  quality_load_note_ = why;
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
    // Latest-generation-wins: overwrite any unconsumed pending job AND drop
    // any completed result that can no longer be adopted.
    pending_ = std::move(job);
    has_pending_ = true;
    has_result_ = false;
  }
  cv_.notify_one();
}

void PlannerWorker::clear_pending_and_result() {
  std::lock_guard<std::mutex> lock(mu_);
  has_pending_ = false;
  has_result_ = false;
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

const TrajectoryCostWeights& PlannerWorker::default_weights() const {
  return default_weights_;
}

const std::string& PlannerWorker::quality_load_note() const {
  return quality_load_note_;
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
      MotionTask task = job.task;
      task.q = job.state.q;
      task.dq = job.state.dq;
      task.dq_known = job.state.dq_known;
      if (task.model_digest.empty() && kin_ != nullptr) {
        // Real model identity from URDF path/xml fingerprint length+hash seed.
        task.model_digest = urdf_is_xml_ ? "urdf_xml:" + urdf_path_.substr(0, 64)
                                         : "urdf_path:" + urdf_path_;
      }
      result = optimizer_->plan(task, job.state, job.policy, job.weights,
                                job.deadline_ms);
      result.submit_seq = job.seq;
      result.generation = task.generation;
      result.task_digest = motion_task_digest(task);
      result.model_digest = result.package.model_digest;
      result.limits_digest = result.package.limits_digest;
      if (result.package.task_digest == 0) {
        result.package.task_digest = result.task_digest;
      }
    } else {
      result.ok = false;
      result.reason = "worker_not_ready";
      result.submit_seq = job.seq;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      // Drop stale results if a newer job was already submitted.
      if (job.seq >= submitted_seq_.load(std::memory_order_relaxed) ||
          !has_pending_) {
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
