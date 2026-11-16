#pragma once

#include <cstdint>
#include <string>

namespace kmpc {

// Fixed production topic names (not ROS parameters).
inline constexpr const char* kStageAckTopic = "/kmpc/stage_ack";
inline constexpr const char* kMotionSequenceTopic = "/kmpc/motion_sequence";

enum class V2PlanPhase : uint8_t {
  Idle = 0,     // cleared / arrived / waiting for a new task
  Pending = 1,  // submitted; wait for worker
  Active = 2,   // adopted and executing
  Failed = 3,   // plan/adopt failed; wait for new identity or explicit replan
};

struct V2TaskId {
  uint64_t generation = 0;
  uint64_t digest = 0;

  bool valid() const { return generation != 0 && digest != 0; }
  bool same_as(const V2TaskId& o) const {
    return generation == o.generation && digest == o.digest;
  }
};

struct V2SubmitCounters {
  uint64_t submits = 0;
  uint64_t worker_done = 0;
  uint64_t adopts = 0;
  uint64_t rejects = 0;
};

// 50 Hz gate: submit once per request identity; poll while Pending; no spam.
class V2SubmitGate {
 public:
  // Explicit NeedReplan may retry once after a failure for the same identity.
  static constexpr int kMaxExplicitFailRetries = 1;

  struct TickDecision {
    bool submit = false;
    bool invalidate_old = false;
    bool poll_worker = false;
    bool follow = false;
    bool hold = false;
    const char* reason = "";
  };

  struct TakeDecision {
    bool identity_ok = false;
    bool adopt = false;
    bool retry_allowed = false;
    const char* reject_reason = "";
  };

  void clear() {
    phase_ = V2PlanPhase::Idle;
    request_ = {};
    submitted_ = {};
    active_ = {};
    submitted_seq_ = 0;
    active_seq_ = 0;
    active_path_id_ = 0;
    explicit_replan_ = false;
    explicit_retries_done_ = 0;
    last_reject_.clear();
  }

  void request_replan() { explicit_replan_ = true; }

  TickDecision on_tick(const V2TaskId& request, bool has_follower_package,
                       bool explicit_request) {
    request_ = request;
    if (explicit_request) {
      explicit_replan_ = true;
    }
    if (!request.valid()) {
      return {false, false, false, false, true, "invalid_request"};
    }

    // Explicit replan wins over Active-follow so NeedReplan is not stuck.
    if (explicit_replan_) {
      if (phase_ == V2PlanPhase::Failed && request.same_as(submitted_) &&
          explicit_retries_done_ >= kMaxExplicitFailRetries) {
        return {false, false, false, false, true, "retry_exhausted"};
      }
      return {true, true, false, false, true, "explicit_replan"};
    }

    if (phase_ == V2PlanPhase::Pending && request.same_as(submitted_)) {
      return {false, false, true, false, true, "await_worker"};
    }

    if (phase_ == V2PlanPhase::Active && request.same_as(active_) &&
        has_follower_package) {
      return {false, false, false, true, false, "executing"};
    }

    if (phase_ == V2PlanPhase::Active && request.same_as(active_) &&
        !has_follower_package) {
      phase_ = V2PlanPhase::Idle;
      return {false, false, false, false, true, "done_idle"};
    }

    if (phase_ == V2PlanPhase::Failed && request.same_as(submitted_)) {
      return {false, false, false, false, true, "failed_wait"};
    }

    // New request identity (includes first submit: submitted is 0/0).
    if (!request.same_as(submitted_)) {
      return {true, true, false, false, true, "submit_once"};
    }

    // Same identity, Idle after arrive/clear-package: do not rebuild.
    return {false, false, false, false, true, "idle_cached"};
  }

  void note_submitted(const V2TaskId& id, uint64_t submit_seq) {
    if (!id.same_as(submitted_)) {
      explicit_retries_done_ = 0;
    } else if (explicit_replan_ && phase_ == V2PlanPhase::Failed) {
      ++explicit_retries_done_;
    }
    submitted_ = id;
    submitted_seq_ = submit_seq;
    phase_ = V2PlanPhase::Pending;
    explicit_replan_ = false;
    ++counters_.submits;
  }

  TakeDecision on_worker_result(const V2TaskId& request, uint64_t plan_gen,
                                uint64_t plan_digest, uint64_t plan_seq,
                                bool plan_ok, bool package_fields_ok) {
    ++counters_.worker_done;
    const bool seq_ok = plan_seq != 0 && plan_seq == submitted_seq_ &&
                        request.same_as(submitted_);
    const bool id_ok =
        request.valid() && plan_gen == request.generation &&
        plan_digest == request.digest && plan_digest != 0 &&
        plan_gen == submitted_.generation &&
        plan_digest == submitted_.digest && package_fields_ok && seq_ok;
    if (!plan_ok || !id_ok) {
      phase_ = V2PlanPhase::Failed;
      ++counters_.rejects;
      last_reject_ = !plan_ok ? "plan_not_ok" : "stale_or_mismatch";
      return {false, false, false, last_reject_.c_str()};
    }
    return {true, true, false, ""};
  }

  void note_adopted(const V2TaskId& id, uint64_t submit_seq,
                    uint32_t path_id) {
    active_ = id;
    active_seq_ = submit_seq;
    active_path_id_ = path_id;
    phase_ = V2PlanPhase::Active;
    explicit_replan_ = false;
    explicit_retries_done_ = 0;
    ++counters_.adopts;
  }

  void note_adopt_failed(const char* why) {
    phase_ = V2PlanPhase::Failed;
    ++counters_.rejects;
    last_reject_ = why ? why : "adopt_failed";
    explicit_replan_ = false;
  }

  void note_plan_failed(const char* why) {
    phase_ = V2PlanPhase::Failed;
    ++counters_.rejects;
    last_reject_ = why ? why : "plan_failed";
    explicit_replan_ = false;
  }

  void note_arrived() {
    if (phase_ == V2PlanPhase::Active) {
      phase_ = V2PlanPhase::Idle;
    }
    explicit_replan_ = false;
  }

  V2PlanPhase phase() const { return phase_; }
  const V2TaskId& request() const { return request_; }
  const V2TaskId& submitted() const { return submitted_; }
  const V2TaskId& active() const { return active_; }
  uint64_t submitted_seq() const { return submitted_seq_; }
  uint64_t active_seq() const { return active_seq_; }
  uint32_t active_path_id() const { return active_path_id_; }
  const V2SubmitCounters& counters() const { return counters_; }
  const std::string& last_reject() const { return last_reject_; }

 private:
  V2PlanPhase phase_ = V2PlanPhase::Idle;
  V2TaskId request_{};
  V2TaskId submitted_{};
  V2TaskId active_{};
  uint64_t submitted_seq_ = 0;
  uint64_t active_seq_ = 0;
  uint32_t active_path_id_ = 0;
  bool explicit_replan_ = false;
  int explicit_retries_done_ = 0;
  V2SubmitCounters counters_{};
  std::string last_reject_;
};

}  // namespace kmpc
