#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace kmpc {

// Owned by the sending thread, under stream_mu_. The 50 Hz observer never
// re-anchors this clock. Pauses consume wall time without consuming path time.
class V2ExecutionState {
 public:
  void adopt(uint64_t generation, uint64_t digest, uint32_t path_id,
             double steady_s) {
    generation_ = generation;
    digest_ = digest;
    path_id_ = path_id;
    exec_s_ = 0.0;
    last_steady_s_ = steady_s;
    fault_.clear();
  }

  double advance(double steady_s, bool frozen, bool barrier_pending,
                 double barrier_s) {
    const double elapsed = std::max(0.0, steady_s - last_steady_s_);
    last_steady_s_ = steady_s;
    if (!faulted() && !frozen) {
      exec_s_ += elapsed;
      if (barrier_pending) exec_s_ = std::min(exec_s_, barrier_s);
    }
    return exec_s_;
  }

  // Also called when nothing was sent: never catch up a disabled/muted gap.
  void pause(double steady_s) { last_steady_s_ = steady_s; }
  void discard_unsent_progress(double previous_exec_s, double steady_s) {
    exec_s_ = previous_exec_s;
    last_steady_s_ = steady_s;
  }
  void latch_fault(const char* reason) {
    if (!faulted()) fault_ = reason;
  }
  bool faulted() const { return !fault_.empty(); }
  bool fault_matches(uint64_t generation, uint64_t digest) const {
    return faulted() && generation_ == generation && digest_ == digest;
  }
  const char* reason() const { return fault_.c_str(); }
  double exec_s() const { return exec_s_; }
  uint64_t generation() const { return generation_; }
  uint32_t path_id() const { return path_id_; }

 private:
  uint64_t generation_ = 0;
  uint64_t digest_ = 0;
  uint32_t path_id_ = 0;
  double exec_s_ = 0.0;
  double last_steady_s_ = 0.0;
  std::string fault_;
};

}  // namespace kmpc
