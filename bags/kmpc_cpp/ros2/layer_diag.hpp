#pragma once
// 阶段 A 分层诊断。热路径只往预分配 SPSC 里写固定样本。
// JSONL 只给导出线程/测试用，禁止在 250Hz 回调里调用 format。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kmpc/types.hpp"

namespace kmpc::ros2io {

enum class LayerDiagLayer : uint8_t {
  Build = 0,
  FollowQp = 1,
  FollowCommit = 2,
  StreamRawRef = 3,
  LineSmoother = 4,
  FinalSend = 5,
  Feedback = 6,
  SdkStatus = 7,
};

inline const char* layer_diag_name(LayerDiagLayer layer) {
  switch (layer) {
    case LayerDiagLayer::Build: return "Build";
    case LayerDiagLayer::FollowQp: return "FollowQp";
    case LayerDiagLayer::FollowCommit: return "FollowCommit";
    case LayerDiagLayer::StreamRawRef: return "stream_raw_ref";
    case LayerDiagLayer::LineSmoother: return "LineSmoother";
    case LayerDiagLayer::FinalSend: return "final_send";
    case LayerDiagLayer::Feedback: return "Feedback";
    case LayerDiagLayer::SdkStatus: return "sdk_status";
  }
  return "unknown";
}

inline LayerDiagLayer layer_diag_next(LayerDiagLayer layer) {
  switch (layer) {
    case LayerDiagLayer::Build: return LayerDiagLayer::FollowQp;
    case LayerDiagLayer::FollowQp: return LayerDiagLayer::FollowCommit;
    case LayerDiagLayer::FollowCommit: return LayerDiagLayer::StreamRawRef;
    case LayerDiagLayer::StreamRawRef: return LayerDiagLayer::LineSmoother;
    case LayerDiagLayer::LineSmoother: return LayerDiagLayer::FinalSend;
    case LayerDiagLayer::FinalSend: return LayerDiagLayer::Feedback;
    case LayerDiagLayer::Feedback: return LayerDiagLayer::Feedback;
    case LayerDiagLayer::SdkStatus: return LayerDiagLayer::SdkStatus;
  }
  return layer;
}

// mono_ns / recv_mono_ns：steady_clock 自纪元起的纳秒，不是 Unix 时间。
// source_stamp_ns：消息 header 的 Unix 纳秒；没有传感器时间戳时为 -1。
// 不能把节点 now() 填进 source_stamp_ns。
struct LayerDiagSample {
  uint64_t mono_ns = 0;
  int64_t source_stamp_ns = -1;
  uint64_t recv_mono_ns = 0;
  uint64_t run_id = 0;
  uint64_t goal_generation = 0;
  uint64_t seq = 0;
  uint32_t line_gen = 0;
  uint32_t drops_before = 0;
  LayerDiagLayer layer = LayerDiagLayer::Build;
  LayerDiagLayer next_layer = LayerDiagLayer::FollowQp;
  double prog = 0.0;
  double q[kNDof]{};
  char note[48]{};
};

inline uint64_t layer_diag_mono_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline void layer_diag_note(LayerDiagSample* sample, const char* note) {
  sample->note[0] = '\0';
  if (note == nullptr || note[0] == '\0') {
    return;
  }
  std::snprintf(sample->note, sizeof(sample->note), "%s", note);
}

inline void layer_diag_copy_q(LayerDiagSample* sample, const Vec18& q) {
  for (int i = 0; i < kNDof; ++i) {
    sample->q[i] = q[i];
  }
}

// 单写单读。满了丢新样本并累加 drops，写者不等待读者。
template <size_t Cap>
class LayerDiagQueue {
 public:
  static_assert(Cap >= 2 && (Cap & (Cap - 1)) == 0, "Cap must be power of two");

  LayerDiagQueue() : slots_(Cap) {}

  bool try_push(LayerDiagSample sample) {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    const size_t head = head_.load(std::memory_order_acquire);
    if (tail - head >= Cap) {
      drops_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    sample.drops_before = static_cast<uint32_t>(drops_.load(std::memory_order_relaxed));
    slots_[tail & (Cap - 1)] = sample;
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  size_t try_pop(LayerDiagSample* out, size_t max_n) {
    size_t got = 0;
    while (got < max_n) {
      const size_t head = head_.load(std::memory_order_relaxed);
      const size_t tail = tail_.load(std::memory_order_acquire);
      if (head == tail) {
        break;
      }
      out[got++] = slots_[head & (Cap - 1)];
      head_.store(head + 1, std::memory_order_release);
    }
    return got;
  }

  uint64_t drops() const { return drops_.load(std::memory_order_relaxed); }
  size_t size_approx() const {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    const size_t head = head_.load(std::memory_order_relaxed);
    return tail - head;
  }

 private:
  std::vector<LayerDiagSample> slots_;
  std::atomic<size_t> head_{0};
  std::atomic<size_t> tail_{0};
  std::atomic<uint64_t> drops_{0};
};

inline std::string layer_diag_jsonl(const LayerDiagSample& sample) {
  char buf[2048];
  int n = std::snprintf(
      buf, sizeof(buf),
      "{\"schema\":1,\"run_id\":%llu,\"goal_generation\":%llu,\"seq\":%llu,"
      "\"line_gen\":%u,\"layer\":\"%s\",\"next_layer\":\"%s\","
      "\"mono_ns\":%llu,\"source_stamp_ns\":%lld,\"recv_mono_ns\":%llu,"
      "\"drops_before\":%u,\"prog\":%.9g,\"note\":\"%s\",\"q\":[",
      static_cast<unsigned long long>(sample.run_id),
      static_cast<unsigned long long>(sample.goal_generation),
      static_cast<unsigned long long>(sample.seq), sample.line_gen,
      layer_diag_name(sample.layer), layer_diag_name(sample.next_layer),
      static_cast<unsigned long long>(sample.mono_ns),
      static_cast<long long>(sample.source_stamp_ns),
      static_cast<unsigned long long>(sample.recv_mono_ns), sample.drops_before,
      sample.prog, sample.note);
  if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) {
    return {};
  }
  for (int i = 0; i < kNDof; ++i) {
    const int used = std::snprintf(
        buf + n, sizeof(buf) - static_cast<size_t>(n), "%s%.9g",
        (i == 0 ? "" : ","), sample.q[i]);
    if (used < 0 || static_cast<size_t>(n + used) >= sizeof(buf)) {
      return {};
    }
    n += used;
  }
  const int tail = std::snprintf(
      buf + n, sizeof(buf) - static_cast<size_t>(n), "]}");
  if (tail < 0 || static_cast<size_t>(n + tail) >= sizeof(buf)) {
    return {};
  }
  return std::string(buf);
}

inline std::string layer_diag_header_jsonl(uint64_t run_id) {
  char buf[640];
  std::snprintf(
      buf, sizeof(buf),
      "{\"schema\":1,\"kind\":\"header\",\"run_id\":%llu,"
      "\"mono_clock\":\"steady_clock_epoch_ns\","
      "\"source_stamp\":\"header_unix_ns_or_-1\","
      "\"recv_clock\":\"steady_clock_epoch_ns\","
      "\"gaps\":[\"imu_not_subscribed\",\"sdk_queue_depth_not_visible\","
      "\"stream_status_has_no_goal_generation\"]}",
      static_cast<unsigned long long>(run_id));
  return std::string(buf);
}

inline std::string layer_diag_overflow_jsonl(uint64_t run_id, uint64_t drops) {
  char buf[192];
  std::snprintf(
      buf, sizeof(buf),
      "{\"schema\":1,\"kind\":\"overflow\",\"run_id\":%llu,\"drops\":%llu}",
      static_cast<unsigned long long>(run_id),
      static_cast<unsigned long long>(drops));
  return std::string(buf);
}

struct ConfigSnapshotRow {
  std::string name;
  std::string value;
  std::string unit;
  std::string source;
  std::string range;
  bool ok = true;
};

inline std::string config_snapshot_json(
    uint64_t run_id, const std::string& git_sha,
    const std::vector<ConfigSnapshotRow>& rows) {
  std::string out;
  out.reserve(rows.size() * 160 + 128);
  char head[192];
  std::snprintf(
      head, sizeof(head),
      "{\"schema\":1,\"kind\":\"effective_config\",\"run_id\":%llu,\"git_sha\":\"%s\",\"rows\":[",
      static_cast<unsigned long long>(run_id), git_sha.c_str());
  out += head;
  for (size_t i = 0; i < rows.size(); ++i) {
    const ConfigSnapshotRow& row = rows[i];
    char item[768];
    std::snprintf(
        item, sizeof(item),
        "%s{\"name\":\"%s\",\"value\":\"%s\",\"unit\":\"%s\",\"source\":\"%s\","
        "\"range\":\"%s\",\"ok\":%s}",
        (i == 0 ? "" : ","), row.name.c_str(), row.value.c_str(), row.unit.c_str(),
        row.source.c_str(), row.range.c_str(), row.ok ? "true" : "false");
    out += item;
  }
  out += "]}";
  return out;
}

}  // namespace kmpc::ros2io
