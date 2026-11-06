#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "layer_diag.hpp"

namespace {

kmpc::ros2io::LayerDiagSample sample_at(uint64_t seq, int64_t source_ns) {
  kmpc::ros2io::LayerDiagSample sample;
  sample.mono_ns = kmpc::ros2io::layer_diag_mono_ns();
  sample.recv_mono_ns = sample.mono_ns;
  sample.source_stamp_ns = source_ns;
  sample.run_id = 42;
  sample.goal_generation = 1000000007ULL;
  sample.seq = seq;
  sample.line_gen = 3;
  sample.layer = kmpc::ros2io::LayerDiagLayer::StreamRawRef;
  sample.next_layer = kmpc::ros2io::layer_diag_next(sample.layer);
  sample.prog = 0.25;
  sample.q[0] = 0.125;
  kmpc::ros2io::layer_diag_note(&sample, "ok");
  return sample;
}

}  // namespace

TEST(LayerDiag, OverflowIsVisibleAndPushDoesNotBlock) {
  kmpc::ros2io::LayerDiagQueue<4> queue;
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(queue.try_push(sample_at(static_cast<uint64_t>(i), -1)));
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    EXPECT_FALSE(queue.try_push(sample_at(1000, -1)));
  }
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  EXPECT_EQ(queue.drops(), 1000u);
  EXPECT_LT(us, 2000.0);
  kmpc::ros2io::LayerDiagSample out[4];
  EXPECT_EQ(queue.try_pop(out, 4), 4u);
  EXPECT_EQ(out[0].seq, 0u);
  EXPECT_EQ(out[0].drops_before, 0u);
  const std::string line = kmpc::ros2io::layer_diag_jsonl(out[0]);
  EXPECT_NE(line.find("\"run_id\":42"), std::string::npos);
  EXPECT_NE(line.find("\"goal_generation\":1000000007"), std::string::npos);
  EXPECT_NE(line.find("\"layer\":\"stream_raw_ref\""), std::string::npos);
  EXPECT_NE(line.find("\"next_layer\":\"LineSmoother\""), std::string::npos);
  EXPECT_NE(line.find("\"source_stamp_ns\":-1"), std::string::npos);
  const std::string marked = kmpc::ros2io::layer_diag_overflow_jsonl(42, queue.drops());
  EXPECT_NE(marked.find("\"kind\":\"overflow\""), std::string::npos);
  EXPECT_NE(marked.find("\"drops\":1000"), std::string::npos);
}

TEST(LayerDiag, SourceStampAndGenerationStayOnTheSample) {
  kmpc::ros2io::LayerDiagQueue<8> queue;
  ASSERT_TRUE(queue.try_push(sample_at(7, 1700000000123456789LL)));
  kmpc::ros2io::LayerDiagSample out;
  ASSERT_EQ(queue.try_pop(&out, 1), 1u);
  EXPECT_EQ(out.source_stamp_ns, 1700000000123456789LL);
  EXPECT_EQ(out.goal_generation, 1000000007ULL);
  EXPECT_EQ(out.run_id, 42u);
  EXPECT_GT(out.mono_ns, 0u);
  EXPECT_EQ(out.recv_mono_ns, out.mono_ns);
  const std::string line = kmpc::ros2io::layer_diag_jsonl(out);
  EXPECT_NE(line.find("\"source_stamp_ns\":1700000000123456789"), std::string::npos);
  EXPECT_NE(line.find("\"seq\":7"), std::string::npos);
}

TEST(LayerDiag, DisabledPathSkipsTheQueue) {
  kmpc::ros2io::LayerDiagQueue<1024> queue;
  constexpr int kN = 200000;
  std::atomic<int> mode{0};
  auto run = [&]() {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kN; ++i) {
      if (mode.load(std::memory_order_relaxed) == 0) {
        continue;
      }
      queue.try_push(sample_at(static_cast<uint64_t>(i), -1));
    }
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - t0)
               .count() /
           kN;
  };
  const double off_ns = run();
  mode.store(1, std::memory_order_relaxed);
  const double on_ns = run();
  std::cout << "layer_diag_ns_per_call off=" << off_ns << " on=" << on_ns
            << " delta=" << (on_ns - off_ns) << "\n";
  EXPECT_LT(off_ns, 50.0);
  EXPECT_GT(on_ns, off_ns);
  EXPECT_GT(queue.size_approx(), 0u);
}

TEST(ConfigSnapshot, JsonRecordsSourceUnitAndRange) {
  const std::vector<kmpc::ros2io::ConfigSnapshotRow> rows = {
      {"stream_hz", "250", "Hz", "node_default", ">0", true},
      {"line_dq_scale", "0.6", "1", "ros_override", "[0,1]", true},
  };
  const std::string json = kmpc::ros2io::config_snapshot_json(9, "abc", rows);
  EXPECT_NE(json.find("\"kind\":\"effective_config\""), std::string::npos);
  EXPECT_NE(json.find("\"name\":\"stream_hz\""), std::string::npos);
  EXPECT_NE(json.find("\"unit\":\"Hz\""), std::string::npos);
  EXPECT_NE(json.find("\"source\":\"ros_override\""), std::string::npos);
  EXPECT_NE(json.find("\"range\":\"[0,1]\""), std::string::npos);
  EXPECT_NE(json.find("\"git_sha\":\"abc\""), std::string::npos);
  std::ofstream out("/tmp/kmpc_config_snapshot_test.json");
  ASSERT_TRUE(out.good());
  out << json << "\n";
}
