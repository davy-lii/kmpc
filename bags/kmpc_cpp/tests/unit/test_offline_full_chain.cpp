#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "kmpc/line_async_handoff.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/offline_full_chain.hpp"
#include "kmpc/path_progress.hpp"
#include "kmpc/path_smooth_verify.hpp"
#include "kmpc/qp_fail_dump_worker.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/unified_time_law.hpp"
#include "line_stream.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

std::string unique_report_dir() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch())
                      .count();
  return std::string(
             "/home/davy/03/home/nvidia/ljw/issues/"
             "20261111_最新运行停线与指令平顺性/evidence/"
             "offline_full_chain_report_") +
         std::to_string(ms);
}

std::string resolve_source_id() {
  if (const char* env = std::getenv("KMPC_CHAIN_SOURCE_ID");
      env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  // Live worktree identity — never hardcode a commit string.
  std::string cmd =
      "git -C \"/home/davy/03/home/nvidia/workspaces/kmpc\" "
      "rev-parse --short=12 HEAD 2>/dev/null";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return "unknown_source";
  }
  char buf[128]{};
  const char* got = std::fgets(buf, sizeof(buf), pipe);
  const int rc = pclose(pipe);
  if (!got || rc != 0) {
    return "unknown_source";
  }
  std::string out(buf);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  if (out.empty()) {
    return "unknown_source";
  }
  FILE* dirty = popen(
      "git -C \"/home/davy/03/home/nvidia/workspaces/kmpc\" "
      "status --porcelain 2>/dev/null | head -1",
      "r");
  if (dirty) {
    char dbuf[8]{};
    if (std::fgets(dbuf, sizeof(dbuf), dirty) != nullptr && dbuf[0] != '\0') {
      out += "+dirty";
    }
    pclose(dirty);
  }
  return out;
}

// Count unexpected waist J1/J2/J3 and pitch-sum reversals on 250 Hz finals.
struct StreamAxisStats {
  int waist_reverse = 0;
  int pitch_reverse = 0;
  double max_waist_step_deg = 0.0;
  double max_pitch_step_deg = 0.0;
  double max_speed = 0.0;
};

StreamAxisStats scan_stream_axes(const kmpc::ChainRunResult& r,
                                 const kmpc::DirectionContract& contract) {
  StreamAxisStats s;
  if (r.stream.size() < 2) {
    return s;
  }
  for (std::size_t i = 1; i < r.stream.size(); ++i) {
    const auto& a = r.stream[i - 1].q_final;
    const auto& b = r.stream[i].q_final;
    const double dt = std::max(1e-9, r.stream[i].dt_s);
    s.max_speed =
        std::max(s.max_speed, ((b - a) / dt).cwiseAbs().maxCoeff());
    for (int j = 0; j < 3; ++j) {
      const double dq = b[j] - a[j];
      s.max_waist_step_deg =
          std::max(s.max_waist_step_deg, std::abs(dq) * 180.0 / kmpc::kPi);
      const int sgn = contract.sign[j];
      if (sgn != 0 && dq * static_cast<double>(sgn) < -1e-6) {
        ++s.waist_reverse;
      }
    }
    const double dp = b.head<3>().sum() - a.head<3>().sum();
    s.max_pitch_step_deg =
        std::max(s.max_pitch_step_deg, std::abs(dp) * 180.0 / kmpc::kPi);
    const int pitch_sgn =
        contract.sign[static_cast<int>(kmpc::DirectionAxisId::PitchSum)];
    if (pitch_sgn != 0 && dp * static_cast<double>(pitch_sgn) < -1e-6) {
      ++s.pitch_reverse;
    }
  }
  return s;
}

kmpc::JointPath corner_samples() {
  // Dense knots: per-segment Δq must stay << joint_eps*2^max_depth so the
  // adaptive Hermite validator can converge (max_depth=6, eps=5e-4 → ~0.032).
  kmpc::JointPath samples(5);
  for (auto& s : samples) {
    s.setZero();
  }
  samples[1][4] = 0.010;
  samples[2][4] = 0.020;
  samples[3][4] = 0.020;
  samples[3][5] = 0.010;  // turn onto axis 5
  samples[4][4] = 0.020;
  samples[4][5] = 0.020;
  return samples;
}

}  // namespace

TEST(OfflineFullChain, LegacyVsNewUnifiedBothMustComplete) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.12, 0.0, 0.0);

  kmpc::OfflineChainConfig opt;
  opt.max_control_ticks = 800;
  opt.speed_scale = 0.55;
  opt.accel_s = 0.15;
  opt.simulate_stream_advance = true;
  opt.stream_advance_frac = 0.15;
  opt.feedback = kmpc::FeedbackMode::IdealFinal;

  kmpc::CartesianTracker legacy_tr(kin, cfg_all);
  const auto legacy = kmpc::run_offline_full_chain(
      legacy_tr, kin, cfg_all, q, goal, quat, /*left=*/true,
      kmpc::CartesianTracker::LineChainProfile::Legacy, opt);

  kmpc::CartesianTracker new_tr(kin, cfg_all);
  const auto neu = kmpc::run_offline_full_chain(
      new_tr, kin, cfg_all, q, goal, quat, /*left=*/true,
      kmpc::CartesianTracker::LineChainProfile::NewUnified, opt);

  ASSERT_FALSE(legacy.ticks.empty());
  ASSERT_FALSE(neu.ticks.empty());
  bool legacy_layers = false;
  bool new_layers = false;
  for (const auto& t : legacy.ticks) {
    if (t.plan_ran && t.qp_ran && t.commit_ran && t.interp_ran && t.final_ran) {
      legacy_layers = true;
    }
  }
  for (const auto& t : neu.ticks) {
    if (t.plan_ran && t.qp_ran && t.commit_ran && t.interp_ran && t.final_ran) {
      new_layers = true;
    }
  }
  EXPECT_TRUE(legacy_layers) << "legacy chain did not run full layers";
  EXPECT_TRUE(new_layers) << "new chain did not run full layers";

  EXPECT_EQ(new_tr.line_pace_mode(),
            kmpc::CartesianTracker::LinePaceMode::UnifiedTimeLaw);
  EXPECT_EQ(new_tr.line_qp_follow_mode(), kmpc::LineQpFollowMode::B);

  // Production hard gate is NewUnified / unified_multistage_v2 only.
  ASSERT_TRUE(neu.completed) << "new refuse=" << neu.refuse_reason
                             << " fault=" << neu.first_fault.reason;
  EXPECT_TRUE(neu.cmd_arrived);
  EXPECT_TRUE(neu.feedback_arrived);
  EXPECT_EQ(neu.reverse_events, 0);
  EXPECT_LT(neu.final_pos_err_m, 0.015);
  EXPECT_LT(neu.final_ori_err_rad, 0.10);
  EXPECT_FALSE(neu.stream.empty());

  // Legacy remains for offline comparison / archive only. It is not a
  // production gate: v2 densify + retired sign/ori stream policy can trip the
  // old stream_filtered_orientation path without meaning NewUnified failed.
  if (!legacy.completed) {
    std::cout << "legacy_nonprod_incomplete refuse=" << legacy.refuse_reason
              << " fault=" << legacy.first_fault.reason << "\n";
  } else {
    EXPECT_TRUE(legacy.cmd_arrived);
    EXPECT_TRUE(legacy.feedback_arrived);
    EXPECT_EQ(legacy.reverse_events, 0);
    EXPECT_LT(legacy.final_pos_err_m, 0.015);
    EXPECT_LT(legacy.final_ori_err_rad, 0.10);
  }

  const auto neu_axes =
      scan_stream_axes(neu, new_tr.line_direction_contract());
  // v2 retires absolute waist/pitch sign hard gates; count is diagnostic only.
  std::cout << "neu_stream_diag waist_reverse=" << neu_axes.waist_reverse
            << " pitch_reverse=" << neu_axes.pitch_reverse << "\n";

  const char* report_env = std::getenv("KMPC_CHAIN_REPORT_DIR");
  const std::string report_parent =
      report_env && report_env[0] ? std::string(report_env) : unique_report_dir();
  // Always write into a slot so a shared KMPC_CHAIN_REPORT_DIR cannot clash
  // with hermite_full_chain (or other suite writers) on summary.md.
  const std::string report_dir =
      kmpc::chain_report_slot_dir(report_parent, "ab_compare");
  ASSERT_EQ(report_dir.find("offline_full_chain_report_1559"),
            std::string::npos)
      << "refusing to write into previously-overwritten 1559 evidence";
  const std::string source_id = resolve_source_id();
  ASSERT_TRUE(kmpc::write_chain_compare_report(report_dir, legacy, neu,
                                               source_id.c_str()));
  EXPECT_TRUE(std::filesystem::exists(report_dir + "/summary.md"));
  EXPECT_TRUE(std::filesystem::exists(report_dir + "/legacy_metrics.csv"));
  EXPECT_TRUE(std::filesystem::exists(report_dir + "/new_metrics.csv"));
  EXPECT_TRUE(std::filesystem::exists(report_dir + "/new_stream250.csv"));
  std::printf("offline_full_chain report: %s/summary.md source=%s\n",
              report_dir.c_str(), source_id.c_str());
}

void run_feedback_suite(kmpc::CartesianTracker::LineChainProfile profile,
                        const char* tag) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.10, 0.0, 0.0);

  {
    kmpc::OfflineChainConfig opt;
    opt.max_control_ticks = 900;
    opt.speed_scale = 0.55;
    opt.accel_s = 0.15;
    opt.simulate_stream_advance = true;
    opt.stream_advance_frac = 0.15;
    opt.feedback = kmpc::FeedbackMode::LagFinal;
    opt.feedback_lag_ticks = 3;
    opt.drain_extra_ticks = 80;
    kmpc::CartesianTracker tr(kin, cfg_all);
    const auto run = kmpc::run_offline_full_chain(
        tr, kin, cfg_all, q, goal, quat, /*left=*/true, profile, opt);
    ASSERT_TRUE(run.completed) << tag << " lag fault=" << run.first_fault.reason;
    EXPECT_TRUE(run.cmd_arrived) << tag;
    EXPECT_TRUE(run.feedback_arrived) << tag;
    bool saw_lag = false;
    for (const auto& s : run.stream) {
      if ((s.q_feedback - s.q_final).cwiseAbs().maxCoeff() > 1e-4) {
        saw_lag = true;
        break;
      }
    }
    EXPECT_TRUE(saw_lag) << tag << " LagFinal produced no delay";
  }
  {
    kmpc::OfflineChainConfig opt;
    opt.max_control_ticks = 500;
    opt.speed_scale = 0.55;
    opt.accel_s = 0.12;
    opt.simulate_stream_advance = true;
    opt.stream_advance_frac = 0.2;
    opt.feedback = kmpc::FeedbackMode::StaleHold;
    opt.feedback_resume_control_ticks = -1;
    opt.drain_extra_ticks = 20;
    kmpc::CartesianTracker tr(kin, cfg_all);
    const auto run = kmpc::run_offline_full_chain(
        tr, kin, cfg_all, q, goal, quat, /*left=*/true, profile, opt);
    EXPECT_FALSE(run.completed) << tag << " freeze must not complete";
    EXPECT_FALSE(run.feedback_arrived) << tag;
    const bool drain_fault =
        run.first_fault.found &&
        std::string(run.first_fault.reason)
                .find("hold_feedback_not_caught_up") != std::string::npos;
    const bool tail_refuse =
        run.refused && run.refuse_reason.find("tail_feedback_not_arrived") !=
                           std::string::npos;
    EXPECT_TRUE(drain_fault || tail_refuse)
        << tag << " fault=" << run.first_fault.reason
        << " refuse=" << run.refuse_reason;
  }
  {
    kmpc::OfflineChainConfig opt;
    opt.max_control_ticks = 900;
    opt.speed_scale = 0.55;
    opt.accel_s = 0.12;
    opt.simulate_stream_advance = true;
    opt.stream_advance_frac = 0.2;
    opt.feedback = kmpc::FeedbackMode::StaleHold;
    opt.feedback_resume_control_ticks = 30;
    opt.drain_extra_ticks = 60;
    kmpc::CartesianTracker tr(kin, cfg_all);
    const auto run = kmpc::run_offline_full_chain(
        tr, kin, cfg_all, q, goal, quat, /*left=*/true, profile, opt);
    ASSERT_TRUE(run.completed) << tag << " resume fault=" << run.first_fault.reason
                               << " refuse=" << run.refuse_reason;
    EXPECT_TRUE(run.cmd_arrived) << tag;
    EXPECT_TRUE(run.feedback_arrived) << tag;
  }
}

TEST(OfflineFullChain, LegacyFeedbackLagFreezeResume) {
  run_feedback_suite(kmpc::CartesianTracker::LineChainProfile::Legacy, "legacy");
}

TEST(OfflineFullChain, NewUnifiedFeedbackLagFreezeResume) {
  run_feedback_suite(kmpc::CartesianTracker::LineChainProfile::NewUnified,
                     "new_unified");
}

TEST(OfflineFullChain, ProgressRecoverRejectsMismatchedChordScale) {
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> samples(3);
  samples[0].setZero();
  samples[1].setZero();
  samples[1][0] = 0.10;
  samples[2].setZero();
  samples[2][0] = 0.20;
  auto at = [&](double s) {
    const int n = 3;
    double p = std::clamp(s, 0.0, 2.0);
    const int i = std::min(n - 2, static_cast<int>(std::floor(p)));
    const double u = p - i;
    return (1 - u) * samples[static_cast<std::size_t>(i)] +
           u * samples[static_cast<std::size_t>(i + 1)];
  };
  kmpc::Vec18 accepted = kmpc::Vec18::Zero();
  accepted[5] = 0.40;  // off-path
  const auto rec = kmpc::recover_path_progress(0.0, 1.0, accepted, at, 0.015);
  EXPECT_TRUE(rec.unknown);
  EXPECT_FALSE(rec.ok);
  EXPECT_NE(std::string(rec.reason).find("residual"), std::string::npos);
}

TEST(OfflineFullChain, ProgressRecoverZeroSpanLargeResidualUnknown) {
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> samples(2);
  samples[0].setZero();
  samples[1].setZero();
  samples[1][0] = 0.1;
  auto at = [&](double s) {
    return (1.0 - s) * samples[0] + s * samples[1];
  };
  kmpc::Vec18 accepted = kmpc::Vec18::Zero();
  accepted[0] = 1.0;  // 1 rad off at zero span
  const auto rec = kmpc::recover_path_progress(0.0, 0.0, accepted, at, 0.015);
  EXPECT_TRUE(rec.unknown);
  EXPECT_FALSE(rec.ok);
  EXPECT_NEAR(rec.residual_rad, 1.0, 1e-9);
}

TEST(OfflineFullChain, TimeLawCornerLookAheadStops) {
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> samples(3);
  samples[0].setZero();
  samples[1].setZero();
  samples[1][0] = 0.2;
  samples[2].setZero();
  samples[2][1] = 0.2;
  auto at = [&](double s) {
    double p = std::clamp(s, 0.0, 2.0);
    const int i = std::min(1, static_cast<int>(std::floor(p)));
    const double u = p - i;
    return (1 - u) * samples[static_cast<std::size_t>(i)] +
           u * samples[static_cast<std::size_t>(i + 1)];
  };
  const auto corners = kmpc::measure_path_corners(samples);
  ASSERT_GE(corners.max_tangent_jump_rad, 0.02);
  ASSERT_FALSE(corners.tangent_jumps.empty());
  kmpc::TimeLawLimits lim;
  lim.dq_max.setConstant(1.0);
  lim.step_cap.setConstant(0.05);
  lim.speed_scale = 1.0;
  lim.accel_s = 1.0;
  lim.dt = 0.02;
  lim.geometry_c1 = false;
  lim.corner_remain_s = 0.08;
  lim.end_remain_s = 2.0;
  lim.rate = 0.9;
  lim.rate_acc = 0.0;
  const auto sample = kmpc::unified_time_law_step(at, 0.92, lim);
  EXPECT_TRUE(sample.look_ahead_brake) << sample.note;
  EXPECT_LE(sample.rate, 0.40 + 1e-6);
  EXPECT_LT(sample.rate, lim.rate);
}

// r31 field: linear corner_stop * end_brake put want under the 1e-4 progress
// gate and froze prog forever. After deleting corner_stop, sqrt look-ahead at
// the same remain must stay above that gate so the knot can be crossed.
TEST(OfflineFullChain, TimeLawNoCornerDeadlockShell) {
  auto at = [](double s) -> kmpc::Vec18 {
    kmpc::Vec18 q = kmpc::Vec18::Zero();
    q[0] = 0.05 * s;
    return q;
  };
  kmpc::TimeLawLimits lim;
  lim.dq_max.setConstant(1.0);
  lim.step_cap.setConstant(0.05);
  lim.speed_scale = 0.30;
  lim.path_remain_m = 0.009779;
  lim.brake_distance_m = 0.08;
  lim.brake_floor = 0.20;
  lim.accel_s = 0.3;
  lim.jerk_s = 0.08;
  lim.dt = 0.02;
  lim.geometry_c1 = false;
  lim.corner_remain_s = 0.000312;  // r31 left-retreat knot gap
  lim.end_remain_s = 2.0;
  lim.rate = 0.0;
  lim.rate_acc = 0.0;
  const auto sample = kmpc::unified_time_law_step(at, 46.999688, lim);
  EXPECT_GT(sample.want, 1e-4) << sample.note;
  EXPECT_GT(sample.rate, 1e-4) << sample.note;
  // Old linear factor would have been remain/0.35 ≈ 8.9e-4, times end_brake
  // ≈0.35 → want≈9.3e-5 < 1e-4. Look-ahead floor is sqrt(2 a δ).
  const double a_bar = 1.0 / lim.accel_s;
  const double stop_r =
      kmpc::stop_limited_rate(lim.corner_remain_s, a_bar);
  EXPECT_NEAR(sample.want, std::min(0.30 * sample.brake, stop_r), 1e-6);
}

TEST(QpFailDumpWorker, BoundedQueueDropsAndJoins) {
  std::atomic<int> written{0};
  kmpc::QpFailDumpWorker<int> worker(2);
  worker.start([&](int&& v) {
    (void)v;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    written.fetch_add(1);
  });
  EXPECT_TRUE(worker.enqueue(1));
  EXPECT_TRUE(worker.enqueue(2));
  EXPECT_FALSE(worker.enqueue(3));
  EXPECT_GE(worker.drops(), 1u);
  worker.stop_join();
  EXPECT_FALSE(worker.running());
  EXPECT_GE(written.load(), 1);
}

TEST(LinePath, PartialCommitWithDifferentStreamFromRecoversOrUnknown) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(0.3, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.12, 0.0, 0.0);
  ASSERT_FALSE(
      tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr)
          .dq.hasNaN());
  ASSERT_TRUE(tracker.line_commit_pending());
  const kmpc::Vec18 proposed = tracker.line_cmd();
  const kmpc::Vec18 planning_from = tracker.line_commit_from();
  kmpc::Vec18 stream_raw = planning_from + 0.3 * (proposed - planning_from);
  const auto bounded = kmpc::ros2io::bound_line_stream_step(
      stream_raw, proposed, tracker.effective_dq_max() * 0.02 * 0.6);
  ASSERT_TRUE(bounded);
  const bool ok =
      tracker.confirm_line_stream_commit(bounded->target, bounded->scale);
  if (ok) {
    EXPECT_GE(tracker.line_progress(), tracker.line_commit_old_prog() - 1e-12);
    EXPECT_LE(tracker.line_progress(), tracker.line_commit_new_prog() + 1e-12);
    EXPECT_FALSE(tracker.last_progress_recover().unknown);
  } else {
    EXPECT_TRUE(tracker.last_progress_recover().unknown);
  }
}

TEST(LinePath, QpModeBRejectsNamedHardRow) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_qp_follow_mode(kmpc::LineQpFollowMode::B);
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(0.25, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.08, 0.0, 0.0);
  tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  ASSERT_TRUE(tracker.line_commit_pending());
  ASSERT_TRUE(tracker.confirm_line_stream_commit(tracker.line_cmd(), 1.0));
  q = tracker.line_cmd();
  kmpc::Vec18 bad = q;
  bad[11] += 0.30;
  tracker.set_line_reference(bad);
  tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
  ASSERT_TRUE(tracker.line_refused());
  const auto& fail = tracker.last_line_qp_fail();
  ASSERT_TRUE(tracker.line_reason().find("qp_") != std::string::npos)
      << tracker.line_reason();
  ASSERT_GE(fail.worst_c, 0) << "expected named hard row, reason="
                             << tracker.line_reason();
  ASSERT_NE(fail.worst_tag[0], '\0') << "worst_c set but tag empty";
  EXPECT_GT(fail.worst_violation, 0.0);
  EXPECT_EQ(std::string(fail.worst_tag).rfind("s0_", 0), 0) << fail.worst_tag;
}

TEST(LinePath, ProposeDoesNotFreezeBeforeConfirm) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(0.5, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.18, 0.0, 0.0);
  tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  ASSERT_TRUE(tracker.line_commit_pending());
  EXPECT_FALSE(tracker.line_frozen())
      << "frozen must wait for accepted confirm";
  tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
  EXPECT_FALSE(tracker.line_frozen())
      << "stale propose must not latch freeze across ticks";
}

TEST(LinePath, LegacyProfilePreservesPaceAndQpModes) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace_mode(
      kmpc::CartesianTracker::LinePaceMode::R11Scurve);
  tracker.set_line_qp_follow_mode(kmpc::LineQpFollowMode::C);
  tracker.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::Legacy);
  EXPECT_EQ(tracker.line_pace_mode(),
            kmpc::CartesianTracker::LinePaceMode::R11Scurve);
  EXPECT_EQ(tracker.line_qp_follow_mode(), kmpc::LineQpFollowMode::C);

  tracker.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  EXPECT_EQ(tracker.line_pace_mode(),
            kmpc::CartesianTracker::LinePaceMode::UnifiedTimeLaw);
  EXPECT_EQ(tracker.line_qp_follow_mode(), kmpc::LineQpFollowMode::B);
  EXPECT_EQ(tracker.line_chain_profile(),
            kmpc::CartesianTracker::LineChainProfile::NewUnified);
}

TEST(LinePath, HermiteCornerVerifiedAndTubeFallback) {
  const auto samples = corner_samples();
  ASSERT_GE(kmpc::measure_path_corners(samples).max_tangent_jump_rad, 1e-4);

  auto ok_check = [](const kmpc::Vec18&, std::string*) { return true; };
  const auto ok = kmpc::try_verified_hermite_smooth(samples, ok_check);
  EXPECT_TRUE(ok.hermite_candidate);
  EXPECT_TRUE(ok.hermite_verified) << ok.reason;
  EXPECT_TRUE(ok.applied) << ok.reason;

  auto tube_fail = [](const kmpc::Vec18& q, std::string* why) {
    if (q[4] > 0.005) {
      if (why) {
        *why = "tube";
      }
      return false;
    }
    return true;
  };
  const auto bad = kmpc::try_verified_hermite_smooth(samples, tube_fail);
  EXPECT_TRUE(bad.hermite_candidate);
  EXPECT_FALSE(bad.applied);
  EXPECT_FALSE(bad.hermite_verified);
  EXPECT_NE(std::string(bad.reason).find("tube"), std::string::npos)
      << bad.reason;
}

TEST(LinePath, HermiteEndpointsCloseMidBulgeIsCaught) {
  // Endpoints nearly identical; Hermite mid bulges off the chord.
  kmpc::JointPath samples(3);
  samples[0].setZero();
  samples[1].setZero();
  samples[2].setZero();
  samples[2][4] = 0.30;
  ASSERT_GE(kmpc::measure_path_corners(samples).max_tangent_jump_rad, 1e-4);
  EXPECT_GT(std::abs(kmpc::hermite_joint_at(samples, 0.5)[4]), 1e-3);

  auto reject_bulge = [](const kmpc::Vec18& q, std::string* why) {
    if (std::abs(q[4]) > 0.01) {
      if (why) {
        *why = "tube";
      }
      return false;
    }
    return true;
  };
  const auto bad = kmpc::try_verified_hermite_smooth(samples, reject_bulge);
  EXPECT_TRUE(bad.hermite_candidate);
  EXPECT_FALSE(bad.applied) << bad.reason;
  EXPECT_FALSE(bad.hermite_verified) << bad.reason;
  EXPECT_NE(std::string(bad.reason).find("tube"), std::string::npos)
      << bad.reason;
}

TEST(LinePath, HermiteAntisymmetricQuarterBulgeIsCaught) {
  // Knots [-0.2, 0, 0, 0.2]: segment [1,2] has endpoints=0 and mid=0, but
  // quarters ≈ ±0.009375. Mid-only early-exit would miss this.
  kmpc::JointPath samples(4);
  for (auto& s : samples) {
    s.setZero();
  }
  samples[0][4] = -0.2;
  samples[1][4] = 0.0;
  samples[2][4] = 0.0;
  samples[3][4] = 0.2;
  ASSERT_GE(kmpc::measure_path_corners(samples).max_tangent_jump_rad, 1e-4);
  EXPECT_NEAR(kmpc::hermite_joint_at(samples, 1.5)[4], 0.0, 1e-9);
  EXPECT_NEAR(kmpc::hermite_joint_at(samples, 1.25)[4], 0.009375, 1e-6);
  EXPECT_NEAR(kmpc::hermite_joint_at(samples, 1.75)[4], -0.009375, 1e-6);

  auto reject_quarter = [](const kmpc::Vec18& q, std::string* why) {
    if (std::abs(q[4]) > 0.005) {
      if (why) {
        *why = "tube";
      }
      return false;
    }
    return true;
  };
  const auto bad = kmpc::try_verified_hermite_smooth(samples, reject_quarter);
  EXPECT_TRUE(bad.hermite_candidate);
  EXPECT_FALSE(bad.applied) << bad.reason;
  EXPECT_FALSE(bad.hermite_verified) << bad.reason;
  EXPECT_NE(std::string(bad.reason).find("tube"), std::string::npos)
      << bad.reason;
}

namespace {

constexpr double kTestTubeTolM = 0.003;

// Production-like Hermite check_q (joint / CoM / pitch / FK tube).
template <class Kin, class Guards, class Limits>
auto make_prod_hermite_check(Kin& kin, const Guards& guards, const Limits& limits,
                             const std::string& frame, const kmpc::Vec3& start_xyz,
                             const kmpc::Vec3& target, double torso_pitch_max,
                             double start_pitch_abs) {
  return [=, &kin, &guards, &limits](const kmpc::Vec18& q,
                                     std::string* why) -> bool {
    if (!q.allFinite()) {
      if (why) {
        *why = "nonfinite";
      }
      return false;
    }
    for (int k = 0; k < 18; ++k) {
      if (q[k] < limits.q_min[k] - 1e-9 || q[k] > limits.q_max[k] + 1e-9) {
        if (why) {
          *why = "joint_limit";
        }
        return false;
      }
    }
    if (!guards.com_xy_inside(kin.com(q))) {
      if (why) {
        *why = "com";
      }
      return false;
    }
    if (torso_pitch_max > 0.0) {
      const double start_ex =
          std::max(0.0, start_pitch_abs - torso_pitch_max);
      const double ex =
          std::max(0.0, std::abs(q.template head<3>().sum()) - torso_pitch_max);
      if (ex > std::max(start_ex, kmpc::kStartPitchAdmitRad) +
                   kmpc::kPitchRoundEpsRad) {
        if (why) {
          *why = "pitch";
        }
        return false;
      }
    }
    const auto pose = kin.fk(q, frame);
    const kmpc::Vec3 ab = target - start_xyz;
    const double along = std::clamp(
        (pose.col(3).template head<3>() - start_xyz).dot(ab) /
            std::max(ab.squaredNorm(), 1e-12),
        0.0, 1.0);
    if ((pose.col(3).template head<3>() - (start_xyz + along * ab)).norm() >
        kTestTubeTolM) {
      if (why) {
        *why = "tube";
      }
      return false;
    }
    return true;
  };
}

bool on_cart_tube(kmpc::Kinematics& kin, const std::string& frame,
                  const kmpc::Vec3& start_xyz, const kmpc::Vec3& target,
                  const kmpc::Vec18& q, double tol_m = kTestTubeTolM) {
  const auto pose = kin.fk(q, frame);
  const kmpc::Vec3 ab = target - start_xyz;
  const double along = std::clamp(
      (pose.col(3).head<3>() - start_xyz).dot(ab) /
          std::max(ab.squaredNorm(), 1e-12),
      0.0, 1.0);
  return (pose.col(3).head<3>() - (start_xyz + along * ab)).norm() <= tol_m;
}

// Dense worker knots with a mild on-tube arm fold — keeps QP tractable while
// creating a Hermite candidate (tangent jump ≥ 1e-4).
bool build_cornered_on_tube(kmpc::Kinematics& kin,
                            const kmpc::CartesianTracker::LineRun& base,
                            kmpc::JointPath* out, std::string* why) {
  if (!out || base.samples.size() < 5) {
    if (why) {
      *why = "too_few_base_samples";
    }
    return false;
  }
  kmpc::JointPath knots = base.samples;
  if (kmpc::measure_path_corners(knots).max_tangent_jump_rad >= 1e-4) {
    *out = std::move(knots);
    return true;
  }
  const std::size_t mid_i = knots.size() / 2;
  const kmpc::Vec3 start_xyz = kin.fk_pos(knots.front(), base.frame);
  kmpc::Vec18 bent = knots[mid_i];
  constexpr int kFoldJ = 12;
  for (double fold : {0.04, 0.06, 0.08, 0.10}) {
    bent = knots[mid_i];
    bent[kFoldJ] += fold;
    for (int i = 0; i < 20 && !on_cart_tube(kin, base.frame, start_xyz,
                                             base.target, bent);
         ++i) {
      bent = knots[mid_i] + (bent - knots[mid_i]) * 0.7;
    }
    if (!on_cart_tube(kin, base.frame, start_xyz, base.target, bent)) {
      continue;
    }
    knots[mid_i] = bent;
    if (kmpc::measure_path_corners(knots).max_tangent_jump_rad >= 1e-4) {
      *out = std::move(knots);
      return true;
    }
  }
  if (why) {
    *why = "corner_too_small";
  }
  return false;
}

}  // namespace

TEST(LinePath, WorkerHermiteRealFkAdoptQpFinalGate) {
  // Hard requirement: prepare_line itself must enable VerifiedHermite.
  // Artificial corner construction belongs in a separate test — never fall
  // back here (would green-wash a near-linear prepare).
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 p0 = posed.col(3).head<3>();

  const kmpc::Vec3 goal_cands[] = {
      p0 + kmpc::Vec3(0.20, 0.08, -0.06),
      p0 + kmpc::Vec3(0.18, 0.05, -0.04),
      p0 + kmpc::Vec3(0.14, -0.06, -0.05),
      p0 + kmpc::Vec3(0.12, 0.10, 0.0),
      p0 + kmpc::Vec3(0.16, 0.0, -0.10),
  };
  kmpc::CartesianTracker::LineRun plan;
  kmpc::Vec3 goal = goal_cands[0];
  bool found = false;
  uint64_t gen = 11;
  for (const auto& g : goal_cands) {
    kmpc::CartesianTracker worker(kin, cfg_all);
    worker.set_line_chain_profile(
        kmpc::CartesianTracker::LineChainProfile::NewUnified);
    ASSERT_TRUE(worker.rearm_line(gen++));
    auto cand = worker.prepare_line(q, /*left=*/true, g, &quat);
    if (!cand.on || cand.samples.size() < 5) {
      continue;
    }
    if (cand.path_interp !=
        kmpc::CartesianTracker::LinePathInterp::VerifiedHermite) {
      continue;
    }
    ASSERT_TRUE(cand.path_smooth.hermite_verified && cand.path_smooth.applied)
        << cand.path_smooth.reason;
    plan = std::move(cand);
    goal = g;
    found = true;
    break;
  }
  ASSERT_TRUE(found) << "prepare_line did not enable Hermite on any candidate; "
                        "do not substitute an artificial path in this test";
  std::printf("hermite_case prepare=1 samples=%zu reason=%s\n",
              plan.samples.size(), plan.path_smooth.reason);

  const auto verified_knots = plan.samples;
  const auto verified_reason = std::string(plan.path_smooth.reason);

  kmpc::CartesianTracker main_tr(kin, cfg_all);
  main_tr.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(main_tr.rearm_line(20));
  main_tr.adopt_line(std::move(plan));
  EXPECT_EQ(main_tr.line_path_interp(),
            kmpc::CartesianTracker::LinePathInterp::VerifiedHermite);
  EXPECT_TRUE(main_tr.last_path_smooth_report().hermite_verified);
  for (std::size_t i = 0; i < verified_knots.size(); ++i) {
    EXPECT_NEAR((main_tr.line_samples()[i] - verified_knots[i]).norm(), 0.0,
                1e-12);
  }

  kmpc::OfflineChainConfig opt;
  opt.max_control_ticks = 1500;
  opt.speed_scale = 0.28;
  opt.accel_s = 0.30;
  opt.simulate_stream_advance = true;
  opt.stream_advance_frac = 0.08;
  opt.feedback = kmpc::FeedbackMode::IdealFinal;
  opt.reuse_active_line = true;
  const auto run = kmpc::run_offline_full_chain(
      main_tr, kin, cfg_all, q, goal, quat, /*left=*/true,
      kmpc::CartesianTracker::LineChainProfile::NewUnified, opt);
  ASSERT_TRUE(run.completed) << "fault=" << run.first_fault.reason
                             << " refuse=" << run.refuse_reason;
  EXPECT_TRUE(run.cmd_arrived);
  EXPECT_TRUE(run.feedback_arrived);
  EXPECT_EQ(main_tr.line_path_interp(),
            kmpc::CartesianTracker::LinePathInterp::VerifiedHermite);
  bool layers = false;
  for (const auto& t : run.ticks) {
    if (t.plan_ran && t.qp_ran && t.commit_ran && t.interp_ran && t.final_ran) {
      layers = true;
    }
  }
  EXPECT_TRUE(layers);
  EXPECT_EQ(scan_stream_axes(run, main_tr.line_direction_contract())
                .waist_reverse,
            0);

  const char* report_env = std::getenv("KMPC_CHAIN_REPORT_DIR");
  const std::string report_parent =
      report_env && report_env[0] ? std::string(report_env)
                                  : unique_report_dir();
  const std::string report_dir =
      kmpc::chain_report_slot_dir(report_parent, "hermite_full_chain");
  kmpc::ChainRunResult legacy_stub;
  legacy_stub.profile_name = "legacy_n_a";
  const std::string source_id = resolve_source_id();
  ASSERT_TRUE(kmpc::write_chain_compare_report(report_dir, legacy_stub, run,
                                               source_id.c_str()));
  std::printf("hermite full-chain report: %s source=%s smooth=%s\n",
              report_dir.c_str(), source_id.c_str(), verified_reason.c_str());
}

TEST(LinePath, ArtificialHermiteCornerAdoptAndVerify) {
  // Separate from prepare_line hard gate: constructs an on-tube corner, runs
  // production FK Hermite verify, adopts, checks interp handoff only.
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal =
      posed.col(3).head<3>() + kmpc::Vec3(0.18, 0.05, -0.04);

  kmpc::CartesianTracker worker(kin, cfg_all);
  worker.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(worker.rearm_line(31));
  auto plan = worker.prepare_line(q, /*left=*/true, goal, &quat);
  ASSERT_TRUE(plan.on) << plan.reason;
  kmpc::JointPath cornered;
  std::string why;
  ASSERT_TRUE(build_cornered_on_tube(kin, plan, &cornered, &why)) << why;
  const kmpc::Vec3 start_xyz = kin.fk_pos(cornered.front(), plan.frame);
  auto check_q = make_prod_hermite_check(
      kin, worker.guards(), cfg_all.limits, plan.frame, start_xyz, plan.target,
      cfg_all.params.torso_pitch_max_deg * kmpc::kPi / 180.0,
      std::abs(cornered.front().head<3>().sum()));
  const auto smooth = kmpc::try_verified_hermite_smooth(cornered, check_q);
  ASSERT_TRUE(smooth.hermite_verified && smooth.applied) << smooth.reason;
  plan.samples = cornered;
  plan.path_interp =
      kmpc::CartesianTracker::LinePathInterp::VerifiedHermite;
  plan.path_smooth = smooth;
  plan.corners = smooth.after;
  plan.geometry_c1 = smooth.geometry_c1;
  plan.direction_contract = kmpc::make_path_direction_contract(cornered);
  plan.direction_check =
      kmpc::check_path_direction(cornered, plan.direction_contract);
  ASSERT_TRUE(plan.direction_check.ok) << plan.direction_check.reason;

  kmpc::CartesianTracker main_tr(kin, cfg_all);
  main_tr.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(main_tr.rearm_line(31));
  main_tr.adopt_line(std::move(plan));
  EXPECT_EQ(main_tr.line_path_interp(),
            kmpc::CartesianTracker::LinePathInterp::VerifiedHermite);
  EXPECT_EQ(main_tr.line_samples().size(), cornered.size());
}

TEST(LinePath, HermiteRealFkTubeRejectKeepsLinear) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal =
      posed.col(3).head<3>() + kmpc::Vec3(0.16, 0.04, -0.03);

  kmpc::CartesianTracker worker(kin, cfg_all);
  worker.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(worker.rearm_line(3));
  auto plan = worker.prepare_line(q, /*left=*/true, goal, &quat);
  ASSERT_TRUE(plan.on) << plan.reason;
  ASSERT_GE(plan.samples.size(), 3u);

  // Endpoints-close / large outgoing tangent: knots stay near the tube, but
  // Hermite mid bulges in joint space and FK leaves the Cartesian tube.
  kmpc::JointPath knots(3);
  knots[0] = plan.samples.front();
  knots[1] = plan.samples.front();  // identical → end_span≈0 on segment 0
  knots[2] = plan.samples.back();
  const kmpc::Vec3 start_xyz = kin.fk_pos(knots[0], plan.frame);
  ASSERT_TRUE(on_cart_tube(kin, plan.frame, start_xyz, plan.target, knots[0]));
  ASSERT_TRUE(on_cart_tube(kin, plan.frame, start_xyz, plan.target, knots[1]));
  ASSERT_TRUE(on_cart_tube(kin, plan.frame, start_xyz, plan.target, knots[2]));
  ASSERT_GE(kmpc::measure_path_corners(knots).max_tangent_jump_rad, 1e-4);
  const auto mid = kmpc::hermite_joint_at(knots, 0.5);
  EXPECT_FALSE(on_cart_tube(kin, plan.frame, start_xyz, plan.target, mid))
      << "fixture must make Hermite mid leave the FK tube";

  const double start_pitch_abs =
      std::abs(knots.front().head<3>().sum());
  auto check_q = make_prod_hermite_check(
      kin, worker.guards(), cfg_all.limits, plan.frame, start_xyz, plan.target,
      cfg_all.params.torso_pitch_max_deg * kmpc::kPi / 180.0, start_pitch_abs);
  const auto smooth = kmpc::try_verified_hermite_smooth(knots, check_q);
  EXPECT_TRUE(smooth.hermite_candidate);
  EXPECT_FALSE(smooth.applied) << smooth.reason;
  EXPECT_FALSE(smooth.hermite_verified) << smooth.reason;
  EXPECT_NE(std::string(smooth.reason).find("tube"), std::string::npos)
      << smooth.reason;
  // Production policy on verify failure: keep linear (do not promote).
  EXPECT_FALSE(smooth.applied);
}

TEST(LinePath, StaleGenerationPlanMustNotAdopt) {
  // Boundary: uses the same line_async_plan_matches_live() as kmpc_node.
  // Covers "old worker future ready after a newer generation is already live".
  // Does not cover ROS future/mutex timing or waist_locked/q_seed drift alone
  // beyond the shared predicate fields.
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) +
                       "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal =
      posed.col(3).head<3>() + kmpc::Vec3(0.10, 0.0, 0.0);
  const kmpc::Vec3 goal_new =
      posed.col(3).head<3>() + kmpc::Vec3(0.12, 0.02, 0.0);

  kmpc::CartesianTracker worker(kin, cfg_all);
  worker.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(worker.rearm_line(7));
  auto plan = worker.prepare_line(q, /*left=*/true, goal, &quat);
  ASSERT_TRUE(plan.on) << plan.reason;
  const auto plan_ok = plan;  // copy for positive-control adopt

  kmpc::LineAsyncPendingSeed pending;
  pending.generation = 7;
  pending.left = true;
  pending.waist_locked = false;
  pending.xyz = goal;
  pending.quat = quat;
  pending.q_seed = q;

  // Newer generation already active with a different target (as if the node
  // rearmed while the old future was still building).
  EXPECT_FALSE(kmpc::line_async_plan_matches_live(
      pending, /*live_generation=*/8, true, false, goal_new, quat, q));

  kmpc::CartesianTracker main_tr(kin, cfg_all);
  main_tr.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(main_tr.rearm_line(8));
  EXPECT_FALSE(main_tr.line_active());
  if (kmpc::line_async_plan_matches_live(pending, 8, true, false, goal_new,
                                         quat, q)) {
    main_tr.adopt_line(std::move(plan));
  }
  EXPECT_FALSE(main_tr.line_active())
      << "stale worker plan must not become the live line";
  EXPECT_EQ(main_tr.goal_generation(), 8u);

  // Matching live seed still adopts (positive control of the shared gate).
  kmpc::CartesianTracker main_ok(kin, cfg_all);
  main_ok.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  ASSERT_TRUE(main_ok.rearm_line(7));
  ASSERT_TRUE(kmpc::line_async_plan_matches_live(pending, 7, true, false, goal,
                                                 quat, q));
  main_ok.adopt_line(plan_ok);
  EXPECT_TRUE(main_ok.line_active());
}

TEST(OfflineFullChain, ReportRefuseOverwriteExisting) {
  namespace fs = std::filesystem;
  const auto dir = fs::temp_directory_path() /
                   ("kmpc_chain_ow_" + std::to_string(
                       std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(dir);
  {
    std::ofstream md(dir / "summary.md");
    md << "sentinel\n";
  }
  unsetenv("KMPC_CHAIN_REPORT_OVERWRITE");
  kmpc::ChainRunResult a;
  a.profile_name = "legacy";
  kmpc::ChainRunResult b;
  b.profile_name = "new_unified";
  EXPECT_FALSE(kmpc::write_chain_compare_report(dir.string(), a, b, "test"));
  {
    std::ifstream in(dir / "summary.md");
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(in, line)));
    EXPECT_EQ(line, "sentinel");
  }
  setenv("KMPC_CHAIN_REPORT_OVERWRITE", "1", 1);
  EXPECT_TRUE(kmpc::write_chain_compare_report(dir.string(), a, b, "test"));
  unsetenv("KMPC_CHAIN_REPORT_OVERWRITE");
  fs::remove_all(dir);
}

TEST(OfflineFullChain, SharedReportDirUsesDistinctSlots) {
  namespace fs = std::filesystem;
  const auto parent = fs::temp_directory_path() /
                      ("kmpc_chain_slots_" + std::to_string(
                          std::chrono::steady_clock::now()
                              .time_since_epoch()
                              .count()));
  kmpc::ChainRunResult a;
  a.profile_name = "legacy";
  kmpc::ChainRunResult b;
  b.profile_name = "new_unified";
  const auto ab = kmpc::chain_report_slot_dir(parent.string(), "ab_compare");
  const auto herm =
      kmpc::chain_report_slot_dir(parent.string(), "hermite_full_chain");
  ASSERT_TRUE(kmpc::write_chain_compare_report(ab, a, b, "slot-ab"));
  ASSERT_TRUE(kmpc::write_chain_compare_report(herm, a, b, "slot-hermite"));
  EXPECT_TRUE(fs::exists(ab + "/summary.md"));
  EXPECT_TRUE(fs::exists(herm + "/summary.md"));
  EXPECT_NE(ab, herm);
  fs::remove_all(parent);
}
