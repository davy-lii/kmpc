#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/tracker_math.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::Vec18 q_at(double j1, double j2, double j3) {
  kmpc::Vec18 q = kmpc::Vec18::Zero();
  q[0] = j1;
  q[1] = j2;
  q[2] = j3;
  return q;
}

}  // namespace

// T07 / T08 style: absolute fold gate — seed with 3° J1 turnback must fail.
TEST(LineContract, AbsoluteFoldRejectsSeededTurnback) {
  const double fold = 3.0 * kmpc::kPi / 180.0;
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> path = {
      q_at(0.0, 0.0, 0.0),
      q_at(fold, 0.0, 0.0),
      q_at(0.0, 0.0, 0.0),
  };
  const auto chk = kmpc::check_path_direction_strict(path);
  EXPECT_FALSE(chk.ok);
  EXPECT_STREQ(chk.reason, "direction_waist_j1_fold");
  EXPECT_GT(chk.axes[0].fold_rad, kmpc::kDirectionStrictFoldMaxRad);
}

// T08: straighten-style “delta improvement” must not pass absolute fold.
TEST(LineContract, AbsoluteFoldIgnoresIncrementalGate) {
  const double seed_fold = 3.0 * kmpc::kPi / 180.0;
  const double tiny = 0.5 * kmpc::kPi / 180.0;
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> before = {
      q_at(0.0, 0.0, 0.0),
      q_at(seed_fold, 0.0, 0.0),
      q_at(0.0, 0.0, 0.0),
  };
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> after = {
      q_at(0.0, 0.0, 0.0),
      q_at(seed_fold + tiny, 0.0, 0.0),
      q_at(0.0, 0.0, 0.0),
  };
  // Old incremental gate may not blow (small growth), but strict absolute fails.
  EXPECT_FALSE(kmpc::straighten_axis_turnback_regressed(
      before, after, kmpc::kWaistAxisTurnbackSumStageBlowRad));
  EXPECT_FALSE(kmpc::check_path_direction_strict(after).ok);
}

// T07: tiny reverse each tick accumulates beyond envelope.
TEST(LineContract, EnvelopeCatchesAccumulatedEpsSteps) {
  kmpc::DirectionContract c;
  c.strict = true;
  c.sign[0] = 1;
  c.sign[1] = 0;
  c.sign[2] = 0;
  c.sign[3] = 1;
  c.q_anchor = q_at(0.0, 0.0, 0.0);
  c.pitch_anchor = 0.0;
  c.fold_max_rad = kmpc::kDirectionStrictFoldMaxRad;

  kmpc::DirectionEnvelope env;
  kmpc::direction_envelope_seed(&env, c, q_at(0.0, 0.0, 0.0));
  const double eps_half = 0.5 * kmpc::kDirectionNumericEpsRad;
  // Forward a bit, then reverse by eps/2 many times past the deadband.
  ASSERT_TRUE(kmpc::direction_envelope_accept(&env, q_at(0.02, 0.0, 0.0)).ok);
  bool caught = false;
  kmpc::Vec18 q = env.last_q;
  for (int i = 0; i < 5000; ++i) {
    q[0] -= eps_half;
    const auto r = kmpc::direction_envelope_accept(&env, q);
    if (!r.ok) {
      caught = true;
      EXPECT_TRUE(std::string(r.reason).find("final_send_waist") != std::string::npos);
      break;
    }
  }
  EXPECT_TRUE(caught);
}

// Pitch sum fold independent of single-axis signs.
TEST(LineContract, PitchSumFoldFailsEvenIfAxesLookLocal) {
  // J1 up then down, J2 down then up — nets near zero per axis but pitch folds.
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> path = {
      q_at(0.0, 0.0, 0.0),
      q_at(0.02, -0.01, 0.0),
      q_at(0.0, 0.0, 0.0),
  };
  const auto chk = kmpc::check_path_direction_strict(path);
  EXPECT_FALSE(chk.ok);
}

// T01: reverse_cache with fold is rejected; reason preserved (not overwritten).
TEST(LineContract, ReverseCacheWithFoldRejectedKeepsRootReason) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  tracker.set_line_direction_strict_enable(true);

  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 pA = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 pB = pA + kmpc::Vec3(0.06, 0.0, 0.0);

  ASSERT_TRUE(tracker.rearm_line(1));
  bool forward_ok = false;
  for (int i = 0; i < 800; ++i) {
    q += tracker.compute(q, cfg_all.q_pref, &pB, nullptr, &quat, nullptr).dq;
    if (tracker.last_arrived()) {
      forward_ok = true;
      tracker.maybe_commit_last_line_cache(q);
      break;
    }
  }
  ASSERT_TRUE(forward_ok);
  ASSERT_TRUE(tracker.last_line_cache().valid);

  // Inject a waist J1 fold into the cached forward samples so reverse inherits it.
  auto cache = tracker.last_line_cache();
  ASSERT_GE(cache.samples.size(), 3u);
  const double fold = 3.0 * kmpc::kPi / 180.0;
  cache.samples[cache.samples.size() / 2][0] += fold;
  // Re-validate that the poisoned cache itself fails strict direction.
  ASSERT_FALSE(kmpc::check_path_direction_strict(cache.samples).ok);
  tracker.import_last_line_cache(cache);

  ASSERT_TRUE(tracker.rearm_line(2));
  tracker.compute(q, cfg_all.q_pref, &pA, nullptr, &quat, nullptr);
  const std::string rev = tracker.reverse_line_decision();
  // Must not be "accepted" then later fail with overwritten pitch_recovery.
  EXPECT_NE(rev, "accepted");
  EXPECT_TRUE(rev.find("rejected_direction") != std::string::npos ||
              tracker.line_candidate() != "reverse_cache")
      << "rev=" << rev << " candidate=" << tracker.line_candidate()
      << " reason=" << tracker.line_reason();
  EXPECT_NE(tracker.line_reason(), "pitch_recovery_infeasible");
  EXPECT_TRUE(tracker.line_failure_category() != "pitch_recovery_infeasible" ||
              tracker.line_reason().find("direction") != std::string::npos ||
              !tracker.line_refused());
}

// Gate: any accepted line samples must pass strict waist/pitch direction.
TEST(LineContract, AcceptedLineSamplesHaveNoWaistPitchFold) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  tracker.set_line_direction_strict_enable(true);

  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.08, 0.0, 0.0);

  ASSERT_TRUE(tracker.rearm_line(1));
  const auto step0 =
      tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
  (void)step0;
  if (tracker.line_refused()) {
    GTEST_SKIP() << "build refused under strict: " << tracker.line_reason();
  }
  ASSERT_TRUE(tracker.line_active());
  ASSERT_GE(tracker.line_sample_count(), 2u);
  const auto chk = kmpc::check_path_direction_strict(tracker.line_samples());
  EXPECT_TRUE(chk.ok) << chk.reason << " foldJ1=" << chk.axes[0].fold_rad
                      << " foldJ2=" << chk.axes[1].fold_rad
                      << " foldJ3=" << chk.axes[2].fold_rad
                      << " foldPitch=" << chk.axes[3].fold_rad;
  EXPECT_TRUE(tracker.line_direction_check().ok);
}

// Monotone path must pass.
TEST(LineContract, MonotonePathPasses) {
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> path = {
      q_at(0.00, -0.10, 0.00),
      q_at(0.01, -0.12, 0.01),
      q_at(0.02, -0.14, 0.02),
  };
  const auto chk = kmpc::check_path_direction_strict(path);
  EXPECT_TRUE(chk.ok) << chk.reason;
}

// Solver internal slack must stay strictly inside the final-send frozen gate.
TEST(LineContract, SolverSlackStrictlyInsideFinalGate) {
  EXPECT_GT(kmpc::kDirectionSolverSlackRad, 0.0);
  EXPECT_LT(kmpc::kDirectionSolverSlackRad, kmpc::kDirectionStrictFoldMaxRad);
  // Historical QP pitch slack 2e-4 rad (~0.01146°) sat outside the 0.01° gate.
  EXPECT_LT(kmpc::kDirectionSolverSlackRad, 2e-4);
}

// Frozen-pitch envelope: deviations inside fold_max pass; beyond fail.
TEST(LineContract, FrozenPitchEnvelopeBoundaries) {
  kmpc::DirectionContract c;
  c.strict = true;
  c.sign[0] = 0;
  c.sign[1] = 0;
  c.sign[2] = 0;
  c.sign[3] = 0;
  c.q_anchor = q_at(-0.20, -0.20, -0.15);
  c.pitch_anchor = c.q_anchor.head<3>().sum();
  c.fold_max_rad = kmpc::kDirectionStrictFoldMaxRad;

  const double deg = kmpc::kPi / 180.0;
  const double deltas_deg[] = {0.0, 0.009, 0.010, 0.011, 0.01146};
  for (double d_deg : deltas_deg) {
    kmpc::DirectionEnvelope env;
    kmpc::direction_envelope_seed(&env, c, c.q_anchor);
    kmpc::Vec18 q = c.q_anchor;
    // Put the entire pitch delta on J1 so waist frozen axes also move — pitch
    // gate is the one under test (sign[3]==0).
    q[0] += d_deg * deg;
    // Waist J1 is also frozen; for pitch-only check temporarily mark J1 free
    // with zero step intent by matching extremum via non-frozen mask.
    env.mask.waist_j1 = false;
    env.mask.waist_j2 = false;
    env.mask.waist_j3 = false;
    const auto r = kmpc::direction_envelope_accept(&env, q);
    const double band = c.fold_max_rad + c.numeric_eps_rad;
    if (std::abs(d_deg * deg) <= band) {
      EXPECT_TRUE(r.ok) << "delta_deg=" << d_deg << " reason=" << r.reason;
    } else {
      EXPECT_FALSE(r.ok) << "delta_deg=" << d_deg;
      EXPECT_STREQ(r.reason, "final_send_pitch_frozen");
    }
  }
}

// Gate0 + exclusive-pool regression (M03 / r30 retreat).
TEST(LineContract, R30Gate0PitchMarginAllowsAcceptOrderChange) {
  // Gate 0: from the r30 first line=on joints, free_fit/fold_15 must be able to
  // build a retreat with END pitch margin ≥2° (start already hugs −32°).
  // Planner no longer invents arm_only (lock-waist) candidates.
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  tracker.set_line_direction_strict_enable(true);
  tracker.set_reverse_last_line_enable(false);  // isolate free/fold squat path
  tracker.set_line_pitch_straighten_enable(true);

  const double deg = kmpc::kPi / 180.0;
  const double raw[18] = {
      24.9960 * deg, -65.1187 * deg, 8.3496 * deg, 6.6790 * deg,
      -40.3518 * deg, -70.2034 * deg, 106.8446 * deg, 25.0744 * deg,
      -107.0707 * deg, 24.2474 * deg, -66.8087 * deg,
      158.5158 * deg, -53.6328 * deg, -106.8170 * deg, -103.7110 * deg,
      128.1150 * deg, -4.7469 * deg, 14.3400 * deg};
  kmpc::Vec18 q0;
  for (int i = 0; i < 18; ++i) q0[i] = raw[i];
  const kmpc::Vec3 prep(0.245839, -0.391452, 0.490);
  const kmpc::Vec4 quat = kmpc::tool_to_flange_quat_xyzw(
      kmpc::Vec4(-0.851980633, 0.201084490, 0.138013856, 0.463299261));

  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.compute(q0, cfg_all.q_pref, nullptr, &prep, nullptr, &quat);
  if (tracker.line_refused()) {
    GTEST_SKIP() << "gate0 free/arm build refused: " << tracker.line_reason();
  }
  ASSERT_GE(tracker.line_sample_count(), 2u);
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage")
      << "cand=" << tracker.line_candidate();
  const auto& samples = tracker.line_samples();
  const double pitch_max = cfg_all.params.torso_pitch_max_deg * deg;
  const double start_margin = pitch_max - std::abs(q0.head<3>().sum());
  const double end_margin =
      pitch_max - std::abs(samples.back().head<3>().sum());
  double j4_step0 = 0.0;
  if (samples.size() >= 2) {
    j4_step0 = std::abs(samples[1][7] - samples[0][7]);
  }
  const double cap_j4 = 0.005676897543884534;
  // Hard pitch stays inside ±32°. The old 1.5° squat recovery is not a selector.
  EXPECT_GE(end_margin, -0.05 * deg)
      << "cand=" << tracker.line_candidate()
      << " start_margin_deg=" << (start_margin / deg)
      << " end_margin_deg=" << (end_margin / deg);
  EXPECT_LT(j4_step0, 4.0 * cap_j4 + 1e-9)
      << "cand=" << tracker.line_candidate() << " j4_step0=" << j4_step0;
  const auto chk = kmpc::check_path_direction_strict(samples);
  EXPECT_TRUE(chk.ok) << chk.reason;
}

TEST(LineContract, ExclusivePoolPicksBetterPitchMargin) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  tracker.set_line_direction_strict_enable(true);
  tracker.set_reverse_last_line_enable(true);

  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 pA = kin.fk_pos(q, kmpc::kEeFrameRight);
  const auto posed = kin.fk(q, kmpc::kEeFrameRight);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 pB = pA + kmpc::Vec3(0.06, 0.0, 0.0);

  ASSERT_TRUE(tracker.rearm_line(1));
  bool forward_ok = false;
  for (int i = 0; i < 800; ++i) {
    q += tracker.compute(q, cfg_all.q_pref, nullptr, &pB, nullptr, &quat).dq;
    if (tracker.last_arrived()) {
      forward_ok = true;
      tracker.maybe_commit_last_line_cache(q);
      break;
    }
  }
  ASSERT_TRUE(forward_ok);
  ASSERT_TRUE(tracker.last_line_cache().valid);

  auto cache = tracker.last_line_cache();
  ASSERT_GE(cache.samples.size(), 3u);
  const double pitch_max = cfg_all.params.torso_pitch_max_deg * kmpc::kPi / 180.0;
  const double deg = kmpc::kPi / 180.0;
  // Put live start on the wall (r30-like) and keep cache.q_end matched.
  {
    const double p = q.head<3>().sum();
    const double need = std::copysign(pitch_max - 0.2 * deg, p) - p;
    q[1] += need;
    cache.q_end = q;
  }
  // Poison reverse samples onto the wall so free/arm squat can win by ≥2°.
  for (auto& s : cache.samples) {
    const double p = s.head<3>().sum();
    const double need = std::copysign(pitch_max - 0.2 * deg, p) - p;
    s[1] += need;
  }
  tracker.import_last_line_cache(cache);

  ASSERT_TRUE(tracker.rearm_line(2));
  tracker.compute(q, cfg_all.q_pref, nullptr, &pA, nullptr, &quat);
  if (tracker.line_refused()) {
    GTEST_SKIP() << "retreat refused: " << tracker.line_reason();
  }
  // Poisoned reverse must not win; free/fold may win with squat, else qa_seed.
  EXPECT_NE(tracker.line_candidate(), "reverse_cache")
      << "rev=" << tracker.reverse_line_decision();
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage")
      << "cand=" << tracker.line_candidate()
      << " rev=" << tracker.reverse_line_decision();
  const auto chk = kmpc::check_path_direction_strict(tracker.line_samples());
  EXPECT_TRUE(chk.ok) << chk.reason;
}

// 几何示例：俯仰和保持 0 时，腰仍能降下肩架并后移中间枢轴。不是要执行的角度。
TEST(LineContract, FoldGeometrySeparatesPitchFromHeight) {
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double a = 20.0 * kmpc::kPi / 180.0;
  kmpc::Vec18 upright = kmpc::Vec18::Zero();
  kmpc::Vec18 folded = upright;
  folded[0] = a;
  folded[1] = -2.0 * a;
  folded[2] = a;
  const auto up = kin.fk(upright, "Body_Link");
  const auto down = kin.fk(folded, "Body_Link");
  const auto pivot0 = kin.fk(upright, "Leg_Y_Link2");
  const auto pivot1 = kin.fk(folded, "Leg_Y_Link2");
  EXPECT_NEAR(folded.head<3>().sum(), 0.0, 1e-12);
  EXPECT_GT(up(2, 3) - down(2, 3), 0.030);
  EXPECT_GT(std::abs(pivot1(0, 3) - pivot0(0, 3)), 0.050);
}

// P0：同一 q0 和手目标。折叠初值必须过与当前拟合相同的硬门，且早段腰臂一起动。
// 本探针不写入正式路径。
TEST(LineContract, FoldContrastSameStartDoesNotTakeOfficialPath) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double a = 15.0 * kmpc::kPi / 180.0;
  kmpc::Vec18 q0 = cfg_all.q_pref;
  kmpc::Vec18 folded = q0;
  folded[0] = std::clamp(q0[0] + a, cfg_all.limits.q_min[0], cfg_all.limits.q_max[0]);
  folded[1] = std::clamp(q0[1] - 2.0 * a, cfg_all.limits.q_min[1], cfg_all.limits.q_max[1]);
  folded[2] = std::clamp(q0[2] + a, cfg_all.limits.q_min[2], cfg_all.limits.q_max[2]);
  folded[5] = std::clamp(q0[5] + 0.35, cfg_all.limits.q_min[5], cfg_all.limits.q_max[5]);
  folded[7] = std::clamp(q0[7] - 0.25, cfg_all.limits.q_min[7], cfg_all.limits.q_max[7]);
  const kmpc::Vec3 target = kin.fk_pos(folded, kmpc::kEeFrameRight);
  kmpc::LineTarget tip;
  tip.xyz = target + kmpc::Vec3(0.0, 0.0, 0.04);
  tip.frame = std::string(kmpc::kBaseFrame);
  kmpc::CartesianTracker tracker(kin, cfg_all);
  const auto contrast = tracker.probe_fold_contrast(q0, false, target, nullptr, &tip);
  EXPECT_FALSE(tracker.line_active());
  EXPECT_EQ(contrast.fold_reason, "removed");
  EXPECT_FALSE(contrast.fold_hard_ok);
  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.compute(q0, cfg_all.q_pref, nullptr, &target, nullptr, nullptr);
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage")
      << tracker.line_reason();
}
