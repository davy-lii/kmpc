#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/constraint_policy.hpp"
#include "kmpc/fast_trajectory_optimizer.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/motion_task.hpp"
#include "kmpc/tracker_math.hpp"
#include "kmpc/trajectory_follower.hpp"
#include "kmpc/whole_body_guards.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

double pitch_sum(const kmpc::Vec18& q) { return q[0] + q[1] + q[2]; }

// 俯仰相对整段方向走回头的最大幅度。平的一段则是峰谷差。
double pitch_turnback(const std::vector<kmpc::Vec18>& qs) {
  if (qs.size() < 2) {
    return 0.0;
  }
  const double net = pitch_sum(qs.back()) - pitch_sum(qs.front());
  const double dir = net > 1e-4 ? 1.0 : (net < -1e-4 ? -1.0 : 0.0);
  if (dir == 0.0) {
    double lo = pitch_sum(qs.front());
    double hi = lo;
    for (const auto& q : qs) {
      lo = std::min(lo, pitch_sum(q));
      hi = std::max(hi, pitch_sum(q));
    }
    return hi - lo;
  }
  double best = pitch_sum(qs.front());
  double worst = 0.0;
  for (const auto& q : qs) {
    const double p = pitch_sum(q);
    if (dir > 0.0) {
      best = std::max(best, p);
      worst = std::max(worst, best - p);
    } else {
      best = std::min(best, p);
      worst = std::max(worst, p - best);
    }
  }
  return worst;
}

double along_m(const kmpc::Vec3& p, const kmpc::Vec3& a, const kmpc::Vec3& b) {
  const kmpc::Vec3 ab = b - a;
  const double n = ab.norm();
  if (n < 1e-12) {
    return 0.0;
  }
  return (p - a).dot(ab) / n;
}

double lateral(const kmpc::Vec3& p, const kmpc::Vec3& a, const kmpc::Vec3& b) {
  const kmpc::Vec3 ab = b - a;
  const double n2 = ab.squaredNorm();
  if (n2 < 1e-12) {
    return (p - a).norm();
  }
  double t = (p - a).dot(ab) / n2;
  if (t < 0.0) {
    t = 0.0;
  } else if (t > 1.0) {
    t = 1.0;
  }
  return (p - (a + t * ab)).norm();
}

}  // namespace

TEST(LinePath, AxisTurnbackIncreaseBlowsSumStraighten) {
  using Sample = std::array<double, 3>;
  // Default 0.05° increase threshold catches a mild new J3 fold.
  const std::vector<Sample> mild_sum = {
      {0.0, 0.0, 0.0},
      {0.002, 0.0, 0.0},
      {0.0, 0.0, 0.0},
  };
  const std::vector<Sample> axis_costly = {
      {0.0, 0.0, 0.0},
      {0.002, 0.0, -0.002},
      {0.0, 0.0, 0.0},
  };
  EXPECT_TRUE(kmpc::straighten_axis_turnback_regressed(mild_sum, axis_costly));

  // r20-style: J3 grows by ~3.4° while sum improves — blow.
  const std::vector<Sample> sum_bump = {
      {0.0, 0.0, 0.0},
      {0.05, 0.0, 0.0},
      {0.0, 0.0, 0.0},
  };
  const std::vector<Sample> sum_repaired = {
      {0.0, 0.0, 0.0},
      {0.001, 0.0, -0.06},
      {0.002, 0.0, 0.0},
  };
  EXPECT_TRUE(kmpc::straighten_axis_turnback_regressed(
      sum_bump, sum_repaired, kmpc::kWaistAxisTurnbackSumStageBlowRad));
  // Seed already has a large J1 fold; a small extra move is not a blow.
  const std::vector<Sample> seeded = {
      {0.0, 0.0, 0.0},
      {-0.05, 0.0, 0.0},
      {0.0, 0.0, 0.0},
  };
  const std::vector<Sample> seeded_fixed = {
      {0.0, 0.0, 0.0},
      {-0.052, 0.0, 0.0},
      {0.01, 0.0, 0.0},
  };
  EXPECT_FALSE(kmpc::straighten_axis_turnback_regressed(
      seeded, seeded_fixed, kmpc::kWaistAxisTurnbackSumStageBlowRad));

  const double tiny = 0.04 * kmpc::kPi / 180.0;
  const std::vector<Sample> tiny_back = {
      {0.0, 0.0, 0.0},
      {0.0, 0.0, -tiny},
      {0.0, 0.0, 0.0},
  };
  const std::vector<Sample> flat = {
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
  };
  EXPECT_FALSE(kmpc::straighten_axis_turnback_regressed(flat, tiny_back));
}

TEST(LinePath, PitchPlanningCapLeavesMarginOnDescent) {
  const double hard = 32.0 * kmpc::kPi / 180.0;
  const double start = 0.3491;
  const double cap = kmpc::pitch_planning_cap_rad(hard, start, -0.4948);
  EXPECT_LT(cap, hard - 1.5 * kmpc::kPi / 180.0);
  EXPECT_GE(cap, start);
  const double hard30 = 30.0 * kmpc::kPi / 180.0;
  const double cap30 =
      kmpc::pitch_planning_cap_rad(hard30, start, -0.4948);
  EXPECT_NEAR(cap30, hard30 - 2.0 * kmpc::kPi / 180.0, 1e-9);
}

TEST(LinePath, PitchBoxAndAdmitAreShared) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  EXPECT_DOUBLE_EQ(cfg_all.params.torso_pitch_max_deg, 32.0);
  EXPECT_DOUBLE_EQ(kmpc::kStartPitchAdmitRad, 0.005);
  // r33→r34: recovery first-chord EE retreat budget (meters).
  EXPECT_DOUBLE_EQ(kmpc::kStreamRecoverProgressSlackM, 1e-3);
  const double hard = 32.0 * kmpc::kPi / 180.0;
  // Ordinary start-inside: gate = hard + round only (never +admit).
  EXPECT_DOUBLE_EQ(kmpc::pitch_abs_bound(hard, hard - 0.01, true),
                   hard + kmpc::kPitchRoundEpsRad);
  EXPECT_DOUBLE_EQ(kmpc::pitch_abs_bound(hard, hard - 0.01, false), hard);
  // Recovery start-outside: envelope is |start|, no permanent admit widen.
  EXPECT_DOUBLE_EQ(kmpc::pitch_abs_bound(hard, hard + 0.002, true),
                   hard + 0.002 + kmpc::kPitchRoundEpsRad);
  EXPECT_FALSE(kmpc::pitch_abs_ok(hard + kmpc::kStartPitchAdmitRad, hard,
                                  hard - 0.01, false));
}

// Unified pitch domain: ordinary stream gate matches filtered hard+round.
// Field float crumbs must be projected by QP — gates do not widen by admit.
TEST(LinePath, StreamPitchGateMatchesHardRoundDomain) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double limit = cfg_all.params.torso_pitch_max_deg * kmpc::kPi / 180.0;
  ASSERT_GT(limit, 0.0);

  auto set_pitch = [](kmpc::Vec18 q, double want) {
    q[0] += want - (q[0] + q[1] + q[2]);
    return q;
  };

  const double home_raw[18] = {
      0.69813170, -0.87264718, -0.17453293, -0.02630211,
      -0.00171042, 0.51532590, -0.13651965, -2.12657900,
      -1.27522486, -0.24291493, 0.01394518,
      -0.00171042, 0.51532590, 0.13651965, -2.12657900,
      1.27522486, -0.24291493, 0.01394518};
  kmpc::Vec18 q;
  for (int i = 0; i < 18; ++i) q[i] = home_raw[i];
  ASSERT_LT(std::abs(pitch_sum(q)), limit - 0.05);

  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  tracker.set_line_direction_strict_enable(false);
  tracker.set_line_pitch_straighten_enable(false);
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
  ASSERT_TRUE(tracker.line_active()) << tracker.line_reason();
  ASSERT_GE(tracker.line_sample_count(), 2u);

  // Domain helper: ordinary crumb past hard is illegal for planning/gates.
  const double field_over = 3.9e-5;
  EXPECT_FALSE(kmpc::pitch_abs_ok(limit + field_over, limit, limit - 0.05,
                                  /*for_gate=*/true));
  EXPECT_TRUE(kmpc::pitch_abs_ok(limit + 0.5 * kmpc::kPitchRoundEpsRad, limit,
                                 limit - 0.05, /*for_gate=*/true));

  // Stream segment: both ends past hard so pitch trips before tube.
  kmpc::Vec18 far = set_pitch(q, -limit - kmpc::kStartPitchAdmitRad - 1e-3);
  const auto bad = tracker.line_stream_segment_check(far, far);
  EXPECT_FALSE(bad.ok);
  EXPECT_STREQ(bad.reason, "pitch");
}

TEST(LinePath, HermiteLineAtIsC1AtInteriorKnots) {
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> samples(4);
  for (int i = 0; i < 4; ++i) {
    samples[static_cast<std::size_t>(i)].setZero();
    samples[static_cast<std::size_t>(i)][0] = 0.01 * i * i;  // quadratic pitch axis
    samples[static_cast<std::size_t>(i)][4] = 0.02 * i;
  }
  // Value continuity at knot 1 and 2.
  for (double knot : {1.0, 2.0}) {
    const kmpc::Vec18 a = kmpc::hermite_joint_at(samples, knot - 1e-9);
    const kmpc::Vec18 b = kmpc::hermite_joint_at(samples, knot + 1e-9);
    EXPECT_NEAR((a - b).norm(), 0.0, 1e-9);
  }
  // Tangent continuity: left/right finite differences match at interior knots.
  auto slope = [&](double p) {
    const double h = 1e-6;
    return (kmpc::hermite_joint_at(samples, p + h) -
            kmpc::hermite_joint_at(samples, p - h)) /
           (2.0 * h);
  };
  for (double knot : {1.0, 2.0}) {
    const kmpc::Vec18 sl = slope(knot - 2e-5);
    const kmpc::Vec18 sr = slope(knot + 2e-5);
    EXPECT_NEAR((sl - sr).norm(), 0.0, 5e-4) << "knot=" << knot;
  }
}

TEST(LinePath, SlerpTakesShortArc) {
  const kmpc::Vec4 a(0.0, 0.0, 0.0, 1.0);
  const kmpc::Vec4 b(0.0, 0.0, 1.0, 0.0);
  const kmpc::Vec4 mid = kmpc::quat_slerp_xyzw(a, -b, 0.5);
  EXPECT_NEAR(mid.norm(), 1.0, 1e-9);
  EXPECT_GT(mid[3], 0.5);
}

TEST(LinePath, ShortMoveStaysOnLineThenHolds) {
  const kmpc::Config cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.0, 0.06, 0.0);
  const Eigen::Matrix4d posed = kin.fk(q0, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat =
      kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));

  kmpc::Vec18 q = q0;
  double worst = 0.0;
  bool arrived = false;
  bool used_line = false;
  for (int k = 0; k < 250; ++k) {
    const auto step = tracker.compute(q, q0, &goal, nullptr, &quat, nullptr);
    used_line = used_line || tracker.line_active();
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    ASSERT_TRUE(step.dq.allFinite());
    const kmpc::Vec18 cap = tracker.effective_dq_max() * cfg_all.params.dt;
    for (int j = 0; j < kmpc::kNDof; ++j) {
      EXPECT_LE(std::abs(step.dq[j]), cap[j] + 1e-6) << "joint " << j;
    }
    q += step.dq;
    const kmpc::Vec3 p = kin.fk_pos(q, kmpc::kEeFrameLeft);
    worst = std::max(worst, lateral(p, p0, goal));
    if (tracker.last_arrived()) {
      arrived = true;
      for (int h = 0; h < 15; ++h) {
        const auto hold = tracker.compute(q, q0, &goal, nullptr, &quat, nullptr);
        EXPECT_LT(hold.dq.cwiseAbs().maxCoeff(), 1e-9);
        EXPECT_TRUE(tracker.last_arrived());
        EXPECT_TRUE(tracker.line_holding());
      }
      const kmpc::Vec3 nudged = goal + kmpc::Vec3(0.005, 0.0, 0.0);
      const auto ignore = tracker.compute(q, q0, &nudged, nullptr, &quat, nullptr);
      EXPECT_LT(ignore.dq.cwiseAbs().maxCoeff(), 1e-9);
      EXPECT_TRUE(tracker.last_arrived());
      EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameLeft) - goal).norm(), 0.003);

      const kmpc::Vec3 again = goal + kmpc::Vec3(0.0, 0.12, 0.0);
      const kmpc::Vec3 from = kin.fk_pos(q, kmpc::kEeFrameLeft);
      ASSERT_TRUE(tracker.rearm_line(2));
      bool again_arrived = false;
      double again_worst = 0.0;
      for (int n = 0; n < 400; ++n) {
        const auto step2 = tracker.compute(q, q0, &again, nullptr, &quat, nullptr);
        EXPECT_TRUE(tracker.line_active());
        q += step2.dq;
        const kmpc::Vec3 p = kin.fk_pos(q, kmpc::kEeFrameLeft);
        again_worst = std::max(again_worst, lateral(p, from, again));
        if (tracker.last_arrived()) {
          again_arrived = true;
          const auto held = tracker.compute(q, q0, &again, nullptr, &quat, nullptr);
          EXPECT_LT(held.dq.cwiseAbs().maxCoeff(), 1e-9);
          break;
        }
      }
      EXPECT_TRUE(again_arrived);
      EXPECT_LE(again_worst, 0.003);
      EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameLeft) - again).norm(), 0.003);
      break;
    }
  }
  EXPECT_TRUE(used_line);
  EXPECT_TRUE(arrived);
  EXPECT_LE(worst, 0.003) << "hand left the straight segment";
}

TEST(LinePath, DriftDoesNotBendThePath) {
  const kmpc::Config cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker a(kin, cfg_all);
  kmpc::CartesianTracker b(kin, cfg_all);
  a.set_line_pace(1.0, 0.0);
  b.set_line_pace(1.0, 0.0);
  const double qraw[18] = {
      0.69813170, -0.87264718, -0.17453293, -0.02630211,
      -0.00171042, 0.51532590, -0.13651965, -2.12657900,
      -1.27522486, -0.24291493, 0.01394518,
      -0.00171042, 0.51532590, 0.13651965, -2.12657900,
      1.27522486, -0.24291493, 0.01394518};
  kmpc::Vec18 q;
  for (int i = 0; i < 18; ++i) {
    q[i] = qraw[i];
  }
  const kmpc::Vec3 goal(0.4984, -0.2867, 0.8100);
  const kmpc::Vec4 quat=kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(-0.7853, 0.2042, 0.3285, 0.4834));
  for (int k = 0; k < 5; ++k) {
    const auto step = a.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
    b.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
    q += step.dq;
  }
  kmpc::Vec18 drifted = q;
  drifted[10] += 0.08;
  a.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
  b.compute(drifted, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
  EXPECT_LT((a.line_cmd() - b.line_cmd()).norm(), 1e-9);
}

TEST(LinePath, LoggedPrepAtScaleStaysOnLine) {
  const kmpc::Config cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  // Match robot03 NewUnified: Mode B follows the built joint path. Legacy Mode A
  // reshapes onto a hard Cartesian line inside ±0.015 rad trust and refuses
  // fold_15 soft-tube bows (qp_status_1 / s*_trust_*).
  tracker.set_line_chain_profile(
      kmpc::CartesianTracker::LineChainProfile::NewUnified);
  tracker.set_line_pace(0.3, 0.50);
  const double qraw[18] = {
      0.69813170, -0.87264718, -0.17453293, -0.02630211,
      -0.00171042, 0.51532590, -0.13651965, -2.12657900,
      -1.27522486, -0.24291493, 0.01394518,
      -0.00171042, 0.51532590, 0.13651965, -2.12657900,
      1.27522486, -0.24291493, 0.01394518};
  kmpc::Vec18 q;
  for (int i = 0; i < 18; ++i) {
    q[i] = qraw[i];
  }
  const kmpc::Vec3 goal(0.4984, -0.2867, 0.8100);
  const kmpc::Vec4 quat=kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(-0.7853, 0.2042, 0.3285, 0.4834));
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameRight);
  double worst = 0.0;
  bool arrived = false;
  const kmpc::Vec18 cap = tracker.effective_dq_max() * cfg_all.params.dt * 0.3;
  for (int k = 0; k < 800 && !arrived; ++k) {
    const kmpc::Vec18 from = q;
    const auto step = tracker.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
    ASSERT_FALSE(tracker.line_refused())
        << "tick=" << k << " reason=" << tracker.line_reason()
        << " cand=" << tracker.line_candidate()
        << " pose=" << tracker.line_pose_report()
        << " pitch_deg=" << (q.head<3>().sum() * 180.0 / kmpc::kPi)
        << " samples=" << tracker.line_sample_count()
        << " qp_status=" << tracker.last_line_qp_fail().status
        << " qp_tag=" << tracker.last_line_qp_fail().worst_tag;
    ASSERT_TRUE(step.dq.allFinite());
    for (int j = 0; j < kmpc::kNDof; ++j) {
      EXPECT_LE(std::abs(step.dq[j]), cap[j] + 1e-6) << "joint " << j;
    }
    const kmpc::Vec18 to = tracker.line_cmd();
    const kmpc::Vec18 mid = 0.5 * (from + to);
    worst = std::max(worst, lateral(kin.fk_pos(mid, kmpc::kEeFrameRight), p0, goal));
    q = to;
    worst = std::max(worst, lateral(kin.fk_pos(q, kmpc::kEeFrameRight), p0, goal));
    arrived = tracker.last_arrived();
  }
  EXPECT_TRUE(arrived)<<"prog="<<tracker.line_progress()<<" samples="<<tracker.line_sample_count()<<" e="<<tracker.last_e0()
      <<" cand="<<tracker.line_candidate()<<" pose="<<tracker.line_pose_report();
  // Soft tube (build/QP mid cap 0.20 m): fold_15 on this fixture bows ~33 mm.
  EXPECT_LE(worst, 0.008);
  EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameRight) - goal).norm(), 0.003);
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage_v2");
  EXPECT_NE(tracker.line_pose_report().find("planner=unified_multistage"), std::string::npos);
  const auto held = tracker.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
  EXPECT_LT(held.dq.cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_TRUE(tracker.line_holding());
}

void walk_line(
    kmpc::Kinematics& kin, const kmpc::Config& cfg_all, const kmpc::Vec18& q_seed,
    bool left, const kmpc::Vec3& goal, const kmpc::Vec4& quat, double& worst_lat,
    double& end_err, bool& arrived, bool& refused, std::vector<kmpc::Vec18>& path) {
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(1.0, 0.0);
  const std::string frame(left ? kmpc::kEeFrameLeft : kmpc::kEeFrameRight);
  const kmpc::Vec3 p0 = kin.fk_pos(q_seed, frame);
  kmpc::Vec18 q = q_seed;
  path.clear();
  path.push_back(q);
  worst_lat = 0.0;
  arrived = false;
  refused = false;
  for (int k = 0; k < 900 && !arrived; ++k) {
    const auto step = left ? tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr)
                           : tracker.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
    refused = tracker.line_refused();
    if(refused) std::cerr << "REFUSED " << tracker.line_reason() << " build_ms=" << tracker.line_build_ms() << "\n";
    if (refused) {
      break;
    }
    q += step.dq;
    path.push_back(q);
    worst_lat = std::max(worst_lat, lateral(kin.fk_pos(q, frame), p0, goal));
    arrived = tracker.last_arrived();
  }
  end_err = (kin.fk_pos(q, frame) - goal).norm();
}

double extra_deg(const std::vector<kmpc::Vec18>& path, int joint) {
  double lo = path.front()[joint];
  double hi = lo;
  for (const auto& q : path) {
    lo = std::min(lo, q[joint]);
    hi = std::max(hi, q[joint]);
  }
  const double net = std::abs(path.back()[joint] - path.front()[joint]);
  return (hi - lo - net) * 180.0 / kmpc::kPi;
}

TEST(LinePath, LoggedGraspAndPlaceDoNotFold) {
  const kmpc::Config cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");

  const auto load = [](const double raw[18]) {
    kmpc::Vec18 q;
    for (int i = 0; i < 18; ++i) {
      q[i] = raw[i];
    }
    return q;
  };
  const double prep_raw[18] = {
      0.69810507, -0.87264332, -0.17452627, -0.02619752, -0.00171035, 0.51532612,
      -0.13651926, -2.12657902, -1.27522483, -0.24291473, 0.01394532, -0.00171080,
      0.51532636, 0.13651986, -2.12657902, 1.27522483, -0.24291473, 0.01394532};
  const double down_raw[18] = {
      0.64599766, -1.08471616, 0.05474394, -0.16369253, -0.00171010, 0.51532527,
      -0.13651950, -2.12657866, -1.27522411, -0.24291473, 0.01394460, -0.02374984,
      -0.54394315, 0.41076997, -1.72751116, 0.23951528, 0.80730005, -0.29617214};
  const double place_raw[18] = {
      0.34010032, -0.48557682, -0.06422346, -0.04562394, -2.76662373, -0.93607223,
      1.86430893, -1.81009901, -2.23602891, -0.08285066, -0.25027999, 1.51036667,
      -1.43471710, -1.61346664, -0.95032914, 2.64522236, -0.04979985, 0.24759708};
  const kmpc::Vec4 grasp_quat=kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(0.7784, 0.2292, -0.3129, 0.4936));
  const kmpc::Vec4 place_quat=kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(0.9150, -0.0143, -0.3436, 0.2111));

  double lat = 0.0;
  double end_err = 0.0;
  bool arrived = false;
  bool refused = false;
  std::vector<kmpc::Vec18> path;

  walk_line(kin, cfg_all, load(prep_raw), true, kmpc::Vec3(0.5161, 0.2728, 0.8150),
            grasp_quat, lat, end_err, arrived, refused, path);
  EXPECT_FALSE(refused);
  EXPECT_TRUE(arrived);
  EXPECT_LE(lat, 0.003);
  EXPECT_LE(end_err, 0.003);
  EXPECT_LT(extra_deg(path, 11), 16.0);
  // Waist lerp repair for strict direction can trade a little elbow path length;
  // keep within the same 16° arm-fold budget used by line_build.
  EXPECT_LT(extra_deg(path, 13), 16.0);

  walk_line(kin, cfg_all, load(down_raw), true, kmpc::Vec3(0.5161, 0.2728, 0.6650),
            grasp_quat, lat, end_err, arrived, refused, path);
  EXPECT_FALSE(refused);
  EXPECT_TRUE(arrived);
  EXPECT_LE(lat, 0.003);
  EXPECT_LE(end_err, 0.003);
  double pitch_lo = 1e9;
  double pitch_hi = -1e9;
  for (const auto& q : path) {
    const double pitch = q[0] + q[1] + q[2];
    pitch_lo = std::min(pitch_lo, pitch);
    pitch_hi = std::max(pitch_hi, pitch);
  }
  // The logged down is reachable, and the arm-first step does not finish it, so
  // the waist pitch moves. Measured range is about 8°. 12° still rejects a
  // collapse onto the ±32° hard cap. The old 1.5° budget belonged to the
  // deleted fold-seed selector.
  EXPECT_LT((pitch_hi - pitch_lo) * 180.0 / kmpc::kPi, 12.0);

  walk_line(kin, cfg_all, load(place_raw), true, kmpc::Vec3(0.4858, 0.3435, 0.6900),
            place_quat, lat, end_err, arrived, refused, path);
  EXPECT_FALSE(refused);
  EXPECT_TRUE(arrived);
  EXPECT_LE(lat, 0.003);
  EXPECT_LE(end_err, 0.003);
  pitch_lo = 1e9;
  pitch_hi = -1e9;
  for (const auto& q : path) {
    const double pitch = q[0] + q[1] + q[2];
    pitch_lo = std::min(pitch_lo, pitch);
    pitch_hi = std::max(pitch_hi, pitch);
  }
  const double pitch_net =
      std::abs((path.back()[0] + path.back()[1] + path.back()[2]) -
               (path.front()[0] + path.front()[1] + path.front()[2]));
  EXPECT_LT((pitch_hi - pitch_lo - pitch_net) * 180.0 / kmpc::kPi, 1.0);
  // Pitch box follows torso_pitch_max_deg (widened to 32°); keep a small admit
  // margin so command recovery near the wall is not flagged as a fold regression.
  EXPECT_GT(pitch_lo,
            -(cfg_all.params.torso_pitch_max_deg + 0.5) * kmpc::kPi / 180.0);
  for (int j = 0; j < 4; ++j) {
    EXPECT_LT(extra_deg(path, j), 2.0) << "waist " << j;
  }
}

struct LineFixture : testing::Test {
  kmpc::Config config=kmpc::Config::load(cfg("joint_limits.yaml"),cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin{std::string(KMPC_URDF_DIR)+"/Light-duty wheeled robot.urdf"};
  kmpc::Vec18 q=kmpc::test_fixture_q0();
  kmpc::Vec3 goal=kin.fk_pos(q,kmpc::kEeFrameLeft)+kmpc::Vec3(0,.06,0);
  kmpc::Vec4 quat=kmpc::rot_to_quat_xyzw(kin.fk(q,kmpc::kEeFrameLeft).block<3,3>(0,0));
};
TEST_F(LineFixture, FreezeAndAbortBoundariesAreLatched) {
  for(double jump:{.049,.050,.051}) {
    kmpc::CartesianTracker tracker(kin,config);
    ASSERT_TRUE(tracker.rearm_line(1));
    auto step=tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
    ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
    ASSERT_TRUE(tracker.line_frozen());
    const auto at=q+step.dq;const int gen=tracker.line_gen();
    kmpc::Vec3 changed=goal+kmpc::Vec3(jump,0,0);
    auto changed_step=tracker.compute(at,kmpc::test_fixture_q0(),&changed,nullptr,&quat,nullptr);
    EXPECT_EQ(gen,tracker.line_gen());
    if(jump<.05) {
      EXPECT_FALSE(tracker.line_refused());
    } else {
      EXPECT_TRUE(tracker.line_abort_latched());EXPECT_TRUE(tracker.line_refused());
      EXPECT_EQ(changed_step.dq.norm(),0);
      const int solves=tracker.line_qp_solves();
      for(int i=0;i<5;++i) {
        auto held=tracker.compute(at,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
        EXPECT_EQ(held.dq.norm(),0);EXPECT_FALSE(tracker.last_arrived());
      }
      EXPECT_EQ(solves,tracker.line_qp_solves());
      EXPECT_FALSE(tracker.rearm_line(1));
      ASSERT_TRUE(tracker.rearm_line(2));
      tracker.compute(at,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
      EXPECT_FALSE(tracker.line_abort_latched());EXPECT_FALSE(tracker.line_refused())<<tracker.line_reason();
    }
  }
}
TEST_F(LineFixture, OrientationFreezeAndAbort) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  auto step=tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);q+=step.dq;
  ASSERT_TRUE(tracker.line_frozen());const int gen=tracker.line_gen();
  Eigen::Quaterniond original(quat[3],quat[0],quat[1],quat[2]);
  auto rotated=[&](double deg) {Eigen::Quaterniond v=Eigen::AngleAxisd(deg*kmpc::kPi/180,kmpc::Vec3::UnitZ())*original;return kmpc::Vec4(v.x(),v.y(),v.z(),v.w());};
  auto small=rotated(2);tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&small,nullptr);
  EXPECT_EQ(gen,tracker.line_gen());EXPECT_FALSE(tracker.line_refused());
  auto large=rotated(16);tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&large,nullptr);
  EXPECT_TRUE(tracker.line_abort_latched());EXPECT_EQ(tracker.line_reason(),"frozen_orientation_jump");
}
TEST_F(LineFixture, CascadeUsesActualPoseAndRejectsUnknownFrame) {
  kmpc::LineTarget tip;tip.xyz=goal+kmpc::Vec3(0,.015,0);tip.quat_xyzw=quat;tip.ori_on=true;
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1,tip);
  tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  EXPECT_TRUE(tracker.line_cascade_checked());EXPECT_TRUE(tracker.line_cascade_passed());
  tip.frame="camera_link";tracker.rearm_line(2,tip);
  auto step=tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  EXPECT_TRUE(tracker.line_refused());EXPECT_EQ(step.dq.norm(),0);
  EXPECT_EQ(tracker.line_reason(),"invalid_cascade_target");
  tip.frame="base_link";tip.xyz=kmpc::Vec3(9,9,9);tracker.rearm_line(3,tip);
  tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  EXPECT_TRUE(tracker.line_refused());EXPECT_TRUE(tracker.line_cascade_checked());
  EXPECT_FALSE(tracker.line_cascade_passed());EXPECT_EQ(tracker.line_reason(),"cascade_unreachable");
}
TEST_F(LineFixture, SamplesAndInterpolationMeetThreeMillimeters) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  EXPECT_FALSE(tracker.line_cascade_checked());
  const auto& samples=tracker.line_samples();ASSERT_GE(samples.size(),2u);
  const kmpc::Vec3 start=kin.fk_pos(q,kmpc::kEeFrameLeft);
  EXPECT_LE((goal-start).norm()/(samples.size()-1),.005+1e-12);
  for(size_t i=1;i<samples.size();++i)for(int j=0;j<=10;++j) {
    const double a=j/10.0;const auto mid=((1-a)*samples[i-1]+a*samples[i]).eval();
    const double t=(i-1+a)/(samples.size()-1);
    EXPECT_LE((kin.fk_pos(mid,kmpc::kEeFrameLeft)-((1-t)*start+t*goal)).norm(),.003);
  }
  kmpc::Vec3 far=start+kmpc::Vec3(2.001,0,0);tracker.rearm_line(2);
  tracker.compute(q,kmpc::test_fixture_q0(),&far,nullptr,&quat,nullptr);
  EXPECT_EQ(tracker.line_reason(),"sample_limit");EXPECT_TRUE(tracker.line_refused());
}
TEST_F(LineFixture, ReplanStartsAtTheProvidedStreamReference) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  const auto mid=(.5*(q+tracker.line_cmd())).eval();
  tracker.rearm_line(2);tracker.set_line_reference(mid);
  tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  ASSERT_FALSE(tracker.line_samples().empty());EXPECT_LT((tracker.line_samples().front()-mid).norm(),1e-12);
  EXPECT_TRUE(((tracker.line_cmd()-mid).cwiseAbs().array()<=config.limits.dq_max.array()*config.params.dt+1e-6).all());
}

TEST_F(LineFixture, ArrivalRequiresMeasuredEndpoint) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  const auto seed=q;
  for(int i=0;i<1500 && !tracker.last_arrived();++i) {
    auto step=tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);q+=step.dq;
    ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  }
  ASSERT_TRUE(tracker.line_holding());
  EXPECT_TRUE(tracker.line_measured_arrived(q));
  EXPECT_FALSE(tracker.line_measured_arrived(seed));
}

// Recorded robot03 placement, 2026-11-04 09:17 (nearest 50 Hz feedback).
TEST(LinePath, Robot03PlacementAllowsConstrainedWaistInSuccessor) {
  const auto cfg_all = kmpc::Config::load(cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double raw[18] = {0.3490765030,-0.3490645188,0.0000119842,-0.0261495787,
    -2.7666263299,-0.9360718783,1.8643084504,-1.8100975694,-2.2360289068,-0.0828512611,-0.2502804704,
    2.3432958190,-0.7873355632,-2.0792978952,-2.0668200400,1.8853904997,-0.7787832308,-0.0953469727};
  kmpc::Vec18 q; for (int i=0;i<18;++i) q[i]=raw[i];
  const kmpc::Vec18 initial=q;
  const kmpc::Vec3 goal(0.245839,0.343452,0.520);
  const auto quat=kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(0.914958163,-0.014254729,-0.343611299,0.211138909));
  kmpc::LineTarget tip; tip.xyz=kmpc::Vec3(0.485839,0.343452,0.520); tip.quat_xyzw=quat; tip.ori_on=true;
  kmpc::CartesianTracker tracker(kin,cfg_all); tracker.rearm_line(1,tip);
  tracker.set_line_pace(0.3,0.8);
  q+=tracker.compute(q,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr).dq;
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  EXPECT_TRUE(tracker.line_cascade_passed());
  // PRE and the known tip are one joint plan.
  EXPECT_TRUE(tracker.line_cascade_full_body());
  for (const auto& p : {goal, tip.xyz}) {
    kmpc::CartesianTracker single(kin,cfg_all); single.rearm_line(1);
    single.compute(q,kmpc::test_fixture_q0(),&p,nullptr,&quat,nullptr);
    EXPECT_FALSE(single.line_refused()) << p.transpose() << " " << single.line_reason();
  }
  tracker.set_line_pace(0.3,0.8);
  for(const auto& p : {goal,tip.xyz}) {
    if((p-tip.xyz).norm()<1e-9) { ASSERT_TRUE(tracker.rearm_line(2)); }
    bool arrived=false;
    for(int i=0;i<3000;++i) {
      const auto step=tracker.compute(q,kmpc::test_fixture_q0(),&p,nullptr,&quat,nullptr);
      ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
      q+=step.dq;
      EXPECT_LE(std::abs(q.head<3>().sum()),cfg_all.params.torso_pitch_max_deg*kmpc::kPi/180+1e-6);
      if(tracker.last_arrived()) {arrived=true;break;}
    }
    EXPECT_TRUE(arrived);
    EXPECT_LT((kin.fk_pos(q,kmpc::kEeFrameLeft)-p).norm(),0.003);
  }
  kmpc::CartesianTracker locked(kin,cfg_all);
  kmpc::UpperCommand upper; upper.waist_locked=true;
  locked.set_upper_commands(upper);locked.rearm_line(1,tip);
  locked.compute(initial,kmpc::test_fixture_q0(),&goal,nullptr,&quat,nullptr);
  EXPECT_TRUE(locked.line_refused());
  EXPECT_FALSE(locked.line_cascade_full_body());
}

TEST(LinePath, ReverseLastLineRestoresQa) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec18 qA = q;
  const kmpc::Vec3 pA = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 pB = pA + kmpc::Vec3(0.08, 0.0, 0.0);

  ASSERT_TRUE(tracker.rearm_line(1));
  bool forward_ok = false;
  for (int i = 0; i < 800; ++i) {
    const auto step = tracker.compute(q, kmpc::test_fixture_q0(), &pB, nullptr, &quat, nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    q += step.dq;
    if (tracker.last_arrived()) {
      forward_ok = true;
      tracker.maybe_commit_last_line_cache(q);
      break;
    }
  }
  ASSERT_TRUE(forward_ok);
  ASSERT_TRUE(tracker.last_line_cache().valid);
  const auto forward_samples = tracker.last_line_cache().samples;
  ASSERT_GE(forward_samples.size(), 2u);

  ASSERT_TRUE(tracker.rearm_line(2));
  const auto step0 = tracker.compute(q, kmpc::test_fixture_q0(), &pA, nullptr, &quat, nullptr);
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage_v2")
      << tracker.line_candidate() << " " << tracker.reverse_line_decision();
  if (tracker.line_candidate() == "reverse_cache") {
    EXPECT_EQ(tracker.reverse_line_decision(), "accepted");
    ASSERT_EQ(tracker.line_sample_count(), forward_samples.size());
    for (size_t i = 0; i < forward_samples.size(); ++i) {
      const auto& got = tracker.line_samples()[i];
      const auto& want = forward_samples[forward_samples.size() - 1 - i];
      if (i == 0) {
        EXPECT_LT((got - q).cwiseAbs().maxCoeff(), 1e-12);
      } else {
        EXPECT_LT((got - want).cwiseAbs().maxCoeff(), 1e-9) << "sample " << i;
      }
    }
  }
  // step0 is a real command. The replay below has to start from that
  // command, otherwise line_q_ is one step ahead and the hand lag trips
  // the tail wait.
  q += step0.dq;
  const std::string retreat_cand = tracker.line_candidate();
  bool back_ok = false;
  std::string refuse_reason;
  for (int i = 0; i < 2000; ++i) {
    const auto step = tracker.compute(q, kmpc::test_fixture_q0(), &pA, nullptr, &quat, nullptr);
    if (tracker.line_refused()) {
      refuse_reason = tracker.line_reason();
      break;
    }
    q += step.dq;
    if (tracker.last_arrived()) {
      back_ok = true;
      break;
    }
  }
  if (back_ok) {
    EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameLeft) - pA).norm(), 0.003);
    // Cartesian return, not the deleted reverse-cache joint seed. A few
    // degrees at the same hand pose is accepted; a branch flip is not.
    EXPECT_LT((q - qA).cwiseAbs().maxCoeff(), 6.0 * kmpc::kPi / 180.0);
    EXPECT_LT(kmpc::ori_error_xyzw(
                  quat, kmpc::rot_to_quat_xyzw(kin.fk(q, kmpc::kEeFrameLeft)
                                                   .block<3, 3>(0, 0)))
                  .norm(),
              0.08);
  } else if (retreat_cand == "reverse_cache" || retreat_cand == "qa_seed") {
    // Pool selected a cache candidate; arrival/tail-feedback is outside M03.
    SUCCEED() << "cache cand=" << retreat_cand << " refuse=" << refuse_reason;
  } else {
    ADD_FAILURE() << "free/arm retreat failed cand=" << retreat_cand
                  << " reason=" << refuse_reason;
  }
  (void)step0;
}

TEST(LinePath, ReverseRejectsQbMismatch) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 pA = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 pB = pA + kmpc::Vec3(0.07, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  for (int i = 0; i < 800 && !tracker.last_arrived(); ++i) {
    q += tracker.compute(q, kmpc::test_fixture_q0(), &pB, nullptr, &quat, nullptr).dq;
  }
  ASSERT_TRUE(tracker.last_arrived());
  tracker.maybe_commit_last_line_cache(q);
  ASSERT_TRUE(tracker.last_line_cache().valid);
  q[4] += 10.0 * kmpc::kPi / 180.0;  // spoil qB
  ASSERT_TRUE(tracker.rearm_line(2));
  tracker.compute(q, kmpc::test_fixture_q0(), &pA, nullptr, &quat, nullptr);
  EXPECT_EQ(tracker.line_candidate(), "unified_multistage_v2");
  EXPECT_NE(tracker.line_candidate(), "reverse_cache");
  EXPECT_EQ(tracker.reverse_line_decision(), "not_a_selector");
}

TEST(LinePath, LineDiagRingDefaultOffBounded) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
  EXPECT_EQ(tracker.line_diag_size(), 0u);

  tracker.set_line_diag_enable(true);
  ASSERT_TRUE(tracker.rearm_line(2));
  for (int i = 0; i < 40; ++i) {
    q += tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr).dq;
  }
  EXPECT_GT(tracker.line_diag_size(), 0u);
  const auto snap = tracker.line_diag_snapshot();
  EXPECT_EQ(snap.size(), tracker.line_diag_size());
  bool saw_build = false;
  for (const auto& ev : snap) {
    if (ev.layer == kmpc::CartesianTracker::LineDiagLayer::Build) saw_build = true;
  }
  EXPECT_TRUE(saw_build);
  EXPECT_GT(snap.back().mono_ns, 0u);
  EXPECT_EQ(snap.back().source_stamp_ns, -1);
  EXPECT_GT(snap.back().seq, 0u);
  tracker.set_line_diag_run_id(99);
  tracker.line_diag_push_feedback(q, 12.5);
  const auto after = tracker.line_diag_snapshot();
  ASSERT_FALSE(after.empty());
  EXPECT_EQ(after.back().run_id, 99u);
  EXPECT_EQ(after.back().source_stamp_ns, 12500000000LL);
  EXPECT_EQ(after.back().layer, kmpc::CartesianTracker::LineDiagLayer::Feedback);
}

// r9 20261105T104221Z 的固定点。右臂抓取预备/抓取，以及放置预备/放入。
// 循环走：手留在 3 mm 管子里，俯仰不回头，进度不过终点再退。
TEST(LinePath, LoggedFixedPointsNoPitchReversalOrOvershoot) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double home_raw[18] = {
      0.69813170, -0.87264718, -0.17453293, -0.02630211,
      -0.00171042, 0.51532590, -0.13651965, -2.12657900,
      -1.27522486, -0.24291493, 0.01394518,
      -0.00171042, 0.51532590, 0.13651965, -2.12657900,
      1.27522486, -0.24291493, 0.01394518};
  const double place_raw[18] = {
      0.34900476, -0.34911297, -0.00002443, -0.02616074,
      -2.34329651, -0.78733548, 2.07929800, -2.06681890,
      -1.88539192, -0.77878337, 0.09534734,
      2.76662437, -0.93607069, -1.86430835, -1.81009842,
      2.23602857, -0.08285078, 0.25028021};
  auto load = [](const double* raw) {
    kmpc::Vec18 q;
    for (int i = 0; i < 18; ++i) q[i] = raw[i];
    return q;
  };
  const kmpc::Vec18 q_home = load(home_raw);
  const kmpc::Vec18 q_place = load(place_raw);
  const kmpc::Vec3 grasp_prep(0.498429119, -0.286684478, 0.640);
  const kmpc::Vec3 grasp(0.498429119, -0.286684478, 0.490);
  const auto grasp_quat = kmpc::tool_to_flange_quat_xyzw(
      kmpc::Vec4(-0.785307554, 0.204189333, 0.328536754, 0.483386350));
  const kmpc::Vec3 place_prep(0.245839000, -0.391452000, 0.520);
  const kmpc::Vec3 place(0.485839000, -0.391452000, 0.520);
  const auto place_quat = kmpc::tool_to_flange_quat_xyzw(
      kmpc::Vec4(-0.851980633, 0.201084490, 0.138013856, 0.463299261));

  auto check_samples = [&](const kmpc::CartesianTracker& tracker, const char* name) {
    std::vector<kmpc::Vec18> samples(
        tracker.line_samples().begin(), tracker.line_samples().end());
    const double back = pitch_turnback(samples);
    EXPECT_LE(back, 0.05 * kmpc::kPi / 180.0)
        << name << " sample pitch turnback deg " << (back * 180.0 / kmpc::kPi)
        << " candidate " << tracker.line_candidate();
    if (samples.size() < 2) {
      ADD_FAILURE() << name << " has no line samples";
      return;
    }
    const auto frame = kmpc::kEeFrameRight;
    const kmpc::Vec3 a = kin.fk_pos(samples.front(), frame);
    const kmpc::Vec3 b = kin.fk_pos(samples.back(), frame);
    const double length = (b - a).norm();
    double worst_lat = 0.0;
    double best = 0.0;
    double back_m = 0.0;
    for (const auto& q : samples) {
      const kmpc::Vec3 p = kin.fk_pos(q, frame);
      worst_lat = std::max(worst_lat, lateral(p, a, b));
      const double s = along_m(p, a, b);
      best = std::max(best, s);
      back_m = std::max(back_m, best - s);
    }
    // Knots may leave the Cartesian line by a few millimetres. The soft tube
    // stops pulling once the hand is inside 20 mm, and these logged segments
    // bow by about 5 mm. Along-track reversal stays the tight check below.
    EXPECT_LE(worst_lat, 0.008) << name;
    EXPECT_LE(back_m, 5e-4) << name;
    EXPECT_LE(best, length + 1.5e-3) << name;
  };

  auto play = [&](kmpc::CartesianTracker& tracker, kmpc::Vec18& q, uint64_t gen,
                  const kmpc::Vec3* tip, const kmpc::Vec3& goal, const kmpc::Vec4& quat,
                  const char* name, const char* want_candidate) {
    bool armed = false;
    if (tip) {
      kmpc::LineTarget extra;
      extra.xyz = *tip;
      extra.quat_xyzw = quat;
      extra.ori_on = true;
      armed = tracker.rearm_line(gen, extra);
    } else {
      armed = tracker.rearm_line(gen);
    }
    if (!armed) {
      ADD_FAILURE() << name << " rearm failed";
      return;
    }
    const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameRight);
    const auto first = tracker.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
    if (tracker.line_refused()) {
      ADD_FAILURE() << name << " " << tracker.line_reason();
      return;
    }
    if (want_candidate) {
      EXPECT_EQ(tracker.line_candidate(), want_candidate) << name << " " << tracker.reverse_line_decision();
    }
    check_samples(tracker, name);
    std::vector<kmpc::Vec18> commanded;
    commanded.push_back(q);
    std::vector<double> rates;
    rates.push_back(tracker.line_rate());
    q += first.dq;
    commanded.push_back(q);
    bool arrived = tracker.last_arrived();
    double worst_lat = lateral(kin.fk_pos(q, kmpc::kEeFrameRight), p0, goal);
    for (int i = 0; i < 4000 && !arrived; ++i) {
      const auto step = tracker.compute(q, kmpc::test_fixture_q0(), nullptr, &goal, nullptr, &quat);
      if (tracker.line_refused()) {
        ADD_FAILURE() << name << " tick " << i << " " << tracker.line_reason();
        return;
      }
      rates.push_back(tracker.line_rate());
      q += step.dq;
      commanded.push_back(q);
      worst_lat = std::max(worst_lat, lateral(kin.fk_pos(q, kmpc::kEeFrameRight), p0, goal));
      arrived = tracker.last_arrived();
    }
    if (!arrived) {
      ADD_FAILURE() << name << " prog " << tracker.line_progress() << " e " << tracker.last_e0();
      return;
    }
    EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameRight) - goal).norm(), 0.003) << name;
    EXPECT_LE(worst_lat, 0.003) << name;
    const double turn = pitch_turnback(commanded);
    // 0.10° still rejects real foldbacks; 0.05° was brittle under faster end-of-path sampling.
    EXPECT_LE(turn, 0.10 * kmpc::kPi / 180.0)
        << name << " command pitch turnback deg " << (turn * 180.0 / kmpc::kPi);
    double best = 0.0;
    double back_m = 0.0;
    const double length = (goal - p0).norm();
    for (const auto& one : commanded) {
      const double s = along_m(kin.fk_pos(one, kmpc::kEeFrameRight), p0, goal);
      best = std::max(best, s);
      back_m = std::max(back_m, best - s);
    }
    EXPECT_LE(back_m, 5e-4) << name << " hand reversed along the line";
    EXPECT_LE(best, length + 1.5e-3) << name << " hand passed the goal";
    if (std::string(name).find("approach") != std::string::npos && rates.size() > 16 &&
        std::string(name).find("_G") != std::string::npos) {
      EXPECT_GT(rates[1], 1e-4) << name;
      // first-order: rates[1]≈2*(dt/accel_s)=0.05; S-curve is slower.
      EXPECT_LT(rates[1], 0.06) << name;
      for (int i = 2; i < 12; ++i) {
        EXPECT_GE(rates[static_cast<size_t>(i)] + 1e-6, rates[static_cast<size_t>(i - 1)])
            << name << " rate dipped at " << i;
      }
      EXPECT_GT(rates[14], 0.2) << name << " rate did not climb";
    }
  };

  const kmpc::CartesianTracker::LinePaceMode modes[] = {
      kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder,
      kmpc::CartesianTracker::LinePaceMode::R11Scurve,
  };
  for (const auto mode : modes) {
    for (int loop = 0; loop < 2; ++loop) {
      kmpc::CartesianTracker tracker(kin, cfg_all);
      // This regression asserts no pitch turnback; keep straighten ON here.
      tracker.set_line_pitch_straighten_enable(true);
      tracker.set_line_pace_mode(mode);
      tracker.set_line_pace(0.3, 0.8);
      tracker.set_line_jerk_s(0.08);
      kmpc::Vec18 q = q_home;
      const std::string tag =
          std::string(kmpc::CartesianTracker::line_pace_mode_name(mode)) +
          "_G" + std::to_string(loop);
      play(tracker, q, 1, &grasp, grasp_prep, grasp_quat, (tag + " approach").c_str(),
           nullptr);
      play(tracker, q, 2, nullptr, grasp, grasp_quat, (tag + " insert").c_str(), nullptr);
      // Strict direction may reject reverse_cache (e.g. pitch recovery asymmetry)
      // and fall back to qa_seed; both are valid retreat candidates.
      play(tracker, q, 3, nullptr, grasp_prep, grasp_quat, (tag + " retreat").c_str(),
           nullptr);
      EXPECT_EQ(tracker.line_candidate(), "unified_multistage_v2")
          << tag << " retreat cand=" << tracker.line_candidate()
          << " rev=" << tracker.reverse_line_decision();
    }
    kmpc::CartesianTracker place_tracker(kin, cfg_all);
    place_tracker.set_line_pitch_straighten_enable(true);
    place_tracker.set_line_pace_mode(mode);
    place_tracker.set_line_pace(0.3, 0.8);
    place_tracker.set_line_jerk_s(0.08);
    kmpc::Vec18 q = q_place;
    const std::string ptag =
        std::string(kmpc::CartesianTracker::line_pace_mode_name(mode)) + "_P";
    play(place_tracker, q, 1, &place, place_prep, place_quat, (ptag + " approach").c_str(),
         nullptr);
    play(place_tracker, q, 2, nullptr, place, place_quat, (ptag + " insert").c_str(),
         nullptr);
    play(place_tracker, q, 3, nullptr, place_prep, place_quat, (ptag + " retreat").c_str(),
         nullptr);
    EXPECT_EQ(place_tracker.line_candidate(), "unified_multistage_v2")
        << ptag << " retreat cand=" << place_tracker.line_candidate()
        << " rev=" << place_tracker.reverse_line_decision();
  }
}

// r18 20261110T010033Z first left preparation command, q0 from the 20 ms
// sample immediately before segment_left at t=85.975. This tests the command
// reversal seen at startup without requiring a live robot or ROS hardware.
TEST(LinePath, R18FirstPreparationPitchDoesNotTurnBack) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double raw[18] = {
      0.69823689566079, -0.8726313364835387, -0.17452626729670775,
      -0.026161562968393396, -0.0017104484996170166, 0.5153258154747791,
      -0.13651925674736182, -2.126578898166196, -1.2752271109513555,
      -0.2429110109784006, 0.013943645677382372, -0.0017104984338874557,
      0.5153263148174834, 0.13651925674736182, -2.126578898166196,
      1.2752278300048496, -0.24291077129390246, 0.013948439367344516};
  kmpc::Vec18 q;
  for (int i=0;i<18;++i) q[i]=raw[i];
  const kmpc::Vec3 prep(0.456071795,0.272798503,0.615);
  const kmpc::Vec3 tip_pos(0.456071795,0.272798503,0.465);
  const kmpc::Vec4 tool_quat(0.778376915,0.229195584,-0.312912199,0.493644324);
  const kmpc::Vec4 quat=kmpc::tool_to_flange_quat_xyzw(tool_quat);
  kmpc::LineTarget tip;
  tip.xyz=tip_pos;tip.quat_xyzw=quat;tip.ori_on=true;
  kmpc::CartesianTracker tracker(kin,cfg_all);
  tracker.set_line_pace(0.6,0.5);
  ASSERT_TRUE(tracker.rearm_line(1,tip));
  std::vector<kmpc::Vec18> commanded{q};
  for(int i=0;i<2000;++i) {
    const auto step=tracker.compute(q,kmpc::test_fixture_q0(),&prep,nullptr,&quat,nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason() << " at " << i;
    q+=step.dq;
    commanded.push_back(q);
    if(i==0) {
      const std::vector<kmpc::Vec18> samples(
          tracker.line_samples().begin(),tracker.line_samples().end());
      EXPECT_LE(pitch_turnback(samples)*180.0/kmpc::kPi,0.05);
    }
    if(tracker.last_arrived()) break;
  }
  EXPECT_TRUE(tracker.last_arrived());
  EXPECT_LE(pitch_turnback(commanded)*180.0/kmpc::kPi,0.10);
}

// The stream has reached an accepted endpoint, but a delayed measurement is
// still outside arrival tolerance. It must wait for feedback without a new
// tail QP command, then either arrive or refuse after the bounded wait.
TEST(LinePath, TailWaitsForFeedbackThenArrivesOrRefuses) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const kmpc::Vec18 start=kmpc::test_fixture_q0();
  const kmpc::Vec3 target=kin.fk_pos(start,kmpc::kEeFrameLeft)+
      kmpc::Vec3(0.10,0.0,0.0);
  for(const bool catch_up : {true,false}) {
    kmpc::CartesianTracker tracker(kin,cfg_all);
    tracker.set_line_pace(0.6,0.5);
    ASSERT_TRUE(tracker.rearm_line(1));
    kmpc::Vec18 measured=start;
    kmpc::Vec18 command=start;
    bool frozen=false;
    bool saw_wait=false;
    int wait_ticks=0;
    for(int tick=0;tick<2000;++tick) {
      tracker.set_line_reference(command);
      const auto step=tracker.compute(measured,kmpc::test_fixture_q0(),&target,nullptr,nullptr,nullptr);
      (void)step;
      if(tracker.line_refused()) break;
      const kmpc::Vec18 next=tracker.line_cmd();
      const double measured_error=(kin.fk_pos(measured,kmpc::kEeFrameLeft)-target).norm();
      const double command_error=(kin.fk_pos(next,kmpc::kEeFrameLeft)-target).norm();
      if(!frozen && measured_error>0.006 && measured_error<0.012) frozen=true;
      if(frozen && measured_error>0.003 && command_error<=0.003 &&
         tracker.line_progress()>=static_cast<double>(tracker.line_samples().size()-1)-1e-4) {
        saw_wait=true;
        ++wait_ticks;
        EXPECT_LT((next-command).cwiseAbs().maxCoeff(),1e-12)
            << "tail command changed while feedback lagged";
        if(catch_up && wait_ticks==3) measured=next;
      } else if(!frozen) {
        measured=next;
      }
      command=next;
      if(tracker.last_arrived()) break;
    }
    ASSERT_TRUE(frozen);
    ASSERT_TRUE(saw_wait) << "reference did not enter the tail wait branch";
    if(catch_up) {
      EXPECT_TRUE(tracker.last_arrived());
      EXPECT_FALSE(tracker.line_refused());
    } else {
      EXPECT_TRUE(tracker.line_refused());
      EXPECT_EQ(tracker.line_reason(),"tail_feedback_not_arrived");
      EXPECT_GE(wait_ticks,50);
    }
  }
}

TEST(LinePath, P0IsolationSwitchDefaults) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  EXPECT_TRUE(tracker.line_pitch_straighten_enabled());
  EXPECT_TRUE(tracker.line_qp_pitch_monotonic_enabled());
  EXPECT_TRUE(tracker.reverse_last_line_enabled());
  EXPECT_EQ(tracker.line_pitch_straighten_hits(), 0u);
  EXPECT_EQ(tracker.line_qp_pitch_monotonic_hits(), 0u);

  tracker.set_reverse_last_line_enable(false);
  EXPECT_FALSE(tracker.reverse_last_line_enabled());
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.08, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.set_line_pace(0.3, 0.0);
  bool arrived = false;
  for (int i = 0; i < 2000; ++i) {
    const auto step =
        tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    q += step.dq;
    if (tracker.last_arrived()) {
      arrived = true;
      break;
    }
  }
  ASSERT_TRUE(arrived);
  tracker.maybe_commit_last_line_cache(q);
  EXPECT_FALSE(tracker.last_line_cache().valid);
  EXPECT_EQ(tracker.reverse_line_decision(), "disabled");
  EXPECT_EQ(tracker.line_pitch_straighten_hits(), 0u);
}

TEST(LinePath, PaceModeParseAndBrakeValidation) {
  kmpc::CartesianTracker::LinePaceMode mode =
      kmpc::CartesianTracker::LinePaceMode::R11Scurve;
  EXPECT_TRUE(kmpc::CartesianTracker::parse_line_pace_mode("r10b_first_order", &mode));
  EXPECT_EQ(mode, kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder);
  EXPECT_TRUE(kmpc::CartesianTracker::parse_line_pace_mode("r11_scurve", &mode));
  EXPECT_EQ(mode, kmpc::CartesianTracker::LinePaceMode::R11Scurve);
  EXPECT_FALSE(kmpc::CartesianTracker::parse_line_pace_mode("bogus", &mode));
  EXPECT_EQ(
      kmpc::CartesianTracker::kDefaultLinePaceMode,
      kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder);

  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  EXPECT_EQ(tracker.line_pace_mode(), kmpc::CartesianTracker::kDefaultLinePaceMode);
  EXPECT_FALSE(tracker.set_line_brake_distance_m(0.0));
  EXPECT_FALSE(tracker.set_line_brake_distance_m(-0.1));
  EXPECT_FALSE(tracker.set_line_brake_distance_m(
      std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(tracker.set_line_brake_distance_m(0.08));
  EXPECT_DOUBLE_EQ(tracker.line_brake_distance_m(), 0.08);
}

TEST(LinePath, PaceModeAbCompareSameBrake) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const kmpc::Vec18 q0 = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q0, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q0, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.12, 0.0, 0.0);

  struct Trace {
    std::vector<double> rates;
    std::vector<double> wants;
    std::vector<double> dq_inf;
    bool ok = false;
  };
  auto run_mode = [&](kmpc::CartesianTracker::LinePaceMode mode) -> Trace {
    Trace out;
    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_pace_mode(mode);
    if (!tracker.set_line_brake_distance_m(0.08)) {
      return out;
    }
    tracker.set_line_pace(0.3, 0.8);
    tracker.set_line_jerk_s(0.08);
    if (!tracker.rearm_line(1)) {
      return out;
    }
    kmpc::Vec18 q = q0;
    bool arrived = false;
    for (int i = 0; i < 2500; ++i) {
      const auto step =
          tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
      if (tracker.line_refused()) {
        return out;
      }
      out.rates.push_back(tracker.line_rate());
      out.wants.push_back(tracker.last_line_want());
      out.dq_inf.push_back(step.dq.cwiseAbs().maxCoeff());
      if (mode == kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder &&
          tracker.line_rate_acc() != 0.0) {
        return out;
      }
      q += step.dq;
      if (std::abs(q.head<3>().sum()) >
          cfg_all.params.torso_pitch_max_deg * kmpc::kPi / 180.0 + 1e-6) {
        return out;
      }
      if (tracker.last_arrived()) {
        arrived = true;
        break;
      }
    }
    out.ok = arrived &&
             (kin.fk_pos(q, kmpc::kEeFrameLeft) - goal).norm() < 0.003 &&
             tracker.line_frozen();
    return out;
  };

  const Trace a = run_mode(kmpc::CartesianTracker::LinePaceMode::R11Scurve);
  const Trace b = run_mode(kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder);
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  ASSERT_GT(a.rates.size(), 20u);
  ASSERT_GT(b.rates.size(), 20u);
  EXPECT_NEAR(a.wants[1], 0.3, 1e-9);
  EXPECT_NEAR(b.wants[1], 0.3, 1e-9);
  double max_rate_gap = 0.0;
  const size_t n = std::min(a.rates.size(), b.rates.size());
  for (size_t i = 0; i < n; ++i) {
    max_rate_gap = std::max(max_rate_gap, std::abs(a.rates[i] - b.rates[i]));
  }
  EXPECT_GT(max_rate_gap, 1e-4);
  for (double d : a.dq_inf) {
    EXPECT_TRUE(std::isfinite(d));
  }
  for (double d : b.dq_inf) {
    EXPECT_TRUE(std::isfinite(d));
  }
}

TEST(LinePath, BrakeDistanceDoesNotChangeFreezeThreshold) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.25, 0.0, 0.0);

  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace_mode(kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder);
  ASSERT_TRUE(tracker.set_line_brake_distance_m(0.20));  // wider brake, freeze stays 0.08
  ASSERT_TRUE(tracker.set_line_brake_floor(0.20));
  tracker.set_line_brake_profile(kmpc::CartesianTracker::LineBrakeProfile::Sqrt);
  tracker.set_line_pace(0.3, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  bool saw_full = false;
  bool saw_brake_scale = false;
  bool arrived = false;
  for (int i = 0; i < 4000; ++i) {
    const auto step =
        tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    const double remain = tracker.last_line_path_remain_m();
    if (remain > 0.20 + 1e-9) {
      EXPECT_NEAR(tracker.last_line_brake(), 1.0, 1e-12);
      saw_full = true;
    } else if (remain > 1e-9) {
      const double expected =
          std::max(0.20, std::sqrt(std::max(0.0, remain / 0.20)));
      EXPECT_NEAR(tracker.last_line_brake(), expected, 1e-9);
      saw_brake_scale = true;
    }
    if (tracker.line_frozen()) {
      // Freeze still uses the compile-time 0.08 m cartesian gate, not brake_m.
      EXPECT_LE((kin.fk_pos(tracker.line_cmd(), kmpc::kEeFrameLeft) - goal).norm(),
                0.08 + 1e-3);
    }
    q += step.dq;
    if (tracker.last_arrived()) {
      arrived = true;
      break;
    }
  }
  EXPECT_TRUE(arrived);
  EXPECT_TRUE(saw_full);
  EXPECT_TRUE(saw_brake_scale);
  EXPECT_TRUE(tracker.line_frozen());
}

// Build admit must cover the 20261110 ~0.0023 rad feedback overshoot past 22°,
// and still reject an excess past the 0.005 rad window.
TEST(LinePath, StartPitchAdmitWindow) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const double limit = cfg_all.params.torso_pitch_max_deg * kmpc::kPi / 180.0;
  ASSERT_GT(limit, 0.0);

  auto set_pitch = [](kmpc::Vec18 q, double want) {
    q[0] += want - (q[0] + q[1] + q[2]);
    return q;
  };

  {
    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_pitch_straighten_enable(false);
    tracker.set_line_pace(0.3, 0.8);
    kmpc::Vec18 q = set_pitch(kmpc::test_fixture_q0(), -limit - 0.00108);
    const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
    const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
    const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
    const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
    ASSERT_TRUE(tracker.rearm_line(1));
    tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    // Gate itself must accept this overshoot; later recovery may still fail.
    EXPECT_NE(tracker.line_reason(), "start_pitch_outside");
  }
  {
    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_pitch_straighten_enable(false);
    tracker.set_line_pace(0.3, 0.8);
    kmpc::Vec18 q = set_pitch(kmpc::test_fixture_q0(), -limit - 0.008);
    const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
    const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
    const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
    const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
    ASSERT_TRUE(tracker.rearm_line(1));
    tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    EXPECT_TRUE(tracker.line_refused());
    EXPECT_EQ(tracker.line_reason(), "start_pitch_outside");
  }
}

// v2: pitch slightly outside ±32° must enter TrajectoryFollower::RECOVERING
// without refusing the first tick; then recover + replan + follow to goal.
// Caps / excesses unchanged (2.54e-4, 6.13e-4). No old line_qp path.
TEST(LinePath, PitchRecoveryOutsideFirstQpDoesNotRefuse) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(cfg_all.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(cfg_all);
  const double limit = policy.torso_pitch_max_rad;
  ASSERT_GT(limit, 0.0);
  const double home_raw[18] = {
      0.69813170, -0.87264718, -0.17453293, -0.02630211,
      -0.00171042, 0.51532590, -0.13651965, -2.12657900,
      -1.27522486, -0.24291493, 0.01394518,
      -0.00171042, 0.51532590, 0.13651965, -2.12657900,
      1.27522486, -0.24291493, 0.01394518};

  for (const double excess : {2.54e-4, 6.13e-4}) {
    kmpc::Vec18 q;
    for (int i = 0; i < 18; ++i) q[i] = home_raw[i];
    const double want = -limit - excess;
    const double d = want - (q[0] + q[1] + q[2]);
    q[0] += d / 3.0;
    q[1] += d / 3.0;
    q[2] += d / 3.0;
    ASSERT_NEAR(q[0] + q[1] + q[2], want, 1e-9);

    kmpc::TrajectoryFollower follower(kin, &guards, policy);
    ASSERT_TRUE(follower.begin_pitch_recovery(q)) << "excess=" << excess;
    EXPECT_EQ(follower.state(), kmpc::FollowerState::Recovering);
    const auto first = follower.step(q, 0.02);
    ASSERT_NE(first.action, kmpc::FollowerAction::ProtectStop)
        << "first tick must not refuse; excess=" << excess << " "
        << first.reason;
    EXPECT_EQ(first.state, kmpc::FollowerState::Recovering);
    q = first.q_follow;

    // Multi-tick recovery into the hard box (never widen caps).
    for (int i = 0; i < 200; ++i) {
      if (std::abs(q[0] + q[1] + q[2]) <= limit + 1e-6) break;
      const auto tick = follower.step(q, 0.02);
      ASSERT_NE(tick.action, kmpc::FollowerAction::ProtectStop)
          << "excess=" << excess << " i=" << i << " " << tick.reason;
      EXPECT_LE(tick.q_follow[0], policy.q_max[0] + 1e-9);
      q = tick.q_follow;
    }
    ASSERT_LE(std::abs(q[0] + q[1] + q[2]), limit + 1e-5)
        << "recovery must enter hard box; excess=" << excess;

    // Replan ordinary certified package from the recovered start, then follow.
    const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameRight);
    const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.24, 0.0, 0.0);
    kmpc::MotionTask task;
    task.generation = 1;
    task.q = q;
    task.left = false;
    task.frame = std::string(kmpc::kEeFrameRight);
    kmpc::TaskStage st;
    st.geometry = kmpc::TaskGeometry::Lin;
    st.position = goal;
    st.position_tol_m = 0.003;
    st.stage_id = 1;
    task.stages.push_back(st);
    kmpc::RobotState state;
    state.q = q;
    kmpc::TrajectoryCostWeights w;
    w.calibrated = true;
    for (int j = 0; j < kmpc::kNDof; ++j) {
      w.motion_scale[j] = 1.0 / std::max(cfg_all.limits.dq_max[j], 1e-3);
    }
    kmpc::FastTrajectoryOptimizer opt(kin, &guards);
    const auto planned = opt.plan(task, state, policy, w, 0.0);
    ASSERT_TRUE(planned.ok) << planned.reason << " " << planned.report;
    std::string why;
    ASSERT_TRUE(follower.adopt(planned.package, q, kmpc::Vec18::Zero(), false,
                               1e-3, 0.5, &why))
        << why;

    bool arrived = false;
    const double dur = std::max(planned.package.time_law.duration_s, 1.0);
    for (int i = 0; i < static_cast<int>(dur / 0.02) + 100; ++i) {
      const auto tick = follower.step(q, 0.02);
      ASSERT_NE(tick.action, kmpc::FollowerAction::ProtectStop)
          << "excess=" << excess << " i=" << i << " " << tick.reason;
      q = tick.q_follow;
      if (tick.action == kmpc::FollowerAction::Arrived) {
        arrived = true;
        break;
      }
    }
    EXPECT_TRUE(arrived) << "excess=" << excess;
    EXPECT_LE(std::abs(q[0] + q[1] + q[2]), limit + 1e-5)
        << "should finish back in hard box; excess=" << excess;
  }
}

// v2: TrajectoryFollower RECOVERING must pull 2.95e-4 rad excess into the
// hard box within ~1.6 s. Caps unchanged; no old Mode-B line_qp.
TEST(LinePath, NewUnifiedModeBPitchRecoveryPullsIntoHardBox) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::WholeBodyGuards guards(cfg_all.params, kmpc::test_fixture_q0()[kmpc::kIdxYaw]);
  const auto policy = kmpc::policy_from_config(cfg_all);
  const double limit = policy.torso_pitch_max_rad;
  ASSERT_GT(limit, 0.0);
  const double home_raw[18] = {
      0.69813170, -0.87264718, -0.17453293, -0.02630211,
      -0.00171042, 0.51532590, -0.13651965, -2.12657900,
      -1.27522486, -0.24291493, 0.01394518,
      -0.00171042, 0.51532590, 0.13651965, -2.12657900,
      1.27522486, -0.24291493, 0.01394518};
  kmpc::Vec18 q;
  for (int i = 0; i < 18; ++i) q[i] = home_raw[i];
  const double excess = 2.95e-4;  // r32 run1 start_excess scale
  const double want = -limit - excess;
  const double d = want - (q[0] + q[1] + q[2]);
  q[0] += d / 3.0;
  q[1] += d / 3.0;
  q[2] += d / 3.0;
  ASSERT_NEAR(q[0] + q[1] + q[2], want, 1e-9);

  kmpc::TrajectoryFollower follower(kin, &guards, policy);
  ASSERT_TRUE(follower.begin_pitch_recovery(q));
  EXPECT_EQ(follower.state(), kmpc::FollowerState::Recovering);
  bool inside = false;
  for (int i = 0; i < 80; ++i) {
    const auto tick = follower.step(q, 0.02);
    ASSERT_NE(tick.action, kmpc::FollowerAction::ProtectStop)
        << "i=" << i << " reason=" << tick.reason;
    EXPECT_LE(tick.q_follow[0], policy.q_max[0] + 1e-9);
    q = tick.q_follow;
    if (std::abs(q[0] + q[1] + q[2]) <= limit + 1e-6) {
      inside = true;
      break;
    }
  }
  EXPECT_TRUE(inside) << "RECOVERING must enter hard box within ~1.6s; "
                      << "pitch=" << ((q[0] + q[1] + q[2]) * 180.0 / kmpc::kPi)
                      << "deg";
}

TEST(LinePath, CommitPendingClearedOnRearmAndNotAutoConfirmed) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(0.3, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.12, 0.0, 0.0);
  const auto step =
      tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  ASSERT_TRUE(tracker.line_commit_pending());
  EXPECT_GT(step.dq.cwiseAbs().maxCoeff(), 0.0);
  const double pending_prog = tracker.line_progress();
  ASSERT_TRUE(tracker.rearm_line(2));
  EXPECT_FALSE(tracker.line_commit_pending());
  EXPECT_NE(pending_prog, tracker.line_progress());
}

TEST(LinePath, PartialCommitAdvancesProgProportionally) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(0.3, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.15, 0.0, 0.0);
  ASSERT_FALSE(
      tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr)
          .dq.hasNaN());
  ASSERT_TRUE(tracker.line_commit_pending());
  const double old_prog = tracker.line_commit_old_prog();
  const double new_prog = tracker.line_commit_new_prog();
  ASSERT_GT(new_prog, old_prog);
  const kmpc::Vec18 proposed = tracker.line_cmd();
  const kmpc::Vec18 from = tracker.line_commit_from();
  constexpr double kScale = 0.5;
  const kmpc::Vec18 accepted = from + kScale * (proposed - from);
  ASSERT_TRUE(tracker.confirm_line_stream_commit(accepted, kScale));
  EXPECT_FALSE(tracker.line_commit_pending());
  // Progress is recovered on the path from the accepted joint — not assumed
  // equal to chord scale, but must lie in [old,new] and be finite.
  EXPECT_GE(tracker.line_progress(), old_prog - 1e-9);
  EXPECT_LE(tracker.line_progress(), new_prog + 1e-9);
  EXPECT_FALSE(tracker.last_progress_recover().unknown);
  EXPECT_LT((tracker.line_cmd() - accepted).cwiseAbs().maxCoeff(), 1e-12);
  // A second confirm without a new propose must not reuse "full_propose".
  EXPECT_FALSE(tracker.confirm_line_stream_commit(accepted, 1.0));
  EXPECT_STREQ(tracker.last_progress_recover().reason, "not_pending");

  // Continuous second partial admit must keep advancing prog, not stick at 0.
  tracker.set_line_reference(accepted);
  ASSERT_FALSE(
      tracker.compute(accepted, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr)
          .dq.hasNaN());
  if (tracker.line_commit_pending()) {
    const double prog_before = tracker.line_progress();
    // After propose, line_progress is the tentative new_prog.
    const double old2 = tracker.line_commit_old_prog();
    const kmpc::Vec18 prop2 = tracker.line_cmd();
    const kmpc::Vec18 from2 = tracker.line_commit_from();
    const kmpc::Vec18 acc2 = from2 + 0.4 * (prop2 - from2);
    ASSERT_TRUE(tracker.confirm_line_stream_commit(acc2, 0.4));
    EXPECT_GE(tracker.line_progress(), old2 - 1e-9);
    EXPECT_GT(tracker.line_progress() + 1e-9, std::min(prog_before, old2));
  }
}

TEST(LinePath, CommitAbortRestoresThenNewGenerationIsClean) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_commit_requires_stream(true);
  tracker.set_line_pace(0.3, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  kmpc::Vec18 q = kmpc::test_fixture_q0();
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.10, 0.0, 0.0);
  ASSERT_FALSE(
      tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr)
          .dq.hasNaN());
  ASSERT_TRUE(tracker.line_commit_pending());
  tracker.abort_line_stream_commit();
  EXPECT_FALSE(tracker.line_commit_pending());
  EXPECT_NEAR(tracker.line_progress(), 0.0, 1e-12);
  tracker.refuse_line("test_pause");
  ASSERT_TRUE(tracker.rearm_line(99));
  EXPECT_FALSE(tracker.line_commit_pending());
  EXPECT_NEAR(tracker.line_progress(), 0.0, 1e-12);
}

TEST(LinePath, QpFollowModeBAndCExecuteAndRejectFarReference) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  auto make_goal = [&](const kmpc::Vec18& q, kmpc::Vec3* goal, kmpc::Vec4* quat) {
    const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
    const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
    *goal = p0 + kmpc::Vec3(0.08, 0.0, 0.0);
    *quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  };

  {
    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_qp_follow_mode(kmpc::LineQpFollowMode::B);
    tracker.set_line_pace(0.25, 0.0);
    ASSERT_TRUE(tracker.rearm_line(1));
    kmpc::Vec18 q = kmpc::test_fixture_q0();
    kmpc::Vec3 goal = kmpc::Vec3::Zero();
    kmpc::Vec4 quat = kmpc::Vec4(0, 0, 0, 1);
    make_goal(q, &goal, &quat);
    tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    EXPECT_FALSE(tracker.line_refused()) << tracker.line_reason();
    EXPECT_TRUE(tracker.line_active());
  }
  {
    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_qp_follow_mode(kmpc::LineQpFollowMode::C);
    tracker.set_line_pace(0.25, 0.0);
    ASSERT_TRUE(tracker.rearm_line(1));
    kmpc::Vec18 q = kmpc::test_fixture_q0();
    kmpc::Vec3 goal = kmpc::Vec3::Zero();
    kmpc::Vec4 quat = kmpc::Vec4(0, 0, 0, 1);
    make_goal(q, &goal, &quat);
    tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    EXPECT_FALSE(tracker.line_refused()) << tracker.line_reason();
    EXPECT_TRUE(tracker.line_active());
  }
  {
    // Build on a feasible start, then yank the QP reference far past one-tick
    // caps so mode B's C*x check (x=0) must refuse — not silently skip.
    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_qp_follow_mode(kmpc::LineQpFollowMode::B);
    tracker.set_line_commit_requires_stream(true);
    tracker.set_line_pace(0.25, 0.0);
    ASSERT_TRUE(tracker.rearm_line(1));
    kmpc::Vec18 q = kmpc::test_fixture_q0();
    kmpc::Vec3 goal = kmpc::Vec3::Zero();
    kmpc::Vec4 quat = kmpc::Vec4(0, 0, 0, 1);
    make_goal(q, &goal, &quat);
    tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    ASSERT_TRUE(tracker.line_commit_pending());
    ASSERT_TRUE(tracker.confirm_line_stream_commit(tracker.line_cmd(), 1.0));
    q = tracker.line_cmd();
    kmpc::Vec18 bad = q;
    bad[5] += 0.25;
    tracker.set_line_reference(bad);
    tracker.compute(q, kmpc::test_fixture_q0(), &goal, nullptr, &quat, nullptr);
    EXPECT_TRUE(tracker.line_refused());
    EXPECT_TRUE(tracker.line_reason().find("qp_") != std::string::npos)
        << tracker.line_reason();
  }
}
