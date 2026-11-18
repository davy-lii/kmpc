// Offline regression: 624 + 525 vision targets → prep → tip → retreat.
// Offset rules match local arm_controller_params.yaml (GRIB/PLACE + pre-approach).
// Waist J1/J2/J3 + pitch must stay contract-clean on samples and commanded path.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/test_fixture.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/tracker_math.hpp"

namespace {

std::string cfg(const char* name) {
  return std::string(KMPC_CONFIG_DIR) + "/" + name;
}

kmpc::Vec18 q_from_deg(const double waist4[4], const double right7[7],
                       const double left7[7]) {
  kmpc::Vec18 q;
  for (int i = 0; i < 4; ++i) q[i] = waist4[i] * kmpc::kPi / 180.0;
  for (int i = 0; i < 7; ++i) q[4 + i] = right7[i] * kmpc::kPi / 180.0;
  for (int i = 0; i < 7; ++i) q[11 + i] = left7[i] * kmpc::kPi / 180.0;
  return q;
}

struct VisionMm {
  const char* name;
  bool left;
  bool grasp;  // false = place
  double x, y, z, qx, qy, qz, qw;
  double exp_prep_x, exp_prep_y, exp_prep_z;
  double exp_tip_x, exp_tip_y, exp_tip_z;
};

// Business offsets from arm_controller_params.yaml (right, left).
constexpr double kGribRight[3] = {-60.0, 0.0, 60.0};
constexpr double kGribLeft[3] = {-38.0, -10.0, 65.0};
constexpr double kPlaceRight[3] = {-40.0, -68.0, 90.0};
constexpr double kPlaceLeft[3] = {-40.0, 20.0, 90.0};
constexpr double kPreApproachX = 240.0;
constexpr double kPreApproachZ = 150.0;

double yaw_from_quat_xyzw(double qx, double qy, double qz, double qw) {
  const double siny_cosp = 2.0 * (qw * qz + qx * qy);
  const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny_cosp, cosy_cosp);
}

void vision_to_prep_tip_mm(const VisionMm& v, double prep[3], double tip[3]) {
  if (v.grasp) {
    const double* off = v.left ? kGribLeft : kGribRight;
    const double yaw = yaw_from_quat_xyzw(v.qx, v.qy, v.qz, v.qw);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    const double rx = c * off[0] - s * off[1];
    const double ry = s * off[0] + c * off[1];
    tip[0] = v.x + rx;
    tip[1] = v.y + ry;
    tip[2] = v.z + off[2];
    prep[0] = tip[0];
    prep[1] = tip[1];
    prep[2] = tip[2] + kPreApproachZ;
  } else {
    const double* off = v.left ? kPlaceLeft : kPlaceRight;
    tip[0] = v.x + off[0];
    tip[1] = v.y + off[1];
    tip[2] = v.z + off[2];
    prep[0] = tip[0] - kPreApproachX;
    prep[1] = tip[1];
    prep[2] = tip[2];
  }
}

struct SegResult {
  std::string name;
  bool arrived = false;
  bool refused = false;
  bool samples_ok = false;
  bool cmd_ok = false;
  std::string reason;
  std::string candidate;
  int ticks = 0;
};

SegResult play(kmpc::CartesianTracker& tracker, kmpc::Vec18* q,
               const kmpc::Vec18& q_pref, uint64_t gen, bool left,
               const kmpc::Vec3& goal, const kmpc::Vec4& quat,
               const kmpc::Vec3* tip, const char* name) {
  SegResult r;
  r.name = name;
  if (tip) {
    kmpc::LineTarget t;
    t.xyz = *tip;
    t.quat_xyzw = quat;
    t.ori_on = true;
    if (!tracker.rearm_line(gen, t)) {
      r.reason = "rearm_failed";
      return r;
    }
  } else if (!tracker.rearm_line(gen)) {
    r.reason = "rearm_failed";
    return r;
  }
  const kmpc::Vec3* L = left ? &goal : nullptr;
  const kmpc::Vec3* R = left ? nullptr : &goal;
  const kmpc::Vec4* qL = left ? &quat : nullptr;
  const kmpc::Vec4* qR = left ? nullptr : &quat;
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> commanded;
  commanded.push_back(*q);
  for (int i = 0; i < 6000; ++i) {
    *q += tracker.compute(*q, q_pref, L, R, qL, qR).dq;
    commanded.push_back(*q);
    r.candidate = tracker.line_candidate();
    r.ticks = i + 1;
    if (tracker.line_refused()) {
      r.refused = true;
      r.reason = tracker.line_reason();
      break;
    }
    if (i == 0 && tracker.line_sample_count() >= 2) {
      r.samples_ok = kmpc::check_path_direction_strict(tracker.line_samples()).ok;
    }
    if (tracker.last_arrived()) {
      r.arrived = true;
      tracker.maybe_commit_last_line_cache(*q);
      break;
    }
  }
  if (!r.arrived && r.reason.empty()) r.reason = "timeout";
  if (commanded.size() >= 2) {
    r.cmd_ok = kmpc::check_path_direction_strict(commanded).ok;
  }
  return r;
}

const VisionMm kTargets[] = {
    {"624_LEFT_GRASP", true, true, 493.430, 284.979, 400.0, 0, 0, 0.028923, 0.999582,
     456.072, 272.799, 615, 456.072, 272.799, 465},
    {"624_LEFT_PLACE", true, false, 624.039, 306.951, 400.0, 0, 0, -0.018784, 0.999824,
     344.039, 326.951, 490, 584.039, 326.951, 490},
    {"624_RIGHT_GRASP", false, true, 488.428, -286.318, 400.0, 0, 0, 0.003054, 0.999995,
     428.429, -286.684, 610, 428.429, -286.684, 460},
    {"624_RIGHT_PLACE", false, false, 625.839, -323.452, 400.0, 0, 0, -0.019123, 0.999817,
     345.839, -391.452, 490, 585.839, -391.452, 490},
    {"525_LEFT_GRASP", true, true, 553.430, 284.979, 400.0, 0, 0, 0.028923, 0.999582,
     516.072, 272.799, 615, 516.072, 272.799, 465},
    {"525_LEFT_PLACE", true, false, 525.839, 323.452, 400.0, 0, 0, -0.018784, 0.999824,
     245.839, 343.452, 490, 485.839, 343.452, 490},
    {"525_RIGHT_GRASP", false, true, 558.428, -286.318, 400.0, 0, 0, 0.003054, 0.999995,
     498.429, -286.684, 610, 498.429, -286.684, 460},
    {"525_RIGHT_PLACE", false, false, 525.839, -323.452, 400.0, 0, 0, -0.019123, 0.999817,
     245.839, -391.452, 490, 485.839, -391.452, 490},
};

}  // namespace

TEST(GraspPlaceTargets, OffsetConversionMatchesPlanTable) {
  for (const auto& v : kTargets) {
    double prep[3], tip[3];
    vision_to_prep_tip_mm(v, prep, tip);
    EXPECT_NEAR(prep[0], v.exp_prep_x, 0.05) << v.name << " prep.x";
    EXPECT_NEAR(prep[1], v.exp_prep_y, 0.05) << v.name << " prep.y";
    EXPECT_NEAR(prep[2], v.exp_prep_z, 0.05) << v.name << " prep.z";
    EXPECT_NEAR(tip[0], v.exp_tip_x, 0.05) << v.name << " tip.x";
    EXPECT_NEAR(tip[1], v.exp_tip_y, 0.05) << v.name << " tip.y";
    EXPECT_NEAR(tip[2], v.exp_tip_z, 0.05) << v.name << " tip.z";
  }
}

TEST(GraspPlaceTargets, EightChainsPrepTipRetreatNoFold) {
  // Field squat from 20261110T051249Z / mock_eye cycle fixture.
  const double waist4[] = {40.0060, -49.9995, -10.0003, -1.4969};
  const double right7[] = {-0.0980, 29.5260, -7.8219, -121.8440, -73.0651, -13.9180,
                           0.7989};
  const double left7[] = {-0.0980, 29.5260, 7.8220, -121.8440, 73.0653, -13.9180,
                          0.7991};
  const kmpc::Vec18 q0 = q_from_deg(waist4, right7, left7);

  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");

  int fail = 0;
  uint64_t gen = 1;
  for (const auto& v : kTargets) {
    double prep_mm[3], tip_mm[3];
    vision_to_prep_tip_mm(v, prep_mm, tip_mm);
    const kmpc::Vec3 prep(prep_mm[0] * 1e-3, prep_mm[1] * 1e-3, prep_mm[2] * 1e-3);
    const kmpc::Vec3 tip(tip_mm[0] * 1e-3, tip_mm[1] * 1e-3, tip_mm[2] * 1e-3);
    const kmpc::Vec4 quat = kmpc::tool_to_flange_quat_xyzw(
        kmpc::Vec4(v.qx, v.qy, v.qz, v.qw));

    kmpc::CartesianTracker tracker(kin, cfg_all);
    tracker.set_line_pace(0.3, 0.8);
    tracker.set_line_direction_strict_enable(true);
    kmpc::Vec18 q = q0;

    const std::string tag = v.name;
    auto approach = play(tracker, &q, kmpc::test_fixture_q0(), gen++, v.left, prep, quat, &tip,
                         (tag + "_approach").c_str());
    auto insert = play(tracker, &q, kmpc::test_fixture_q0(), gen++, v.left, tip, quat, nullptr,
                       (tag + "_insert").c_str());
    auto retreat = play(tracker, &q, kmpc::test_fixture_q0(), gen++, v.left, prep, quat, nullptr,
                        (tag + "_retreat").c_str());

    auto score = [&](const SegResult& r) {
      const bool ok =
          r.arrived && !r.refused && r.samples_ok && r.cmd_ok &&
          r.reason.find("final_send_pitch_frozen") == std::string::npos &&
          r.reason.find("direction_") == std::string::npos;
      std::printf(
          "%s arrived=%d refused=%d samples_ok=%d cmd_ok=%d cand=%s reason=%s "
          "ticks=%d\n",
          r.name.c_str(), r.arrived ? 1 : 0, r.refused ? 1 : 0, r.samples_ok ? 1 : 0,
          r.cmd_ok ? 1 : 0, r.candidate.c_str(),
          r.reason.empty() ? "-" : r.reason.c_str(), r.ticks);
      if (!ok) {
        ADD_FAILURE() << r.name << " failed reason=" << r.reason
                      << " cand=" << r.candidate << " arrived=" << r.arrived
                      << " samples_ok=" << r.samples_ok << " cmd_ok=" << r.cmd_ok;
        ++fail;
      }
    };
    score(approach);
    if (approach.arrived) score(insert);
    else ++fail;
    if (approach.arrived && insert.arrived) score(retreat);
    else if (approach.arrived) ++fail;
  }
  EXPECT_EQ(fail, 0) << "failed_checks=" << fail;
}
