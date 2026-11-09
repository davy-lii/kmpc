#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/config.hpp"
#include "kmpc/kinematics.hpp"
#include "kmpc/tracker_math.hpp"

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

TEST(LinePath, SlerpTakesShortArc) {
  const kmpc::Vec4 a(0.0, 0.0, 0.0, 1.0);
  const kmpc::Vec4 b(0.0, 0.0, 1.0, 0.0);
  const kmpc::Vec4 mid = kmpc::quat_slerp_xyzw(a, -b, 0.5);
  EXPECT_NEAR(mid.norm(), 1.0, 1e-9);
  EXPECT_GT(mid[3], 0.5);
}

TEST(LinePath, ShortMoveStaysOnLineThenHolds) {
  const kmpc::Config cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  const kmpc::Vec18 q0 = cfg_all.q_pref;
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
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
    const auto step = a.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
    b.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
    q += step.dq;
  }
  kmpc::Vec18 drifted = q;
  drifted[10] += 0.08;
  a.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
  b.compute(drifted, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
  EXPECT_LT((a.line_cmd() - b.line_cmd()).norm(), 1e-9);
}

TEST(LinePath, LoggedPrepAtScaleStaysOnLine) {
  const kmpc::Config cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
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
    const auto step = tracker.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
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
  EXPECT_TRUE(arrived)<<"prog="<<tracker.line_progress()<<" samples="<<tracker.line_sample_count()<<" e="<<tracker.last_e0();
  EXPECT_LE(worst, 0.003);
  EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameRight) - goal).norm(), 0.003);
  const auto held = tracker.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
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
    const auto step = left ? tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr)
                           : tracker.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
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
  EXPECT_LT(extra_deg(path, 13), 12.0);

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
  EXPECT_LT((pitch_hi - pitch_lo) * 180.0 / kmpc::kPi, 1.5);

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
  EXPECT_GT(pitch_lo, -22.2 * kmpc::kPi / 180.0);
  for (int j = 0; j < 4; ++j) {
    EXPECT_LT(extra_deg(path, j), 2.0) << "waist " << j;
  }
}

struct LineFixture : testing::Test {
  kmpc::Config config=kmpc::Config::load(cfg("joint_limits.yaml"),cfg("q_pref.yaml"),cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin{std::string(KMPC_URDF_DIR)+"/Light-duty wheeled robot.urdf"};
  kmpc::Vec18 q=config.q_pref;
  kmpc::Vec3 goal=kin.fk_pos(q,kmpc::kEeFrameLeft)+kmpc::Vec3(0,.06,0);
  kmpc::Vec4 quat=kmpc::rot_to_quat_xyzw(kin.fk(q,kmpc::kEeFrameLeft).block<3,3>(0,0));
};
TEST_F(LineFixture, FreezeAndAbortBoundariesAreLatched) {
  for(double jump:{.049,.050,.051}) {
    kmpc::CartesianTracker tracker(kin,config);
    ASSERT_TRUE(tracker.rearm_line(1));
    auto step=tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
    ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
    ASSERT_TRUE(tracker.line_frozen());
    const auto at=q+step.dq;const int gen=tracker.line_gen();
    kmpc::Vec3 changed=goal+kmpc::Vec3(jump,0,0);
    auto changed_step=tracker.compute(at,config.q_pref,&changed,nullptr,&quat,nullptr);
    EXPECT_EQ(gen,tracker.line_gen());
    if(jump<.05) {
      EXPECT_FALSE(tracker.line_refused());
    } else {
      EXPECT_TRUE(tracker.line_abort_latched());EXPECT_TRUE(tracker.line_refused());
      EXPECT_EQ(changed_step.dq.norm(),0);
      const int solves=tracker.line_qp_solves();
      for(int i=0;i<5;++i) {
        auto held=tracker.compute(at,config.q_pref,&goal,nullptr,&quat,nullptr);
        EXPECT_EQ(held.dq.norm(),0);EXPECT_FALSE(tracker.last_arrived());
      }
      EXPECT_EQ(solves,tracker.line_qp_solves());
      EXPECT_FALSE(tracker.rearm_line(1));
      ASSERT_TRUE(tracker.rearm_line(2));
      tracker.compute(at,config.q_pref,&goal,nullptr,&quat,nullptr);
      EXPECT_FALSE(tracker.line_abort_latched());EXPECT_FALSE(tracker.line_refused())<<tracker.line_reason();
    }
  }
}
TEST_F(LineFixture, OrientationFreezeAndAbort) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  auto step=tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);q+=step.dq;
  ASSERT_TRUE(tracker.line_frozen());const int gen=tracker.line_gen();
  Eigen::Quaterniond original(quat[3],quat[0],quat[1],quat[2]);
  auto rotated=[&](double deg) {Eigen::Quaterniond v=Eigen::AngleAxisd(deg*kmpc::kPi/180,kmpc::Vec3::UnitZ())*original;return kmpc::Vec4(v.x(),v.y(),v.z(),v.w());};
  auto small=rotated(2);tracker.compute(q,config.q_pref,&goal,nullptr,&small,nullptr);
  EXPECT_EQ(gen,tracker.line_gen());EXPECT_FALSE(tracker.line_refused());
  auto large=rotated(16);tracker.compute(q,config.q_pref,&goal,nullptr,&large,nullptr);
  EXPECT_TRUE(tracker.line_abort_latched());EXPECT_EQ(tracker.line_reason(),"frozen_orientation_jump");
}
TEST_F(LineFixture, CascadeUsesActualPoseAndRejectsUnknownFrame) {
  kmpc::LineTarget tip;tip.xyz=goal+kmpc::Vec3(0,.015,0);tip.quat_xyzw=quat;tip.ori_on=true;
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1,tip);
  tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
  ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  EXPECT_TRUE(tracker.line_cascade_checked());EXPECT_TRUE(tracker.line_cascade_passed());
  tip.frame="camera_link";tracker.rearm_line(2,tip);
  auto step=tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
  EXPECT_TRUE(tracker.line_refused());EXPECT_EQ(step.dq.norm(),0);
  EXPECT_EQ(tracker.line_reason(),"invalid_cascade_target");
  tip.frame="base_link";tip.xyz=kmpc::Vec3(9,9,9);tracker.rearm_line(3,tip);
  tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
  EXPECT_TRUE(tracker.line_refused());EXPECT_TRUE(tracker.line_cascade_checked());
  EXPECT_FALSE(tracker.line_cascade_passed());EXPECT_EQ(tracker.line_reason(),"cascade_unreachable");
}
TEST_F(LineFixture, SamplesAndInterpolationMeetThreeMillimeters) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
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
  tracker.compute(q,config.q_pref,&far,nullptr,&quat,nullptr);
  EXPECT_EQ(tracker.line_reason(),"sample_limit");EXPECT_TRUE(tracker.line_refused());
}
TEST_F(LineFixture, ReplanStartsAtTheProvidedStreamReference) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
  const auto mid=(.5*(q+tracker.line_cmd())).eval();
  tracker.rearm_line(2);tracker.set_line_reference(mid);
  tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);
  ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  ASSERT_FALSE(tracker.line_samples().empty());EXPECT_LT((tracker.line_samples().front()-mid).norm(),1e-12);
  EXPECT_TRUE(((tracker.line_cmd()-mid).cwiseAbs().array()<=config.limits.dq_max.array()*config.params.dt+1e-6).all());
}

TEST_F(LineFixture, ArrivalRequiresMeasuredEndpoint) {
  kmpc::CartesianTracker tracker(kin,config);tracker.rearm_line(1);
  const auto seed=q;
  for(int i=0;i<1500 && !tracker.last_arrived();++i) {
    auto step=tracker.compute(q,config.q_pref,&goal,nullptr,&quat,nullptr);q+=step.dq;
    ASSERT_FALSE(tracker.line_refused())<<tracker.line_reason();
  }
  ASSERT_TRUE(tracker.line_holding());
  EXPECT_TRUE(tracker.line_measured_arrived(q));
  EXPECT_FALSE(tracker.line_measured_arrived(seed));
}

// Recorded robot03 placement, 2026-11-04 09:17 (nearest 50 Hz feedback).
TEST(LinePath, Robot03PlacementAllowsConstrainedWaistInSuccessor) {
  const auto cfg_all = kmpc::Config::load(cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
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
  q+=tracker.compute(q,cfg_all.q_pref,&goal,nullptr,&quat,nullptr).dq;
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  EXPECT_TRUE(tracker.line_cascade_passed());
  EXPECT_TRUE(tracker.line_cascade_full_body());
  for (const auto& p : {goal, tip.xyz}) {
    kmpc::CartesianTracker single(kin,cfg_all); single.rearm_line(1);
    single.compute(q,cfg_all.q_pref,&p,nullptr,&quat,nullptr);
    EXPECT_FALSE(single.line_refused()) << p.transpose() << " " << single.line_reason();
  }
  tracker.set_line_pace(0.3,0.8);
  for(const auto& p : {goal,tip.xyz}) {
    if((p-tip.xyz).norm()<1e-9) { ASSERT_TRUE(tracker.rearm_line(2)); }
    bool arrived=false;
    for(int i=0;i<3000;++i) {
      const auto step=tracker.compute(q,cfg_all.q_pref,&p,nullptr,&quat,nullptr);
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
  locked.compute(initial,cfg_all.q_pref,&goal,nullptr,&quat,nullptr);
  EXPECT_TRUE(locked.line_refused());
  EXPECT_FALSE(locked.line_cascade_full_body());
}

TEST(LinePath, ReverseLastLineRestoresQa) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec18 qA = q;
  const kmpc::Vec3 pA = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 pB = pA + kmpc::Vec3(0.08, 0.0, 0.0);

  ASSERT_TRUE(tracker.rearm_line(1));
  bool forward_ok = false;
  for (int i = 0; i < 800; ++i) {
    const auto step = tracker.compute(q, cfg_all.q_pref, &pB, nullptr, &quat, nullptr);
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
  const auto step0 = tracker.compute(q, cfg_all.q_pref, &pA, nullptr, &quat, nullptr);
  ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
  EXPECT_EQ(tracker.line_candidate(), "reverse_cache")
      << tracker.reverse_line_decision();
  EXPECT_EQ(tracker.reverse_line_decision(), "accepted");
  ASSERT_EQ(tracker.line_sample_count(), forward_samples.size());
  for (size_t i = 0; i < forward_samples.size(); ++i) {
    const auto& got = tracker.line_samples()[i];
    const auto& want = forward_samples[forward_samples.size() - 1 - i];
    // First sample is re-anchored to live qB; remaining match reverse cache.
    if (i == 0) {
      EXPECT_LT((got - q).cwiseAbs().maxCoeff(), 1e-12);
    } else {
      EXPECT_LT((got - want).cwiseAbs().maxCoeff(), 1e-9) << "sample " << i;
    }
  }
  bool back_ok = false;
  for (int i = 0; i < 800; ++i) {
    const auto step = tracker.compute(q, cfg_all.q_pref, &pA, nullptr, &quat, nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    q += step.dq;
    if (tracker.last_arrived()) {
      back_ok = true;
      break;
    }
  }
  EXPECT_TRUE(back_ok);
  EXPECT_LT((kin.fk_pos(q, kmpc::kEeFrameLeft) - pA).norm(), 0.003);
  EXPECT_LT((q - qA).cwiseAbs().maxCoeff(), 2.0 * kmpc::kPi / 180.0)
      << "qA recovery " << (q - qA).cwiseAbs().transpose();
  (void)step0;
}

TEST(LinePath, ReverseRejectsQbMismatch) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.6, 0.0);
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 pA = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 pB = pA + kmpc::Vec3(0.07, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  for (int i = 0; i < 800 && !tracker.last_arrived(); ++i) {
    q += tracker.compute(q, cfg_all.q_pref, &pB, nullptr, &quat, nullptr).dq;
  }
  ASSERT_TRUE(tracker.last_arrived());
  tracker.maybe_commit_last_line_cache(q);
  ASSERT_TRUE(tracker.last_line_cache().valid);
  q[4] += 10.0 * kmpc::kPi / 180.0;  // spoil qB
  ASSERT_TRUE(tracker.rearm_line(2));
  tracker.compute(q, cfg_all.q_pref, &pA, nullptr, &quat, nullptr);
  EXPECT_NE(tracker.line_candidate(), "reverse_cache");
  EXPECT_EQ(tracker.reverse_line_decision(), "rejected_qb_mismatch");
  EXPECT_FALSE(tracker.last_line_cache().valid);
}

TEST(LinePath, LineDiagRingDefaultOffBounded) {
  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.05, 0.0, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
  EXPECT_EQ(tracker.line_diag_size(), 0u);

  tracker.set_line_diag_enable(true);
  ASSERT_TRUE(tracker.rearm_line(2));
  for (int i = 0; i < 40; ++i) {
    q += tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr).dq;
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
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
    EXPECT_LE(worst_lat, 0.003) << name;
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
    const auto first = tracker.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
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
      const auto step = tracker.compute(q, cfg_all.q_pref, nullptr, &goal, nullptr, &quat);
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
    EXPECT_LE(turn, 0.05 * kmpc::kPi / 180.0)
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
      play(tracker, q, 3, nullptr, grasp_prep, grasp_quat, (tag + " retreat").c_str(),
           "reverse_cache");
    }
    kmpc::CartesianTracker place_tracker(kin, cfg_all);
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
         "reverse_cache");
  }
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  const kmpc::Vec18 q0 = cfg_all.q_pref;
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
          tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
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
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::Vec18 q = cfg_all.q_pref;
  const kmpc::Vec3 p0 = kin.fk_pos(q, kmpc::kEeFrameLeft);
  const auto posed = kin.fk(q, kmpc::kEeFrameLeft);
  const kmpc::Vec4 quat = kmpc::rot_to_quat_xyzw(posed.block<3, 3>(0, 0));
  const kmpc::Vec3 goal = p0 + kmpc::Vec3(0.25, 0.0, 0.0);

  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace_mode(kmpc::CartesianTracker::LinePaceMode::R10bFirstOrder);
  ASSERT_TRUE(tracker.set_line_brake_distance_m(0.20));  // wider brake, freeze stays 0.08
  tracker.set_line_pace(0.3, 0.0);
  ASSERT_TRUE(tracker.rearm_line(1));
  bool saw_full = false;
  bool saw_brake_scale = false;
  bool arrived = false;
  for (int i = 0; i < 4000; ++i) {
    const auto step =
        tracker.compute(q, cfg_all.q_pref, &goal, nullptr, &quat, nullptr);
    ASSERT_FALSE(tracker.line_refused()) << tracker.line_reason();
    const double remain = tracker.last_line_path_remain_m();
    if (remain > 0.20 + 1e-9) {
      EXPECT_NEAR(tracker.last_line_brake(), 1.0, 1e-12);
      saw_full = true;
    } else if (remain > 1e-9) {
      EXPECT_NEAR(tracker.last_line_brake(),
                  std::max(0.05, remain / 0.20), 1e-9);
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
