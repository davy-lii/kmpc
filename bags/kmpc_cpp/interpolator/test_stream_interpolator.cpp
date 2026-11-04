#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "kmpc/stream_interpolator.hpp"

namespace {

kmpc::Vec18 axis0(double x) {
  kmpc::Vec18 q = kmpc::Vec18::Zero();
  q[0] = x;
  return q;
}

void arm(kmpc::StreamInterpolator& ip, double v_bar, double a_bar,
         double jerk_s = 0.01) {
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(0.02);
  ip.set_jerk_s(jerk_s);
}

}  // namespace

TEST(StreamInterpolator, StepSettlesMonotone) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(0.1));
  arm(ip, 0.1, 2.0);
  double prev = 0.0;
  for (int k = 0; k < 150; ++k) {
    const kmpc::Vec18& q = ip.step(0.01);
    EXPECT_GE(q[0] + 1e-15, prev);
    prev = q[0];
    EXPECT_LE(std::abs(ip.v_ref()[0]), 0.1 + 1e-12);
  }
  EXPECT_NEAR(ip.q_ref()[0], 0.1, 1e-12);
  EXPECT_NEAR(ip.v_ref()[0], 0.0, 1e-12);
}

TEST(StreamInterpolator, VelocityLimitEachStep) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(1.0));
  arm(ip, 0.1, 2.0);
  for (int k = 0; k < 80; ++k) {
    ip.step(0.01);
    EXPECT_LE(std::abs(ip.v_ref()[0]), 0.1 + 1e-12);
  }
}

TEST(StreamInterpolator, BrakeThenPin) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(1.0));
  arm(ip, 0.1, 2.0);
  for (int k = 0; k < 20; ++k) {
    ip.step(0.01);
  }
  ASSERT_NEAR(ip.v_ref()[0], 0.1, 1e-12);
  const double q_now = ip.q_ref()[0];
  const double goal = q_now + 0.001;
  ip.set_goal(axis0(goal));
  double v_prev = ip.v_ref()[0];
  bool decreased = false;
  double overshoot = 0.0;
  for (int k = 0; k < 40; ++k) {
    ip.step(0.01);
    const double v = ip.v_ref()[0];
    if (v < v_prev - 1e-15) {
      decreased = true;
    }
    overshoot = std::max(overshoot, ip.q_ref()[0] - goal);
    v_prev = v;
  }
  EXPECT_TRUE(decreased);
  EXPECT_LT(overshoot, 0.1 * 0.01 + 1e-12);
  EXPECT_NEAR(ip.q_ref()[0], goal, 1e-12);
  EXPECT_NEAR(ip.v_ref()[0], 0.0, 1e-12);
}

TEST(StreamInterpolator, NewGoalDoesNotReverse) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(0.05));
  arm(ip, 0.1, 2.0);
  for (int k = 0; k < 8; ++k) {
    ip.step(0.01);
  }
  ASSERT_GT(ip.v_ref()[0], 0.0);
  const double q_before = ip.q_ref()[0];
  ip.set_goal(axis0(0.2));
  for (int k = 0; k < 20; ++k) {
    ip.step(0.01);
    EXPECT_GE(ip.q_ref()[0] + 1e-15, q_before);
  }
}

TEST(StreamInterpolator, NanGoalKeepsPrevious) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(0.1));
  arm(ip, 0.1, 2.0);
  ip.step(0.01);
  const kmpc::Vec18 goal_before = ip.q_goal();
  const kmpc::Vec18 q_before = ip.q_ref();
  kmpc::Vec18 nan_g = axis0(0.5);
  nan_g[0] = std::numeric_limits<double>::quiet_NaN();
  ip.set_goal(nan_g);
  EXPECT_EQ(ip.q_goal()[0], goal_before[0]);
  ip.step(0.01);
  EXPECT_GE(ip.q_ref()[0], q_before[0]);
  EXPECT_LE(ip.q_ref()[0], 0.1 + 1e-12);
  const double q_keep = ip.q_ref()[0];
  const double v_keep = ip.v_ref()[0];
  ip.reset(axis0(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(ip.ready());
  EXPECT_NEAR(ip.q_ref()[0], q_keep, 1e-12);
  EXPECT_NEAR(ip.v_ref()[0], v_keep, 1e-12);
}

TEST(StreamInterpolator, JerkBoxAt200Hz) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(1.0));
  arm(ip, 0.1, 2.0, 0.01);
  const double dt = 0.005;
  const double j_dt = (2.0 / 0.01) * dt;
  double a_prev = 0.0;
  bool saw_slew = false;
  for (int k = 0; k < 80; ++k) {
    ip.step(dt);
    const double a = ip.a_ref()[0];
    EXPECT_LE(std::abs(a - a_prev), j_dt + 1e-12);
    if (k == 0) {
      EXPECT_NEAR(a, j_dt, 1e-12);
      saw_slew = std::abs(a) < 2.0 - 1e-12;
    }
    EXPECT_LE(std::abs(ip.v_ref()[0]), 0.1 + 1e-12);
    a_prev = a;
  }
  EXPECT_TRUE(saw_slew);
  EXPECT_NEAR(ip.v_ref()[0], 0.1, 1e-12);
  EXPECT_NEAR(ip.a_ref()[0], 0.0, 1e-9);
}

TEST(StreamInterpolator, CatchSDoesNotDelayBrake) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(0.22));
  const double v_bar = 0.32;
  const double a_bar = v_bar / 0.30;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(0.05);
  ip.set_jerk_s(0.08);
  const double dt = 0.02;
  const double s_stop = (v_bar * v_bar) / (2.0 * a_bar);
  double v_peak = 0.0;
  bool cruised = false;
  double remain_at_drop = -1.0;
  for (int k = 0; k < 200; ++k) {
    ip.step(dt);
    const double v = ip.v_ref()[0];
    const double remain = 0.22 - ip.q_ref()[0];
    v_peak = std::max(v_peak, v);
    if (v > 0.85 * v_bar) {
      cruised = true;
    }
    if (cruised && remain_at_drop < 0.0 && v < 0.90 * v_peak) {
      remain_at_drop = remain;
      break;
    }
  }
  EXPECT_TRUE(cruised);
  ASSERT_GT(remain_at_drop, 0.0);
  EXPECT_GT(remain_at_drop, 0.02) << "catch_s delayed brake to last 0.02 rad";
  EXPECT_NEAR(remain_at_drop, s_stop, 0.04);
}

TEST(StreamInterpolator, TwentyTwoRadDoesNotPinFromCruise) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(0.22));
  const double v_bar = 0.32;
  const double a_bar = v_bar / 0.30;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(0.05);
  ip.set_jerk_s(0.08);
  const double dt = 0.02;
  double v_prev = 0.0;
  bool saw_cruise = false;
  bool saw_taper = false;
  for (int k = 0; k < 200; ++k) {
    ip.step(dt);
    const double v = ip.v_ref()[0];
    if (v > 0.85 * v_bar) {
      saw_cruise = true;
    }
    if (saw_cruise && v < v_prev - 1e-4 && v > 0.05) {
      saw_taper = true;
    }
    if (saw_cruise && std::abs(v_prev) > 0.20 && std::abs(v) < 1e-6) {
      ADD_FAILURE() << "pinned from cruise in one tick v " << v_prev << " -> "
                    << v;
    }
    v_prev = v;
  }
  EXPECT_TRUE(saw_cruise);
  EXPECT_TRUE(saw_taper);
  EXPECT_NEAR(ip.q_ref()[0], 0.22, 3e-3);
  EXPECT_NEAR(ip.v_ref()[0], 0.0, 2e-3);
}

TEST(StreamInterpolator, FollowMovingWaypointKeepsCruise) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  const double v_bar = 0.32;
  const double a_bar = v_bar / 0.30;
  const double dt = 0.02;
  const double step = 0.016;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(dt);
  ip.set_jerk_s(0.08);
  for (int k = 0; k < 50; ++k) {
    ip.set_goal(axis0(ip.q_ref()[0] + step));
    ip.step(dt);
  }
  EXPECT_GT(ip.v_ref()[0], 0.25) << "follow must not rest-to-stop each QP step";
  EXPECT_GT(ip.q_ref()[0], 0.08);
  EXPECT_LE(ip.v_ref()[0], v_bar + 1e-12);
}

TEST(StreamInterpolator, VBarDropDecelsAtABar) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(2.0));
  const double v_bar = 0.32;
  const double a_bar = v_bar / 0.30;
  const double dt = 0.02;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(0.05);
  ip.set_jerk_s(0.08);
  for (int k = 0; k < 80; ++k) {
    ip.step(dt);
  }
  ASSERT_GT(ip.v_ref()[0], 0.25);
  const double v_before = ip.v_ref()[0];
  ip.set_limits(kmpc::Vec18::Constant(0.01), kmpc::Vec18::Constant(a_bar));
  ip.step(dt);
  const double dropped = v_before - ip.v_ref()[0];
  EXPECT_GT(ip.v_ref()[0], 0.15) << "v_bar drop must not clip v in one tick";
  EXPECT_LE(dropped, a_bar * dt + 1e-6);
  EXPECT_GT(dropped, 0.0);
}

TEST(StreamInterpolator, FiftyHzCruisesThenBrakesToSticky) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_goal(axis0(1.0));
  const double v_bar = 0.40;
  const double a_bar = v_bar / 0.30;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(0.05);
  ip.set_jerk_s(0.08);
  const double dt = 0.02;
  double v_max = 0.0;
  bool saw_cruise = false;
  bool saw_brake = false;
  double v_at_mid = 0.0;
  for (int k = 0; k < 200; ++k) {
    ip.step(dt);
    const double v = ip.v_ref()[0];
    const double remain = 1.0 - ip.q_ref()[0];
    v_max = std::max(v_max, v);
    if (v > 0.35) {
      saw_cruise = true;
    }
    if (saw_cruise && remain < 0.12 && v < v_max - 0.05) {
      saw_brake = true;
    }
    if (k == 40) {
      v_at_mid = v;
    }
  }
  EXPECT_GT(v_at_mid, 0.30);
  EXPECT_TRUE(saw_cruise);
  EXPECT_TRUE(saw_brake);
  EXPECT_NEAR(ip.q_ref()[0], 1.0, 2e-3);
  EXPECT_NEAR(ip.v_ref()[0], 0.0, 1e-3);
}

TEST(StreamInterpolator, QpSegmentKeepsCruiseAt250Hz) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  const double v_bar = 0.11;
  const double a_bar = v_bar / 0.50;
  const double dt = 0.004;
  const double period = 0.02;
  const double qp_step = 0.0022;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(period);
  ip.set_jerk_s(0.08);
  double q_from = 0.0;
  double q_to = qp_step;
  int a_sign_flips = 0;
  double a_prev = 0.0;
  bool have_a = false;
  for (int p = 0; p < 40; ++p) {
    ip.begin_qp_segment(axis0(q_from), axis0(q_to), period);
    for (int k = 0; k < 5; ++k) {
      ip.step(dt);
      const double a = ip.a_ref()[0];
      if (have_a && a_prev * a < -1e-12 && std::abs(a_prev) > 1e-6 &&
          std::abs(a) > 1e-6) {
        ++a_sign_flips;
      }
      a_prev = a;
      have_a = true;
    }
    q_from = q_to;
    q_to += qp_step;
  }
  EXPECT_GT(ip.v_ref()[0], 0.06) << "segment follow must not rest-to-stop each QP";
  EXPECT_LE(ip.v_ref()[0], v_bar + 1e-12);
  EXPECT_LT(a_sign_flips, 20) << "must not flip accel every 20 ms";
  EXPECT_GT(ip.q_ref()[0], 0.04);
}

TEST(StreamInterpolator, SetGoalClearsSegment) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  arm(ip, 0.11, 0.11 / 0.50, 0.08);
  ip.begin_qp_segment(axis0(0.0), axis0(0.01), 0.02);
  ip.step(0.004);
  ip.set_stop_at_goal(true);
  ip.set_goal(axis0(0.02));
  EXPECT_NEAR(ip.q_goal()[0], 0.02, 1e-12);
  for (int k = 0; k < 400; ++k) {
    ip.step(0.004);
  }
  EXPECT_NEAR(ip.q_ref()[0], 0.02, 3e-3);
  EXPECT_NEAR(ip.v_ref()[0], 0.0, 2e-3);
}

TEST(StreamInterpolator, ShortAxisBrakesWhileLongTracks) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  const double v_bar = 0.11;
  const double a_bar = v_bar / 0.50;
  const double dt = 0.004;
  const double period = 0.02;
  const double qp_step = 0.0022;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(period);
  ip.set_jerk_s(0.08);
  kmpc::Vec18 star = kmpc::Vec18::Zero();
  star[0] = 1.0;
  star[1] = 0.0035;
  ip.set_q_star(star);
  double q_from0 = 0.0;
  double q_to0 = qp_step;
  int zcross = 0;
  double prev_e1 = star[1];
  for (int p = 0; p < 40; ++p) {
    kmpc::Vec18 from = kmpc::Vec18::Zero();
    kmpc::Vec18 to = kmpc::Vec18::Zero();
    from[0] = q_from0;
    to[0] = q_to0;
    ip.begin_qp_segment(from, to, period);
    for (int k = 0; k < 5; ++k) {
      ip.step(dt);
      const double e1 = star[1] - ip.q_ref()[1];
      if (prev_e1 * e1 < 0.0 && std::abs(prev_e1) > 1e-8 &&
          std::abs(e1) > 1e-8) {
        ++zcross;
      }
      prev_e1 = e1;
    }
    q_from0 = q_to0;
    q_to0 += qp_step;
  }
  EXPECT_GT(ip.v_ref()[0], 0.06);
  EXPECT_EQ(ip.axis_mode(0), kmpc::AxisMode::Track);
  EXPECT_NE(ip.axis_mode(1), kmpc::AxisMode::Track);
  EXPECT_NEAR(ip.q_ref()[1], star[1], 2e-3);
  EXPECT_LT(std::abs(ip.v_ref()[1]), 0.02);
  EXPECT_EQ(zcross, 0) << "short axis must not hunt through q_star";
  EXPECT_GE(ip.n_axis_brake(), 1);
}

TEST(StreamInterpolator, GeometricTailEntersBrake) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  const double v_bar = 0.11;
  const double a_bar = v_bar / 0.50;
  ip.set_limits(kmpc::Vec18::Constant(v_bar), kmpc::Vec18::Constant(a_bar));
  ip.set_catch_s(0.02);
  ip.set_jerk_s(0.08);
  kmpc::Vec18 star = kmpc::Vec18::Zero();
  star[0] = 0.0087;
  ip.set_q_star(star);
  ip.step(0.004);
  EXPECT_NE(ip.axis_mode(0), kmpc::AxisMode::Track)
      << "e=0.5deg < 1deg must enter via e_near, not s_stop";
  for (int k = 0; k < 400; ++k) {
    ip.step(0.004);
  }
  EXPECT_NEAR(ip.q_ref()[0], star[0], 2e-3);
  EXPECT_NEAR(ip.v_ref()[0], 0.0, 2e-3);
}

TEST(StreamInterpolator, BrakesAtInnerJointLimit) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  kmpc::Vec18 q_min = kmpc::Vec18::Constant(-2.0);
  kmpc::Vec18 q_max = kmpc::Vec18::Constant(0.20);
  const double inner = 0.005236;
  ip.set_joint_box(q_min, q_max, inner);
  arm(ip, 0.35, 0.70, 0.08);
  ip.set_goal(axis0(1.0));
  ip.begin_qp_segment(kmpc::Vec18::Zero(), axis0(1.0), 0.02);
  double max_q = 0.0;
  for (int k = 0; k < 800; ++k) {
    const double q = ip.step(0.004)[0];
    max_q = std::max(max_q, q);
  }
  EXPECT_LE(max_q, 0.20 - inner + 1e-6);
  const double at_stop = ip.q_ref()[0];
  ip.set_goal(axis0(-0.5));
  ip.begin_qp_segment(axis0(at_stop), axis0(-0.5), 0.02);
  for (int k = 0; k < 80; ++k) {
    ip.step(0.004);
  }
  EXPECT_LT(ip.q_ref()[0], at_stop - 0.01);
}

TEST(StreamInterpolator, ResetAndHoldStayInsideInnerBox) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  kmpc::Vec18 q_min = kmpc::Vec18::Constant(-2.0);
  kmpc::Vec18 q_max = kmpc::Vec18::Constant(0.20);
  const double inner = 0.005236;
  ip.set_joint_box(q_min, q_max, inner);
  ip.reset(axis0(1.0));
  EXPECT_NEAR(ip.q_ref()[0], 0.20 - inner, 1e-9);
  ip.hold_axis(0);
  ip.set_q_ref(axis0(-3.0));
  EXPECT_NEAR(ip.q_ref()[0], -2.0 + inner, 1e-9);
  ip.step(0.004);
  EXPECT_NEAR(ip.q_ref()[0], -2.0 + inner, 1e-9);
}

TEST(StreamInterpolator, StarJumpReleasesBrake) {
  kmpc::StreamInterpolator ip;
  ip.reset(kmpc::Vec18::Zero());
  ip.set_stop_at_goal(false);
  arm(ip, 0.11, 0.11 / 0.50, 0.08);
  kmpc::Vec18 star = kmpc::Vec18::Zero();
  star[0] = 0.01;
  ip.set_q_star(star);
  ip.step(0.004);
  ASSERT_NE(ip.axis_mode(0), kmpc::AxisMode::Track);
  star[0] = 0.05;
  ip.set_q_star(star);
  ip.step(0.004);
  EXPECT_EQ(ip.axis_mode(0), kmpc::AxisMode::Track);
}

TEST(StreamBrakeEnter, RestAndFarDoesNotBrake) {
  const kmpc::Vec18 z = kmpc::Vec18::Zero();
  const kmpc::Vec18 a_bar = kmpc::Vec18::Constant(0.22);
  EXPECT_FALSE(kmpc::stream_brake_enter(
      z, z, z, axis0(0.5), a_bar, 0.08));
}

TEST(StreamBrakeEnter, SmallRemainForcesFixedGoal) {
  const kmpc::Vec18 z = kmpc::Vec18::Zero();
  const kmpc::Vec18 a_bar = kmpc::Vec18::Constant(0.22);
  EXPECT_TRUE(kmpc::stream_brake_enter(
      z, z, z, axis0(0.04), a_bar, 0.08));
}

TEST(StreamBrakeEnter, CruiseStopDistance) {
  kmpc::Vec18 q_ref = kmpc::Vec18::Zero();
  kmpc::Vec18 v_ref = kmpc::Vec18::Zero();
  kmpc::Vec18 a_ref = kmpc::Vec18::Zero();
  kmpc::Vec18 q_star = kmpc::Vec18::Zero();
  const kmpc::Vec18 a_bar = kmpc::Vec18::Constant(0.22);
  v_ref[0] = 0.11;
  q_star[0] = 0.20;
  EXPECT_FALSE(kmpc::stream_brake_enter(
      q_ref, v_ref, a_ref, q_star, a_bar, 0.08));
  q_star[0] = 0.02;
  EXPECT_TRUE(kmpc::stream_brake_enter(
      q_ref, v_ref, a_ref, q_star, a_bar, 0.08));
}

TEST(StreamAxisBrakeEnter, GeometricTailNeedsENear) {
  const double e = 0.0087;
  const double v = 5.0 * e;
  const double a_bar = 0.22;
  const double s = kmpc::stream_s_stop(v, 0.0, a_bar, 0.08);
  EXPECT_LT(s, e) << "pure s_stop must miss the geometric tail";
  EXPECT_FALSE(kmpc::stream_axis_brake_enter(e, v, 0.0, a_bar, 0.08, 0.0));
  EXPECT_TRUE(kmpc::stream_axis_brake_enter(e, v, 0.0, a_bar, 0.08, 0.05));
}

TEST(StreamAxisBrakeEnter, RestAndFarStillFalse) {
  EXPECT_FALSE(kmpc::stream_axis_brake_enter(0.5, 0.0, 0.0, 0.22, 0.08, 0.05));
}
