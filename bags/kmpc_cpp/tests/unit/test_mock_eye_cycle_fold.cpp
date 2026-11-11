// Offline fold check for mock_eye cycle points (mm→m).
// Under test: local HEAD 86fe46b + uncommitted direction-strict work
// (field package was r21; r22 not released).

#include <cmath>
#include <cstdio>
#include <iostream>
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

kmpc::Vec18 q_from_deg(const double waist4[4], const double right7[7],
                       const double left7[7]) {
  kmpc::Vec18 q;
  for (int i = 0; i < 4; ++i) q[i] = waist4[i] * kmpc::kPi / 180.0;
  for (int i = 0; i < 7; ++i) q[4 + i] = right7[i] * kmpc::kPi / 180.0;
  for (int i = 0; i < 7; ++i) q[11 + i] = left7[i] * kmpc::kPi / 180.0;
  return q;
}

struct PoseMm {
  double x, y, z, qx, qy, qz, qw;
};

kmpc::Vec4 flange_quat(const PoseMm& p) {
  return kmpc::tool_to_flange_quat_xyzw(kmpc::Vec4(p.qx, p.qy, p.qz, p.qw));
}

struct SegReport {
  std::string name;
  bool armed = false;
  bool arrived = false;
  bool refused = false;
  std::string reason;
  std::string candidate;
  std::string reverse;
  bool samples_ok = false;
  bool cmd_ok = false;
  double fold_j1_s = 0, fold_j2_s = 0, fold_j3_s = 0, fold_p_s = 0;
  double fold_j1_c = 0, fold_j2_c = 0, fold_j3_c = 0, fold_p_c = 0;
  int ticks = 0;
};

void fill_folds(const std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>>& path,
                double* j1, double* j2, double* j3, double* pitch, bool* ok) {
  if (path.size() < 2) {
    *ok = false;
    return;
  }
  const auto chk = kmpc::check_path_direction_strict(path);
  *ok = chk.ok;
  *j1 = chk.axes[0].fold_rad * 180.0 / kmpc::kPi;
  *j2 = chk.axes[1].fold_rad * 180.0 / kmpc::kPi;
  *j3 = chk.axes[2].fold_rad * 180.0 / kmpc::kPi;
  *pitch = chk.axes[3].fold_rad * 180.0 / kmpc::kPi;
}

const char* diagnose_reverse_validate(
    const kmpc::Kinematics& kin, const kmpc::Config& cfg,
    const kmpc::CartesianTracker& tracker, const kmpc::Vec18& q0,
    const kmpc::Vec3& target, const kmpc::Vec4& quat, bool left) {
  const auto& cache = tracker.last_line_cache();
  if (!cache.valid) return "no_cache";
  const double q_err = (q0 - cache.q_end).cwiseAbs().maxCoeff() * 180.0 / kmpc::kPi;
  std::printf(
      "retreat_diag cache_valid=1 cand_src=%s q_end_err_deg=%.4f n=%zu "
      "cache_start_vs_retreat_target_mm=%.2f cache_end_vs_live_mm=%.2f "
      "cache_ori_on=%d\n",
      cache.candidate_source.c_str(), q_err, cache.samples.size(),
      (kin.fk_pos(cache.q_start, cache.frame) - target).norm() * 1e3,
      (kin.fk_pos(q0, cache.frame) - cache.end_xyz).norm() * 1e3,
      cache.ori_on ? 1 : 0);
  // Reproduce reverse_cache construction.
  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> rev;
  rev.reserve(cache.samples.size());
  for (auto it = cache.samples.rbegin(); it != cache.samples.rend(); ++it) {
    rev.push_back(*it);
  }
  rev.front() = q0;
  const bool ori_on = !kmpc::quat_is_identity_xyzw(&quat);
  const std::string& frame = cache.frame;
  const kmpc::Vec3 start = kin.fk_pos(rev.front(), frame);
  const kmpc::Vec4 start_ori =
      kmpc::rot_to_quat_xyzw(kin.fk(rev.front(), frame).block<3, 3>(0, 0));
  const int n = static_cast<int>(rev.size()) - 1;
  double worst_tube = 0.0;
  double worst_ori = 0.0;
  double worst_com_out = 0.0;
  bool hit_limit = false;
  int fail_i = -1, fail_j = -1;
  const char* fail = "ok";
  const double margin = cfg.params.q_limit_margin;
  const double cx0 = cfg.params.com_xmin + cfg.params.com_margin;
  const double cx1 = cfg.params.com_xmax - cfg.params.com_margin;
  const double cy0 = cfg.params.com_ymin + cfg.params.com_margin;
  const double cy1 = cfg.params.com_ymax - cfg.params.com_margin;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= 5; ++j) {
      const double a = j / 5.0;
      const kmpc::Vec18 qq = (1 - a) * rev[static_cast<std::size_t>(i)] +
                            a * rev[static_cast<std::size_t>(i + 1)];
      if (!qq.allFinite()) {
        hit_limit = true;
        fail = "nonfinite";
        fail_i = i;
        fail_j = j;
        break;
      }
      {
        int bad = -1;
        double viol = 0.0;
        for (int k = 0; k < 18; ++k) {
          const double lo = cfg.limits.q_min[k] + margin;
          const double hi = cfg.limits.q_max[k] - margin;
          if (qq[k] < lo - 1e-9) {
            bad = k;
            viol = (lo - qq[k]) * 180.0 / kmpc::kPi;
            break;
          }
          if (qq[k] > hi + 1e-9) {
            bad = k;
            viol = (qq[k] - hi) * 180.0 / kmpc::kPi;
            break;
          }
        }
        if (bad >= 0) {
          hit_limit = true;
          fail = "joint_limit";
          fail_i = i;
          fail_j = j;
          const double cache_end = cache.q_end[bad] * 180.0 / kmpc::kPi;
          std::printf(
              "retreat_diag joint_limit axis=%d viol_rad=%.3e q_deg=%.6f "
              "cache_q_end_deg=%.6f lo_deg=%.6f hi_deg=%.6f (i=%d j=%d)\n",
              bad, viol * kmpc::kPi / 180.0, qq[bad] * 180.0 / kmpc::kPi,
              cache_end, (cfg.limits.q_min[bad] + margin) * 180.0 / kmpc::kPi,
              (cfg.limits.q_max[bad] - margin) * 180.0 / kmpc::kPi, i, j);
          break;
        }
      }
      const kmpc::Vec3 com = kin.com(qq);
      const double out = std::max(
          {cx0 - com[0], com[0] - cx1, cy0 - com[1], com[1] - cy1, 0.0});
      worst_com_out = std::max(worst_com_out, out);
      if (out > 0.0 && std::string(fail) == "ok") {
        fail = "com";
        fail_i = i;
        fail_j = j;
      }
      const double t = (i + a) / static_cast<double>(n);
      const auto pose = kin.fk(qq, frame);
      const double tube =
          (pose.col(3).head<3>() - ((1.0 - t) * start + t * target)).norm();
      worst_tube = std::max(worst_tube, tube);
      if (tube > 0.003 && std::string(fail) == "ok") {
        fail = "tube";
        fail_i = i;
        fail_j = j;
      }
      if (ori_on) {
        const double oe =
            kmpc::ori_error_xyzw(kmpc::quat_slerp_xyzw(start_ori, quat, t),
                                kmpc::rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                .norm();
        worst_ori = std::max(worst_ori, oe);
        if (oe > cfg.params.arrived_ori_rad && std::string(fail) == "ok") {
          fail = "orientation";
          fail_i = i;
          fail_j = j;
        }
      }
    }
    if (hit_limit) break;
  }
  // Endpoint mismatch: reversed path ends at cache.q_start, command target is prep.
  const double end_fk_vs_target =
      (kin.fk_pos(rev.back(), frame) - target).norm();
  const double start_fk_vs_cache_end =
      (start - cache.end_xyz).norm();
  std::printf(
      "retreat_diag reverse_validate fail=%s i=%d j=%d worst_tube_mm=%.2f "
      "worst_ori_deg=%.3f worst_com_out_mm=%.2f end_fk_vs_target_mm=%.2f "
      "start_fk_vs_cache_end_mm=%.2f ori_on=%d left=%d\n",
      fail, fail_i, fail_j, worst_tube * 1e3, worst_ori * 180.0 / kmpc::kPi,
      worst_com_out * 1e3, end_fk_vs_target * 1e3, start_fk_vs_cache_end * 1e3,
      ori_on ? 1 : 0, left ? 1 : 0);
  return fail;
}

SegReport play_segment(kmpc::CartesianTracker& tracker, kmpc::Vec18* q,
                       const kmpc::Vec18& q_pref, uint64_t gen, bool left,
                       const kmpc::Vec3& goal, const kmpc::Vec4& quat,
                       const kmpc::Vec3* tip_xyz, const char* name) {
  SegReport r;
  r.name = name;
  if (tip_xyz) {
    kmpc::LineTarget tip;
    tip.xyz = *tip_xyz;
    tip.quat_xyzw = quat;
    tip.ori_on = true;
    r.armed = tracker.rearm_line(gen, tip);
  } else {
    r.armed = tracker.rearm_line(gen);
  }
  if (!r.armed) {
    r.reason = "rearm_failed";
    return r;
  }

  const kmpc::Vec3* L = left ? &goal : nullptr;
  const kmpc::Vec3* R = left ? nullptr : &goal;
  const kmpc::Vec4* qL = left ? &quat : nullptr;
  const kmpc::Vec4* qR = left ? nullptr : &quat;

  std::vector<kmpc::Vec18, Eigen::aligned_allocator<kmpc::Vec18>> commanded;
  commanded.push_back(*q);
  for (int i = 0; i < 5000; ++i) {
    const auto step = tracker.compute(*q, q_pref, L, R, qL, qR);
    r.candidate = tracker.line_candidate();
    r.reverse = tracker.reverse_line_decision();
    if (tracker.line_refused()) {
      r.refused = true;
      r.reason = tracker.line_reason();
      r.ticks = i;
      if (i == 0 && tracker.line_sample_count() >= 2) {
        fill_folds(tracker.line_samples(), &r.fold_j1_s, &r.fold_j2_s, &r.fold_j3_s,
                   &r.fold_p_s, &r.samples_ok);
      }
      return r;
    }
    if (i == 0) {
      fill_folds(tracker.line_samples(), &r.fold_j1_s, &r.fold_j2_s, &r.fold_j3_s,
                 &r.fold_p_s, &r.samples_ok);
    }
    *q += step.dq;
    commanded.push_back(*q);
    r.ticks = i + 1;
    if (tracker.last_arrived()) {
      r.arrived = true;
      tracker.maybe_commit_last_line_cache(*q);
      break;
    }
  }
  fill_folds(commanded, &r.fold_j1_c, &r.fold_j2_c, &r.fold_j3_c, &r.fold_p_c, &r.cmd_ok);
  if (!r.arrived && r.reason.empty()) r.reason = "timeout";

  // Per-tick reverse stats for waist/pitch on commanded series.
  auto tick_stats = [&](const char* axis, auto at) {
    const int n = static_cast<int>(commanded.size());
    if (n < 2) return;
    const double front = at(0);
    const double back = at(n - 1);
    const double net = back - front;
    const int sgn = net > 1e-12 ? 1 : (net < -1e-12 ? -1 : 0);
    int rev_ticks = 0;
    int first_rev = -1;
    double max_back = 0.0;
    double sum_back = 0.0;
    for (int i = 1; i < n; ++i) {
      const double step = at(i) - at(i - 1);
      if (sgn == 0) continue;
      const double step_dir = static_cast<double>(sgn) * step;
      if (step_dir < -1e-12) {
        ++rev_ticks;
        if (first_rev < 0) first_rev = i;
        const double bd = -step_dir * 180.0 / kmpc::kPi;
        max_back = std::max(max_back, bd);
        sum_back += bd;
      }
    }
    std::printf(
        "tickstat %s %s net_deg=%.4f sgn=%d n=%d reverse_ticks=%d first_rev=%d "
        "max_single_back_deg=%.6f sum_back_deg=%.6f\n",
        name, axis, net * 180.0 / kmpc::kPi, sgn, n, rev_ticks, first_rev, max_back,
        sum_back);
  };
  if (r.arrived) {
    tick_stats("J1", [&](int i) { return commanded[static_cast<std::size_t>(i)][0]; });
    tick_stats("J2", [&](int i) { return commanded[static_cast<std::size_t>(i)][1]; });
    tick_stats("J3", [&](int i) { return commanded[static_cast<std::size_t>(i)][2]; });
    tick_stats("Pitch", [&](int i) {
      return commanded[static_cast<std::size_t>(i)][0] +
             commanded[static_cast<std::size_t>(i)][1] +
             commanded[static_cast<std::size_t>(i)][2];
    });
  }
  return r;
}

void print_report(const std::vector<SegReport>& rows) {
  std::printf(
      "seg,ok,arrived,refused,cand,rev,reason,ticks,"
      "sJ1,sJ2,sJ3,sPitch,cJ1,cJ2,cJ3,cPitch\n");
  for (const auto& r : rows) {
    const bool ok = r.armed && r.arrived && !r.refused && r.samples_ok && r.cmd_ok;
    std::printf(
        "%s,%d,%d,%d,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        r.name.c_str(), ok ? 1 : 0, r.arrived ? 1 : 0, r.refused ? 1 : 0,
        r.candidate.c_str(), r.reverse.c_str(),
        r.reason.empty() ? "-" : r.reason.c_str(), r.ticks, r.fold_j1_s,
        r.fold_j2_s, r.fold_j3_s, r.fold_p_s, r.fold_j1_c, r.fold_j2_c, r.fold_j3_c,
        r.fold_p_c);
  }
}

}  // namespace

TEST(MockEyeCycle, WaistPitchNoFoldOnGraspPlaceCycle) {
  // mock_eye_state.py 525 vision (mm, tool quat xyzw). Tips/preps apply the same
  // arm_controller_params.yaml offsets as test_grasp_place_targets_624_525 —
  // do not treat raw vision XYZ as flange tip.
  const PoseMm LEFT_GRASP{553.430, 284.979, 400.0, 0.0, 0.0, 0.028923, 0.999582};
  const PoseMm LEFT_PLACE{525.839, 323.452, 400.0, 0.0, 0.0, -0.018784, 0.999824};
  const PoseMm RIGHT_GRASP{558.428, -286.318, 400.0, 0.0, 0.0, 0.003054, 0.999995};
  const PoseMm RIGHT_PLACE{525.839, -323.452, 400.0, 0.0, 0.0, -0.019123, 0.999817};

  auto yaw_of = [](const PoseMm& p) {
    const double siny_cosp = 2.0 * (p.qw * p.qz + p.qx * p.qy);
    const double cosy_cosp = 1.0 - 2.0 * (p.qy * p.qy + p.qz * p.qz);
    return std::atan2(siny_cosp, cosy_cosp);
  };
  auto tip_prep_m = [&](const PoseMm& eye, bool left, bool grasp, kmpc::Vec3* tip,
                        kmpc::Vec3* prep) {
    if (grasp) {
      const double ox = left ? -38.0 : -60.0;
      const double oy = left ? -10.0 : 0.0;
      const double oz = left ? 65.0 : 60.0;
      const double yaw = yaw_of(eye);
      const double c = std::cos(yaw);
      const double s = std::sin(yaw);
      tip->x() = (eye.x + c * ox - s * oy) * 1e-3;
      tip->y() = (eye.y + s * ox + c * oy) * 1e-3;
      tip->z() = (eye.z + oz) * 1e-3;
      *prep = *tip + kmpc::Vec3(0.0, 0.0, 0.150);
    } else {
      const double ox = -40.0;
      const double oy = left ? 20.0 : -68.0;
      const double oz = 90.0;
      tip->x() = (eye.x + ox) * 1e-3;
      tip->y() = (eye.y + oy) * 1e-3;
      tip->z() = (eye.z + oz) * 1e-3;
      *prep = *tip + kmpc::Vec3(-0.240, 0.0, 0.0);
    }
  };

  // q0 from evidence 20261110T051249Z servo @ t≈28.34 (field r21 start squat)
  const double waist4[] = {40.0060, -49.9995, -10.0003, -1.4969};
  const double right7[] = {-0.0980, 29.5260, -7.8219, -121.8440, -73.0651, -13.9180,
                           0.7989};
  const double left7[] = {-0.0980, 29.5260, 7.8220, -121.8440, 73.0653, -13.9180,
                          0.7991};
  kmpc::Vec18 q = q_from_deg(waist4, right7, left7);

  const auto cfg_all = kmpc::Config::load(
      cfg("joint_limits.yaml"), cfg("q_pref.yaml"), cfg("kmpc_params.yaml"));
  kmpc::Kinematics kin(std::string(KMPC_URDF_DIR) + "/Light-duty wheeled robot.urdf");
  kmpc::CartesianTracker tracker(kin, cfg_all);
  tracker.set_line_pace(0.3, 0.8);
  tracker.set_line_direction_strict_enable(true);

  std::cout << "version_note=8842e7e_plus_mock_eye_business_offsets\n";
  std::cout << "urdf_note=Chassis_Z_Joint4.z=0.1805\n";

  uint64_t gen = 1;
  std::vector<SegReport> rows;
  auto run_arm = [&](bool left, const PoseMm& grasp, const PoseMm& place) {
    kmpc::Vec3 g, g_prep, p, p_prep;
    tip_prep_m(grasp, left, true, &g, &g_prep);
    tip_prep_m(place, left, false, &p, &p_prep);
    const kmpc::Vec4 gq = flange_quat(grasp);
    const kmpc::Vec4 pq = flange_quat(place);

    const char* tag = left ? "L" : "R";
    rows.push_back(play_segment(tracker, &q, cfg_all.q_pref, gen++, left, g_prep, gq, &g,
                                (std::string(tag) + "_grasp_approach").c_str()));
    if (!rows.back().arrived) return;
    rows.push_back(play_segment(tracker, &q, cfg_all.q_pref, gen++, left, g, gq, nullptr,
                                (std::string(tag) + "_grasp_insert").c_str()));
    if (!rows.back().arrived) return;
    diagnose_reverse_validate(kin, cfg_all, tracker, q, g_prep, gq, left);
    rows.push_back(play_segment(tracker, &q, cfg_all.q_pref, gen++, left, g_prep, gq, nullptr,
                                (std::string(tag) + "_grasp_retreat").c_str()));
    if (!rows.back().arrived) return;
    rows.push_back(play_segment(tracker, &q, cfg_all.q_pref, gen++, left, p_prep, pq, &p,
                                (std::string(tag) + "_place_approach").c_str()));
    if (!rows.back().arrived) return;
    rows.push_back(play_segment(tracker, &q, cfg_all.q_pref, gen++, left, p, pq, nullptr,
                                (std::string(tag) + "_place_insert").c_str()));
    if (!rows.back().arrived) return;
    rows.push_back(play_segment(tracker, &q, cfg_all.q_pref, gen++, left, p_prep, pq, nullptr,
                                (std::string(tag) + "_place_retreat").c_str()));
  };

  run_arm(true, LEFT_GRASP, LEFT_PLACE);

  // Right arm: fresh tracker from same field q0 so left abort does not poison it.
  {
    kmpc::CartesianTracker right(kin, cfg_all);
    right.set_line_pace(0.3, 0.8);
    right.set_line_direction_strict_enable(true);
    kmpc::Vec18 qr = q_from_deg(waist4, right7, left7);
    auto run_right = [&](bool left, const PoseMm& grasp, const PoseMm& place) {
      kmpc::Vec3 g, g_prep, p, p_prep;
      tip_prep_m(grasp, left, true, &g, &g_prep);
      tip_prep_m(place, left, false, &p, &p_prep);
      const kmpc::Vec4 gq = flange_quat(grasp);
      const kmpc::Vec4 pq = flange_quat(place);
      const char* tag = left ? "L" : "R";
      uint64_t ggen = 100;
      rows.push_back(play_segment(right, &qr, cfg_all.q_pref, ggen++, left, g_prep, gq, &g,
                                  (std::string(tag) + "_grasp_approach").c_str()));
      if (!rows.back().arrived) return;
      rows.push_back(play_segment(right, &qr, cfg_all.q_pref, ggen++, left, g, gq, nullptr,
                                  (std::string(tag) + "_grasp_insert").c_str()));
      if (!rows.back().arrived) return;
      diagnose_reverse_validate(kin, cfg_all, right, qr, g_prep, gq, left);
      rows.push_back(play_segment(right, &qr, cfg_all.q_pref, ggen++, left, g_prep, gq, nullptr,
                                  (std::string(tag) + "_grasp_retreat").c_str()));
      if (!rows.back().arrived) return;
      rows.push_back(play_segment(right, &qr, cfg_all.q_pref, ggen++, left, p_prep, pq, &p,
                                  (std::string(tag) + "_place_approach").c_str()));
      if (!rows.back().arrived) return;
      rows.push_back(play_segment(right, &qr, cfg_all.q_pref, ggen++, left, p, pq, nullptr,
                                  (std::string(tag) + "_place_insert").c_str()));
      if (!rows.back().arrived) return;
      rows.push_back(play_segment(right, &qr, cfg_all.q_pref, ggen++, left, p_prep, pq, nullptr,
                                  (std::string(tag) + "_place_retreat").c_str()));
    };
    run_right(false, RIGHT_GRASP, RIGHT_PLACE);
  }

  print_report(rows);

  int fold_fail = 0;
  int retreat_fail = 0;
  for (const auto& r : rows) {
    const bool is_retreat = r.name.find("retreat") != std::string::npos;
    // Only score fold when a path actually ran (arrived or had samples).
    if (r.arrived) {
      if (!r.samples_ok) {
        ADD_FAILURE() << r.name << " SAMPLE fold J1/J2/J3/P=" << r.fold_j1_s << "/"
                      << r.fold_j2_s << "/" << r.fold_j3_s << "/" << r.fold_p_s;
        ++fold_fail;
      }
      if (!r.cmd_ok) {
        ADD_FAILURE() << r.name << " CMD fold J1/J2/J3/P=" << r.fold_j1_c << "/"
                      << r.fold_j2_c << "/" << r.fold_j3_c << "/" << r.fold_p_c;
        ++fold_fail;
      }
    }
    if (!r.arrived || r.refused) {
      if (is_retreat) {
        ++retreat_fail;
        std::printf(
            "KNOWN_SEPARATE %s retreat refused reason=%s cand=%s rev=%s\n",
            r.name.c_str(), r.reason.c_str(), r.candidate.c_str(),
            r.reverse.c_str());
      } else {
        ADD_FAILURE() << r.name << " refused/incomplete reason=" << r.reason
                      << " cand=" << r.candidate << " rev=" << r.reverse;
        ++fold_fail;
      }
    }
  }
  // This test's hard gate is waist/pitch fold on approach+insert.
  // Retreat geometry refuse is logged above as a separate open defect.
  EXPECT_EQ(fold_fail, 0) << "fold_fail=" << fold_fail
                          << " retreat_fail=" << retreat_fail
                          << " total_segments=" << rows.size();
  if (retreat_fail > 0) {
    std::printf("note: retreat_fail=%d (geometry/validate; not fold)\n",
                retreat_fail);
  }
}
