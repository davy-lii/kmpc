// Line-only full-stage QP. Every predicted stage has its own 11-DOF correction;
// the generic 36-variable endpoint QP is deliberately untouched.
#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/line_contract.hpp"
#include "kmpc/path_progress.hpp"
#include "kmpc/tracker_math.hpp"  // kStartPitchAdmitRad
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <Eigen/SparseCore>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <proxsuite/proxqp/sparse/sparse.hpp>
#pragma GCC diagnostic pop

namespace kmpc {
namespace {

const char* proxqp_status_name(proxsuite::proxqp::QPSolverOutput status) {
  using S = proxsuite::proxqp::QPSolverOutput;
  switch (status) {
    case S::PROXQP_SOLVED:
      return "SOLVED";
    case S::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE:
      return "SOLVED_CLOSEST";
    case S::PROXQP_MAX_ITER_REACHED:
      return "MAX_ITER";
    case S::PROXQP_PRIMAL_INFEASIBLE:
      return "PRIMAL_INFEASIBLE";
    case S::PROXQP_DUAL_INFEASIBLE:
      return "DUAL_INFEASIBLE";
    case S::PROXQP_NOT_RUN:
      return "NOT_RUN";
    default:
      return "UNKNOWN";
  }
}

// Soft-line build (line_path kLineTubeHardM / kLineOriHardRad): mid knots may
// leave the chord; endpoint stays tight. QP used to keep a hard 3 mm / arrival
// ori tube everywhere, so Mode B DIRECT rejected fold_15 soft bows
// (qp_primal_residual / s*_tube_* / s*_ori_*).
constexpr double kQpTubeEndM = 0.003;
constexpr double kQpTubeMidM = 0.20;
constexpr double kQpOriMidRad = 0.40;
// Only the very last chord fraction uses the arrival tube. Soft-path joint
// lerps near the end still leave the start→target chord by more than 3 mm
// (LoggedPrep fold_15 hit qp_nonlinear_tube_or_com around along≈0.92–0.99).
constexpr double kQpTubeEndFrac = 0.999;
inline double qp_cross_tube_m(double along01) {
  return (along01 >= kQpTubeEndFrac) ? kQpTubeEndM : kQpTubeMidM;
}
inline double qp_ori_tol_rad(double along01, double arrived_ori_rad) {
  return (along01 >= kQpTubeEndFrac) ? arrived_ori_rad : kQpOriMidRad;
}

}  // namespace

bool CartesianTracker::take_line_qp_fail_dump(LineQpFailDump* out) {
  if (out == nullptr || !last_line_qp_fail_dump_.pending_write) {
    return false;
  }
  *out = last_line_qp_fail_dump_;
  last_line_qp_fail_dump_.pending_write = false;
  return out->valid;
}

bool CartesianTracker::line_path_qp(const Vec18& from, double next_prog,
                                     const Vec18& cap, Vec18& to) {
  std::vector<std::string> row_tags;
  row_tags.reserve(512);
  auto reject=[this,&row_tags](const std::string& reason) {
    line_.reason=reason;
    std::snprintf(last_line_qp_fail_.reason, sizeof(last_line_qp_fail_.reason), "%s",
                  reason.c_str());
    last_line_qp_fail_.valid = true;
    if (last_line_qp_fail_.worst_c >= 0 &&
        last_line_qp_fail_.worst_c < static_cast<int>(row_tags.size())) {
      std::snprintf(last_line_qp_fail_.worst_tag,
                    sizeof(last_line_qp_fail_.worst_tag), "%s",
                    row_tags[static_cast<size_t>(last_line_qp_fail_.worst_c)].c_str());
    }
    last_line_qp_fail_dump_.reason[0] = '\0';
    std::snprintf(last_line_qp_fail_dump_.reason,
                  sizeof(last_line_qp_fail_dump_.reason), "%s", reason.c_str());
    last_line_qp_fail_dump_.valid = true;
    last_line_qp_fail_dump_.pending_write = true;
    return false;
  };
  last_line_qp_fail_ = LineQpFailSnapshot{};
  last_line_qp_fail_.path_remain_m = last_line_path_remain_m_;
  last_line_qp_fail_.prog = next_prog;
  last_line_qp_fail_.pitch_deg = rad_to_deg(from.head<3>().sum());
  last_line_qp_fail_.ee_err_m =
      (line_.target - kin_.fk_pos(from, line_.frame)).norm();
  last_line_qp_solve_us_ = 0.0;
  last_line_qp_dx_norm_ = 0.0;
  last_line_qp_proj_max_ = 0.0;
  last_line_qp_nominal_ = Vec18::Zero();
  last_line_qp_pre_ = Vec18::Zero();

  constexpr int d=kNWaist+kNArm;
  const int n=std::clamp(n_horizon_,1,LineQpFailDump::kMaxStages), nx=n*d;
  const int arm=line_.left?kSliceLeft.begin:kSliceRight.begin;
  const double end=static_cast<double>(line_.samples.size()-1);
  const Vec3 p_start=kin_.fk_pos(line_.samples.front(),line_.frame);
  const Vec4 ori_start=rot_to_quat_xyzw(kin_.fk(line_.samples.front(),line_.frame).block<3,3>(0,0));
  Eigen::Matrix<double,18,d> T=Eigen::Matrix<double,18,d>::Zero();
  T.block<kNWaist,kNWaist>(0,0).setIdentity();
  T.block<kNArm,kNArm>(arm,kNWaist).setIdentity();
  using Triplet=Eigen::Triplet<double>;
  std::vector<Triplet> ht,ct;
  std::vector<double> lower,upper;
  Eigen::VectorXd g=Eigen::VectorXd::Zero(nx);
  std::vector<Vec18,Eigen::aligned_allocator<Vec18>> nominal;
  std::vector<double> progress;
  nominal.reserve(n); progress.reserve(n);
  auto tag_row=[&](const char* tag) {
    row_tags.emplace_back(tag);
  };
  auto row=[&](int k,const Eigen::Matrix<double,1,d>& a,double lo,double hi,
               const char* tag, bool previous=false) {
    const int r=static_cast<int>(lower.size()); lower.push_back(lo); upper.push_back(hi);
    tag_row(tag);
    for(int j=0;j<d;++j) if(a[j]!=0.0) {
      ct.emplace_back(r,k*d+j,a[j]);
      if(previous) ct.emplace_back(r,(k-1)*d+j,-a[j]);
    }
  };
  auto finish_dump=[&]() {
    LineQpFailDump& dump = last_line_qp_fail_dump_;
    dump = LineQpFailDump{};
    dump.valid = true;
    dump.pending_write = false;  // reject() sets pending after fill
    dump.truncated = false;
    dump.mono_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    dump.goal_generation = goal_generation_;
    dump.line_gen = line_gen_;
    // Early rejects may fire before every stage is assembled.
    const int n_filled = static_cast<int>(nominal.size());
    dump.n_horizon = n_filled > 0 ? n_filled : n;
    dump.nx = std::min(nx, dump.n_horizon * d);
    dump.prog = next_prog;
    dump.path_remain_m = last_line_path_remain_m_;
    dump.pitch_deg = last_line_qp_fail_.pitch_deg;
    dump.ee_err_m = last_line_qp_fail_.ee_err_m;
    dump.torso_pitch_max = torso_pitch_max_;
    dump.q_limit_margin = q_limit_margin_;
    dump.com_xmin = com_xmin_;
    dump.com_xmax = com_xmax_;
    dump.com_ymin = com_ymin_;
    dump.com_ymax = com_ymax_;
    dump.com_margin = com_margin_;
    dump.waist_locked = line_.waist_locked;
    dump.ori_on = line_.ori_on;
    dump.pitch_mono_enable = line_qp_pitch_monotonic_enable_;
    dump.direction_strict = line_direction_strict_enable_;
    for (int a = 0; a < 4; ++a) {
      dump.direction_sign[a] = line_.direction_contract.sign[a];
    }
    std::snprintf(dump.candidate, sizeof(dump.candidate), "%s",
                  line_.candidate.c_str());
    std::snprintf(dump.frame, sizeof(dump.frame), "%s", line_.frame.c_str());
    dump.from = from;
    dump.cap = cap;
    dump.target = line_.target;
    dump.quat = line_.quat;
    for (int k = 0; k < dump.n_horizon; ++k) {
      if (k < static_cast<int>(progress.size())) {
        dump.progress[k] = progress[static_cast<size_t>(k)];
      }
      if (k < static_cast<int>(nominal.size())) {
        for (int j = 0; j < 18; ++j) {
          dump.nominal[k][j] = nominal[static_cast<size_t>(k)][j];
        }
      }
    }
    for (int i = 0; i < dump.nx; ++i) {
      dump.g[i] = g[i];
    }
    const int nc_cap =
        std::min(static_cast<int>(lower.size()), LineQpFailDump::kMaxNc);
    dump.nc = nc_cap;
    if (static_cast<int>(lower.size()) > LineQpFailDump::kMaxNc) {
      dump.truncated = true;
    }
    for (int i = 0; i < nc_cap; ++i) {
      dump.lo[i] = lower[static_cast<size_t>(i)];
      dump.hi[i] = upper[static_cast<size_t>(i)];
      if (i < static_cast<int>(row_tags.size())) {
        std::snprintf(dump.tags[i], LineQpFailDump::kTagLen, "%s",
                      row_tags[static_cast<size_t>(i)].c_str());
      }
    }
    const int nt_cap = std::min(static_cast<int>(ct.size()),
                                LineQpFailDump::kMaxTriplets);
    dump.n_triplets = nt_cap;
    if (static_cast<int>(ct.size()) > LineQpFailDump::kMaxTriplets) {
      dump.truncated = true;
    }
    for (int i = 0; i < nt_cap; ++i) {
      dump.trip_r[i] = ct[static_cast<size_t>(i)].row();
      dump.trip_c[i] = ct[static_cast<size_t>(i)].col();
      dump.trip_v[i] = ct[static_cast<size_t>(i)].value();
    }
    dump.worst_c = last_line_qp_fail_.worst_c;
    dump.worst_violation = last_line_qp_fail_.worst_violation;
    dump.status = last_line_qp_fail_.status;
    dump.iter = last_line_qp_fail_.iter;
    dump.pri_res = last_line_qp_fail_.pri_res;
    dump.dua_res = last_line_qp_fail_.dua_res;
    dump.solve_ms = last_line_qp_fail_.solve_ms;
    std::snprintf(dump.status_name, sizeof(dump.status_name), "%s",
                  last_line_qp_fail_.status_name);
    if (dump.worst_c >= 0 && dump.worst_c < dump.nc) {
      std::snprintf(dump.worst_tag, sizeof(dump.worst_tag), "%s",
                    dump.tags[dump.worst_c]);
    }
  };
  const double pitch0=line_.samples.front().head<3>().sum();
  const double pitchN=line_.samples.back().head<3>().sum();
  // Live stream excess: NewUnified Mode B is x=0 and cannot pull pitch back.
  // While recovering, force ProxQP so waist nullspace can meet the shrinking box.
  const double from_pitch_ex =
      (torso_pitch_max_ > 0.0)
          ? std::max(0.0, std::abs(from.head<3>().sum()) - torso_pitch_max_)
          : 0.0;
  const bool pitch_recovering_from = from_pitch_ex > 1e-9;
  double pitch_recover_step = 0.0;
  for (int j = 0; j < 3; ++j) pitch_recover_step += std::abs(cap[j]);
  Vec18 prev=from;
  double s=next_prog;
  for(int k=0;k<n;++k) {
    if(k>0) {
      s=bounded_path_progress(s,end,prev,cap,[this](double p){return line_at(p);});
    }
    const Vec18 q=line_at(s); nominal.push_back(q); progress.push_back(s);
    const double t=s/std::max(end,1.0);
    const auto pose=kin_.fk(q,line_.frame);
    const Eigen::Matrix<double,6,d> J=kin_.jacobian_spatial(q,line_.frame)*T;
    const Vec3 ep=(1-t)*p_start+t*line_.target-pose.col(3).head<3>();
    const Vec3 eo=line_.ori_on?ori_error_xyzw(quat_slerp_xyzw(ori_start,line_.quat,t),
        rot_to_quat_xyzw(pose.block<3,3>(0,0))):Vec3::Zero();
    // Mode A: Cartesian reshape (legacy). Mode C: min deviation from nominal.
    // Mode B skips the ProxQP solve but still builds the same hard rows below.
    Eigen::Matrix<double,d,d> H=Eigen::Matrix<double,d,d>::Zero();
    Eigen::Matrix<double,d,1> grad=Eigen::Matrix<double,d,1>::Zero();
    if(line_qp_follow_mode_==LineQpFollowMode::C) {
      H.setIdentity();
      // Prefer staying on the planned joint path; no large Cartesian re-pose.
    } else {
      H=400.0*J.topRows<3>().transpose()*J.topRows<3>();
      grad=-400.0*J.topRows<3>().transpose()*ep;
      if(line_.ori_on) {
        H+=20.0*J.bottomRows<3>().transpose()*J.bottomRows<3>();
        grad-=20.0*J.bottomRows<3>().transpose()*eo;
      }
    }
    // Fixed nominal path regularizer; original line pitch/yaw/arm/wrist ratios.
    Eigen::Matrix<double,d,1> w;
    w << 0.04,0.04,0.04,0.16,0.0016,0.0016,0.0016,0.0016,0.0016,0.0144,0.0144;
    H.diagonal()+=w*100.0;
    for(int i=0;i<d;++i) {
      g[k*d+i]=2*grad[i];
      for(int j=0;j<d;++j) ht.emplace_back(k*d+i,k*d+j,2*H(i,j));
      // Smooth corrections across all stages, anchored to the stream reference.
      ht.emplace_back(k*d+i,k*d+i,0.2);
      if(k>0) {
        ht.emplace_back((k-1)*d+i,(k-1)*d+i,0.2);
        ht.emplace_back(k*d+i,(k-1)*d+i,-0.2);
        ht.emplace_back((k-1)*d+i,k*d+i,-0.2);
      }
      Eigen::Matrix<double,1,d> a=Eigen::Matrix<double,1,d>::Zero(); a[i]=1;
      const double trust=(line_.waist_locked && i<kNWaist)?0.0:0.015;
      char tag[LineQpFailDump::kTagLen];
      std::snprintf(tag, sizeof(tag), "s%d_trust_d%d", k, i);
      row(k,a,-trust,trust,tag);
    }
    for(int j=0;j<18;++j) {
      if(T.row(j).squaredNorm()==0) continue;
      char tag[LineQpFailDump::kTagLen];
      // Hard joint box only. Soft margin is a planner preference / growth gate
      // (line_validate), not a follow stop: r39 rise failed s0_qlim_j0 when
      // J1 sat ~0.003° past soft while moving *away* from the hard wall.
      std::snprintf(tag, sizeof(tag), "s%d_qlim_j%d", k, j);
      row(k,T.row(j),limits_.q_min[j]-q[j],limits_.q_max[j]-q[j],tag);
      std::snprintf(tag, sizeof(tag), "s%d_step_j%d", k, j);
      row(k,T.row(j),-cap[j]-(q[j]-prev[j]),cap[j]-(q[j]-prev[j]),tag,k>0);
    }
    // Box inscribed in the progress-dependent Euclidean tube (soft mid / 3 mm end).
    const double tube=qp_cross_tube_m(t)/1.7320508075688772;
    for(int j=0;j<3;++j) {
      char tag[LineQpFailDump::kTagLen];
      std::snprintf(tag, sizeof(tag), "s%d_tube_%c", k, "xyz"[j]);
      row(k,J.row(j),ep[j]-tube,ep[j]+tube,tag);
      if(line_.ori_on) {
        const double ori_box=qp_ori_tol_rad(t,arrived_ori_rad_)/1.7321;
        std::snprintf(tag, sizeof(tag), "s%d_ori_%c", k, "xyz"[j]);
        row(k,J.row(3+j),eo[j]-ori_box,eo[j]+ori_box,tag);
      }
    }
    const Vec3 com=kin_.com(q);
    const Eigen::Matrix<double,3,d> jc=kin_.jacobian_com(q)*T;
    {
      char tag[LineQpFailDump::kTagLen];
      std::snprintf(tag, sizeof(tag), "s%d_com_x", k);
      row(k,jc.row(0),com_xmin_+com_margin_-com[0],com_xmax_-com_margin_-com[0],tag);
      std::snprintf(tag, sizeof(tag), "s%d_com_y", k);
      row(k,jc.row(1),com_ymin_+com_margin_-com[1],com_ymax_-com_margin_-com[1],tag);
    }
    Eigen::Matrix<double,1,d> pitch=Eigen::Matrix<double,1,d>::Zero(); pitch.head<3>().setOnes();
    // Unified with line_build / 250 Hz stream_pitch_recovery (kStartPitchAdmitRad):
    // Active recovery (from_ex>0): shrink ceiling by waist rate; no nom_ex floor
    // (that froze r32). Non-recovery Mode B: keep nom_ex so x=0 stays feasible
    // when the seed sample sits slightly outside while live `from` is inside
    // (r33 run1 s0_pitch_box viol≈2e-5). Never exceed admit; never widen hard.
    const double limit=torso_pitch_max_;
    bool pitch_recovering=false;
    if(limit>0) {
      auto excess=[&](double p){ return std::max(0.0,std::abs(p)-limit); };
      const double p_from=from.head<3>().sum();
      const double from_ex=excess(p_from);
      const double start_ex=excess(pitch0);
      const double admit_excess=
          std::min(kStartPitchAdmitRad,std::max(start_ex,from_ex));
      double recover_step=0.0;
      for(int j=0;j<3;++j) recover_step+=std::abs(cap[j]);
      // Active recovery follows the live stream state, not only the sample start.
      pitch_recovering=from_ex>1e-9;
      // Ceiling shrinks with stage index; stage 0 keeps a one-step reachability
      // floor so field-B's ~1e-5 rad shortfall stays feasible.
      const double staged=
          std::max(0.0,from_ex-static_cast<double>(k)*recover_step);
      const double first_floor=
          (k==0)?std::max(0.0,from_ex-recover_step):0.0;
      // nom_ex only when NOT recovering — Mode B x=0 needs the sample's own
      // crumb; recovering ProxQP must still drive toward the hard box.
      const double nom_ex=pitch_recovering?0.0:excess(q.head<3>().sum());
      const double allowed_ex=
          std::min(admit_excess,std::max({staged,first_floor,nom_ex}));
      const double soft=limit+allowed_ex;
      char tag[LineQpFailDump::kTagLen];
      std::snprintf(tag, sizeof(tag), "s%d_pitch_box", k);
      row(k,pitch,-soft-q.head<3>().sum(),soft-q.head<3>().sum(),tag);
    }
    // Pitch-sum and waist axes share the accepted DirectionContract. Internal
    // slack is kDirectionSolverSlackRad (strictly < final-send fold_max) so QP
    // cannot mint a step that final_send_pitch_frozen must reject. Never re-judge
    // sign from clamped path endpoints — that disagreed with the contract.
    if (line_qp_pitch_monotonic_enable_ && !pitch_recovering) {
      const double kPitchSlack = kDirectionSolverSlackRad;
      const double p_prev=prev.head<3>().sum();
      const double p_q=q.head<3>().sum();
      const double p_from=from.head<3>().sum();
      // Prefer the committed contract sign; fall back to raw path net only when
      // the contract was not accepted (legacy / strict-off path).
      int psgn = 0;
      if (line_direction_strict_enable_ && line_.direction_check.ok) {
        psgn = line_.direction_contract
                   .sign[static_cast<int>(DirectionAxisId::PitchSum)];
      } else {
        psgn = direction_sign_from_net(pitchN - pitch0, kDirectionZeroNetEpsRad);
      }
      double lo=-1e9, hi=1e9;
      char tag[LineQpFailDump::kTagLen];
      std::snprintf(tag, sizeof(tag), "s%d_pitch_mono", k);
      if (psgn == 0) {
        // Frozen pitch: hold every stage to the live stream pitch (from), not a
        // sliding nominal-prev band. Adjacent-only freeze can stack slack past
        // the final-send envelope. Band stays inside kDirectionStrictFoldMaxRad.
        lo = p_from - kPitchSlack - p_q;
        hi = p_from + kPitchSlack - p_q;
        if (lo <= hi) {
          row(k, pitch, lo, hi, tag, false);
          ++line_qp_pitch_monotonic_hits_;
        } else {
          finish_dump();
          return reject("qp_pitch_monotonic_infeasible");
        }
      } else if (k == 0) {
        const double dp0 = p_q - p_from;
        if (psgn > 0) {
          lo = -dp0 - kPitchSlack;
        } else {
          hi = -dp0 + kPitchSlack;
        }
        const double p_goal =
            (limit > 0) ? std::clamp(pitchN, -limit, limit) : pitchN;
        lo = std::max(lo, std::min(p_from, p_goal) - kPitchSlack - p_q);
        hi = std::min(hi, std::max(p_from, p_goal) + kPitchSlack - p_q);
        if (lo < hi) {
          row(k, pitch, lo, hi, tag, false);
          ++line_qp_pitch_monotonic_hits_;
        } else {
          finish_dump();
          return reject("qp_pitch_monotonic_infeasible");
        }
      } else {
        const double dp = p_q - p_prev;
        if (psgn > 0) {
          lo = -dp - kPitchSlack;
        } else {
          hi = -dp + kPitchSlack;
        }
        const double p_goal =
            (limit > 0) ? std::clamp(pitchN, -limit, limit) : pitchN;
        lo = std::max(lo, std::min(p_prev, p_goal) - kPitchSlack - p_q);
        hi = std::min(hi, std::max(p_prev, p_goal) + kPitchSlack - p_q);
        if (lo < hi) {
          row(k, pitch, lo, hi, tag, /*previous=*/true);
          ++line_qp_pitch_monotonic_hits_;
        } else {
          finish_dump();
          return reject("qp_pitch_monotonic_infeasible");
        }
      }
    }

    // Strict waist-axis adjacent direction from the accepted line contract.
    // Zero slack on signed axes; frozen axes hold the live command chain.
    if (line_direction_strict_enable_ && line_.direction_check.ok && !pitch_recovering) {
      constexpr double kAxisSlack = 0.0;
      for(int a=0;a<3;++a) {
        const int sgn=line_.direction_contract.sign[a];
        Eigen::Matrix<double,1,d> axis=Eigen::Matrix<double,1,d>::Zero();
        axis[a]=1.0;
        char tag[LineQpFailDump::kTagLen];
        std::snprintf(tag, sizeof(tag), "s%d_waist_j%d", k, a + 1);
        if(sgn==0) {
          // Frozen net axis: hold at the live command chain (from/prev), not
          // the path-contract anchor. Anchoring to the contract after tracking
          // drift made insert-stage QP primal-infeasible.
          if (k == 0) {
            row(k, axis, from[a] - q[a], from[a] - q[a], tag, false);
          } else {
            row(k, axis, prev[a] - q[a], prev[a] - q[a], tag, /*previous=*/true);
          }
          continue;
        }
        if(k==0) {
          const double step0=q[a]-from[a];
          double lo=-1e9, hi=1e9;
          if(sgn>0) lo=-step0-kAxisSlack;
          else hi=-step0+kAxisSlack;
          if(lo<=hi) row(k,axis,lo,hi,tag,false);
        } else {
          const double step=q[a]-prev[a];
          double lo=-1e9, hi=1e9;
          if(sgn>0) lo=-step-kAxisSlack;
          else hi=-step+kAxisSlack;
          if(lo<=hi) row(k,axis,lo,hi,tag,/*previous=*/true);
        }
      }
    }
    if(yaw_max_>0) {
      Eigen::Matrix<double,1,d> yaw=Eigen::Matrix<double,1,d>::Zero();yaw[kIdxYaw]=1;
      char tag[LineQpFailDump::kTagLen];
      std::snprintf(tag, sizeof(tag), "s%d_yaw_box", k);
      row(k,yaw,q_yaw_pref_-yaw_max_-q[kIdxYaw],q_yaw_pref_+yaw_max_-q[kIdxYaw],tag);
    }
    prev=q;
  }
  const int nc=static_cast<int>(lower.size());
  Eigen::SparseMatrix<double> H(nx,nx),C(nc,nx);
  H.setFromTriplets(ht.begin(),ht.end()); C.setFromTriplets(ct.begin(),ct.end());
  const Eigen::Map<Eigen::VectorXd> lo(lower.data(),nc),hi(upper.data(),nc);
  auto fill_worst=[&](const Eigen::VectorXd& x_cand) {
    if (x_cand.size() != nx || nc <= 0) {
      return;
    }
    const Eigen::VectorXd cx = C * x_cand;
    double worst = 0.0;
    int wi = -1;
    for (int i = 0; i < nc; ++i) {
      const double v = std::max(lo[i] - cx[i], cx[i] - hi[i]);
      if (v > worst) {
        worst = v;
        wi = i;
      }
    }
    last_line_qp_fail_.worst_c = wi;
    last_line_qp_fail_.worst_violation = worst;
  };
  if((lo.array()>hi.array()).any()) {
    double worst = 0.0;
    int wi = -1;
    for (int i = 0; i < nc; ++i) {
      const double v = lo[i] - hi[i];
      if (v > worst) {
        worst = v;
        wi = i;
      }
    }
    last_line_qp_fail_.worst_c = wi;
    last_line_qp_fail_.worst_violation = worst;
    finish_dump();
    return reject("qp_bounds");
  }
  Eigen::VectorXd x=Eigen::VectorXd::Zero(nx);
  if(line_qp_follow_mode_==LineQpFollowMode::B && !pitch_recovering_from) {
    // Direct nominal path: skip Cartesian reshape, still enforce C*x∈[lo,hi].
    // Recovery must not take this shortcut — x=0 cannot shrink pitch excess.
    ++line_qp_solves_;
    last_line_qp_fail_.solve_ms = 0.0;
    last_line_qp_solve_us_ = 0.0;
    last_line_qp_fail_.status = 0;
    std::snprintf(last_line_qp_fail_.status_name,
                  sizeof(last_line_qp_fail_.status_name), "DIRECT");
  } else {
    try {
      proxsuite::proxqp::sparse::QP<double,int> qp(nx,0,nc);
      qp.settings.eps_abs=1e-7; qp.settings.eps_rel=0.0;
      // ProxSuite 0.7.x can prematurely certify an infeasible primal problem
      // with its default certificate tolerance (r29 feasible constraint dump).
      // Reduce certificate stationarity error; keep hard rows/primal checks.
      qp.settings.eps_primal_inf=1e-9;
      qp.settings.max_iter=line_qp_max_iter_;
      qp.settings.verbose=false;
      qp.init(H,g,proxsuite::nullopt,proxsuite::nullopt,C,lo,hi);
      ++line_qp_solves_;
      const auto t0 = std::chrono::steady_clock::now();
      qp.solve();
      const auto t1 = std::chrono::steady_clock::now();
      last_line_qp_fail_.solve_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      last_line_qp_solve_us_ = last_line_qp_fail_.solve_ms * 1000.0;
      last_line_qp_fail_.status = static_cast<int>(qp.results.info.status);
      std::snprintf(last_line_qp_fail_.status_name,
                    sizeof(last_line_qp_fail_.status_name), "%s",
                    proxqp_status_name(qp.results.info.status));
      last_line_qp_fail_.iter = static_cast<int>(qp.results.info.iter);
      last_line_qp_fail_.pri_res = static_cast<double>(qp.results.info.pri_res);
      last_line_qp_fail_.dua_res = static_cast<double>(qp.results.info.dua_res);
      fill_worst(qp.results.x);
      if(qp.results.info.status!=proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
        finish_dump();
        return reject("qp_status_"+std::to_string(static_cast<int>(qp.results.info.status)));
      }
      x=qp.results.x;
    } catch(...) {
      finish_dump();
      return reject("qp_exception");
    }
    if(!x.allFinite()) {
      finish_dump();
      return reject("qp_nonfinite");
    }
  }
  // Hard constraint matrix check for A/B/C — B must not skip this.
  if(line_.waist_locked) for(int k=0;k<n;++k) x.segment(k*d,kNWaist).setZero();
  {
    const Eigen::VectorXd cx=C*x;
    if((cx.array()<lo.array()-2e-7).any() || (cx.array()>hi.array()+2e-7).any()) {
      fill_worst(x);
      finish_dump();
      std::ostringstream os;
      os << std::setprecision(8) << "qp_primal_residual v="
         << last_line_qp_fail_.worst_violation << " "
         << (last_line_qp_fail_.worst_c >= 0 &&
                     last_line_qp_fail_.worst_c < static_cast<int>(row_tags.size())
                 ? row_tags[static_cast<std::size_t>(last_line_qp_fail_.worst_c)]
                 : "?");
      return reject(os.str());
    }
  }
  last_line_qp_dx_norm_ = x.norm();
  if(!nominal.empty()) {
    last_line_qp_nominal_ = nominal.front();
  }
  prev=from;
  // The first interval starts at the actual stream reference, not the old endpoint.
  double previous_t=(kin_.fk_pos(from,line_.frame)-p_start).dot(line_.target-p_start)/
      std::max((line_.target-p_start).squaredNorm(),1e-12);
  previous_t=std::clamp(previous_t,0.0,1.0);
  for(int k=0;k<n;++k) {
    const Vec18 q=nominal[k]+T*x.segment(k*d,d);
    const double t=progress[k]/std::max(end,1.0);
    if(((q-prev).cwiseAbs().array()>cap.array()+3e-7).any()) {
      finish_dump();
      return reject("qp_velocity");
    }
    for(int j=1;j<=5;++j) {
      const double a=j/5.0;
      const Vec18 mid=(1-a)*prev+a*q;
      const auto pose=kin_.fk(mid,line_.frame);
      const Vec3 ab=line_.target-p_start;
      const double along=std::clamp((pose.col(3).head<3>()-p_start).dot(ab)/std::max(ab.squaredNorm(),1e-12),0.0,1.0);
      if((pose.col(3).head<3>()-(p_start+along*ab)).norm()>qp_cross_tube_m(along) ||
          !guards_.com_xy_inside(kin_.com(mid))) {
        finish_dump();
        return reject("qp_nonlinear_tube_or_com");
      }
      if(line_.ori_on && ori_error_xyzw(quat_slerp_xyzw(ori_start,line_.quat,(1-a)*previous_t+a*t),
          rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm()>
              qp_ori_tol_rad(along,arrived_ori_rad_)+1e-6) {
        finish_dump();
        return reject("qp_nonlinear_orientation");
      }
    }
    if (k == 0) {
      last_line_qp_pre_ = q;
      to = q;
      // Final emit projection: QP half-spaces stop intentional reverse, but
      // float crumbs and pitch-sum coupling can still wiggle. Enforce the live
      // command monotone half-spaces on stage-0 before it leaves the tracker.
      // Direction and pitch are projected jointly — never per-axis hard-clamp
      // alone, which can break the pitch/direction contract.
      const int psgn =
          line_.direction_contract
              .sign[static_cast<int>(DirectionAxisId::PitchSum)];
      const double p_from = from[0] + from[1] + from[2];
      auto apply_pitch_need = [&](double need) {
        // Prefer frozen waist axes (they can absorb pitch without breaking
        // per-axis mono). Then only move a signed axis in its allowed dir.
        for (int a = 0; a < 3 && std::abs(need) > 1e-15; ++a) {
          if (line_.direction_contract.sign[a] == 0) {
            to[a] += need;
            need = 0.0;
          }
        }
        for (int a = 0; a < 3 && std::abs(need) > 1e-15; ++a) {
          const int s = line_.direction_contract.sign[a];
          if (s == 0) {
            continue;
          }
          if (need * static_cast<double>(s) > 0.0) {
            to[a] += need;
            need = 0.0;
          }
        }
      };
      auto reclamp_axes = [&]() {
        for (int a = 0; a < 3; ++a) {
          const int sgn = line_.direction_contract.sign[a];
          if (sgn > 0) {
            to[a] = std::max(to[a], from[a]);
          } else if (sgn < 0) {
            to[a] = std::min(to[a], from[a]);
          } else {
            to[a] = from[a];
          }
        }
      };
      // Recovery may need to move pitch against the path contract sign (hold
      // wall-hug: path wants more fold, recovery wants back into ±hard).
      auto apply_pitch_need_recover = [&](double need) {
        for (int ax : {1, 0, 2}) {
          if (std::abs(need) <= 1e-15) break;
          const double lo = limits_.q_min[ax];
          const double hi = limits_.q_max[ax];
          const double room = need > 0.0 ? (hi - to[ax]) : (lo - to[ax]);
          const double capped =
              need > 0.0 ? std::min(need, std::min(room, from[ax] + cap[ax] - to[ax]))
                         : std::max(need, std::max(room, from[ax] - cap[ax] - to[ax]));
          to[ax] += capped;
          need -= capped;
        }
      };
      // Direction freeze must not cancel pitch recovery: wall-hug hold paths
      // often have psgn==0, and pinning to `from` made stream_pitch_recovery
      // timeout deterministic (r32). While recovering, only shrink |pitch|.
      // Certified unified paths already chose the joint motion, including a
      // single smooth reversal. Do not force the net-sign monotone clamp.
      if (line_direction_strict_enable_ && line_.direction_check.ok &&
          !pitch_recovering_from && line_.candidate != "unified_multistage") {
        reclamp_axes();
        double p_to = to[0] + to[1] + to[2];
        if (psgn == 0) {
          apply_pitch_need(p_from - p_to);
        } else if (psgn > 0 && p_to < p_from) {
          apply_pitch_need(p_from - p_to);
        } else if (psgn < 0 && p_to > p_from) {
          apply_pitch_need(p_from - p_to);
        }
        reclamp_axes();
        p_to = to[0] + to[1] + to[2];
        if (psgn == 0 &&
            std::abs(p_to - p_from) > kDirectionStrictFoldMaxRad +
                                          kDirectionNumericEpsRad) {
          finish_dump();
          return reject("qp_pitch_frozen_projection");
        }
      }
      // Joint pitch-box projection after direction repair. Ordinary segments
      // target the hard box. Recovery shrinks by one waist-rate step toward
      // the hard box (never grow; never keep the old no-pull envelope forever).
      if (torso_pitch_max_ > 0.0) {
        const double start_abs = std::abs(pitch0);
        const double from_abs = std::abs(p_from);
        const double bound =
            pitch_recovering_from
                ? (torso_pitch_max_ +
                   std::max(0.0, from_pitch_ex - pitch_recover_step) +
                   kPitchRoundEpsRad)
                : pitch_abs_bound(torso_pitch_max_,
                                  std::max(start_abs, from_abs),
                                  /*for_gate=*/true);
        double p_to = to[0] + to[1] + to[2];
        if (std::abs(p_to) > bound) {
          const double target_p =
              std::copysign(bound, p_to == 0.0 ? p_from : p_to);
          if (pitch_recovering_from) {
            apply_pitch_need_recover(target_p - p_to);
          } else {
            apply_pitch_need(target_p - p_to);
          }
          if (line_direction_strict_enable_ && line_.direction_check.ok &&
              !pitch_recovering_from) {
            reclamp_axes();
            p_to = to[0] + to[1] + to[2];
            if (psgn == 0) {
              apply_pitch_need(p_from - p_to);
            } else if (psgn > 0 && p_to < p_from) {
              apply_pitch_need(p_from - p_to);
            } else if (psgn < 0 && p_to > p_from) {
              apply_pitch_need(p_from - p_to);
            }
            reclamp_axes();
          }
          p_to = to[0] + to[1] + to[2];
          if (std::abs(p_to) > bound + kPitchRoundEpsRad) {
            finish_dump();
            return reject("qp_pitch_box_projection");
          }
        }
      }
      // Pitch redistribution can inflate a single waist step past cap — scale
      // the whole delta back into the velocity box, then re-check pitch/dir.
      {
        Vec18 dq = to - from;
        double scale = 1.0;
        for (int j = 0; j < 18; ++j) {
          if (cap[j] > 1e-12 && std::abs(dq[j]) > cap[j] + 3e-7) {
            scale = std::min(scale, cap[j] / std::abs(dq[j]));
          }
        }
        if (scale < 1.0 - 1e-12) {
          to = from + scale * dq;
          if (line_direction_strict_enable_ && line_.direction_check.ok &&
              !pitch_recovering_from) {
            reclamp_axes();
            double p_scaled = to[0] + to[1] + to[2];
            if (psgn == 0) {
              apply_pitch_need(p_from - p_scaled);
            } else if (psgn > 0 && p_scaled < p_from) {
              apply_pitch_need(p_from - p_scaled);
            } else if (psgn < 0 && p_scaled > p_from) {
              apply_pitch_need(p_from - p_scaled);
            }
            reclamp_axes();
          }
          if (torso_pitch_max_ > 0.0) {
            const double start_abs = std::abs(pitch0);
            const double from_abs = std::abs(p_from);
            const double bound =
                pitch_recovering_from
                    ? (torso_pitch_max_ +
                       std::max(0.0, from_pitch_ex - pitch_recover_step) +
                       kPitchRoundEpsRad)
                    : pitch_abs_bound(torso_pitch_max_,
                                      std::max(start_abs, from_abs),
                                      /*for_gate=*/true);
            const double p_scaled = to[0] + to[1] + to[2];
            if (std::abs(p_scaled) > bound + kPitchRoundEpsRad) {
              finish_dump();
              return reject("qp_pitch_box_projection");
            }
          }
        }
        if (((to - from).cwiseAbs().array() > cap.array() + 3e-7).any()) {
          finish_dump();
          return reject("qp_projection_velocity");
        }
      }
      for (int j = 0; j < 18; ++j) {
        if (to[j] < limits_.q_min[j] - 1e-7 ||
            to[j] > limits_.q_max[j] + 1e-7) {
          finish_dump();
          return reject("qp_projection_joint_limit");
        }
      }
      {
        const auto pose = kin_.fk(to, line_.frame);
        const Vec3 ab = line_.target - p_start;
        const double along = std::clamp(
            (pose.col(3).head<3>() - p_start).dot(ab) /
                std::max(ab.squaredNorm(), 1e-12),
            0.0, 1.0);
        if ((pose.col(3).head<3>() - (p_start + along * ab)).norm() >
                qp_cross_tube_m(along) ||
            !guards_.com_xy_inside(kin_.com(to))) {
          finish_dump();
          return reject("qp_projection_tube_or_com");
        }
      }
      // Recovery may move frozen waist axes / pitch toward the hard box; the
      // QP already skipped direction rows while pitch_recovering.
      if (line_direction_strict_enable_ && line_.direction_check.ok &&
          !pitch_recovering_from && line_.candidate != "unified_multistage") {
        for (int a = 0; a < 3; ++a) {
          const int sgn = line_.direction_contract.sign[a];
          const double dq = to[a] - from[a];
          if (sgn > 0 && dq < -kDirectionNumericEpsRad) {
            finish_dump();
            return reject("qp_projection_direction");
          }
          if (sgn < 0 && dq > kDirectionNumericEpsRad) {
            finish_dump();
            return reject("qp_projection_direction");
          }
          if (sgn == 0 && std::abs(dq) > kDirectionNumericEpsRad) {
            finish_dump();
            return reject("qp_projection_direction");
          }
        }
        const double p_to = to[0] + to[1] + to[2];
        if (psgn > 0 && p_to + kDirectionNumericEpsRad < p_from) {
          finish_dump();
          return reject("qp_projection_direction");
        }
        if (psgn < 0 && p_to - kDirectionNumericEpsRad > p_from) {
          finish_dump();
          return reject("qp_projection_direction");
        }
        if (psgn == 0 &&
            std::abs(p_to - p_from) >
                kDirectionStrictFoldMaxRad + kDirectionNumericEpsRad) {
          finish_dump();
          return reject("qp_projection_direction");
        }
      }
      // Full first-stage chord after projection (not endpoint-only).
      for (int j = 1; j <= 5; ++j) {
        const double a = j / 5.0;
        const Vec18 mid = (1 - a) * from + a * to;
        const auto pose = kin_.fk(mid, line_.frame);
        const Vec3 ab = line_.target - p_start;
        const double along = std::clamp(
            (pose.col(3).head<3>() - p_start).dot(ab) /
                std::max(ab.squaredNorm(), 1e-12),
            0.0, 1.0);
        if ((pose.col(3).head<3>() - (p_start + along * ab)).norm() >
                qp_cross_tube_m(along) ||
            !guards_.com_xy_inside(kin_.com(mid))) {
          finish_dump();
          return reject("qp_projection_chord_tube_or_com");
        }
        if (line_.ori_on &&
            ori_error_xyzw(quat_slerp_xyzw(ori_start, line_.quat,
                                           (1 - a) * previous_t + a * t),
                           rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                    .norm() > qp_ori_tol_rad(along, arrived_ori_rad_) + 1e-6) {
          finish_dump();
          return reject("qp_projection_chord_orientation");
        }
      }
    }
    // Subsequent prediction must continue from the actual first-stage result
    // (projected), not the pre-projection QP state.
    if (k == 0) {
      prev = to;
      previous_t = t;
      if ((to - q).cwiseAbs().maxCoeff() > 1e-6) {
        // Rebuild remaining horizon from the projected first stage along the
        // nominal path (not the stale ProxQP x for k>0). Re-anchor onto the
        // path from the accepted `to` — comparing against progress[0] (the
        // pre-projection nominal) falsely rejects feasible projected steps.
        auto at = [this](double p) { return line_at(p); };
        auto chord_ok = [&](const Vec18& a, const Vec18& b, double ta,
                            double tb, const char* tube_reason,
                            const char* ori_reason) -> bool {
          for (int j = 1; j <= 5; ++j) {
            const double alpha = j / 5.0;
            const Vec18 mid = (1.0 - alpha) * a + alpha * b;
            const auto pose = kin_.fk(mid, line_.frame);
            const Vec3 ab = line_.target - p_start;
            const double along = std::clamp(
                (pose.col(3).head<3>() - p_start).dot(ab) /
                    std::max(ab.squaredNorm(), 1e-12),
                0.0, 1.0);
            if ((pose.col(3).head<3>() - (p_start + along * ab)).norm() >
                    qp_cross_tube_m(along) ||
                !guards_.com_xy_inside(kin_.com(mid))) {
              finish_dump();
              reject(tube_reason);
              return false;
            }
            if (line_.ori_on &&
                ori_error_xyzw(
                    quat_slerp_xyzw(ori_start, line_.quat,
                                    (1.0 - alpha) * ta + alpha * tb),
                    rot_to_quat_xyzw(pose.block<3, 3>(0, 0)))
                        .norm() >
                    qp_ori_tol_rad(along, arrived_ori_rad_) + 1e-6) {
              finish_dump();
              reject(ori_reason);
              return false;
            }
          }
          return true;
        };
        auto fill_postproj_diag = [&](int kk, const Vec18& pred,
                                      const Vec18& qk, double pred_s) {
          int worst_j = 0;
          double worst_ratio = -1.0;
          for (int j = 0; j < 18; ++j) {
            const double dq = std::abs(qk[j] - pred[j]);
            const double c = std::max(cap[j], 1e-12);
            const double ratio = dq / c;
            if (ratio > worst_ratio) {
              worst_ratio = ratio;
              worst_j = j;
            }
          }
          last_line_qp_fail_.worst_c = worst_j;
          last_line_qp_fail_.worst_violation =
              std::abs(qk[worst_j] - pred[worst_j]) - cap[worst_j];
          std::snprintf(last_line_qp_fail_.worst_tag,
                        sizeof(last_line_qp_fail_.worst_tag),
                        "kk%d_j%d", kk, worst_j);
          last_line_qp_fail_.prog = pred_s;
        };

        Vec18 pred = to;
        const double search_lo = std::max(0.0, progress[0] - 3.0);
        const double search_hi =
            std::min(end, progress[0] + static_cast<double>(std::max(n, 1)));
        // recover still reports the nearest sample even when residual>tol;
        // residual/ambiguous/cross_knot must bridge, not hard-stop (r27 field).
        const auto recovered =
            recover_path_progress(search_lo, search_hi, to, at);
        const bool nearest_usable =
            recovered.ok ||
            std::strncmp(recovered.reason, "residual_", 9) == 0 ||
            std::strncmp(recovered.reason, "zero_span_residual", 18) == 0 ||
            std::strcmp(recovered.reason, "ambiguous_minima") == 0 ||
            std::strcmp(recovered.reason, "cross_knot_uncertain") == 0;
        if (!nearest_usable) {
          std::snprintf(last_line_qp_fail_.worst_tag,
                        sizeof(last_line_qp_fail_.worst_tag), "%.27s",
                        recovered.reason);
          finish_dump();
          return reject("qp_postproj_horizon_anchor");
        }
        double pred_s = recovered.prog;
        // Prefer a path sample already inside the velocity box from `to`.
        bool horizon_done = false;
        if (const auto on_path =
                find_progress_within_cap(search_lo, search_hi, to, cap, at)) {
          pred_s = *on_path;
        } else {
          // Off-path after projection: bridge toward nearest nominal sample
          // even when recover residual exceeded the 15 mrad map tolerance.
          const double goal_s = recovered.prog;
          const Vec18 q_goal = at(goal_s);
          const int bridge = count_cap_substeps(pred, q_goal, cap);
          if (bridge < 0 || bridge > n - 1) {
            fill_postproj_diag(1, pred, q_goal, goal_s);
            finish_dump();
            return reject(bridge < 0 ? "qp_postproj_horizon_bridge"
                                     : "qp_postproj_horizon_velocity");
          }
          if (bridge > 0) {
            Vec18 cursor = pred;
            const double bridge_t = goal_s / std::max(end, 1.0);
            double cursor_t = bridge_t;
            for (int s = 1; s <= bridge; ++s) {
              const double alpha =
                  static_cast<double>(s) / static_cast<double>(bridge);
              const Vec18 step = pred + alpha * (q_goal - pred);
              if (!within_velocity_cap(cursor, step, cap)) {
                fill_postproj_diag(s, cursor, step, goal_s);
                finish_dump();
                return reject("qp_postproj_horizon_velocity");
              }
              if (!chord_ok(cursor, step, cursor_t, bridge_t,
                            "qp_postproj_horizon_tube_or_com",
                            "qp_postproj_horizon_orientation")) {
                return false;
              }
              for (int j = 0; j < 18; ++j) {
                if (step[j] < limits_.q_min[j] - 1e-7 ||
                    step[j] > limits_.q_max[j] + 1e-7) {
                  fill_postproj_diag(s, cursor, step, goal_s);
                  finish_dump();
                  return reject("qp_postproj_horizon_bridge");
                }
              }
              cursor = step;
              cursor_t = bridge_t;
            }
            pred = q_goal;
            pred_s = goal_s;
            const int stages_left = n - 1 - bridge;
            if (stages_left <= 0) {
              horizon_done = true;
            } else {
              double pred_t = pred_s / std::max(end, 1.0);
              for (int kk = 0; kk < stages_left; ++kk) {
                pred_s = bounded_path_progress(pred_s, end, pred, cap, at);
                const Vec18 qk = at(pred_s);
                if (!within_velocity_cap(pred, qk, cap)) {
                  fill_postproj_diag(bridge + 1 + kk, pred, qk, pred_s);
                  finish_dump();
                  return reject("qp_postproj_horizon_velocity");
                }
                const double tk = pred_s / std::max(end, 1.0);
                if (!chord_ok(pred, qk, pred_t, tk,
                              "qp_postproj_horizon_tube_or_com",
                              "qp_postproj_horizon_orientation")) {
                  return false;
                }
                pred = qk;
                pred_t = tk;
              }
              horizon_done = true;
            }
          } else {
            pred_s = goal_s;
          }
        }
        if (horizon_done) {
          break;
        }
        double pred_t = pred_s / std::max(end, 1.0);
        for (int kk = 1; kk < n; ++kk) {
          pred_s = bounded_path_progress(pred_s, end, pred, cap, at);
          const Vec18 qk = at(pred_s);
          if (!within_velocity_cap(pred, qk, cap)) {
            // One more catch attempt: advance to a reachable path sample.
            const auto catch_s = find_progress_within_cap(
                pred_s, std::min(end, pred_s + static_cast<double>(n - kk)),
                pred, cap, at);
            if (!catch_s) {
              fill_postproj_diag(kk, pred, qk, pred_s);
              finish_dump();
              return reject("qp_postproj_horizon_velocity");
            }
            const Vec18 q_catch = at(*catch_s);
            const double tk = *catch_s / std::max(end, 1.0);
            if (!chord_ok(pred, q_catch, pred_t, tk,
                          "qp_postproj_horizon_tube_or_com",
                          "qp_postproj_horizon_orientation")) {
              return false;
            }
            pred = q_catch;
            pred_s = *catch_s;
            pred_t = tk;
            continue;
          }
          const double tk = pred_s / std::max(end, 1.0);
          if (!chord_ok(pred, qk, pred_t, tk, "qp_postproj_horizon_tube_or_com",
                        "qp_postproj_horizon_orientation")) {
            return false;
          }
          pred = qk;
          pred_t = tk;
        }
        break;  // horizon rebuilt and checked; skip stale ProxQP stages
      }
    } else {
      prev = q;
      previous_t = t;
    }
  }
  last_line_qp_proj_max_ = (to - last_line_qp_pre_).cwiseAbs().maxCoeff();
  // Success clears the fail latch so notes only show the last real refusal.
  last_line_qp_fail_ = LineQpFailSnapshot{};
  last_line_qp_fail_dump_.pending_write = false;
  return true;
}
} // namespace kmpc
