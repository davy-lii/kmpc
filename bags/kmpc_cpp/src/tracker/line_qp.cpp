// Line-only full-stage QP. Every predicted stage has its own 11-DOF correction;
// the generic 36-variable endpoint QP is deliberately untouched.
#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/path_progress.hpp"
#include "kmpc/tracker_math.hpp"  // kStartPitchAdmitRad
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

}  // namespace

bool CartesianTracker::line_path_qp(const Vec18& from, double next_prog,
                                     const Vec18& cap, Vec18& to) {
  auto reject=[this](const std::string& reason) {
    line_.reason=reason;
    std::snprintf(last_line_qp_fail_.reason, sizeof(last_line_qp_fail_.reason), "%s",
                  reason.c_str());
    last_line_qp_fail_.valid = true;
    return false;
  };
  last_line_qp_fail_ = LineQpFailSnapshot{};
  last_line_qp_fail_.path_remain_m = last_line_path_remain_m_;
  last_line_qp_fail_.prog = next_prog;
  last_line_qp_fail_.pitch_deg = rad_to_deg(from.head<3>().sum());
  last_line_qp_fail_.ee_err_m =
      (line_.target - kin_.fk_pos(from, line_.frame)).norm();

  constexpr int d=kNWaist+kNArm;
  const int n=std::clamp(n_horizon_,1,30), nx=n*d;
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
  auto row=[&](int k,const Eigen::Matrix<double,1,d>& a,double lo,double hi,
               bool previous=false) {
    const int r=static_cast<int>(lower.size()); lower.push_back(lo); upper.push_back(hi);
    for(int j=0;j<d;++j) if(a[j]!=0.0) {
      ct.emplace_back(r,k*d+j,a[j]);
      if(previous) ct.emplace_back(r,(k-1)*d+j,-a[j]);
    }
  };
  const double pitch0=line_.samples.front().head<3>().sum();
  const double pitchN=line_.samples.back().head<3>().sum();
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
    Eigen::Matrix<double,d,d> H=400.0*J.topRows<3>().transpose()*J.topRows<3>();
    Eigen::Matrix<double,d,1> grad=-400.0*J.topRows<3>().transpose()*ep;
    if(line_.ori_on) {
      H+=20.0*J.bottomRows<3>().transpose()*J.bottomRows<3>();
      grad-=20.0*J.bottomRows<3>().transpose()*eo;
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
      row(k,a,-trust,trust);
    }
    for(int j=0;j<18;++j) {
      if(T.row(j).squaredNorm()==0) continue;
      row(k,T.row(j),limits_.q_min[j]+q_limit_margin_-q[j],limits_.q_max[j]-q_limit_margin_-q[j]);
      row(k,T.row(j),-cap[j]-(q[j]-prev[j]),cap[j]-(q[j]-prev[j]),k>0);
    }
    // Box inscribed in the 3 mm Euclidean tube; nonlinear validation follows.
    constexpr double tube=0.0029/1.7320508075688772;
    for(int j=0;j<3;++j) {
      row(k,J.row(j),ep[j]-tube,ep[j]+tube);
      if(line_.ori_on) row(k,J.row(3+j),eo[j]-arrived_ori_rad_/1.7321,eo[j]+arrived_ori_rad_/1.7321);
    }
    const Vec3 com=kin_.com(q);
    const Eigen::Matrix<double,3,d> jc=kin_.jacobian_com(q)*T;
    row(k,jc.row(0),com_xmin_+com_margin_-com[0],com_xmax_-com_margin_-com[0]);
    row(k,jc.row(1),com_ymin_+com_margin_-com[1],com_ymax_-com_margin_-com[1]);
    Eigen::Matrix<double,1,d> pitch=Eigen::Matrix<double,1,d>::Zero(); pitch.head<3>().setOnes();
    // Unified with line_build / 250 Hz stream_pitch_recovery (kStartPitchAdmitRad):
    // existing tiny excess may persist (no worsen vs stream `from`), shrink toward
    // the hard box by the waist rate budget, then stay inside. Never permanently
    // widen the box; never treat MAX_ITER as success.
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
      const double allowed_ex=
          std::min(admit_excess,std::max(staged,first_floor));
      const double soft=limit+allowed_ex;
      row(k,pitch,-soft-q.head<3>().sum(),soft-q.head<3>().sum());
    }
    // 名义路径的俯仰和已经定好。这一拍的修正不能把它改回反方向。x=0 不在界内就不加，避免把原来能走的段判失败。
    // P0: independent of line_fit pitch straighten; first field对照 keeps this ON.
    // While `from` is still outside, flat mono can lock the path outside the hard
    // box (field A). After back in box, mono uses clamped endpoints so the flat
    // band sits on ±limit instead of fighting it.
    if (line_qp_pitch_monotonic_enable_ && !pitch_recovering) {
      constexpr double kPitchSlack=2e-4;
      const double p_prev=prev.head<3>().sum();
      const double p_q=q.head<3>().sum();
      const double dp=p_q-p_prev;
      const double p0m=
          (limit>0)?std::clamp(pitch0,-limit,limit):pitch0;
      const double pNm=
          (limit>0)?std::clamp(pitchN,-limit,limit):pitchN;
      const double netm=pNm-p0m;
      double lo=-1e9, hi=1e9;
      if(std::abs(netm)<=1e-4) {
        lo=p0m-kPitchSlack-p_q;
        hi=p0m+kPitchSlack-p_q;
      } else if(netm>0.0) {
        lo=-dp-kPitchSlack;
      } else {
        hi=-dp+kPitchSlack;
      }
      lo=std::max(lo,std::min(p_prev,pNm)-kPitchSlack-p_q);
      hi=std::min(hi,std::max(p_prev,pNm)+kPitchSlack-p_q);
      if(lo<=0.0 && hi>=0.0 && lo<hi) {
        row(k,pitch,lo,hi);
        ++line_qp_pitch_monotonic_hits_;
      }
    }
    if(yaw_max_>0) {
      Eigen::Matrix<double,1,d> yaw=Eigen::Matrix<double,1,d>::Zero();yaw[kIdxYaw]=1;
      row(k,yaw,q_yaw_pref_-yaw_max_-q[kIdxYaw],q_yaw_pref_+yaw_max_-q[kIdxYaw]);
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
    return reject("qp_bounds");
  }
  Eigen::VectorXd x;
  try {
    proxsuite::proxqp::sparse::QP<double,int> qp(nx,0,nc);
    qp.settings.eps_abs=1e-7; qp.settings.eps_rel=0.0;
    qp.settings.max_iter=line_qp_max_iter_;
    qp.settings.verbose=false;
    qp.init(H,g,proxsuite::nullopt,proxsuite::nullopt,C,lo,hi);
    ++line_qp_solves_;
    const auto t0 = std::chrono::steady_clock::now();
    qp.solve();
    const auto t1 = std::chrono::steady_clock::now();
    last_line_qp_fail_.solve_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_line_qp_fail_.status = static_cast<int>(qp.results.info.status);
    std::snprintf(last_line_qp_fail_.status_name,
                  sizeof(last_line_qp_fail_.status_name), "%s",
                  proxqp_status_name(qp.results.info.status));
    last_line_qp_fail_.iter = static_cast<int>(qp.results.info.iter);
    last_line_qp_fail_.pri_res = static_cast<double>(qp.results.info.pri_res);
    last_line_qp_fail_.dua_res = static_cast<double>(qp.results.info.dua_res);
    fill_worst(qp.results.x);
    if(qp.results.info.status!=proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
      return reject("qp_status_"+std::to_string(static_cast<int>(qp.results.info.status)));
    }
    x=qp.results.x;
  } catch(...) { return reject("qp_exception"); }
  if(!x.allFinite()) return reject("qp_nonfinite");
  if(line_.waist_locked) for(int k=0;k<n;++k) x.segment(k*d,kNWaist).setZero();
  const Eigen::VectorXd cx=C*x;
  if((cx.array()<lo.array()-2e-7).any() || (cx.array()>hi.array()+2e-7).any()) {
    fill_worst(x);
    return reject("qp_primal_residual");
  }
  prev=from;
  // The first interval starts at the actual stream reference, not the old endpoint.
  double previous_t=(kin_.fk_pos(from,line_.frame)-p_start).dot(line_.target-p_start)/
      std::max((line_.target-p_start).squaredNorm(),1e-12);
  previous_t=std::clamp(previous_t,0.0,1.0);
  for(int k=0;k<n;++k) {
    const Vec18 q=nominal[k]+T*x.segment(k*d,d);
    const double t=progress[k]/std::max(end,1.0);
    if(((q-prev).cwiseAbs().array()>cap.array()+3e-7).any()) return reject("qp_velocity");
    for(int j=1;j<=5;++j) {
      const double a=j/5.0;
      const Vec18 mid=(1-a)*prev+a*q;
      const auto pose=kin_.fk(mid,line_.frame);
      const Vec3 ab=line_.target-p_start;
      const double along=std::clamp((pose.col(3).head<3>()-p_start).dot(ab)/std::max(ab.squaredNorm(),1e-12),0.0,1.0);
      if((pose.col(3).head<3>()-(p_start+along*ab)).norm()>0.003 ||
          !guards_.com_xy_inside(kin_.com(mid))) return reject("qp_nonlinear_tube_or_com");
      if(line_.ori_on && ori_error_xyzw(quat_slerp_xyzw(ori_start,line_.quat,(1-a)*previous_t+a*t),
          rot_to_quat_xyzw(pose.block<3,3>(0,0))).norm()>arrived_ori_rad_+1e-6) return reject("qp_nonlinear_orientation");
    }
    if(k==0) to=q;
    prev=q; previous_t=t;
  }
  // Success clears the fail latch so notes only show the last real refusal.
  last_line_qp_fail_ = LineQpFailSnapshot{};
  return true;
}
} // namespace kmpc
