// Standalone offline command simulation: no ROS, sockets or hardware output.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "kmpc/cartesian_tracker.hpp"
#include "kmpc/tracker_math.hpp"
using namespace kmpc;
struct Case { std::string record; int index=0,left=0; double time=0; Vec18 q; Vec3 xyz,tip; Vec4 quat,tquat; int has_tip=0; };
double quantile(std::vector<double> x,double p) { if(x.empty()) return 0; std::sort(x.begin(),x.end());return x[std::min(x.size()-1,size_t(p*(x.size()-1)))]; }
int main(int argc,char**argv) {
  if(argc<3) {std::cerr<<"replay_recent cases.tsv results.csv [limit] [chain]\n";return 2;}
  const int limit=argc>3?std::stoi(argv[3]):100000;
  const bool chain=argc>4 && std::string(argv[4])=="chain";
  Config cfg=Config::load(std::string(KMPC_CONFIG_DIR)+"/joint_limits.yaml",std::string(KMPC_CONFIG_DIR)+"/q_pref.yaml",std::string(KMPC_CONFIG_DIR)+"/kmpc_params.yaml");
  Kinematics kin(std::string(KMPC_URDF_DIR)+"/Light-duty wheeled robot.urdf");
  if(std::string(argv[1])=="--fixture") {
    std::ofstream fixture(argv[2]);
    const auto pose=kin.fk(cfg.q_pref,kEeFrameLeft);
    const Vec4 quat=rot_to_quat_xyzw(pose.block<3,3>(0,0));
    fixture<<std::setprecision(16)<<"{\"q\":[";
    for(int i=0;i<18;++i) fixture<<(i?",":"")<<cfg.q_pref[i];
    fixture<<"],\"position\":[";
    for(int i=0;i<3;++i) fixture<<(i?",":"")<<pose(i,3);
    fixture<<"],\"flange_quat\":[";
    for(int i=0;i<4;++i) fixture<<(i?",":"")<<quat[i];
    fixture<<"]}\n";return 0;
  }

  std::ifstream in(argv[1]);std::ofstream out(argv[2]);
  out<<"record,index,left,tip,arrived,refused,reason,ticks,length_m,lateral_mm,end_mm,ori_deg,step_ratio,waist_extra_deg,build_ms,solve_p95_ms,solve_p99_ms,first_step_ratio,cascade,forced,qp_solves\n";
  std::string row,record;Vec18 continued=cfg.q_pref;bool previous_ok=false;int count=0,arrivals=0,refusals=0;
  while(std::getline(in,row) && count<limit) {
    Case c;std::istringstream ss(row);ss>>c.record>>c.index>>c.left>>c.time;
    for(int j=0;j<18;++j)ss>>c.q[j];for(int j=0;j<3;++j)ss>>c.xyz[j];for(int j=0;j<4;++j)ss>>c.quat[j];
    ss>>c.has_tip;for(int j=0;j<3;++j)ss>>c.tip[j];for(int j=0;j<4;++j)ss>>c.tquat[j];
    if(!ss) return 3;
    c.quat=tool_to_flange_quat_xyzw(c.quat);c.tquat=tool_to_flange_quat_xyzw(c.tquat);
    if(chain && record==c.record && c.index%3!=0 && previous_ok)c.q=continued;
    record=c.record;
    CartesianTracker tracker(kin,cfg);tracker.set_line_pace(0.6,0.5);
#ifndef KMPC_BASELINE
    std::optional<LineTarget> tip;
    if(c.has_tip) {LineTarget t;t.xyz=c.tip;t.quat_xyzw=c.tquat;t.ori_on=true;tip=t;}
    tracker.rearm_line(count+1,tip);
#endif
    const std::string frame(c.left?kEeFrameLeft:kEeFrameRight);
    const Vec3 p0=kin.fk_pos(c.q,frame),ab=c.xyz-p0;
    const Vec18 seed=c.q;Vec18 q=c.q,qmin=q,qmax=q;
    double lat=0,ratio=0,first=0,build_ms=0;int ticks=0;
    std::vector<double> times;
    for(;ticks<2000;++ticks) {
      const auto start=std::chrono::steady_clock::now();
      auto step=tracker.compute(q,cfg.q_pref,c.left?&c.xyz:nullptr,c.left?nullptr:&c.xyz,c.left?&c.quat:nullptr,c.left?nullptr:&c.quat);
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(ticks)times.push_back(ms);else build_ms=ms;
      if(tracker.line_refused())break;
      const Vec18 next=tracker.line_cmd();
      if(!next.allFinite())return 4;
#ifndef KMPC_BASELINE
      if(!tracker.line_holding()) {
        const auto chk=tracker.line_stream_segment_check(q,next);
        if(!chk.ok) {
          tracker.refuse_line(std::string("stream_")+chk.reason);break;
        }
      }
#endif
      const double sr=((next-q).cwiseAbs().array()/(cfg.limits.dq_max.array()*cfg.params.dt*0.6)).maxCoeff();
      ratio=std::max(ratio,sr);if(!ticks)first=sr;
      for(int j=1;j<=5;++j) {
        const Vec18 mid=q+(next-q)*(j/5.0);const Vec3 pos=kin.fk_pos(mid,frame);
        const double s=std::clamp((pos-p0).dot(ab)/std::max(ab.squaredNorm(),1e-12),0.0,1.0);
        lat=std::max(lat,(pos-(p0+s*ab)).norm());
      }
      q=next;qmin=qmin.cwiseMin(q);qmax=qmax.cwiseMax(q);
      if(tracker.last_arrived())break;
    }
    continued=q;previous_ok=tracker.last_arrived();
    const double end=(kin.fk_pos(q,frame)-c.xyz).norm();
    const double ori=ori_error_xyzw(c.quat,rot_to_quat_xyzw(kin.fk(q,frame).block<3,3>(0,0))).norm()*180/kPi;
    double extra=0;for(int j=0;j<4;++j)extra=std::max(extra,qmax[j]-qmin[j]-std::abs(q[j]-seed[j]));
    std::string reason="none";int cascade=0,forced=0,solves=0;
#ifndef KMPC_BASELINE
    reason=tracker.line_reason();if(reason.empty())reason="none";
    build_ms=tracker.line_build_ms();cascade=tracker.line_cascade_passed();forced=tracker.line_cascade_forced_waist();solves=tracker.line_qp_solves();
#endif
    out<<std::setprecision(10)<<c.record<<','<<c.index<<','<<c.left<<','<<c.has_tip<<','<<tracker.last_arrived()<<','<<tracker.line_refused()<<','<<reason<<','<<ticks<<','<<ab.norm()<<','<<lat*1000<<','<<end*1000<<','<<ori<<','<<ratio<<','<<extra*180/kPi<<','<<build_ms<<','<<quantile(times,.95)<<','<<quantile(times,.99)<<','<<first<<','<<cascade<<','<<forced<<','<<solves<<'\n';out.flush();
    ++count;arrivals+=tracker.last_arrived();refusals+=tracker.line_refused();
  }
  std::cout<<"segments="<<count<<" arrived="<<arrivals<<" refused="<<refusals<<" unfinished="<<(count-arrivals-refusals)<<'\n';
  return 0;
}
