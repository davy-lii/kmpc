#include <iostream>
#include <iomanip>
#include "kmpc/cartesian_tracker.hpp"
using namespace kmpc;
int main() {
  Config cfg=Config::load(std::string(KMPC_CONFIG_DIR)+"/joint_limits.yaml", std::string(KMPC_CONFIG_DIR)+"/kmpc_params.yaml");
  Kinematics kin(std::string(KMPC_URDF_DIR)+"/Light-duty wheeled robot.urdf");
  std::cout<<std::hexfloat;
  for(int mode=0;mode<3;++mode) {
    CartesianTracker tracker(kin,cfg);Vec18 q=kmpc::test_fixture_q0();
    Vec3 l=kin.fk_pos(q,kEeFrameLeft)+Vec3(.015,.01,-.01);
    Vec3 r=kin.fk_pos(q,kEeFrameRight)+Vec3(.01,-.01,-.015);
    if(mode==1) {UpperCommand c;c.tracking_state=TrackingState::Calibrating;c.has_q_calib=true;c.q_calib=kmpc::test_fixture_q0();c.q_calib[6]+=.01;tracker.set_upper_commands(c);}
    for(int tick=0;tick<30;++tick) {
      const auto step=tracker.compute(q,kmpc::test_fixture_q0(),mode==0?&l:nullptr,mode==0?&r:nullptr);
      for(int j=0;j<18;++j)std::cout<<step.dq[j]<<' ';
      for(int j=0;j<36;++j) {std::cout<<tracker.last_g()[j]<<' ';for(int k=0;k<36;++k)std::cout<<tracker.last_P()(j,k)<<' ';}
      std::cout<<int(tracker.last_qp_status())<<'\n';q+=step.dq;
    }
  }
}
