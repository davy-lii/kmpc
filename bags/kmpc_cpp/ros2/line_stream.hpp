#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include "kmpc/types.hpp"

namespace kmpc::ros2io {
struct LineStreamStep {
  Vec18 target;
  double scale;
};

// The stream can advance while the control thread solves. Keep the complete
// joint chord and shorten it uniformly from the reference at commit time.
// Independent per-axis clipping would change that chord and its FK geometry.
inline std::optional<LineStreamStep> bound_line_stream_step(
    const Vec18& from, const Vec18& planned, const Vec18& cap) {
  if (!from.allFinite() || !planned.allFinite() || !cap.allFinite() ||
      (cap.array()<0.0).any()) return std::nullopt;
  const Vec18 delta=planned-from;
  double scale=1.0;
  for (int j=0;j<kNDof;++j) {
    if (std::abs(delta[j])>cap[j]) scale=std::min(scale,cap[j]/std::abs(delta[j]));
  }
  if (scale<1.0 && scale>0.0) scale=std::nextafter(scale,0.0);
  return LineStreamStep{from+scale*delta,scale};
}
} // namespace kmpc::ros2io

namespace kmpc::ros2io {
// M09/P02: deleted the 3×15 causal mean FIR. Passthrough keeps speed_inf() for
// arrival settle checks; v/a/j belong to unified s(t), not a joint-space FIR.
class LineSmoother {
 public:
  void reset(const Vec18& q) {
    last_=q;velocity_.setZero();ready_=true;
  }
  Vec18 push(Vec18 q,double dt) {
    if(!ready_)reset(q);
    if(!(dt>1e-9)) dt=1e-9;
    velocity_=(q-last_)/dt;last_=q;return q;
  }
  double speed_inf() const {return velocity_.cwiseAbs().maxCoeff();}
 private:
  Vec18 last_=Vec18::Zero(),velocity_=Vec18::Zero();
  bool ready_=false;
};
}
