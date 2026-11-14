#include "kmpc/motion_evaluator.hpp"

namespace kmpc {

void MotionEvaluator::evaluate(
    const Vec18& q, bool left, MotionSnapshot* out) const {
  const std::string_view frame = left ? kEeFrameLeft : kEeFrameRight;
  out->tcp = kin_.fk_pos(q, frame);
  out->tcp_jac = kin_.jacobian_pos(q, frame);
  out->body = kin_.fk_pos(q, "Body_Link");
  out->leg_y = kin_.fk_pos(q, "Leg_Y_Link2");
  out->com = kin_.com(q);
  out->j_com = kin_.jacobian_com(q);
}

}  // namespace kmpc
