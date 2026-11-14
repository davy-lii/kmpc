#include "kmpc/motion_evaluator.hpp"

namespace kmpc {

void MotionEvaluator::evaluate(
    const Vec18& q, bool left, MotionSnapshot* out) const {
  if (out == nullptr) {
    return;
  }
  const std::string_view frame = left ? kEeFrameLeft : kEeFrameRight;
  out->tcp = kin_.fk_pos(q, frame);
  out->tcp_jac = kin_.jacobian_pos(q, frame);
  out->body = kin_.fk_pos(q, "Body_Link");
  out->body_jac = kin_.jacobian_pos(q, "Body_Link");
  out->leg_y = kin_.fk_pos(q, "Leg_Y_Link2");
  out->leg_jac = kin_.jacobian_pos(q, "Leg_Y_Link2");
  out->com = kin_.com(q);
  out->j_com = kin_.jacobian_com(q);
  out->body_minus_leg_x = out->body.x() - out->leg_y.x();
  out->pitch = q[0] + q[1] + q[2];
}

void MotionEvaluator::batch(
    const std::vector<Vec18, Eigen::aligned_allocator<Vec18>>& knots, bool left,
    std::vector<MotionSnapshot, Eigen::aligned_allocator<MotionSnapshot>>* out)
    const {
  if (out == nullptr) {
    return;
  }
  out->clear();
  out->reserve(knots.size());
  for (const Vec18& q : knots) {
    MotionSnapshot snap;
    evaluate(q, left, &snap);
    out->push_back(snap);
  }
}

}  // namespace kmpc
