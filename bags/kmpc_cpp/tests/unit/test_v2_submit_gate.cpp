#include <gtest/gtest.h>

#include "kmpc/v2_submit_gate.hpp"

using kmpc::V2PlanPhase;
using kmpc::V2SubmitGate;
using kmpc::V2TaskId;

TEST(V2SubmitGate, FirstTaskSubmitsOnceThenPolls) {
  V2SubmitGate gate;
  const V2TaskId id{5, 1001};
  auto d0 = gate.on_tick(id, false, false);
  EXPECT_TRUE(d0.submit);
  EXPECT_STREQ(d0.reason, "submit_once");
  gate.note_submitted(id, 7);

  for (int i = 0; i < 20; ++i) {
    auto d = gate.on_tick(id, false, false);
    EXPECT_FALSE(d.submit) << i;
    EXPECT_TRUE(d.poll_worker);
    EXPECT_EQ(gate.counters().submits, 1u);
  }
}

TEST(V2SubmitGate, WorkerOkThenAdoptThenNoResubmitWhileCached) {
  V2SubmitGate gate;
  const V2TaskId id{5, 1001};
  ASSERT_TRUE(gate.on_tick(id, false, false).submit);
  gate.note_submitted(id, 3);
  auto take = gate.on_worker_result(id, 5, 1001, 3, true, true);
  EXPECT_TRUE(take.identity_ok);
  EXPECT_TRUE(take.adopt);
  gate.note_adopted(id, 3, 42);
  EXPECT_EQ(gate.phase(), V2PlanPhase::Active);

  auto follow = gate.on_tick(id, true, false);
  EXPECT_TRUE(follow.follow);
  EXPECT_FALSE(follow.submit);

  gate.note_arrived();
  for (int i = 0; i < 10; ++i) {
    auto d = gate.on_tick(id, false, false);
    EXPECT_FALSE(d.submit) << i;
    EXPECT_TRUE(d.hold);
  }
  EXPECT_EQ(gate.counters().submits, 1u);
  EXPECT_EQ(gate.counters().adopts, 1u);
}

TEST(V2SubmitGate, PlanFailDoesNotSpam) {
  V2SubmitGate gate;
  const V2TaskId id{8, 2002};
  ASSERT_TRUE(gate.on_tick(id, false, false).submit);
  gate.note_submitted(id, 1);
  auto take = gate.on_worker_result(id, 8, 2002, 1, false, true);
  EXPECT_FALSE(take.identity_ok);
  gate.note_plan_failed("plan_not_ok");
  for (int i = 0; i < 30; ++i) {
    auto d = gate.on_tick(id, false, false);
    EXPECT_FALSE(d.submit) << i;
    EXPECT_STREQ(d.reason, "failed_wait");
  }
  EXPECT_EQ(gate.counters().submits, 1u);
}

TEST(V2SubmitGate, NewGenerationReplacesOld) {
  V2SubmitGate gate;
  const V2TaskId a{1, 11};
  const V2TaskId b{2, 22};
  ASSERT_TRUE(gate.on_tick(a, false, false).submit);
  gate.note_submitted(a, 1);
  auto d = gate.on_tick(b, false, false);
  EXPECT_TRUE(d.submit);
  EXPECT_TRUE(d.invalidate_old);
  gate.note_submitted(b, 2);
  EXPECT_EQ(gate.counters().submits, 2u);
  EXPECT_TRUE(gate.submitted().same_as(b));
}

TEST(V2SubmitGate, ClearPreventsReactivation) {
  V2SubmitGate gate;
  const V2TaskId id{9, 99};
  ASSERT_TRUE(gate.on_tick(id, false, false).submit);
  gate.note_submitted(id, 4);
  gate.note_adopted(id, 4, 1);
  gate.clear();
  EXPECT_EQ(gate.phase(), V2PlanPhase::Idle);
  EXPECT_EQ(gate.submitted().generation, 0u);
  // Same numbers after clear still count as new identity vs empty submitted.
  auto d = gate.on_tick(id, false, false);
  EXPECT_TRUE(d.submit);
}

TEST(V2SubmitGate, ExplicitReplanRetryBudget) {
  V2SubmitGate gate;
  const V2TaskId id{3, 33};
  ASSERT_TRUE(gate.on_tick(id, false, false).submit);
  gate.note_submitted(id, 1);
  gate.note_plan_failed("x");
  // Without explicit: no submit.
  EXPECT_FALSE(gate.on_tick(id, false, false).submit);
  // With explicit: one retry allowed.
  auto d = gate.on_tick(id, false, true);
  EXPECT_TRUE(d.submit);
  gate.note_submitted(id, 2);
  gate.note_plan_failed("x2");
  // Budget exhausted.
  auto d2 = gate.on_tick(id, false, true);
  EXPECT_FALSE(d2.submit);
  EXPECT_STREQ(d2.reason, "retry_exhausted");
}

TEST(V2SubmitGate, FixedTopicsAreNotEmpty) {
  EXPECT_STREQ(kmpc::kStageAckTopic, "/kmpc/stage_ack");
  EXPECT_STREQ(kmpc::kMotionSequenceTopic, "/kmpc/motion_sequence");
}
