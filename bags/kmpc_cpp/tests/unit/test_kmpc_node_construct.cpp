#include <string>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "kmpc_declared_params.hpp"
#include "kmpc/v2_submit_gate.hpp"

namespace {

class ParamProbeNode : public rclcpp::Node {
 public:
  ParamProbeNode() : Node("kmpc_param_probe") {
    kmpc::ros2_params::declare_kmpc_node_params(*this);
  }
};

}  // namespace

TEST(KmpcNodeConstruct, DeclaredParamsCoverAllStrOrReads) {
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<ParamProbeNode>();
  for (const auto& name :
       kmpc::ros2_params::string_params_read_by_str_or()) {
    EXPECT_TRUE(node->has_parameter(name)) << name;
    EXPECT_NO_THROW({
      (void)node->get_parameter(name).as_string();
    }) << name;
  }
  for (const auto& name : kmpc::ros2_params::forbidden_topic_params()) {
    EXPECT_FALSE(node->has_parameter(name)) << name;
  }
  // Fixed topics remain available as constants for subscriptions.
  EXPECT_FALSE(std::string(kmpc::kStageAckTopic).empty());
  EXPECT_FALSE(std::string(kmpc::kMotionSequenceTopic).empty());
  rclcpp::shutdown();
}

TEST(KmpcNodeConstruct, UndeclaredParamStillThrows) {
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("kmpc_undeclared_probe");
  EXPECT_THROW((void)node->get_parameter("stage_ack_topic").as_string(),
               rclcpp::exceptions::ParameterNotDeclaredException);
  rclcpp::shutdown();
}
