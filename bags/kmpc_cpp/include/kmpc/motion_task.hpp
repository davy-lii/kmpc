#pragma once
// Geometric task. No target-height pose switch lives here.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "kmpc/types.hpp"

namespace kmpc {

enum class TaskGeometry {
  Lin = 0,
  Ptp = 1,
  FollowCurve = 2,
};

enum class StageStop {
  Continuous = 0,
  StopRequired = 1,
};

struct TaskStage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TaskGeometry geometry = TaskGeometry::Lin;
  Vec3 position = Vec3::Zero();
  bool has_orientation = false;
  Vec4 quat = Vec4(0.0, 0.0, 0.0, 1.0);
  // Endpoint / line tolerance asked by the task. Not the old 0.20 m hard tube.
  double position_tol_m = 0.003;
  double orientation_tol_rad = 0.05;
  StageStop stop = StageStop::Continuous;
  bool wait_for_external_ack = false;
  uint32_t stage_id = 0;
};

struct MotionTask {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t generation = 0;
  Vec18 q = Vec18::Zero();
  Vec18 dq = Vec18::Zero();
  bool dq_known = false;
  bool left = true;
  bool waist_locked = false;
  std::string frame;
  std::string model_digest;
  std::vector<TaskStage, Eigen::aligned_allocator<TaskStage>> stages;
};

struct RobotState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec18 q = Vec18::Zero();
  Vec18 dq = Vec18::Zero();
  bool dq_known = false;
  double timestamp_s = 0.0;
  bool fresh = true;
};

inline uint64_t motion_task_digest(const MotionTask& task) {
  uint64_t h = 14695981039346656037ull;
  auto mix = [&](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  auto mixd = [&](double v) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v), "double 8");
    std::memcpy(&bits, &v, sizeof(bits));
    mix(bits);
  };
  auto mix_str = [&](const std::string& s) {
    mix(static_cast<uint64_t>(s.size()));
    for (unsigned char c : s) {
      mix(static_cast<uint64_t>(c));
    }
  };
  mix(task.generation);
  mix(task.left ? 1u : 0u);
  mix(task.waist_locked ? 1u : 0u);
  mix_str(task.frame);
  mix_str(task.model_digest);
  mix(static_cast<uint64_t>(task.stages.size()));
  for (const auto& st : task.stages) {
    mix(static_cast<uint64_t>(st.geometry));
    mix(st.stage_id);
    mix(st.has_orientation ? 1u : 0u);
    mix(static_cast<uint64_t>(st.stop));
    mix(st.wait_for_external_ack ? 1u : 0u);
    mixd(st.position.x());
    mixd(st.position.y());
    mixd(st.position.z());
    mixd(st.quat.x());
    mixd(st.quat.y());
    mixd(st.quat.z());
    mixd(st.quat.w());
    mixd(st.position_tol_m);
    mixd(st.orientation_tol_rad);
  }
  return h;
}

}  // namespace kmpc
