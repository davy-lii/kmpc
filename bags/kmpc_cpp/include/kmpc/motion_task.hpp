#pragma once
// Geometric task. No target-height pose switch lives here.

#include <cstdint>
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
};

struct MotionTask {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t generation = 0;
  Vec18 q = Vec18::Zero();
  Vec18 dq = Vec18::Zero();
  bool dq_known = false;
  bool left = true;
  std::vector<TaskStage, Eigen::aligned_allocator<TaskStage>> stages;
};

}  // namespace kmpc
