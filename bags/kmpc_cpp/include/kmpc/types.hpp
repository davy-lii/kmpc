#pragma once
// 对应 Python kmpc/types.py；对照 09-04 §0。
// I/O 边界：核内一律弧度、米；pack_stream 出口才变成度（joint_stream_cmd）。

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

#include <Eigen/Core>

namespace kmpc {

constexpr int kNDof = 18;
constexpr int kNWaist = 4;
constexpr int kNArm = 7;
constexpr int kNQp = 2;  // 决策 (y0, y_{N-1})，禁止把中间点塞进求解器
constexpr int kNx = kNDof * kNQp;  // 36
// 笛卡尔两端点：俯仰 4 + yaw 4 + CoM 8 = 16。工作墙 CoM 打 y0 与 y_{N-1}。
constexpr int kMaxIneq = 16;
constexpr double kPi = 3.14159265358979323846;

using Mat36 = Eigen::Matrix<double, kNx, kNx>;
using Vec36 = Eigen::Matrix<double, kNx, 1>;
using Vec18 = Eigen::Matrix<double, kNDof, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat3x18 = Eigen::Matrix<double, 3, kNDof>;
using Mat2x18 = Eigen::Matrix<double, 2, kNDof>;
using Mat6x18 = Eigen::Matrix<double, 6, kNDof>;
using Mat18 = Eigen::Matrix<double, kNDof, kNDof>;
using MatIneq = Eigen::Matrix<double, kMaxIneq, kNx>;
using MatIneq18 = Eigen::Matrix<double, kMaxIneq, kNDof>;
using VecIneq = Eigen::Matrix<double, kMaxIneq, 1>;

constexpr double kHessRidge = 1e-8;  // assemble 对角；对照 horizon_qp.py

// 腕 J5–J7 左右共 6 轴：λ 使 mean(M_ii) 对上 r_arm。对照 urdf_kinematics.WRIST_INDICES。
inline constexpr std::array<int, 6> kWristIndices = {8, 9, 10, 15, 16, 17};

struct Slice {
  int begin = 0;  // 含
  int end = 0;    // 不含，与 Python slice 相同
  constexpr int size() const { return end - begin; }
  constexpr bool operator==(const Slice& o) const {
    return begin == o.begin && end == o.end;
  }
};

constexpr Slice kSliceWaist{0, 4};
constexpr Slice kSliceRight{4, 11};
constexpr Slice kSliceLeft{11, 18};

// 肩俯仰 J2、肘 J4。禁止把 q[4]/q[8] 当肘（那是指南示例的错下标）。
constexpr int kIdxJ2R = 5;
constexpr int kIdxJ4R = 7;
constexpr int kIdxJ2L = 12;
constexpr int kIdxJ4L = 14;
constexpr int kIdxYaw = 3;  // 腰 J4，Python _IDX_YAW

constexpr int kRobotIdRight = 1;
constexpr int kRobotIdLeft = 2;
constexpr int kRobotIdWaist = 3;
constexpr int kRobotIdHead = 4;

inline constexpr std::array<int, 3> kStreamRobotIds = {
    kRobotIdWaist, kRobotIdRight, kRobotIdLeft};

inline constexpr std::array<std::string_view, kNDof> kJointNames = {
    "Chassis_Y_Joint1", "Chassis_Y_Joint2", "Chassis_Y_Joint3", "Chassis_Z_Joint4",
    "RightArm_Z_Joint1", "RightArm_Z_Joint2", "RightArm_Z_Joint3", "RightArm_Z_Joint4",
    "RightArm_Z_Joint5", "RightArm_Z_Joint6", "RightArm_Z_Joint7",
    "LeftArm_Z_Joint1",  "LeftArm_Z_Joint2",  "LeftArm_Z_Joint3",  "LeftArm_Z_Joint4",
    "LeftArm_Z_Joint5",  "LeftArm_Z_Joint6",  "LeftArm_Z_Joint7",
};

inline constexpr std::string_view kModeSingleEe = "single_ee";
inline constexpr std::string_view kModeDualCarry = "dual_carry";
inline constexpr std::string_view kModeHold = "hold";

inline bool is_valid_mode(std::string_view s) {
  return s == kModeSingleEe || s == kModeDualCarry || s == kModeHold;
}

inline constexpr std::string_view kEeFrameRight = "Link7_R";
inline constexpr std::string_view kEeFrameLeft = "Link7_L";
inline constexpr std::string_view kPalmFrameRight = "r_hand_link";
inline constexpr std::string_view kPalmFrameLeft = "l_hand_link";
inline constexpr std::string_view kBaseFrame = "base_link";

// 跟踪点是法兰 Link7_L / Link7_R。
// /kmpc/target_* 上仍是 1 号机录的工具姿态。收到后转成法兰 −Y = 工具 +X，末端跟着朝下的那根轴。
// 不要把工具四元数贴到 l_hand_link / r_hand_link，那两帧的 +Y 就是法兰 −Y。

enum class QpStatus {
  Success = 0,
  IllegalInput,
  SolverFailed,
  NonFinite,
};

inline constexpr double deg_to_rad(double degrees) {
  return degrees * (kPi / 180.0);
}

inline constexpr double rad_to_deg(double radians) {
  return radians * (180.0 / kPi);
}

inline Vec18 deg_array_to_rad(const Vec18& degrees) {
  return degrees * (kPi / 180.0);
}

inline Vec18 rad_array_to_deg(const Vec18& radians) {
  return radians * (180.0 / kPi);
}

inline std::string_view palm_frame(std::string_view ee_or_palm) {
  if (ee_or_palm == kEeFrameLeft || ee_or_palm == "LeftArm_Z_Link7") {
    return kPalmFrameLeft;
  }
  if (ee_or_palm == kEeFrameRight || ee_or_palm == "RightArm_Z_Link7") {
    return kPalmFrameRight;
  }
  return ee_or_palm;
}

inline int name_to_index(std::string_view name) {
  for (int i = 0; i < kNDof; ++i) {
    if (kJointNames[static_cast<std::size_t>(i)] == name) {
      return i;
    }
  }
  return -1;
}

inline Slice robot_id_to_slice(int robot_id) {
  if (robot_id == kRobotIdWaist) {
    return kSliceWaist;
  }
  if (robot_id == kRobotIdRight) {
    return kSliceRight;
  }
  if (robot_id == kRobotIdLeft) {
    return kSliceLeft;
  }
  return Slice{0, 0};
}

// robot_id → 度。Head 不下发。
struct PackedStream {
  std::array<double, kNWaist> waist_deg{};
  std::array<double, kNArm> right_deg{};
  std::array<double, kNArm> left_deg{};
};

inline PackedStream pack_stream_deg(const Vec18& q_rad) {
  PackedStream out;
  const Vec18 deg = rad_array_to_deg(q_rad);
  for (int i = 0; i < kNWaist; ++i) {
    out.waist_deg[static_cast<std::size_t>(i)] = deg[i];
  }
  for (int i = 0; i < kNArm; ++i) {
    out.right_deg[static_cast<std::size_t>(i)] = deg[kSliceRight.begin + i];
    out.left_deg[static_cast<std::size_t>(i)] = deg[kSliceLeft.begin + i];
  }
  return out;
}

inline Vec18 q_from_name_map(
    const std::unordered_map<std::string, double>& positions,
    double default_value = 0.0) {
  Vec18 q = Vec18::Constant(default_value);
  for (int i = 0; i < kNDof; ++i) {
    const auto it = positions.find(std::string(kJointNames[static_cast<std::size_t>(i)]));
    if (it != positions.end()) {
      q[i] = it->second;
    }
  }
  return q;
}

}  // namespace kmpc
