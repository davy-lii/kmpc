#pragma once
// 对应 Python kmpc/upper_command.py；对照 09-04 §1.1。
// UpperCommand 不进决策量 x；默认 TRACKING + 不锁腰 = noop。

#include <string_view>

#include "kmpc/types.hpp"

namespace kmpc {

enum class TrackingState {
  Calibrating,
  Tracking,
};

inline constexpr std::string_view kTrackingStateCalibrating = "CALIBRATING";
inline constexpr std::string_view kTrackingStateTracking = "TRACKING";

inline const char* to_cstring(TrackingState s) {
  return s == TrackingState::Calibrating ? "CALIBRATING" : "TRACKING";
}

inline TrackingState tracking_state_from_string(std::string_view s) {
  if (s == kTrackingStateCalibrating) {
    return TrackingState::Calibrating;
  }
  return TrackingState::Tracking;
}

struct UpperCommand {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrackingState tracking_state = TrackingState::Tracking;
  bool waist_locked = false;
  bool has_q_calib = false;
  Vec18 q_calib = Vec18::Zero();
  // 现码未用；闲臂由本拍笛卡尔目标推断，不要在 Tracker 里接这个字段。
  bool has_active_arm = false;
  char active_arm[16]{};

  UpperCommand copy() const { return *this; }
};

inline const UpperCommand& default_upper_command() {
  static const UpperCommand kDefault{};
  return kDefault;
}

}  // namespace kmpc
