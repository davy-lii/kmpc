#pragma once
// 金标容差集中在这里。禁止在各 test 里写魔法数。对照开发方案 §6。

namespace kmpc {

struct NumericalPolicy {
  static constexpr double kPAbsTol = 1e-8;
  static constexpr double kGAbsTol = 1e-7;
  static constexpr double kInteriorY0Tol = 1e-6;
  static constexpr double kBoxY0Tol = 1e-5;
  // ProxQP vs Python 三轮牛顿：同一 P 在长 replay 贴约束时 y0 可到 ~1e-4，不是调权重。
  static constexpr double kReplayY0Tol = 2e-4;
  static constexpr double kTcpAbsTol = 1e-6;  // m
  static constexpr double kOriAbsTol = 1e-6;  // rad
  static constexpr double kJacAbsTol = 1e-5;
  static constexpr double kHotMeanMs = 5.0;
  static constexpr double kHotP99Ms = 10.0;
  static constexpr int kHotSkipTicks = 5;  // 首次 ProxQP init 不算热路径
};

}  // namespace kmpc
