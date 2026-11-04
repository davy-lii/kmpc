# kmpc_cpp

全身运动学 MPC 的 C++ 实现。Python 金标在 `../whole_body_kmpc/`。

仿真怎么开、RViz 在哪看：见 **[`sim/README.md`](sim/README.md)**。  
真机 Thor 怎么拷、怎么建容器：[`Thor部署/`](../Thor部署/README.md)。

```bash
source /home/davy/dual_arm_mpc_papers/kmpc_cpp/sim/setup_sim.sh
ros2 launch kmpc_cpp kmpc_cpp_sim.launch.py
```

会弹出 RViz + 关节滑条窗口（pose_gui）。无头检查：`sim/tools/run_closed_loop.sh`。

公式以 `项目核心公式版本迭代/现码核心公式-2026-09-04.md` 为准（略蹲 HOME + 腰锁 / 标定开关；Hessian 骨架同 08-31）。C++ 是同一套代数的工业实现，不是第二套算法。

计划：`../待开发/C++移植/计划表.md`  
方案：`../待开发/C++移植/开发方案.md`

`src/core` / `src/solver` / `src/planner` / `src/tracker` 禁止 `#include <rclcpp/...>`。C8 节点在 `ros2/`。

## 移植铁律（优化只能发生在实现层）

1. 不根据「理论更优」修改 Python 算法。
2. 不自行调整 `Q` / `R` / `W` / `beta` / horizon。
3. 不自行重排 Hessian 块。
4. 不把 Eigen 优化结果当成算法修改（`noalias`、换加法顺序若改变 `P` 逐元值，先证明仍过金标）。
5. 任何数值差异先定位 Python/C++ **第一处 divergence**（运动学 → Hessian 分块 → `g` → `y0`），禁止调权重凑齐。
6. 热路径禁止「看起来等价」的循环改写，除非单测证明与 Python 同式同顺序。

HOME 读 yaml `stations.default`（肘 `-75°`），不要写死 `-π/2`（那是 `arms_side`）。

## 金标环境（开工机 2026-09）

| 项 | 钉死 |
|----|------|
| Ubuntu | 22.04.5 LTS |
| Compiler | g++ 11.4.0 |
| C++ | 17 |
| Eigen | 3.4.0 |
| yaml-cpp | 0.7.0 |
| GTest | 1.11.0 |
| Pinocchio | **4.0.0**（`ros-humble-pinocchio`） |
| ProxSuite | **0.7.3**（`ros-humble-proxsuite`；C4 才链进求解器） |

CMake 对 Eigen / Pinocchio 版本 `FATAL_ERROR`。ProxQP 用 `-DKMPC_REQUIRE_PROXSUITE=ON` 才强制（C4）。

Python dump 与 C++ 必须同一 URDF、同一 Pinocchio 大版本。数字以 `config/` 软链到的 Python yaml 为准，禁止复制改数。

## 编译 / 测试

```bash
cd kmpc_cpp
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`config/` 与 `urdf/` 是指向 `whole_body_kmpc` 的符号链接，由 CMake 创建。

C8 节点（需本机已 `source /opt/ros/humble/setup.bash`，stream 还要 overlay `install/xoip_ros2`）：

```bash
cmake -S . -B build
cmake --build build -j --target kmpc_node
```

仿真与真机 **分目录**（假植物体 vs 本机 RViz 映关节）：

| 目录 | 用途 |
|------|------|
| `sim/` | C9：`robot_tf_sim` 假植物体 + 控制器。`ros2 launch kmpc_cpp kmpc_cpp_sim.launch.py` |
| `real/` | C10：无假植物体。`kmpc_cpp_viz` 只看 `/joint_states`；`kmpc_cpp_real` 带控制（默认 `dq_scale=0.1`） |

```bash
# 仿真（GUI 仍 Python）
ros2 launch kmpc_cpp kmpc_cpp_sim.launch.py
# C9 无头出门
kmpc_cpp/sim/tools/run_closed_loop.sh
# 真机本机看姿态
ros2 launch kmpc_cpp kmpc_cpp_viz.launch.py
```

未 `colcon install` 时也可直接跑 `./build/kmpc_node`。核内 `ctest` 不 spin ROS。真机上传包不含 `sim/` 与 `build/`，见根目录 `thor_kmpc/`。

## 进度

| 阶段 | 状态 |
|------|------|
| C0 仓库骨架 | 本目录 |
| C1 类型与配置 | `types` / `Config` / `JointVelocityLimits` |
| C3 约束盒（提前，不依赖 Pinocchio） | `box_dq` / `y_bounds` 两端点 / 腰锁 ε / Safety |
| C2 运动学 | Pinocchio FK / \(J\) / CoM / \(\mathrm{diag}(M)\)，三姿态金标 |
| C4 QP | `assemble_endpoints` + ProxQP；`qp_gold.json` 对拍 $P,g,y_0$ |
| C5 Tracker | `CartesianTracker`；`tracker_gold.json` 首拍 $P,g,dq$ |
| C6 Planner | `UpperPlanner` 状态机，无 QP |
| C7 金标回归 | `tracker_gold` / `planner_gold` / 1000-tick replay；L1+L2 时序门 |
| C8 ROS2 | `ros2/kmpc_node.cpp` |
| C9 仿真闭环 | `sim/` 假植物体；无头 `sim/tools/run_closed_loop.sh` |
| C10 真机 | `real/`：`/joint_states` → 本机 RViz；`dq_scale` 默认 0.1 |

分层金标 json 放 `tests/gold/`，Replay 放 `tests/replay/`，Python dump 脚本放 `tools/dump_gold/`。容差一律走 `NumericalPolicy`。
