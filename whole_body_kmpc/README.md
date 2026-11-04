# whole_body_kmpc

轻载轮臂 **腰 + 双臂** 全身运动学 MPC。独立 ROS 2 工作空间，执行口与真机相同：

`/xoip/robot/joint_stream_cmd`（**度**，按 `robot_id` 分包：1 右臂 / 2 左臂 / 3 腰）。

内部计算一律 **弧度**。头（robot_id=4）不进优化。

末端 frame：手臂法兰 `Link7_L` / `Link7_R`，跟踪点是法兰原点沿 **−Y 往前 12 cm**（工具/手指方向，相对 `base_link`）。  
感知点如果已经变到 `base_link`，直接发给 `/kmpc/target_left` 或 `/kmpc/target_right` 即可。

---

## 1. 终端要 source 什么

不要 `source huacheng_sdk8.11`（那是机械臂 aarch64，x86 上会把 `xoip_ros2` 搞挂）。

每个新终端：

```bash
source /home/davy/dual_arm_mpc_papers/whole_body_kmpc/setup_kmpc.sh
```

第一次还没编译时：

```bash
cd /home/davy/dual_arm_mpc_papers/whole_body_kmpc
source /opt/ros/humble/setup.bash
source /home/davy/dual_arm_mpc_papers/robot_tf/setup_xoip.sh
colcon build --symlink-install
source setup_kmpc.sh
```

`setup_kmpc.sh` = Humble + `robot_tf`（仿真 / `xoip_ros2`）+ 本仓库。成功时应打印 `whole_body_kmpc at .../install/whole_body_kmpc`。若仍提示 package not found，在同一终端再执行一次 `source setup_kmpc.sh`（不要只用 `source install/setup.bash`）。

看 RViz + **关节设置窗口**（滑条改姿态，再回 HOME、再抓球）：

```bash
cd /home/davy/dual_arm_mpc_papers/whole_body_kmpc
source setup_kmpc.sh
ros2 launch whole_body_kmpc kmpc_sim.launch.py rviz:=true demo:=false
```

弹出的窗口和 `robot_tf` 里的 `joint_state_publisher_gui` 同一套交互：每个关节可拉滑条，也可在框里输入**角度（度）**。

1. 拖滑条 / 输入数值 → 机器人跟过去（先停红球，走关节 QP）。
2. 点 **回 HOME** → 平滑回到准备姿态（略蹲、肩前探 35°、肘 75°，不是直腰+肘 90°）。
3. 点 **抓目标点** → 红球出现，左手去够。

不要窗口、启动约 2 秒后直接够球：加 `pose_gui:=false`。

**先标定再够球（无窗口一键）**：

```bash
ros2 launch whole_body_kmpc kmpc_sim.launch.py rviz:=true demo:=false pose_gui:=false calibrate_first:=true
```

若刚改过代码：`colcon build --symlink-install --packages-select whole_body_kmpc`，再 `source setup_kmpc.sh`（必须在 `whole_body_kmpc` 目录，不要在仓库根目录）。

启动后可**热改**球位置（不用重启），左手灵巧手掌心会跟着走：

- RViz 工具栏选 **Interact**（最左边箭头，不是转视角），拖红球或它旁边的轴。
- 或另开终端：

```bash
source /home/davy/dual_arm_mpc_papers/whole_body_kmpc/setup_kmpc.sh
ros2 topic pub --once /kmpc/grasp_xyz geometry_msgs/msg/Point "{x: 0.45, y: 0.20, z: 1.15}"
```

也可以 `ros2 param set /send_ee_target x 0.45`（y/z 同理）。坐标是 `base_link`、单位米。

---

## 2. 两个工具分别干什么（容易混）

| 命令 | 是不是控制器 | 干什么 |
|------|----------------|--------|
| `ros2 run robot_tf_sim tf_probe` | 否 | **读 TF 的显示器**。每秒打印 `base_link → Link7_L / Link7_R / camera_head_1 / …` 的 xyz 和 rpy。用来看左手现在在哪、目标发完后有没有靠近。 |
| `ros2 run whole_body_kmpc p0_check` | 否 | **通路冒烟测试**。只检查右臂 `joint_arm_R_2` 是否从 0 转到约 -90°（说明 `joint_stream_cmd` 能驱动仿真）。**不能**用来测「左手去感知点」。 |

它们都**不会**发目标、也不会控制机器人。仿真 + MPC 要先用 launch 拉起来。

笛卡尔到位请用：

```bash
ros2 run whole_body_kmpc reach_check --ros-args -p side:=left
```

---

## 3. 启动仿真（RViz 里能看见机器人）

**终端 A：**

```bash
source /home/davy/dual_arm_mpc_papers/whole_body_kmpc/setup_kmpc.sh
ros2 launch whole_body_kmpc kmpc_sim.launch.py rviz:=true demo:=false
```

默认 `auto_enable:=true`：仿真从抓取准备姿态起步（腰略蹲 J1=+8°/J2=-12°，肩前探 J2=-35°，肘 J4=-75°）。  

日志里应有：

```text
Subscribed /xoip/robot/joint_stream_cmd (xoip_ros2 JointStreamCommand, deg)
Cartesian tracker ready (URDF FK/J)
```

`demo:=true` 会让腰 yaw 来回摆，只适合看关节通路；**测感知到达时务必 `demo:=false`**，否则目标和摆腰抢控制。

等几秒手臂抬到位。RViz Fixed Frame 选 `base_link`。

---

## 4. 模拟感知：左手末端去 base_link 下的一个点

场景：相机 `camera_head_1` 检出一点，**已经用 `robot_tf_transform` 变到 `base_link`**。现在只要让左手灵巧手（`l_hand_link` 掌心 TCP）去这个 xyz（米）。姿态这一版只跟位置，四元数填单位即可。

### 4.1 先看左手现在在哪

**终端 B：**

```bash
source /home/davy/dual_arm_mpc_papers/whole_body_kmpc/setup_kmpc.sh
ros2 run robot_tf_sim tf_probe
```

或只看左手：

```bash
ros2 run tf2_ros tf2_echo base_link l_hand_link
```

名义构型（准备姿态）附近，左手掌心大约：

```text
l_hand_link TCP ≈ (0.65, 0.27, 1.19) m
```

（以你终端里 tf_probe 打印的为准。）

### 4.2 发一个假感知点（推荐）

目标要比当前末端近一点（先 5–10 cm），确认方向对了再拉远。

**终端 C：**

```bash
source /home/davy/dual_arm_mpc_papers/whole_body_kmpc/setup_kmpc.sh

# 左手去 base_link 下 (0.50, 0.12, 1.18) m —— 相对 q_pref 往左前、略降低
ros2 run whole_body_kmpc send_ee_target --ros-args \
  -p side:=left \
  -p x:=0.50 -p y:=0.12 -p z:=1.18 \
  -p frame_id:=base_link
```

这个节点会 **10 Hz 重复**发：

- `/kmpc/target_left`（`PoseStamped`，`frame_id=base_link`）
- `/kmpc/enable = true`
- `/kmpc/mode = single_ee`

RViz 里左手应慢慢跟过去；腰可以小幅帮忙，右臂尽量留在 `q_pref`。Ctrl+C 停发后，最后一次目标仍留在 MPC 里，手臂会停在附近。

若已经用默认 `grasp`（或显式 `grasp:=true`）启动，**不要再开第二个** `send_ee_target`（会抢话题）。改位置用热改：RViz 拖球，或 `ros2 topic pub --once /kmpc/grasp_xyz ...`。

换成你的感知坐标，只改 `x/y/z`（米，`base_link`）。

### 4.3 不用辅助节点、自己 pub（等价）

```bash
ros2 topic pub /kmpc/enable std_msgs/msg/Bool "{data: true}" -r 1

ros2 topic pub /kmpc/mode std_msgs/msg/String "{data: single_ee}" -r 1

# 把 0.50 0.12 1.18 换成感知给出的 base_link xyz
ros2 topic pub /kmpc/target_left geometry_msgs/msg/PoseStamped \
"{
  header: {frame_id: 'base_link'},
  pose: {
    position: {x: 0.50, y: 0.12, z: 1.18},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}" -r 10
```

`-r 10` 要带着：MPC 晚启动也能收到；`--once` 容易丢。

右臂同理，话题改成 `/kmpc/target_right`，`send_ee_target` 加 `-p side:=right`。

### 4.4 看有没有到位

继续开着 tf_probe，看 `l_hand_link` 的 xyz 是否靠近目标（掌心再沿手指约 8 cm 才是跟踪点）。

或：

```bash
ros2 topic echo /kmpc/debug/status
# 里有 eL=...m （左手位置误差）

ros2 run whole_body_kmpc reach_check --ros-args -p side:=left -p tol_m:=0.02
```

`reach_check` 对比左手掌心 TCP 和 `/kmpc/target_left`，误差 < 2 cm 打印 PASS。

---

## 5. 建议的联调顺序

1. **通路**：`kmpc_sim.launch.py demo:=true`，再 `p0_check`。右臂动了 → stream 通。
2. **看 TF**：`demo:=false` 重启，`tf_probe` 能打出 `l_hand_link`。
3. **近处笛卡尔**：当前末端附近 ±8 cm 发 `/kmpc/target_left`，RViz 跟手。
4. **假感知点**：换成真实（已在 `base_link`）的零件/箱点。够不着时腰会补一点；再远会顶限位，误差掉不下去。

---

## 6. 话题一览

| 话题 | 方向 | 单位 / 说明 |
|------|------|-------------|
| `/kmpc/target_left` | 订 | `PoseStamped`，**米**，`frame_id=base_link`。左手灵巧手掌心 TCP。 |
| `/kmpc/target_right` | 订 | 同上，右手灵巧手 `r_hand_link` 掌心 TCP。 |
| `/kmpc/enable` | 订 | `false` 停止下发 stream。 |
| `/kmpc/mode` | 订 | `single_ee` / `dual_carry` / `hold`（相对约束 dual_carry 尚未启用）。 |
| `/kmpc/q_target` | 订 | 关节目标（弧度）。有笛卡尔目标时优先跟笛卡尔。 |
| `/joint_states` | 订 | 仿真反馈，弧度，URDF 名。 |
| `/xoip/robot/joint_stream_cmd` | 发 | **度**；每拍腰/右/左各一条。 |
| `/kmpc/debug/ee_error` | 发 | `[eLx,eLy,eLz, eRx,eRy,eRz]` 米。 |
| `/kmpc/debug/status` | 发 | enable、eL/eR、求解时间、`tracking_state`、`waist_locked`。 |
| `/kmpc/status` | 发 | 同上（精简版，给上层 Planner 看状态）。 |
| `/kmpc/tracking_state` | 订 | `CALIBRATING` / `TRACKING`（Planner 层）。 |
| `/kmpc/waist_locked` | 订 | `true` 焊死腰 4 轴。 |
| `/kmpc/q_calib` | 订 | 标定目标关节角（弧度，`JointState`）。 |
| `/kmpc/grasp_start` | 订 | 标定完成后由 `calibrate_then_grasp` 发出，红球才开始发。 |
| `/kmpc/debug/q` | 发 | 当前命令，弧度。 |
| `/kmpc/debug/collision` | 发 | 最近碰撞对、`dmin`、是否硬约束/缩步 |
| `/kmpc/debug/collision_markers` | 发 | 胶囊可视化（绿=安全，黄=警告，红=过近） |

内部 18 维顺序：

```text
q[0:4]   腰   robot_id=3
q[4:11]  右臂 robot_id=1
q[11:18] 左臂 robot_id=2
```

---

## 7. Launch 参数

```bash
ros2 launch whole_body_kmpc kmpc_sim.launch.py \
  rviz:=true \
  demo:=false \
  auto_enable:=true \
  use_sim:=true
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `rviz` | true | 开 RViz |
| `demo` | **false** | true=摆腰（关节冒烟）；笛卡尔测试用 false |
| `auto_enable` | true | 启动就下发（先去 `q_pref`） |
| `use_sim` | true | 同时起 `robot_tf_sim`。仿真已在跑则 `use_sim:=false` |
| `pose_gui` | **true** | 弹出关节滑条窗口；此时红球等你点「抓目标点」 |
| `calibrate_first` | false | true=先 CALIBRATING 回 HOME，再自动红球够球（建议 `pose_gui:=false`） |

仿真已启动、只起 MPC：

```bash
ros2 launch whole_body_kmpc kmpc_sim.launch.py use_sim:=false rviz:=false demo:=false
```

---

## 8. 控制器在做什么（现在）

没有 `/kmpc/target_*` 时：关节空间对角 QP，跟踪 `q_pref`（或 `/kmpc/q_target`）。

有左手/右手 `PoseStamped` 时：冻结雅可比一步 QP（位置）

\[
\min_{\Delta q}\ \|J\Delta q - e\|_Q^2 + \|q-q^{\mathrm{pref}}\|_W^2 + \|\Delta q\|_R^2 + \|\Delta q-\Delta q_{\mathrm{prev}}\|_S^2
\]

硬限位 + 速度限。单臂时另一臂 \(W\) 加大，少乱动。对角 \(W=w_{\mathrm{joint}}\)、\(R=r_{\mathrm{arm}}\) 全轴相同。  
笛卡尔时手臂名义取当前 \(q_0\)。目标 z 低于 `z_fold_height` 时允许折腰（高处锁腰）；\(W_o\) 名义是手臂盒子盖不住的剩余高度，不是折腰表。手够到球高则停腰。贴球 ARRIVED 后腰也停。目标抬高后起身回 `q_pref`，贴球同样停腰。

自碰（臂–臂 / 臂–身 / 臂–头）：URDF 连杆上挂胶囊/球，默认软代价；距离小于 `d_warn` 时加线性化硬约束；SafetyFilter 对 `q_cmd` 再查一遍，过近则缩步或保持。RViz 里半透明胶囊：绿安全、黄接近、红过近。

开关在 `config/kmpc_params.yaml` 的 `collision_enable`（启动默认）。**运行中不用重启**，热关/开：

```bash
ros2 param set /kmpc_node collision_enable false
ros2 param set /kmpc_node collision_enable true
```

关掉后 QP 不再加自碰项，SafetyFilter 也不再缩步，RViz 胶囊会清掉。`d_safe` / `d_warn` / `w_col` 同样可以 `ros2 param set` 热改。改 YAML 文件本身只影响下次启动。

本机未装 Pinocchio/OSQP，FK/J 用同一份 URDF 解析。时域 \(N>1\)、姿态跟踪、搬箱相对约束还在后面。

---

## 9. 单测

```bash
cd /home/davy/dual_arm_mpc_papers/whole_body_kmpc/src/whole_body_kmpc
python3 -m pytest test/ -q
```

---

## 10. 常见问题

**RViz 全是 No transform / 手臂不动**  
没 overlay 本仓库的 `xoip_ros2`，或误 source 了 aarch64 SDK。伪控制器日志必须有 `Subscribed /xoip/robot/joint_stream_cmd`。

**发了 target 手不动**  
1. `demo:=false` 重启。  
2. `/kmpc/enable` 为 true。  
3. `frame_id` 必须是 `base_link`，xyz 是 **米**（不是毫米）。  
4. `ros2 topic echo /kmpc/target_left` 能刷到。  
5. `eL` 很大且不再掉：点在工作空间外，先改到当前左手掌心附近。

**tf_probe 有 Link7、p0_check 也 PASS，但手不去我给的点**  
那只说明关节通路通。必须发 `/kmpc/target_left`；p0_check 不测笛卡尔。

**Ctrl+C 后 launch 报 RCLError shutdown**  
已在 `kmpc_node` 里规避；若仍看到，可以忽略，下一次 launch 即可。

**手动发 `ros2 topic pub --once /kmpc/tracking_state ...` 是啥？**  
这是给 **Planner 层**下命令：`CALIBRATING`=双臂用弹簧慢慢回到 `q_calib`（默认 HOME），且**暂时忽略红球**；`TRACKING`=恢复够球。必须 **`kmpc_node` 已在跑**且已 `colcon build` 过带 Planner 的版本；`--once` 只发一帧，丢了就再发一次。看效果：`ros2 topic echo /kmpc/status`。自动收敛后 Planner 会自己切 `TRACKING`，不必手切。仿真一键流程用 `calibrate_first:=true` 或 `ros2 service call /kmpc/calibrate_then_grasp std_srvs/srv/Trigger`。

**标定完再够球（launch 已在跑）**  
另开终端：`ros2 service call /kmpc/calibrate_then_grasp std_srvs/srv/Trigger`（需 launch 里 `send_ee_target` 的 `autostart:=false`，或直接用上面的 `calibrate_first:=true` launch）。
