# 真机（C10 起）

没有 `robot_tf_sim`。Thor 容器、上传包、上电顺序：[`Thor部署/真机Thor部署注意事项.md`](../../Thor部署/真机Thor部署注意事项.md)。驱动 yaml 250 Hz 配套：[`Thor部署/华成驱动yaml配套.md`](../../Thor部署/华成驱动yaml配套.md)。

机器人（或驱动）发 `/joint_states`（弧度、URDF 名）。`kmpc_node` 读同一份 `/joint_states`，规划仍 50 Hz。

每个 `kmpc` 终端先：

```bash
source /home/nvidia/workspaces/kmpc/bags/kmpc_cpp/real/setup_real.sh
```

`ROS_DOMAIN_ID` 必须和华成驱动一样（默认 20）。

## 启动驱动（先于 kmpc）

华成窗口，**不要**在这里发 `/kmpc/*`，也**不要**再启 `robot_brain`：

```bash
docker exec -it integrate-ros2-humble-latest bash
export ROS_DOMAIN_ID=20
cd /home/workspaces/integrate/huacheng/equip_service
source /home/workspaces/integrate/huacheng/huacheng_sdk/20260824/install/setup.bash
source install/setup.bash
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
ros2 launch xoip_robot_service xoip_robot_service_bringup.launch.py
```

必须看到 `ROS_DOMAIN_ID=20`。等到 `robot_id=1、2、3、4 servo enable success`，以及 `登录成功` / `脱离成功`。改过 `stream_control_rate` / `stream_command_timeout_ms` 必须 Ctrl+C 后重新 launch，参数只在启动时读。已经在容器里、提示符是 `root@nvidia:.../equip_service#` 的，不要再 `docker exec` 一层，直接 launch。

核对（另开已 source 的驱动窗口，不会动机器人）：

```bash
ros2 param get /xoip_robot_node stream_control_rate
```

本趟必须对上 `250.0`（驱动 yaml `stream_control_rate` 与 kmpc `stream_hz` 都是 250）。说明：[`Thor部署/华成驱动yaml配套.md`](../../Thor部署/华成驱动yaml配套.md)。

## RViz 在哪开

`rviz:=true` 会在 **你执行 launch 的那台机器** 上起 `rviz2`，不是自动弹到笔记本。

| 你在哪 launch 控制 | 默认 | 怎么看姿态 |
|--------------------|------|------------|
| Thor `kmpc` 容器 | `rviz:=false` | 容器里一般没显示器，不要开。笔记本另开可视化 |
| 有屏幕的电脑（笔记本） | 控制仍建议 `rviz:=false` | 另开 `kmpc_cpp_viz.launch.py`，和 Thor 同一 `ROS_DOMAIN_ID` |

```bash
# Thor：只跑控制器（默认已经不起 RViz）
ros2 launch kmpc_cpp kmpc_cpp_real.launch.py

# 笔记本：只看 /joint_states，不起控制器
ros2 launch kmpc_cpp kmpc_cpp_viz.launch.py
```

控制端硬要带 RViz（需要 DISPLAY）：`rviz:=true`。驱动已经发 TF 时控制端加 `use_rsp:=false`。

## 常用 launch

默认：`dq_scale=0.1`、`auto_enable=false`、`rviz=false`。下发口只有透传 `/xoip/robot/multi_joint_servo_cmd`，250 Hz，节点自己插补。

```bash
ros2 launch kmpc_cpp kmpc_cpp_real.launch.py

# 速度：dq_scale 只乘 Safety 之后的下发步，不改 Hessian。钉在 [0, 1]
ros2 launch kmpc_cpp kmpc_cpp_real.launch.py dq_scale:=0.3
```

启动日志要能对上：

`servo topic=/xoip/robot/multi_joint_servo_cmd publish=250Hz catch=0.020 accel_s=0.500 jerk_s=0.080 dq_scale=0.100`

`dq_scale` 没写就是 0.1。C9 仿真不过禁止上机。透传走 `PluseMultiToServo`，没有 `specify_global_speed`。

## 回 HOME / 去姿态 A

`enable true` **不是** HOME 话题，也**不会**自动回 HOME。没发过 `q_target`、也没有笛卡尔球时，原地钉住。要回家必须发 HOME 的 `/kmpc/q_target`。`q_target` **会粘住**：发过 A 之后只靠再 enable **不会**回 HOME，必须再发 HOME 的 `q_target`。笛卡尔目标会挡住关节 `q_target`，先 `clear_targets`。乱跳立刻 `enable false`。

先清笛卡尔（不动）：

```bash
ros2 topic pub --once /kmpc/clear_targets std_msgs/msg/Empty {}
```

人让开。开控制（不再自动回 HOME，原地钉住）：

```bash
ros2 topic pub --once /kmpc/enable std_msgs/msg/Bool "{data: true}"
```

回 HOME（enable 必须已是 true；发过 A 之后也用这条，不要手改数）：

```bash
ros2 topic pub --once /kmpc/q_target sensor_msgs/msg/JointState "{
  name: [Chassis_Y_Joint1, Chassis_Y_Joint2, Chassis_Y_Joint3, Chassis_Z_Joint4,
         RightArm_Z_Joint1, RightArm_Z_Joint2, RightArm_Z_Joint3, RightArm_Z_Joint4,
         RightArm_Z_Joint5, RightArm_Z_Joint6, RightArm_Z_Joint7,
         LeftArm_Z_Joint1, LeftArm_Z_Joint2, LeftArm_Z_Joint3, LeftArm_Z_Joint4,
         LeftArm_Z_Joint5, LeftArm_Z_Joint6, LeftArm_Z_Joint7],
  position: [0.13962634016, -0.20943951024, 0.0, 0.0,
             0.0, -0.61086523820, 0.0, -1.30899693899, 0.0, 0.0, 0.0,
             0.0, -0.61086523820, 0.0, -1.30899693899, 0.0, 0.0, 0.0]
}"
```

去姿态 A（enable 必须已是 true；不要手改数）：

```bash
ros2 topic pub --once /kmpc/q_target sensor_msgs/msg/JointState "{
  name: [Chassis_Y_Joint1, Chassis_Y_Joint2, Chassis_Y_Joint3, Chassis_Z_Joint4,
         RightArm_Z_Joint1, RightArm_Z_Joint2, RightArm_Z_Joint3, RightArm_Z_Joint4,
         RightArm_Z_Joint5, RightArm_Z_Joint6, RightArm_Z_Joint7,
         LeftArm_Z_Joint1, LeftArm_Z_Joint2, LeftArm_Z_Joint3, LeftArm_Z_Joint4,
         LeftArm_Z_Joint5, LeftArm_Z_Joint6, LeftArm_Z_Joint7],
  position: [0.18198548190439706, -0.2467721018742138, -0.23006930479421167, -5.235987801495996e-05,
             -0.1962971764444284, -0.23821999253418324, -0.1486845999095033, -1.435812542205631,
             -0.052691491049815194, -0.13435544501957516, -0.19270180910428053,
             0.2853613393589756, -0.6142686249130821, 0.10957176830967513, -1.3878683188912069,
             0.06625269847228378, 0.10386454751460808, 0.36779323247381673]
}"
```

停：

```bash
ros2 topic pub --once /kmpc/enable std_msgs/msg/Bool "{data: false}"
```

完整往返说明：仓库根目录 [`真机双姿态往返测试.md`](../../真机双姿态往返测试.md)。

抖动没有历史话题可查。下次 enable **之前**另开窗口：

```bash
python3 /home/nvidia/workspaces/kmpc/bags/kmpc_cpp/real/record_jitter.py
```

HOME → 左手够远点（分段）用 `record_reach.py`，流程见仓库根目录 [`真机笛卡尔够点.md`](../../真机笛卡尔够点.md)。
