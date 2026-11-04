# 单臂路径任务协议

`/kmpc/segment_left`、`/kmpc/segment_right` 使用 `geometry_msgs/PoseArray`。
`header.frame_id` 必须为 `base_link`。`header.stamp` 是非零、严格递增的任务编号
（`sec * 1e9 + nanosec`），不是超时依据。上游可用 ROS 当前纳秒时间并取
`max(previous+1, now)`。KMPC 重启会清除已见编号；发布端重启不能倒退编号。

- 一个 pose：当前段目标，未提供级联检查。
- 两个 pose：当前段目标和真正的下一段目标；两者各自携带位置、工具姿态。
  节点按既有 `tool_to_flange_quat_xyzw` 规则转为法兰姿态。
- 零个 pose：只清除同侧、同编号的活动段。旧编号的清除消息不会删除新段。
- 新编号会原子替换另一侧目标、清除上一段冻结/abort，并安装本段级联目标。
- 同编号只更新当前位置/姿态输入，不替换级联目标，不解除已锁存 abort。

距离终点 8 cm 内冻结目标；同编号位置变化达到 50 mm 或姿态达到 15 度时停止。
小于门槛的漂移不改变冻结目标。停止后必须发新编号。旧 `PoseStamped` 接口保留，
但它依赖明确清除后再发布来开启新段，也不提供级联证明。

上游必须只接受匹配 `goal_generation` 的状态。`line=hold arrived=True` 表示命令
已保持且当前实测 FK 满足 3 mm/姿态到位门槛。收到成功或失败后，发布同编号零
pose 清除，再等待匹配编号的 `line=off segment_active=0`，才交接下一动作。
`line=off segment_active=1 line_planning=1` 是异步建路，不能视为清除确认。

# 算法边界

几何建路使用四个独立腰关节和活动臂的七个关节；限制腰部俯仰和，而非强制三个
俯仰关节等增量。采样间距最多 5 mm，最多 400 段；最终样点及关节插补验证 3 mm。
级联探针使用下一段实际位置、姿态，不能放松姿态来通过。候选搜索有界，无法证明
全局最优；候选均失败时在起步前停止。

每拍路径 QP 为每个预测阶段使用 11 个变量，通用端点 QP 不变。建路在独立运动学
实例的异步线程中完成；流线程保持快照参考，提交时再核对编号、目标和起点。
运行时还验证从实际流参考到新命令的首步限幅、3 mm 路径管和前进方向。
历史起点本身超出腰部俯仰限制时，报告 `start_pitch_outside`，不放宽配置硬过。

# 离线验证

`tests/replay/extract_recent.py` 只读最近五次 `notes.txt`、`jitter.csv`，从目标之前
的实测关节获取独立种子，并从已知固定点位预备→下一动作复制真实级联目标。
`replay_recent cases.tsv results.csv 119` 逐段使用记录种子；加 `chain` 参数时按
每个三步循环连续传递模拟关节状态。模型回放不包含伺服动力学、环境碰撞或负载。

`tests/replay/test_node_offline.py` 使用模拟反馈测试 ROS 状态流和任务生命周期。
只能在 `docker run --network none`、不挂载硬件设备、不启动真实驱动的容器内执行。
ROS_DOMAIN_ID 额外设为 213。验证输出写入独立验证目录，不替换运行中安装包。
