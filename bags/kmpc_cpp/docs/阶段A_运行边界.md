# KMPC 运行边界（阶段 A）

日期：2026-11-06。这一阶段只加配置快照和分层诊断，不改控制输出。

## 谁拥有什么

| 模块 | 线程 / 回调 | 拥有的状态 | 不做什么 |
| --- | --- | --- | --- |
| `core` / `solver` / `planner` / `tracker` | 控制回调，或建线用的短命 worker | 数学量和直线诊断环 | 不写文件、不发 ROS、不自己起线程 |
| `KmpcNode` 50Hz `on_timer` | 单线程 executor | QP、安全、把路径交给 stream | 不在这拍往伺服口发 50Hz 台阶 |
| `kmpc_stream` | `stream_dedicated_thread=true` 时的独立线程 | `stream_mu_` 下的插补、`stream_raw_ref_`、`LineSmoother` | 不调用 tracker，不格式化 JSON |
| 诊断导出定时器 | 同一个 executor，10Hz | 读队列、写 JSONL | 不改 `q` / `dq` |
| SDK `StreamStatus` | executor 回调 | 只读回执 | 看不到控制器内部队列深度，也没有 goal generation |

`stream_mu_` 保护插补器和直线流状态。tracker 诊断环只在 executor 上读写（worker 写自己的副本，`adopt_line` 再拼回来）。250Hz 线程只写 `LayerDiagQueue`，导出定时器只读。

## 命令怎么走

```text
目标 /joint_states
    → TaskReceiver / StateAggregator
    → CartesianTracker：Build → FollowQp → FollowCommit
    → stream_on_line 提交 from/to（带 goal generation）
    → 250Hz：stream_raw_ref → LineSmoother → 限位/CoM/路径检查
    → publish_stream：MultiJointServoCommand
    → 反馈回到 /joint_states（保留 header stamp）
```

关节空间跟踪不经过 `LineSmoother`。这种拍只有 `final_send`，没有 `stream_raw_ref` 行。这是缺口，不是补出来的值。

## 时钟

- `mono_ns`、`recv_mono_ns`：`steady_clock` 从纪元起的纳秒。
- `source_stamp_ns`：消息头的 Unix 纳秒。没有头时是 `-1`。
- 节点 `now()` 只打在伺服消息上，不写入 `source_stamp_ns`。

对齐用 `run_id + goal_generation + layer + mono_ns`。不要用两个话题的行号硬配对。

## 诊断开关

默认 `line_diag_enable=false` 且 `layer_diag_path` 为空：热路径不进队列，不写文件。

两个都打开才导出。队列容量 8192，满了丢新样本，JSONL 里另起 `kind=overflow` 行，样本上的 `drops_before` 也会涨。导出关掉时，原来的 line diag 环和状态字符串保持原样。

## 配置快照

控制定时器创建前写一份 `effective_config`。YAML 里的限值和 QP 参数用来源文件路径标注；ROS 参数和节点默认值相同则标 `node_default`，否则标 `ros_override`（launch、YAML 覆盖和命令行在 rclcpp 这里不分家）。`git_sha` 在二进制里记为 `unrecorded`，发布记录里写提交号。

## 现在还看不到的接口

- IMU：节点没有订阅。
- SDK 发送队列深度：`StreamStatus` 没有。
- `StreamStatus` 没有 goal generation，所以这一层 `goal_generation=0`。

## 以后再拆

不要和这次一起做：拆 `KmpcNode`、改线程、改 QP/stream 频率、改 `dq_scale` / `line_dq_scale` / 限值。下一刀只在复采证明某一层先出错之后，改那一层。
