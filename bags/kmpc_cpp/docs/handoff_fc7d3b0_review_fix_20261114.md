# 交接：fc7d3b0 审查修复进度（给 ChatGPT / 后续）

**日期：** 2026-11-14  
**仓库：** https://github.com/davy-lii/kmpc  
**起点：** `fc7d3b0`  
**硬约束：** 仍禁止真机发布/启机/下发，直到申请并获准。

## 状态（已改码）

P0-1…P0-7 与 P1-1…P1-3 **已在本地工作树实现**（见 `改码留痕_20261114_p0p1_exec_chain.md`）。  
Git：**可能尚未 commit/push**——以 `git status` 为准。

### 已通过离线测
`test_exec_chain_50_250`、`test_pitch_recovery_v2`、`test_trajectory_follower`、`test_stream_v2`、`test_multistage_v2`、`test_trajectory_adoption`

### 未完成 / 缺口
| 硬门 | 状态 |
| --- | --- |
| A 50/250 调度延迟等扩展场景 | 基础测有；延迟/换向/滞后可再加 |
| E 250Hz 指令统计报表 | 未单独落地 |
| F r40 18D 回放 | 未本轮重跑 |
| G Jetson P50/P95/P99 | 未本轮重跑 |
| H 容器内 kmpc_msgs+equip 联编 | 依赖真机/容器；msgs 已加 StageAck |
| I 全量针对性回归 | 仅跑了上表子集 |

### 上游
`foreign_code_patches/20261114_equip_motion_sequence_ack/`  
equip 需能找到已 install 的 `kmpc_msgs`（含 StageAck）。

## 关键生产行为变化
1. 250Hz：`sample_trajectory_at(package, t_now-t0)` → Supervisor（`q_last_send` + 真实 meas）
2. 50Hz：写 `V2ExecSlot`（offset/recovering/claim_time_done），不把 follow 写入 interpolator 当发送真值
3. Arrived：仅 `v2_safety_arrived_`（Supervisor 接受 claim + 实测到位）
4. TIP：Follower `WaitAck`；`/kmpc/stage_ack`

## 建议下一步
1. `git status` / commit（用户要求时）并 push  
2. 真机 kmpc 容器编节点 + msgs；equip 编上游补丁  
3. 补 F/G/E 报告后申请一次真机抓放  
4. 发布走 `kmpc-release`（auto）
