# 交接：fc7d3b0 代码审查修复 → 给 ChatGPT

**日期：** 2026-11-14  
**仓库：** https://github.com/davy-lii/kmpc  
**审查/开发基准：** `fc7d3b0e257b0fbc0e6ade50ea979e9dcb29494a`  
**本地权威树：** `/home/davy/03/home/nvidia/workspaces/kmpc/`  
**硬约束：** **禁止真机发布、禁止启机、禁止下发动作**；离线硬门全部通过后才可申请一次真机验证。

---

## 0. 当前事实（请勿误判为已完成）

| 项 | 状态 |
| --- | --- |
| unified_multistage_v2 已合入并 push 的基线 | **是** — HEAD = `fc7d3b0`（与 origin/master 对齐） |
| 本审查指出的 P0/P1 **生产执行链缺陷修复** | **否 — 尚未改码** |
| 子代理模型统一 `auto`（Cursor/Codex 调度配置） | **是** — 仅本地 `/home/davy/03` 配置，不在 kmpc Git |
| 真机发布 / 启机 / 动作 | **禁止** |

上一轮 Final v2 交付留痕：`改码留痕_20261114_final_v2_delivery.md`、`交付_KMPC_Final_v2_20261114.md`。  
那是 **v2 上线基线**，不是本审查修复的完成声明。

---

## 1. ChatGPT 本轮目标

修复已发现的生产执行链缺陷，**保留统一多阶段规划**；完成真实 50/250Hz 端到端离线验证后，才申请真机测试。  
**不能只把单测刷绿。**

GitHub 副本（同内容）：`bags/kmpc_cpp/docs/handoff_fc7d3b0_review_fix_20261114.md`

---

## 2. 待修清单（P0 → P1）

### P0-1 重构 50Hz→250Hz 真实轨迹执行
- 文件：`ros2/kmpc_node.cpp`、`src/tracker/trajectory_follower.cpp`、`src/tracker/safety_supervisor.cpp` + 对应 `include/kmpc/`
- 禁止：把 `tick.q_follow` 提前写入 `interpolator_.set_q_ref()`，再让 250Hz 用 `q_ref()` 当上一条命令
- 要求：轨迹包按单调时钟提供 `q(t),dq(t),ddq(t)`；250Hz 按实际发送时刻采样；Supervisor 自持 `q_last_send`；步长/acc/jerk 用相邻发送记录；50Hz 只做监控/状态/有界纠偏（纠偏经 250Hz）；禁止命令伪装实测；用真实 `/joint_states` 快照+接收时间；50/250 锁或双缓冲；安全拒绝后不得继续推进旧任务

### P0-2 SafetySupervisor 测量与认证输入
- 删除 `q_meas=raw`、`meas_timestamp_s=now_s` 代理写法
- 真实测量与命令分离；新鲜度、跟踪偏差、硬盒、俯仰、CoM、vel/acc
- generation/path_id/digest 与轨迹包比对，不能把包内值同时当期望
- Arrived 须最终安全闸接受 + 实测到位
- **不得降低 ±32°、J1 58°、单拍速度安全要求**

### P0-3 RECOVERING 与最终安全闸冲突
- Follower 可多拍俯仰恢复，Supervisor 见超 32° 就拒 → 需独立恢复合同
- 仅允许从已实测小幅超额进入；超额不得增长；不扩大硬盒；回限内后重新认证；无安全路径则 PROTECT_STOP
- 集成测：俯仰外超 `0.000254` 与 `0.000613` rad，50Hz→250Hz 贯通

### P0-4 真正可执行的时间轨迹
- `time_parameterizer` / `trajectory_validator` / `trajectory_follower` / `trajectory_package.hpp`
- C2 连续 `q(t)`；`validate_actual_q_of_t()` 验**实际执行**轨迹；未知 jerk 报软件设计值；验证后冻结，执行层不得偷改

### P0-5 多阶段停止与外部 ACK
- 实现 `StageBoundary.stop_required` 与 `wait_for_external_ack`（当前 Follower 未执行）
- 新 ACK 消息/服务（generation、stage_id、成功）；仅匹配当前任务/阶段；TIP 等待真实夹爪；无 ACK 不得自动 LIFT
- 涉及 `bags/kmpc_msgs/`

### P0-6 旧 Worker 结果误采用
- `motion_task_digest()` 含全部阶段关键字段
- 提交序号 + generation + digest + 模型/限位版本同时比对；真实 URDF 模型摘要；任务替换清旧结果；250Hz 不得推进已撤销任务

### P0-7 刹车与 arrived 生命周期
- 普通 Brake **不得**立刻 `hold_at()` 放弃制动
- 区分正常停止 / 重规划减速 / PROTECT_STOP
- `publish_arrived(true)` 以 250Hz 安全接受 + 实测到位为准

### P1-1 QP 实际反馈
- `try_qp_correction()` 须用真实 `q_meas`；正常优先跟已认证 `q(t)`；纠偏经 250Hz 连续输出

### P1-2 动作质量权重接入生产
- 真加载 `trajectory_quality.yaml`；记录生效值与版本；`calibrated` 与 yaml 一致
- 分离中立姿态 vs 起点保持；禁止 `q_neutral_=q0_` 混目标；禁止场景规则式调参

### P1-3 现场三段输入（上游）
- 改 `xoip_robot_control_service` 需用户明确允许；未在工作区则列阻塞，不得宣称三段前瞻已通
- 路径参考：`/home/davy/03/home/nvidia/workspaces/integrate/huacheng/equip_service/src/xoip_robot_control_service`
- 腰锁定后续任务须能清除

---

## 3. 离线硬门（A–I，全部通过才可申请真机）

A. 真实节点 50Hz+250Hz 交替发包集成测  
B. 三段 TIP 暂停 / ACK 错、丢、成功 / 反馈偏移状态机  
C. 旧 generation/digest/起点拒绝  
D. 俯仰恢复 Follower+Supervisor 集成  
E. 250Hz 最终指令统计（vel/acc/jerk/换向/突跳/TCP/CoM/限位）  
F. 真实 r40 回放；标明 18D 真实 vs 合成  
G. Jetson 规划性能 5/15/24cm 与 2/3 段 P50/P95/P99（争 10ms，不砍安全）  
H. MotionSequence + 容器消息加载；旧业务不因缺消息挂  
I. 针对性单测 + 完整离线链，无新回归  

---

## 4. 本环境已完成的旁路事项

- 子代理全部 `model: auto`（见 `改码留痕_20261114_subagent_auto.md`）
- 工作约定：主代理委派 Task **不要**再传 model 钉选

---

## 5. 建议开工顺序

1. 先读现生产链：`kmpc_node` 50/250 交接、`SafetySupervisorInput` 构造、`admit_package`、Brake/Arrived  
2. P0-1 + P0-2 + P0-7（发送真相与安全闸）  
3. P0-4（可执行 C2 时间律）  
4. P0-3（恢复合同）  
5. P0-5 / P0-6（ACK 与旧结果）  
6. P1-1 / P1-2；P1-3 仅在获准改上游后  
7. 硬门 A–I → 写 `改码留痕_*.md` + 交付报告 → **再**申请真机（仍走 `kmpc-release`）

---

## 6. 给 ChatGPT 的克隆/定位

```bash
git clone git@github.com:davy-lii/kmpc.git
cd kmpc && git checkout fc7d3b0e257b0fbc0e6ade50ea979e9dcb29494a
# 或在最新 master（应仍为 fc7d3b0）上开分支修
```

本地 davy 机权威副本：`/home/davy/03/home/nvidia/workspaces/kmpc`  
Issue 目录：`/home/davy/03/home/nvidia/ljw/issues/20261113_r40_dq03右手抓停_dq08左手抓停/`
