# unified_multistage_v2 production bypass map

Production single-arm Cartesian path (planner=`unified_multistage_v2`):

```text
MotionTask / MotionSequence
  → PlannerWorker::submit_snapshot (resident thread)
  → FastTrajectoryOptimizer → certified TrajectoryPackage
  → PlannerWorker::try_take
  → TrajectoryFollower::adopt / step (50 Hz)
  → SafetySupervisor::check (250 Hz)
  → send
```

## Deleted / bypassed old production call sites

| Old call site | Status under v2 |
| --- | --- |
| `ros2/kmpc_node.cpp` `std::async` + `CartesianTracker::prepare_line` | Bypassed when `planner=unified_multistage_v2` (single_ee / `use_v2_chain_`) |
| `CartesianTracker::line_path_qp` (QP-A/B/C posture reshape) | Bypassed for NewUnified follow (`v2_package_follow` → `v2_sample`); never used as pose selector |
| `direction_envelope_accept` sign hard gate on 250 Hz send | Bypassed on v2: only reachable under `!use_v2_chain_` inside legacy `stream_line_`; v2 uses `SafetySupervisor::check` |
| `line_pose_pref_enable` / fold / reverse_cache as pose selector | Ignored on v2 (logged once); candidate always `unified_multistage_v2` |
| `line_pitch_straighten_enable` / `line_qp_pitch_monotonic_enable` / `reverse_last_line_enable` | No decision power on v2 production chain |

## Still present (non-production / legacy)

- `line_path_qp` body remains for Legacy profile offline tests.
- `direction_envelope_accept` remains for Legacy offline_full_chain, unit contract tests, and the legacy `stream_line_` path when `!use_v2_chain_`.
- Dual-arm / joint-space / calibrating modes keep the prior node paths.

## rg proof (run from bags/kmpc_cpp)

```bash
# Production node must not schedule async prepare_line under v2 default
rg -n 'std::async|prepare_line\(' ros2/kmpc_node.cpp

# v2 follow must not enter line_path_qp as pose selector
rg -n 'v2_package_follow|line_path_qp\(' src/tracker/line_path.cpp

# direction_envelope_accept in the node must be behind !use_v2_chain_
rg -n 'direction_envelope_accept' ros2/kmpc_node.cpp
# expect: call site inside `if (... !use_v2_chain_ && ...)`; v2 branch uses SafetySupervisor

rg -n 'SafetySupervisor' ros2/kmpc_node.cpp
```
