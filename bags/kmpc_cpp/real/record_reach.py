#!/usr/bin/env python3
"""Record HOME → cartesian reach for phase + latency diagnosis.

Does not enable the robot. Start BEFORE enable. Ctrl+C to stop. Writes under
  /home/nvidia/workspaces/kmpc/logs/reach/<stamp>/
"""

from __future__ import annotations

import csv
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray, String

try:
    from xoip_ros2.msg import (
        JointMotionCommand,
        JointStreamCommand,
        RobotState,
        StreamStatus,
    )
except Exception:  # noqa: BLE001
    JointMotionCommand = None  # type: ignore[misc, assignment]
    JointStreamCommand = None  # type: ignore[misc, assignment]
    RobotState = None  # type: ignore[misc, assignment]
    StreamStatus = None  # type: ignore[misc, assignment]

JOINT_NAMES: List[str] = [
    'Chassis_Y_Joint1',
    'Chassis_Y_Joint2',
    'Chassis_Y_Joint3',
    'Chassis_Z_Joint4',
    'RightArm_Z_Joint1',
    'RightArm_Z_Joint2',
    'RightArm_Z_Joint3',
    'RightArm_Z_Joint4',
    'RightArm_Z_Joint5',
    'RightArm_Z_Joint6',
    'RightArm_Z_Joint7',
    'LeftArm_Z_Joint1',
    'LeftArm_Z_Joint2',
    'LeftArm_Z_Joint3',
    'LeftArm_Z_Joint4',
    'LeftArm_Z_Joint5',
    'LeftArm_Z_Joint6',
    'LeftArm_Z_Joint7',
]
N = 18
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
# robot_id → [begin, end) in JOINT_NAMES / Vec18
ROBOT_SLICE = {1: (4, 11), 2: (11, 18), 3: (0, 4)}
# |meas-cmd|_∞ below this → that motion packet has been reached.
ARRIVE_EPS_DEG = float(os.environ.get('KMPC_ARRIVE_EPS_DEG', '0.15'))
ARRIVE_TIMEOUT_S = float(os.environ.get('KMPC_ARRIVE_TIMEOUT_S', '1.5'))
XCORR_MAX_LAG = 50  # ticks × 20 ms = 1.0 s
XCORR_MIN_N = 40

_RE_BOOL = {
    'enable': re.compile(r'\benable=(True|False)\b'),
    'arrived': re.compile(r'\barrived=(True|False)\b'),
    'com_hold': re.compile(r'\bcom_hold=(True|False)\b'),
    'rise_hold': re.compile(r'\brise_hold=(True|False)\b'),
    'com_work_wall': re.compile(r'\bcom_work_wall=(True|False)\b'),
    'held': re.compile(r'\bheld=(True|False)\b'),
}
_RE_NUM = {
    'fail': re.compile(r'\bfail=(\d+)\b'),
    'solve_ms': re.compile(r'\bsolve_ms=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'),
    'eL': re.compile(r'\beL=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)m'),
    'eR': re.compile(r'\beR=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)m'),
    'e0': re.compile(r'\be0=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'),
    'pitch_deg': re.compile(r'\bpitch=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)deg'),
    'theta_des_deg': re.compile(r'\btheta_des=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)deg'),
    'c_des_x': re.compile(r'\bc_des_x=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'),
    'com_slack_x': re.compile(r'\bcom_slack_x=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'),
}
_RE_COM_XY = re.compile(
    r'\bcom_xy=\(([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?),'
    r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)'
)
_RE_MODE = re.compile(r'\bmode=(\S+)')
_RE_TRACK = re.compile(r'\btracking_state=(\S+)')


def _named18(msg: JointState) -> Optional[List[float]]:
    q = [float('nan')] * N
    if msg.name:
        named = {str(n): float(p) for n, p in zip(msg.name, msg.position)}
        hit = 0
        for i, name in enumerate(JOINT_NAMES):
            if name in named:
                q[i] = named[name]
                hit += 1
        return q if hit else None
    if len(msg.position) >= N:
        return [float(x) for x in msg.position[:N]]
    return None


def _stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def parse_status(raw: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        'cart': 1 if 'com_hold=' in raw else 0,
        'status': raw.replace('\n', ' ')[:400],
    }
    for key, cre in _RE_BOOL.items():
        m = cre.search(raw)
        out[key] = '' if m is None else int(m.group(1) == 'True')
    for key, cre in _RE_NUM.items():
        m = cre.search(raw)
        out[key] = '' if m is None else m.group(1)
    m = _RE_COM_XY.search(raw)
    out['com_x'] = '' if m is None else m.group(1)
    out['com_y'] = '' if m is None else m.group(2)
    m = _RE_MODE.search(raw)
    out['mode'] = '' if m is None else m.group(1)
    m = _RE_TRACK.search(raw)
    out['tracking_state'] = '' if m is None else m.group(1)
    return out


def xcorr_lag_ticks(
    meas_hist: List[List[float]], cmd_hist: List[List[float]], max_lag: int
) -> Optional[int]:
    n = min(len(meas_hist), len(cmd_hist))
    if n < XCORR_MIN_N:
        return None
    meas_hist = meas_hist[-n:]
    cmd_hist = cmd_hist[-n:]
    best_d = 0
    best_s = float('inf')
    max_lag = min(max_lag, n - 20)
    if max_lag < 0:
        return None
    for d in range(0, max_lag + 1):
        acc = 0.0
        count = 0
        for k in range(d, n):
            m = meas_hist[k]
            c = cmd_hist[k - d]
            for i in range(N):
                a = m[i]
                b = c[i]
                if math.isfinite(a) and math.isfinite(b):
                    diff = a - b
                    acc += diff * diff
                    count += 1
        if count < 20 * N:
            continue
        mse = acc / count
        if mse < best_s:
            best_s = mse
            best_d = d
    if not math.isfinite(best_s):
        return None
    return best_d


@dataclass
class PendingCmd:
    cmd_id: int
    robot_id: int
    t_pub_mono: float
    t_pub_ros: float
    t_recv_mono: float
    joints_rad: List[float]
    acc_time: float
    dec_time: float


class ReachRecorder(Node):
    def __init__(self, out_dir: str) -> None:
        super().__init__('kmpc_reach_recorder')
        self._t0 = time.monotonic()
        self._meas: Optional[List[float]] = None
        self._cmd: Optional[List[float]] = None
        self._dq: Optional[List[float]] = None
        self._ee = [float('nan')] * 6
        self._status_raw = ''
        self._parsed = parse_status('')
        self._enable: Optional[bool] = None
        self._rows = 0
        self._motion_t: Dict[int, Deque[float]] = {
            1: deque(), 2: deque(), 3: deque(),
        }
        self._motion_last: Dict[int, Optional[float]] = {1: None, 2: None, 3: None}
        self._motion_gap: Dict[int, float] = {1: float('nan'), 2: float('nan'), 3: float('nan')}
        self._stream_cmd_t: Dict[int, Deque[float]] = {
            1: deque(), 2: deque(), 3: deque(),
        }
        self._stream_cmd_last: Dict[int, Optional[float]] = {
            1: None, 2: None, 3: None,
        }
        self._stream_cmd_gap: Dict[int, float] = {
            1: float('nan'), 2: float('nan'), 3: float('nan'),
        }
        self._stale: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self._stream_mode: Dict[int, str] = {1: '', 2: '', 3: ''}
        self._paused: Dict[int, object] = {1: '', 2: '', 3: ''}
        self._move_state: Dict[int, object] = {1: '', 2: '', 3: ''}
        self._velocity: Dict[int, float] = {
            1: float('nan'), 2: float('nan'), 3: float('nan'),
        }
        self._speed_fb_max: Dict[int, float] = {
            1: float('nan'), 2: float('nan'), 3: float('nan'),
        }

        self._meas_ros = float('nan')
        self._cmd_ros = float('nan')
        self._meas_age_ms = float('nan')
        self._cmd_age_ms = float('nan')
        self._cmd_id = 0
        self._pending: Dict[int, List[PendingCmd]] = {1: [], 2: [], 3: []}
        self._last_lat_ms: Dict[int, float] = {
            1: float('nan'), 2: float('nan'), 3: float('nan'),
        }
        self._arrive_eps = ARRIVE_EPS_DEG * DEG2RAD
        self._hist_meas: Deque[List[float]] = deque(maxlen=150)
        self._hist_cmd: Deque[List[float]] = deque(maxlen=150)
        self._xcorr_lag_ms = float('nan')
        self._lat_n_arrived = 0
        self._lat_n_timeout = 0

        reach_path = os.path.join(out_dir, 'reach.csv')
        self._fp = open(reach_path, 'w', newline='', encoding='utf-8')
        self._w = csv.writer(self._fp)
        header = [
            't_s', 'enable', 'cart', 'arrived', 'com_hold', 'rise_hold',
            'com_work_wall', 'held', 'fail', 'solve_ms',
            'eL', 'eR', 'e0', 'eL_x', 'eL_y', 'eL_z', 'eR_x', 'eR_y', 'eR_z',
            'eL_xy', 'pitch_deg', 'theta_des_deg', 'c_des_x', 'com_x', 'com_y',
            'com_slack_x', 'mode', 'tracking_state',
            'motion_hz_right', 'motion_hz_left', 'motion_hz_waist',
            'motion_gap_ms_right', 'motion_gap_ms_left', 'motion_gap_ms_waist',
            'stream_cmd_hz_right', 'stream_cmd_hz_left', 'stream_cmd_hz_waist',
            'stream_cmd_gap_ms_right', 'stream_cmd_gap_ms_left',
            'stream_cmd_gap_ms_waist',
            'stream_stale_right', 'stream_stale_left', 'stream_stale_waist',
            'stream_mode_right', 'stream_mode_left', 'stream_mode_waist',
            'paused_right', 'paused_left', 'paused_waist',
            'move_state_right', 'move_state_left', 'move_state_waist',
            'velocity_right', 'velocity_left', 'velocity_waist',
            'speed_fb_max_right', 'speed_fb_max_left', 'speed_fb_max_waist',
            'meas_ros', 'cmd_ros', 'meas_age_ms', 'cmd_age_ms', 'xcorr_lag_ms',
            'lat_ms_right', 'lat_ms_left', 'lat_ms_waist',
            'pending_right', 'pending_left', 'pending_waist',
        ]
        for name in JOINT_NAMES:
            header += [f'{name}_meas', f'{name}_cmd', f'{name}_dq']
        header += ['err_inf_deg', 'status']
        self._w.writerow(header)

        self._ev_fp = open(
            os.path.join(out_dir, 'events.csv'), 'w', newline='', encoding='utf-8')
        self._ev = csv.writer(self._ev_fp)
        self._ev.writerow(['t_s', 'kind', 'detail'])

        self._mo_fp = open(
            os.path.join(out_dir, 'motion.csv'), 'w', newline='', encoding='utf-8')
        self._mo = csv.writer(self._mo_fp)
        self._mo.writerow([
            'cmd_id', 't_s', 't_ros', 'recv_ros', 'pub_age_ms',
            'robot_id', 'n_joints', 'acc_time', 'dec_time',
            'speed_ratio', 'dt_ms', 'joints_deg',
        ])

        self._st_fp = open(
            os.path.join(out_dir, 'stream.csv'), 'w', newline='', encoding='utf-8')
        self._st = csv.writer(self._st_fp)
        self._st.writerow([
            't_s', 't_ros', 'recv_ros', 'age_ms', 'robot_id', 'mode',
            'active', 'stale', 'last_send_ok', 'command_age_ms', 'command_count',
        ])

        self._sc_fp = open(
            os.path.join(out_dir, 'stream_cmd.csv'), 'w', newline='', encoding='utf-8')
        self._sc = csv.writer(self._sc_fp)
        self._sc.writerow([
            'cmd_id', 't_s', 't_ros', 'recv_ros', 'pub_age_ms',
            'robot_id', 'n_joints', 'dt_ms', 'joints_deg',
        ])

        self._lat_fp = open(
            os.path.join(out_dir, 'latency.csv'), 'w', newline='', encoding='utf-8')
        self._lat = csv.writer(self._lat_fp)
        self._lat.writerow([
            'cmd_id', 'robot_id', 'status',
            't_pub', 't_arrive', 'latency_ms', 'latency_ros_ms',
            'err_inf_deg', 't_pub_ros', 't_meas_ros',
        ])

        self.create_subscription(JointState, '/joint_states', self._on_js, 50)
        self.create_subscription(JointState, '/kmpc/debug/q', self._on_cmd, 50)
        self.create_subscription(Float64MultiArray, '/kmpc/debug/dq', self._on_dq, 20)
        self.create_subscription(
            Float64MultiArray, '/kmpc/debug/ee_error', self._on_ee, 20)
        self.create_subscription(String, '/kmpc/status', self._on_status, 10)
        self.create_subscription(Bool, '/kmpc/enable', self._on_en, 10)
        self.create_subscription(PoseStamped, '/kmpc/target_left', self._on_left, 10)
        self.create_subscription(PoseStamped, '/kmpc/target_right', self._on_right, 10)
        self.create_subscription(Empty, '/kmpc/clear_targets', self._on_clear, 10)
        self.create_subscription(JointState, '/kmpc/q_target', self._on_q_target, 10)
        if JointMotionCommand is not None:
            self.create_subscription(
                JointMotionCommand, '/xoip/robot/joint_motion_cmd',
                self._on_motion, 50)
        else:
            self.get_logger().warn('xoip JointMotionCommand missing; motion.csv empty')
        if JointStreamCommand is not None:
            self.create_subscription(
                JointStreamCommand, '/xoip/robot/joint_stream_cmd',
                self._on_stream_cmd, 50)
        else:
            self.get_logger().warn(
                'xoip JointStreamCommand missing; stream_cmd.csv empty')
        if StreamStatus is not None:
            self.create_subscription(
                StreamStatus, '/xoip/robot/stream_status', self._on_stream, 20)
        if RobotState is not None:
            for rid in (1, 2, 3):
                self.create_subscription(
                    RobotState, f'/xoip/robot_{rid}/state',
                    self._on_robot_state, 20)
        else:
            self.get_logger().warn('xoip RobotState missing; paused/velocity empty')
        self.create_timer(0.02, self._tick)
        self.create_timer(1.0, self._heartbeat)
        self.get_logger().info(
            f'recording → {out_dir}  arrive_eps={ARRIVE_EPS_DEG:.3f}deg  '
            f'timeout={ARRIVE_TIMEOUT_S:.1f}s')

    def _now(self) -> float:
        return time.monotonic() - self._t0

    def _ros_now(self) -> float:
        return _stamp_sec(self.get_clock().now().to_msg())

    def _age_ms(self, stamp_sec: float, recv_ros: float) -> float:
        if not math.isfinite(stamp_sec) or stamp_sec <= 1.0:
            return float('nan')
        return (recv_ros - stamp_sec) * 1000.0

    def _event(self, kind: str, detail: str) -> None:
        self._ev.writerow([f'{self._now():.4f}', kind, detail.replace('\n', ' ')[:300]])
        self._ev_fp.flush()
        self.get_logger().info(f'event {kind} {detail[:120]}')

    def _on_js(self, msg: JointState) -> None:
        q = _named18(msg)
        if q is None:
            return
        now = self._now()
        recv_ros = self._ros_now()
        self._meas = q
        self._meas_ros = _stamp_sec(msg.header.stamp)
        self._meas_age_ms = self._age_ms(self._meas_ros, recv_ros)
        self._match_pending(q, now, self._meas_ros)

    def _on_cmd(self, msg: JointState) -> None:
        q = _named18(msg)
        if q is None:
            return
        recv_ros = self._ros_now()
        self._cmd = q
        self._cmd_ros = _stamp_sec(msg.header.stamp)
        self._cmd_age_ms = self._age_ms(self._cmd_ros, recv_ros)

    def _on_dq(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= N:
            self._dq = [float(x) for x in msg.data[:N]]

    def _on_ee(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 6:
            self._ee = [float(x) for x in msg.data[:6]]

    def _on_status(self, msg: String) -> None:
        self._status_raw = str(msg.data)
        self._parsed = parse_status(self._status_raw)

    def _on_en(self, msg: Bool) -> None:
        self._enable = bool(msg.data)
        self._event('enable', f'data={int(self._enable)}')

    def _on_left(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._event(
            'target_left',
            f'frame={msg.header.frame_id} xyz=({p.x:.4f},{p.y:.4f},{p.z:.4f})')

    def _on_right(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._event(
            'target_right',
            f'frame={msg.header.frame_id} xyz=({p.x:.4f},{p.y:.4f},{p.z:.4f})')

    def _on_clear(self, _msg: Empty) -> None:
        self._event('clear_targets', '')

    def _on_q_target(self, msg: JointState) -> None:
        n = min(4, len(msg.position))
        head = ','.join(f'{float(x):.4f}' for x in msg.position[:n])
        self._event('q_target', f'n={len(msg.position)} waist=[{head}]')

    def _prune_hz(self, store: Dict[int, Deque[float]], now: float) -> None:
        for dq in store.values():
            while dq and now - dq[0] > 1.0:
                dq.popleft()

    def _track_dt(
        self,
        last: Dict[int, Optional[float]],
        gap: Dict[int, float],
        ticks: Dict[int, Deque[float]],
        rid: int,
        now: float,
    ) -> float:
        prev = last.get(rid)
        dt_ms = float('nan') if prev is None else (now - prev) * 1000.0
        last[rid] = now
        gap[rid] = dt_ms
        if rid in ticks:
            ticks[rid].append(now)
        return dt_ms

    def _enqueue_pending(
        self, rid: int, cid: int, now: float, t_ros: float, joints_deg: List[float],
        acc_time: float, dec_time: float,
    ) -> None:
        if rid not in ROBOT_SLICE:
            return
        self._pending[rid].append(PendingCmd(
            cmd_id=cid,
            robot_id=rid,
            t_pub_mono=now,
            t_pub_ros=t_ros,
            t_recv_mono=now,
            joints_rad=[x * DEG2RAD for x in joints_deg],
            acc_time=acc_time,
            dec_time=dec_time,
        ))

    def _write_lat(
        self,
        cmd: PendingCmd,
        status: str,
        t_arrive: float,
        err_inf_deg: float,
        t_meas_ros: float,
    ) -> None:
        latency_ms = (t_arrive - cmd.t_pub_mono) * 1000.0
        latency_ros = float('nan')
        if math.isfinite(t_meas_ros) and t_meas_ros > 1.0 and cmd.t_pub_ros > 1.0:
            latency_ros = (t_meas_ros - cmd.t_pub_ros) * 1000.0
        self._lat.writerow([
            cmd.cmd_id, cmd.robot_id, status,
            f'{cmd.t_pub_mono:.4f}', f'{t_arrive:.4f}',
            f'{latency_ms:.2f}',
            '' if not math.isfinite(latency_ros) else f'{latency_ros:.2f}',
            f'{err_inf_deg:.4f}',
            '' if cmd.t_pub_ros <= 1.0 else f'{cmd.t_pub_ros:.6f}',
            '' if t_meas_ros <= 1.0 else f'{t_meas_ros:.6f}',
        ])
        self._lat_fp.flush()
        if status == 'arrived':
            self._last_lat_ms[cmd.robot_id] = latency_ms
            self._lat_n_arrived += 1
        elif status == 'timeout':
            self._lat_n_timeout += 1

    def _match_pending(
        self, meas: List[float], now: float, meas_ros: float
    ) -> None:
        eps = self._arrive_eps
        for rid, sl in ROBOT_SLICE.items():
            lo, hi = sl
            pending = self._pending[rid]
            if not pending:
                continue
            keep: List[PendingCmd] = []
            best_i = -1
            best_err = float('inf')
            errs: List[float] = []
            for i, cmd in enumerate(pending):
                n = min(hi - lo, len(cmd.joints_rad))
                err = 0.0
                ok = True
                for k in range(n):
                    m = meas[lo + k]
                    c = cmd.joints_rad[k]
                    if not (math.isfinite(m) and math.isfinite(c)):
                        ok = False
                        break
                    err = max(err, abs(m - c))
                if not ok:
                    errs.append(float('inf'))
                    continue
                errs.append(err)
                if err < best_err:
                    best_err = err
                    best_i = i
            if best_i >= 0 and best_err <= eps:
                for i, cmd in enumerate(pending):
                    if i < best_i:
                        self._write_lat(
                            cmd, 'passed', now, errs[i] * RAD2DEG, meas_ros)
                    elif i == best_i:
                        self._write_lat(
                            cmd, 'arrived', now, errs[i] * RAD2DEG, meas_ros)
                    else:
                        keep.append(cmd)
            else:
                for i, cmd in enumerate(pending):
                    if now - cmd.t_pub_mono >= ARRIVE_TIMEOUT_S:
                        self._write_lat(
                            cmd, 'timeout', now,
                            errs[i] * RAD2DEG if i < len(errs) else float('nan'),
                            meas_ros)
                    else:
                        keep.append(cmd)
            self._pending[rid] = keep[-80:]

    def _on_motion(self, msg) -> None:
        now = self._now()
        recv_ros = self._ros_now()
        t_ros = _stamp_sec(msg.header.stamp)
        rid = int(msg.robot_id)
        dt_ms = self._track_dt(
            self._motion_last, self._motion_gap, self._motion_t, rid, now)
        joints_deg = [float(x) for x in msg.joints]
        self._cmd_id += 1
        cid = self._cmd_id
        pub_age = self._age_ms(t_ros, recv_ros)
        self._mo.writerow([
            cid, f'{now:.4f}',
            '' if t_ros <= 1.0 else f'{t_ros:.6f}',
            f'{recv_ros:.6f}',
            '' if not math.isfinite(pub_age) else f'{pub_age:.2f}',
            rid, len(joints_deg),
            float(msg.acc_time), float(msg.dec_time), float(msg.specify_global_speed),
            '' if not math.isfinite(dt_ms) else f'{dt_ms:.2f}',
            ' '.join(f'{x:.4f}' for x in joints_deg),
        ])
        self._mo_fp.flush()
        self._enqueue_pending(
            rid, cid, now, t_ros, joints_deg,
            float(msg.acc_time), float(msg.dec_time))

    def _on_stream_cmd(self, msg) -> None:
        now = self._now()
        recv_ros = self._ros_now()
        t_ros = _stamp_sec(msg.header.stamp)
        rid = int(msg.robot_id)
        dt_ms = self._track_dt(
            self._stream_cmd_last, self._stream_cmd_gap, self._stream_cmd_t,
            rid, now)
        joints_deg = [float(x) for x in msg.joints]
        self._cmd_id += 1
        cid = self._cmd_id
        pub_age = self._age_ms(t_ros, recv_ros)
        self._sc.writerow([
            cid, f'{now:.4f}',
            '' if t_ros <= 1.0 else f'{t_ros:.6f}',
            f'{recv_ros:.6f}',
            '' if not math.isfinite(pub_age) else f'{pub_age:.2f}',
            rid, len(joints_deg),
            '' if not math.isfinite(dt_ms) else f'{dt_ms:.2f}',
            ' '.join(f'{x:.4f}' for x in joints_deg),
        ])
        self._sc_fp.flush()
        self._enqueue_pending(rid, cid, now, t_ros, joints_deg, 0.0, 0.0)

    def _on_robot_state(self, msg) -> None:
        rid = int(msg.robot_id)
        if rid not in self._paused:
            return
        self._paused[rid] = int(bool(msg.paused))
        self._move_state[rid] = int(msg.move_state)
        self._velocity[rid] = float(msg.velocity)
        fb = list(msg.speed_feedback) if msg.speed_feedback is not None else []
        if fb:
            self._speed_fb_max[rid] = max(abs(int(x)) for x in fb)
        else:
            self._speed_fb_max[rid] = float('nan')

    def _on_stream(self, msg) -> None:
        now = self._now()
        recv_ros = self._ros_now()
        t_ros = _stamp_sec(msg.header.stamp)
        rid = int(msg.robot_id)
        if rid in self._stale:
            self._stale[rid] = int(bool(msg.stale))
            self._stream_mode[rid] = str(msg.mode)
        age = self._age_ms(t_ros, recv_ros)
        self._st.writerow([
            f'{now:.4f}',
            '' if t_ros <= 1.0 else f'{t_ros:.6f}',
            f'{recv_ros:.6f}',
            '' if not math.isfinite(age) else f'{age:.2f}',
            rid, str(msg.mode), int(bool(msg.active)),
            int(bool(msg.stale)), int(bool(msg.last_send_ok)),
            int(msg.command_age_ms), int(msg.command_count),
        ])
        self._st_fp.flush()

    def _hz(self, store: Dict[int, Deque[float]], rid: int, now: float) -> str:
        self._prune_hz(store, now)
        n = len(store.get(rid, ()))
        return f'{n:.1f}' if n else ''

    def _tick(self) -> None:
        if self._meas is None:
            return
        now = self._now()
        cmd = self._cmd if self._cmd is not None else [float('nan')] * N
        dq = self._dq if self._dq is not None else [float('nan')] * N
        ee = self._ee
        el_xy = math.hypot(ee[0], ee[1]) if all(math.isfinite(x) for x in ee[:2]) else float('nan')
        p = self._parsed
        err_inf = 0.0
        if self._cmd is not None:
            self._hist_meas.append(list(self._meas))
            self._hist_cmd.append(list(cmd))
            if self._rows % 5 == 0:
                lag = xcorr_lag_ticks(
                    list(self._hist_meas), list(self._hist_cmd), XCORR_MAX_LAG)
                if lag is not None:
                    self._xcorr_lag_ms = lag * 20.0
        row = [
            f'{now:.4f}',
            '' if self._enable is None else int(self._enable),
            p.get('cart', ''),
            p.get('arrived', ''),
            p.get('com_hold', ''),
            p.get('rise_hold', ''),
            p.get('com_work_wall', ''),
            p.get('held', ''),
            p.get('fail', ''),
            p.get('solve_ms', ''),
            p.get('eL', ''),
            p.get('eR', ''),
            p.get('e0', ''),
            ee[0], ee[1], ee[2], ee[3], ee[4], ee[5],
            '' if not math.isfinite(el_xy) else f'{el_xy:.6f}',
            p.get('pitch_deg', ''),
            p.get('theta_des_deg', ''),
            p.get('c_des_x', ''),
            p.get('com_x', ''),
            p.get('com_y', ''),
            p.get('com_slack_x', ''),
            p.get('mode', ''),
            p.get('tracking_state', ''),
            self._hz(self._motion_t, 1, now), self._hz(self._motion_t, 2, now),
            self._hz(self._motion_t, 3, now),
            self._gap_ms(self._motion_gap, 1), self._gap_ms(self._motion_gap, 2),
            self._gap_ms(self._motion_gap, 3),
            self._hz(self._stream_cmd_t, 1, now),
            self._hz(self._stream_cmd_t, 2, now),
            self._hz(self._stream_cmd_t, 3, now),
            self._gap_ms(self._stream_cmd_gap, 1),
            self._gap_ms(self._stream_cmd_gap, 2),
            self._gap_ms(self._stream_cmd_gap, 3),
            self._stale.get(1, ''), self._stale.get(2, ''), self._stale.get(3, ''),
            self._stream_mode.get(1, ''), self._stream_mode.get(2, ''),
            self._stream_mode.get(3, ''),
            self._paused.get(1, ''), self._paused.get(2, ''),
            self._paused.get(3, ''),
            self._move_state.get(1, ''), self._move_state.get(2, ''),
            self._move_state.get(3, ''),
            self._fmt(self._velocity[1]), self._fmt(self._velocity[2]),
            self._fmt(self._velocity[3]),
            self._fmt(self._speed_fb_max[1]), self._fmt(self._speed_fb_max[2]),
            self._fmt(self._speed_fb_max[3]),
            '' if self._meas_ros <= 1.0 else f'{self._meas_ros:.6f}',
            '' if self._cmd_ros <= 1.0 else f'{self._cmd_ros:.6f}',
            self._fmt(self._meas_age_ms), self._fmt(self._cmd_age_ms),
            self._fmt(self._xcorr_lag_ms),
            self._fmt(self._last_lat_ms[1]),
            self._fmt(self._last_lat_ms[2]),
            self._fmt(self._last_lat_ms[3]),
            len(self._pending[1]), len(self._pending[2]), len(self._pending[3]),
        ]
        for i in range(N):
            m = self._meas[i]
            c = cmd[i]
            e = m - c if (math.isfinite(m) and math.isfinite(c)) else float('nan')
            if math.isfinite(e):
                err_inf = max(err_inf, abs(e))
            row += [m, c, dq[i]]
        row += [err_inf * RAD2DEG, p.get('status', '')]
        self._w.writerow(row)
        self._rows += 1
        if self._rows % 50 == 0:
            self._fp.flush()

    @staticmethod
    def _fmt(v: float) -> str:
        return '' if not math.isfinite(v) else f'{v:.2f}'

    def _gap_ms(self, gap_map: Dict[int, float], rid: int) -> str:
        gap = gap_map.get(rid, float('nan'))
        return '' if not math.isfinite(gap) else f'{gap:.1f}'

    def _heartbeat(self) -> None:
        if self._meas is None:
            self.get_logger().warn('waiting for /joint_states')
            return
        cart = self._parsed.get('cart', '')
        el = self._parsed.get('eL', '')
        hold = self._parsed.get('com_hold', '')
        pend = (
            len(self._pending[3]) + len(self._pending[2]) + len(self._pending[1])
        )
        self.get_logger().info(
            f'rows={self._rows} enable={self._enable} cart={cart} '
            f'com_hold={hold} eL={el} xcorr_lag={self._fmt(self._xcorr_lag_ms)}ms '
            f'arrive={self._lat_n_arrived} timeout={self._lat_n_timeout} '
            f'pending={pend}')

    def close(self) -> None:
        now = self._now()
        for rid, pending in self._pending.items():
            for cmd in pending:
                self._write_lat(cmd, 'open', now, float('nan'), float('nan'))
            pending.clear()
        for fp in (
            self._fp, self._ev_fp, self._mo_fp, self._st_fp, self._sc_fp,
            self._lat_fp,
        ):
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = os.environ.get(
        'KMPC_REACH_DIR',
        os.path.join('/home/nvidia/workspaces/kmpc/logs/reach', stamp))
    os.makedirs(out, exist_ok=True)
    note = os.path.join(out, 'README.txt')
    with open(note, 'w', encoding='utf-8') as f:
        f.write(
            'HOME ⇄ A (or cartesian reach) + latency / settle.\n'
            'Start this first. Then: clear_targets, enable true, wait HOME, '
            'q_target A (or target_left). Stop: enable false, Ctrl+C.\n'
            'reach.csv 50 Hz (meas/cmd, err_inf_deg, paused, velocity).\n'
            'motion.csv every joint_motion_cmd; stream_cmd.csv every '
            'joint_stream_cmd.\n'
            'latency.csv one row per motion/stream packet: arrived | passed | '
            'timeout | open.\n'
            f'arrive when |meas-cmd|_∞ < {ARRIVE_EPS_DEG} deg; timeout {ARRIVE_TIMEOUT_S}s.\n'
            'xcorr_lag_ms: meas vs /kmpc/debug/q, 20 ms ticks, max 1 s.\n'
            f'summarize: python3 /home/nvidia/workspaces/kmpc/kmpc_cpp/real/'
            f'summarize_reach.py {out}\n'
        )
    rclpy.init()
    node = ReachRecorder(out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    print(f'saved {out}', file=__import__('sys').stderr)


if __name__ == '__main__':
    main()
