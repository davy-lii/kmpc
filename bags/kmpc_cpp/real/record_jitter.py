#!/usr/bin/env python3
"""Record meas vs cmd for shake diagnosis. Does not enable or command the robot.

Start this BEFORE enable. Ctrl+C to stop. Writes CSV under
  /home/nvidia/workspaces/kmpc/logs/jitter/<stamp>/
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

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


def _stamp_s(msg: JointState) -> float:
    """Header stamp as epoch seconds; nan if unset (外部评审：时间对齐/新鲜度用)."""
    s = msg.header.stamp
    if s.sec == 0 and s.nanosec == 0:
        return float('nan')
    return float(s.sec) + float(s.nanosec) * 1e-9


class JitterRecorder(Node):
    def __init__(self, out_dir: str) -> None:
        super().__init__('kmpc_jitter_recorder')
        self._t0 = time.monotonic()
        self._meas: Optional[List[float]] = None
        self._cmd: Optional[List[float]] = None
        self._meas_stamp = float('nan')
        self._cmd_stamp = float('nan')
        self._dq: Optional[List[float]] = None
        self._status = ''
        self._enable: Optional[bool] = None
        self._prev_dq: Optional[List[float]] = None
        self._flips = 0
        self._rows = 0
        path = os.path.join(out_dir, 'jitter.csv')
        self._fp = open(path, 'w', newline='', encoding='utf-8')
        self._w = csv.writer(self._fp)
        header = ['t_s', 'enable', 'status']
        for name in JOINT_NAMES:
            header += [f'{name}_meas', f'{name}_cmd', f'{name}_err', f'{name}_dq']
        header += ['err_inf_deg', 'dq_sign_flips_cum', 'js_stamp_s', 'cmd_stamp_s']
        self._w.writerow(header)
        self.create_subscription(JointState, '/joint_states', self._on_js, 20)
        self.create_subscription(JointState, '/kmpc/debug/q', self._on_cmd, 20)
        self.create_subscription(Float64MultiArray, '/kmpc/debug/dq', self._on_dq, 20)
        self.create_subscription(String, '/kmpc/status', self._on_status, 10)
        self.create_subscription(Bool, '/kmpc/enable', self._on_en, 10)
        self.create_timer(0.02, self._tick)
        self.create_timer(1.0, self._heartbeat)
        self.get_logger().info(f'recording → {path}  (enable the robot in another terminal)')

    def _on_js(self, msg: JointState) -> None:
        q = _named18(msg)
        if q is not None:
            self._meas = q
            self._meas_stamp = _stamp_s(msg)

    def _on_cmd(self, msg: JointState) -> None:
        q = _named18(msg)
        if q is not None:
            self._cmd = q
            self._cmd_stamp = _stamp_s(msg)

    def _on_dq(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= N:
            dq = [float(x) for x in msg.data[:N]]
            if self._prev_dq is not None:
                for a, b in zip(self._prev_dq, dq):
                    if a * b < 0.0 and abs(a) > 1e-6 and abs(b) > 1e-6:
                        self._flips += 1
            self._prev_dq = dq
            self._dq = dq

    def _on_status(self, msg: String) -> None:
        self._status = str(msg.data).replace('\n', ' ')[:240]

    def _on_en(self, msg: Bool) -> None:
        self._enable = bool(msg.data)
        self.get_logger().info(f'enable={self._enable}')

    def _tick(self) -> None:
        if self._meas is None:
            return
        t = time.monotonic() - self._t0
        cmd = self._cmd if self._cmd is not None else [float('nan')] * N
        dq = self._dq if self._dq is not None else [float('nan')] * N
        err_inf = 0.0
        row = [f'{t:.4f}', '' if self._enable is None else int(self._enable), self._status]
        for i in range(N):
            m = self._meas[i]
            c = cmd[i]
            e = m - c if (math.isfinite(m) and math.isfinite(c)) else float('nan')
            if math.isfinite(e):
                err_inf = max(err_inf, abs(e))
            row += [m, c, e, dq[i]]
        row += [err_inf * RAD2DEG, self._flips,
                f'{self._meas_stamp:.6f}', f'{self._cmd_stamp:.6f}']
        self._w.writerow(row)
        self._rows += 1
        if self._rows % 50 == 0:
            self._fp.flush()

    def _heartbeat(self) -> None:
        if self._meas is None:
            self.get_logger().warn('waiting for /joint_states')
            return
        cmd_ok = self._cmd is not None
        err = 0.0
        if cmd_ok:
            for m, c in zip(self._meas, self._cmd):
                if math.isfinite(m) and math.isfinite(c):
                    err = max(err, abs(m - c))
        self.get_logger().info(
            f'rows={self._rows} enable={self._enable} cmd={"yes" if cmd_ok else "no"} '
            f'|meas-cmd|_∞={err * RAD2DEG:.2f}deg dq_sign_flips={self._flips}')

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = os.environ.get(
        'KMPC_JITTER_DIR',
        os.path.join(
            '/home/nvidia/workspaces/kmpc/logs/jitter', stamp))
    os.makedirs(out, exist_ok=True)
    note = os.path.join(out, 'README.txt')
    with open(note, 'w', encoding='utf-8') as f:
        f.write(
            'Start this first, then enable in another window.\n'
            'Ctrl+C stops. jitter.csv: meas=joint_states, cmd=/kmpc/debug/q.\n'
            'err_inf_deg = max |meas-cmd| that tick. dq_sign_flips_cum counts '
            'per-joint dq sign changes (shake proxy).\n'
            'js_stamp_s/cmd_stamp_s = header stamps of latest meas/cmd message\n'
            '(epoch s, nan if unset); use them to time-align err and to check\n'
            'feedback freshness (repeated values under 50 Hz sampling).\n'
        )
    rclpy.init()
    node = JitterRecorder(out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    print(f'saved {out}/jitter.csv', file=sys.stderr)


if __name__ == '__main__':
    main()
