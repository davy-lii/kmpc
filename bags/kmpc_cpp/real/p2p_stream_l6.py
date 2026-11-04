#!/usr/bin/env python3
"""L6 右臂平移：一次规划的 rest-to-rest S 曲线，250 Hz 发 joint_stream_cmd。

不跑 kmpc。没有 20 ms QP 路点。限速与 L6/T0 相同：
  v̄ = dq_scale × 1.10 rad/s，ā = v̄/accel_s，j̄ = ā/jerk_s。

用法见问题整理/.../测试记录-P2P点到点L6.md
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import time
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

try:
    from xoip_ros2.msg import JointStreamCommand
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        '找不到 JointStreamCommand。先 source kmpc_cpp/real/setup_real.sh'
    ) from exc

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
# kmpc pack_stream：rid 1 右臂、2 左臂、3 腰
SLICE = {3: (0, 4), 1: (4, 11), 2: (11, 18)}

HOME = [
    0.13962634016, -0.20943951024, 0.0, 0.0,
    0.0, -0.61086523820, 0.0, -1.30899693899, 0.0, 0.0, 0.0,
    0.0, -0.61086523820, 0.0, -1.30899693899, 0.0, 0.0, 0.0,
]
SWEEP_R1 = -0.61086523820  # −35°


def named18(msg: JointState) -> Optional[List[float]]:
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


def plan_scurve(dist: float, v_max: float, a_max: float, j_max: float) -> Tuple[List[float], List[float]]:
    """Rest-to-rest jerk 段。返回 (durations, jerks)，与 eval 共用，避免短行程 duration/eval 不一致。"""
    d = abs(dist)
    if d < 1e-12:
        return [0.0], [0.0]
    tj = a_max / j_max
    # 4 段 ±j：d = 2 j Tj^3，a_pk = j Tj，v_pk = j Tj^2
    tj4 = (d / (2.0 * j_max)) ** (1.0 / 3.0)
    a_pk = j_max * tj4
    v_pk = j_max * tj4 * tj4
    if a_pk <= a_max + 1e-12 and v_pk <= v_max + 1e-12:
        return [tj4, tj4, tj4, tj4], [j_max, -j_max, -j_max, j_max]
    if v_max <= a_max * tj + 1e-15:
        tjv = math.sqrt(v_max / j_max)
        s_acc = v_max * tjv
        if 2.0 * s_acc >= d:
            return [tj4, tj4, tj4, tj4], [j_max, -j_max, -j_max, j_max]
        tv = (d - 2.0 * s_acc) / v_max
        return [tjv, tjv, max(tv, 0.0), tjv, tjv], [j_max, -j_max, 0.0, -j_max, j_max]
    # 6 段：到 a_max 但不到 v_max。d = a (ta+tj)(ta+2 tj)
    disc = tj * tj + 4.0 * d / a_max
    u = 0.5 * (-tj + math.sqrt(max(disc, 0.0)))
    ta = u - tj
    v6 = a_max * (ta + tj)
    if ta >= -1e-12 and v6 <= v_max + 1e-12:
        ta = max(ta, 0.0)
        return [tj, ta, tj, tj, ta, tj], [j_max, 0.0, -j_max, -j_max, 0.0, j_max]
    ta = v_max / a_max - tj
    t_acc = ta + 2.0 * tj
    s_acc = 0.5 * v_max * t_acc
    tv = max((d - 2.0 * s_acc) / v_max, 0.0)
    return [tj, ta, tj, tv, tj, ta, tj], [j_max, 0.0, -j_max, 0.0, -j_max, 0.0, j_max]


def s_curve_duration(dist: float, v_max: float, a_max: float, j_max: float) -> float:
    return float(sum(plan_scurve(dist, v_max, a_max, j_max)[0]))


def eval_scurve(t: float, dist: float, v_max: float, a_max: float, j_max: float) -> Tuple[float, float, str]:
    """从静止到静止。返回 (s, v, phase)，s 与 dist 同号，|s| 从 0 到 |dist|。"""
    sign = 1.0 if dist >= 0.0 else -1.0
    d = abs(dist)
    if d < 1e-12 or t <= 0.0:
        return 0.0, 0.0, 'hold'
    segs, js = plan_scurve(d, v_max, a_max, j_max)
    T = float(sum(segs))
    if t >= T:
        return dist, 0.0, 'done'
    return _integrate_jerk(t, segs, js, sign)


def _integrate_jerk(
    t: float, segs: Sequence[float], js: Sequence[float], sign: float
) -> Tuple[float, float, str]:
    names = ['j+', 'a+', 'j-', 'cruise', 'j-', 'a-', 'j+']
    a = 0.0
    v = 0.0
    s = 0.0
    remain = t
    phase = 'j+'
    for i, (dur, j) in enumerate(zip(segs, js)):
        phase = names[i] if i < len(names) else 'seg'
        if remain <= dur + 1e-15:
            dt = max(remain, 0.0)
            s += v * dt + 0.5 * a * dt * dt + j * dt ** 3 / 6.0
            v += a * dt + 0.5 * j * dt * dt
            a += j * dt
            return sign * s, sign * v, phase
        s += v * dur + 0.5 * a * dur * dur + j * dur ** 3 / 6.0
        v += a * dur + 0.5 * j * dur * dur
        a += j * dur
        remain -= dur
    return sign * s, 0.0, 'done'


def ptp_duration(q0: Sequence[float], q1: Sequence[float], v_max: float, a_max: float, j_max: float) -> float:
    T = 0.0
    for a, b in zip(q0, q1):
        T = max(T, s_curve_duration(abs(b - a), v_max, a_max, j_max))
    return T


def ptp_q(t: float, q0: Sequence[float], q1: Sequence[float], v_max: float, a_max: float, j_max: float) -> Tuple[List[float], float, str]:
    """各轴独立 rest-to-rest；几乎不动的轴保持 q0。返回 (q, v_inf, phase_of_longest)。"""
    q = list(q0)
    v_inf = 0.0
    phase = 'hold'
    longest = 0
    for i, (a, b) in enumerate(zip(q0, q1)):
        if abs(b - a) > abs(q1[longest] - q0[longest]):
            longest = i
    for i, (a, b) in enumerate(zip(q0, q1)):
        if abs(b - a) < 1e-8:
            q[i] = a
            continue
        s, v, ph = eval_scurve(t, b - a, v_max, a_max, j_max)
        q[i] = a + s
        v_inf = max(v_inf, abs(v))
        if i == longest:
            phase = ph
    return q, v_inf, phase


class P2PStream(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('p2p_stream_l6')
        self._args = args
        self._v_max = args.dq_scale * 1.10
        self._a_max = self._v_max / args.accel_s
        self._j_max = self._a_max / args.jerk_s
        self._dt = 1.0 / args.hz
        self._meas: Optional[List[float]] = None
        self._q = list(HOME)
        self._lock: Optional[List[float]] = None
        self._phase = 'wait_js'
        self._seg_t0 = time.monotonic()
        self._q_from = list(HOME)
        self._q_to = list(HOME)
        self._hold_s = args.hold_s
        self._catch_s = args.catch_s
        self._stop = False
        self._want_exit = False
        self._closed = False
        self._rows = 0
        r1 = -abs(SWEEP_R1) if args.sign < 0 else abs(SWEEP_R1)
        self._r1_home = 0.0
        self._r1_sweep = r1
        os.makedirs(args.out_dir, exist_ok=True)
        self._cmd_path = os.path.join(args.out_dir, 'p2p_cmd_250hz.csv')
        self._fp = open(self._cmd_path, 'w', newline='', encoding='utf-8')
        self._w = csv.writer(self._fp)
        self._w.writerow(
            ['t_s', 'phase', 'v_inf_deg_s']
            + [f'{n}_cmd' for n in JOINT_NAMES]
        )
        qos = QoSProfile(depth=10)
        self._pub = self.create_publisher(JointStreamCommand, '/xoip/robot/joint_stream_cmd', qos)
        self._dbg = self.create_publisher(JointState, '/kmpc/debug/q', qos)
        self._en = self.create_publisher(Bool, '/kmpc/enable', 10)
        self._st = self.create_publisher(String, '/kmpc/status', 10)
        self.create_subscription(JointState, '/joint_states', self._on_js, 20)
        self.create_subscription(String, '/xoip/system/alarm', self._on_alarm, 20)
        self.create_timer(self._dt, self._tick)
        self.create_timer(1.0, self._hb)
        self._t_log0 = time.monotonic()
        self.get_logger().warn(
            'P2P S-curve 250Hz — 不是 kmpc，没有 20ms QP。'
            f' v={self._v_max:.4f}rad/s={self._v_max * RAD2DEG:.2f}deg/s'
            f' accel_s={args.accel_s} jerk_s={args.jerk_s} R1→{r1 * RAD2DEG:.1f}°'
            ' 只动 R1，其余轴钉住 catch 测量（不去整机 HOME）'
        )
        self.get_logger().warn('若 kmpc 还在跑，先 Ctrl+C 掉，否则两路命令会打架。')
        self.get_logger().warn(
            f'先 catch {self._catch_s:.1f}s 钉住测量再 to_home。'
            '窗口①出现「伺服电机报警」整趟作废，脚本会冻结，不要等驱动自己再上使能。'
        )

    def _on_js(self, msg: JointState) -> None:
        q = named18(msg)
        if q is not None:
            self._meas = q

    def _on_alarm(self, msg: String) -> None:
        text = (msg.data or '').strip()
        if not text:
            return
        self._freeze(f'alarm:{text}')

    def _freeze(self, why: str) -> None:
        if self._meas is not None:
            self._q = list(self._meas)
        self._stop = True
        if self._phase != 'alarm_hold':
            self.get_logger().error(
                f'冻结 {why}。不再走 S 曲线，250Hz 钉住当前测量，避免驱动清故障后命令已经跑飞。'
            )
        self._phase = 'alarm_hold'

    def _pose_r1(self, r1: float) -> List[float]:
        q = list(self._lock if self._lock is not None else self._q)
        q[4] = r1
        return q

    def _begin(self, name: str, q_to: List[float]) -> None:
        self._q_from = list(self._q)
        self._q_to = list(q_to)
        self._seg_t0 = time.monotonic()
        self._phase = name
        T = ptp_duration(self._q_from, self._q_to, self._v_max, self._a_max, self._j_max)
        self.get_logger().info(
            f'{name} T={T:.2f}s  R1 {self._q_from[4] * RAD2DEG:.2f}→{self._q_to[4] * RAD2DEG:.2f} deg'
        )

    def _publish(self, q: List[float], phase: str, v_inf: float) -> None:
        if not rclpy.ok() or self._closed:
            return
        stamp = self.get_clock().now().to_msg()
        for rid, (lo, hi) in SLICE.items():
            msg = JointStreamCommand()
            msg.header.stamp = stamp
            msg.robot_id = rid
            msg.joints = [float(x * RAD2DEG) for x in q[lo:hi]]
            self._pub.publish(msg)
        js = JointState()
        js.header.stamp = stamp
        js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in q]
        self._dbg.publish(js)
        en = Bool()
        en.data = True
        self._en.publish(en)
        st = String()
        st.data = (
            f'p2p=1 qp=0 held={phase.startswith("hold")} '
            f'phase={phase} v_inf_deg={v_inf * RAD2DEG:.3f}'
        )
        self._st.publish(st)
        t = time.monotonic() - self._t_log0
        self._w.writerow([f'{t:.6f}', phase, f'{v_inf * RAD2DEG:.6f}'] + [f'{x:.10f}' for x in q])
        self._rows += 1
        if self._rows % 50 == 0:
            self._fp.flush()

    def _tick(self) -> None:
        if self._want_exit:
            if rclpy.ok():
                rclpy.shutdown()
            return
        if not rclpy.ok():
            return
        if self._want_exit:
            rclpy.shutdown()
            return
        if self._stop:
            self._publish(self._q, self._phase if self._phase == 'alarm_hold' else 'estop_hold', 0.0)
            return
        if self._meas is None:
            return
        # 规划还在走、测量已经掉出好几度：多半已经下使能，立刻钉住测量
        if self._phase != 'wait_js':
            err = max(abs(a - b) for a, b in zip(self._meas, self._q))
            if err * RAD2DEG > 3.0:
                self._freeze(f'follow_err={err * RAD2DEG:.2f}deg')
                self._publish(self._q, 'alarm_hold', 0.0)
                return
        if self._phase == 'wait_js':
            self._lock = list(self._meas)
            self._q = list(self._lock)
            self._seg_t0 = time.monotonic()
            self._phase = 'catch'
            self.get_logger().info(
                f'catch {self._catch_s:.1f}s 钉住测量再动  R1={self._q[4] * RAD2DEG:.2f}deg'
            )
            return
        t = time.monotonic() - self._seg_t0
        if self._phase == 'catch':
            self._q = list(self._lock) if self._lock is not None else list(self._meas)
            self._publish(self._q, 'catch', 0.0)
            if t >= self._catch_s:
                self._begin('to_home', self._pose_r1(self._r1_home))
            return
        if self._phase.startswith('to_') or self._phase == 'to_sweep' or self._phase == 'to_home_2':
            q, v_inf, ph = ptp_q(t, self._q_from, self._q_to, self._v_max, self._a_max, self._j_max)
            self._q = q
            T = ptp_duration(self._q_from, self._q_to, self._v_max, self._a_max, self._j_max)
            moving = self._phase
            self._publish(q, f'{moving}:{ph}', v_inf)
            if t >= T:
                self._q = list(self._q_to)
                if self._phase == 'to_home':
                    self._seg_t0 = time.monotonic()
                    self._phase = 'hold_home'
                    self.get_logger().info(f'at HOME, hold {self._hold_s:.1f}s')
                elif self._phase == 'to_sweep':
                    self._seg_t0 = time.monotonic()
                    self._phase = 'hold_sweep'
                    self.get_logger().info(f'at SWEEP, hold {self._hold_s:.1f}s')
                elif self._phase == 'to_home_2':
                    self._seg_t0 = time.monotonic()
                    self._phase = 'hold_done'
                    self.get_logger().info('back HOME. 继续 250Hz 保持，Ctrl+C 结束')
            return
        if self._phase == 'hold_home':
            self._publish(self._q, 'hold_home', 0.0)
            if t >= self._hold_s:
                self._begin('to_sweep', self._pose_r1(self._r1_sweep))
            return
        if self._phase == 'hold_sweep':
            self._publish(self._q, 'hold_sweep', 0.0)
            if t >= self._hold_s:
                self._begin('to_home_2', self._pose_r1(self._r1_home))
            return
        if self._phase == 'hold_done':
            self._publish(self._q, 'hold_done', 0.0)
            return

    def _hb(self) -> None:
        r1 = self._q[4] * RAD2DEG
        self.get_logger().info(f'phase={self._phase} R1={r1:.2f}deg rows={self._rows}')

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fp.flush()
            self._fp.close()
        except ValueError:
            pass
        self.get_logger().info(f'saved {self._cmd_path}')


def main() -> None:
    p = argparse.ArgumentParser(description='L6 一次规划 250Hz P2P stream（无 kmpc）')
    p.add_argument('--dq-scale', type=float, default=0.10)
    p.add_argument('--accel-s', type=float, default=0.50)
    p.add_argument('--jerk-s', type=float, default=0.08)
    p.add_argument('--hz', type=float, default=250.0)
    p.add_argument('--hold-s', type=float, default=3.0)
    p.add_argument(
        '--catch-s',
        type=float,
        default=2.0,
        help='收到 joint_states 后先按测量 250Hz 钉住再 to_home',
    )
    p.add_argument('--sign', type=int, default=-1, help='R1 符号：-1=SWEEP−，+1=SWEEP+')
    p.add_argument(
        '--out-dir',
        default='/home/nvidia/workspaces/kmpc/logs/jitter/p2p_l6_sweep',
    )
    args = p.parse_args()
    rclpy.init()
    node = P2PStream(args)

    def _sig(*_a: object) -> None:
        if node._want_exit:
            return
        if node._stop:
            node._want_exit = True
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            return
        node._stop = True
        try:
            node.get_logger().warn('SIGINT：冻结最后一拍继续发，再按一次才退出')
        except Exception:
            pass

    signal.signal(signal.SIGINT, _sig)
    try:
        rclpy.spin(node)
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
