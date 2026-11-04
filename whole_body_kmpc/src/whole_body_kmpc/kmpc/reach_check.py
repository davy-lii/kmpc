"""Wait until the flange origin is within a distance of /kmpc/target_*."""

from __future__ import annotations

import math
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from kmpc.cartesian_tracker import ori_error_xyzw, quat_is_identity_xyzw
from kmpc.types import EE_FRAME_LEFT, EE_FRAME_RIGHT


class ReachCheck(Node):
    def __init__(self) -> None:
        super().__init__('reach_check')
        self.declare_parameter('side', 'left')
        self.declare_parameter('timeout_sec', 15.0)
        self.declare_parameter('tol_m', 0.02)
        self.declare_parameter('tol_rad', float('inf'))
        side = str(self.get_parameter('side').value).strip().lower()
        self._ee = EE_FRAME_LEFT if side == 'left' else EE_FRAME_RIGHT
        self._topic = f'/kmpc/target_{side}'
        self._timeout = float(self.get_parameter('timeout_sec').value)
        self._tol = float(self.get_parameter('tol_m').value)
        self._tol_rad = float(self.get_parameter('tol_rad').value)
        self._target_xyz: Optional[tuple] = None
        self._target_quat: Optional[np.ndarray] = None
        self._t0 = time.monotonic()
        self.exit_code: Optional[int] = None
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)
        self.create_subscription(PoseStamped, self._topic, self._on_target, 10)
        self.get_logger().info(
            f'reach_check: {self._ee} vs {self._topic}, tol={self._tol:.3f}m, '
            f'tol_rad={self._tol_rad}, timeout={self._timeout:.1f}s'
        )

    def _on_target(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        self._target_xyz = (float(p.x), float(p.y), float(p.z))
        self._target_quat = np.array(
            [float(o.x), float(o.y), float(o.z), float(o.w)], dtype=float)

    def poll(self) -> None:
        if self.exit_code is not None:
            return
        if time.monotonic() - self._t0 > self._timeout:
            self.get_logger().error(
                f'FAIL: no reach within {self._timeout:.1f}s '
                f'(target={self._target_xyz})'
            )
            self.exit_code = 1
            return
        if self._target_xyz is None:
            return
        try:
            tf = self._buffer.lookup_transform(
                'base_link', self._ee, rclpy.time.Time(),
                Duration(seconds=0.2),
            )
        except TransformException:
            return
        t = tf.transform.translation
        tcp = np.array([t.x, t.y, t.z])
        err = math.sqrt(
            (tcp[0] - self._target_xyz[0]) ** 2
            + (tcp[1] - self._target_xyz[1]) ** 2
            + (tcp[2] - self._target_xyz[2]) ** 2
        )
        ori_ok = True
        e_ori = 0.0
        if (math.isfinite(self._tol_rad) and self._target_quat is not None
                and not quat_is_identity_xyzw(self._target_quat)):
            q = tf.transform.rotation
            q_now = np.array([q.x, q.y, q.z, q.w], dtype=float)
            e_ori = float(np.linalg.norm(
                ori_error_xyzw(self._target_quat, q_now)))
            ori_ok = e_ori < self._tol_rad
        if err < self._tol and ori_ok:
            self.get_logger().info(
                f'PASS: {self._ee} err={err*1000:.1f} mm  '
                f'e_ori={math.degrees(e_ori):.1f}deg  '
                f'target=({self._target_xyz[0]:.3f},{self._target_xyz[1]:.3f},'
                f'{self._target_xyz[2]:.3f})'
            )
            self.exit_code = 0


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ReachCheck()
    try:
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.poll()
    except KeyboardInterrupt:
        if node.exit_code is None:
            node.exit_code = 130
    code = 1 if node.exit_code is None else node.exit_code
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
