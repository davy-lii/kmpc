"""Headless P0 check: after enable, right-arm J2 should leave zero toward -90deg."""

from __future__ import annotations

import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from kmpc.types import NAME_TO_INDEX


class P0Check(Node):
    def __init__(self) -> None:
        super().__init__('p0_check')
        self.declare_parameter('timeout_sec', 8.0)
        self.declare_parameter('j2_threshold_rad', -0.5)
        self._timeout = float(self.get_parameter('timeout_sec').value)
        self._thresh = float(self.get_parameter('j2_threshold_rad').value)
        self._j2: Optional[float] = None
        self._t0 = time.monotonic()
        self.exit_code: Optional[int] = None
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.get_logger().info(
            f'P0 check: wait up to {self._timeout:.1f}s for '
            f'RightArm_Z_Joint2 < {self._thresh:.2f} rad'
        )

    def _on_js(self, msg: JointState) -> None:
        names = list(msg.name)
        key = 'RightArm_Z_Joint2'
        if key not in names:
            return
        self._j2 = float(msg.position[names.index(key)])

    def poll(self) -> None:
        if self.exit_code is not None:
            return
        if self._j2 is not None and self._j2 < self._thresh:
            self.get_logger().info(
                f'PASS: RightArm_Z_Joint2={self._j2:.3f} rad '
                f'(idx={NAME_TO_INDEX["RightArm_Z_Joint2"]})'
            )
            self.exit_code = 0
            return
        if time.monotonic() - self._t0 > self._timeout:
            self.get_logger().error(
                f'FAIL: RightArm_Z_Joint2={self._j2} after {self._timeout:.1f}s'
            )
            self.exit_code = 1


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = P0Check()
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
