"""One-shot sim flow: CALIBRATING (HOME) → auto TRACKING → start grasp target."""

from __future__ import annotations

import re
import time
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from kmpc.limits import load_q_pref
from kmpc.types import JOINT_NAMES, N_DOF


class _Phase(Enum):
    WAIT_START = auto()
    WAIT_TRACKING = auto()
    DONE = auto()


class CalibrateThenGrasp(Node):
    """Publish CALIBRATING, wait for Planner convergence, then /kmpc/grasp_start."""

    def __init__(self) -> None:
        super().__init__('calibrate_then_grasp')
        self.declare_parameter('q_pref_file', '')
        self.declare_parameter('q_pref_station', 'default')
        self.declare_parameter('start_delay_sec', 2.0)
        self.declare_parameter('calib_timeout_sec', 120.0)
        self.declare_parameter('status_topic', '/kmpc/status')
        self.declare_parameter('autostart', True)

        q_file = str(self.get_parameter('q_pref_file').value).strip()
        station = str(self.get_parameter('q_pref_station').value).strip()
        if not q_file:
            raise RuntimeError('calibrate_then_grasp: q_pref_file parameter is required')
        self._q_pref = load_q_pref(q_file, station)
        self._timeout = max(float(self.get_parameter('calib_timeout_sec').value), 1.0)
        self._start_delay = max(float(self.get_parameter('start_delay_sec').value), 0.0)

        self._phase = _Phase.WAIT_START
        self._calib_sent_at: Optional[float] = None
        self._tracking_state = ''
        self._saw_calibrating = False
        self._state_re = re.compile(r'tracking_state=(\w+)')
        self._kick_timer = None

        self._state_pub = self.create_publisher(
            String, '/kmpc/tracking_state', 10)
        self._qcalib_pub = self.create_publisher(
            JointState, '/kmpc/q_calib', 10)
        self._grasp_start_pub = self.create_publisher(
            Empty, '/kmpc/grasp_start', 10)

        status_topic = str(self.get_parameter('status_topic').value)
        self.create_subscription(String, status_topic, self._on_status, 10)
        self.create_service(
            Trigger, '/kmpc/calibrate_then_grasp', self._on_trigger)

        if self._as_bool(self.get_parameter('autostart').value):
            self._kick_timer = self.create_timer(self._start_delay, self._kickoff_once)
        self.get_logger().info(
            'calibrate_then_grasp ready: CALIBRATING → TRACKING → /kmpc/grasp_start. '
            'Manual: ros2 service call /kmpc/calibrate_then_grasp std_srvs/srv/Trigger')

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def _kickoff_once(self) -> None:
        if self._kick_timer is not None:
            self._kick_timer.cancel()
            self._kick_timer = None
        if self._phase != _Phase.WAIT_START:
            return
        self._begin_calibrating()

    def _on_trigger(self, _request, response):
        if self._phase == _Phase.WAIT_TRACKING:
            response.success = False
            response.message = 'already waiting for TRACKING'
            return response
        self._begin_calibrating()
        response.success = True
        response.message = 'calibrating; grasp starts after tracking_state=TRACKING'
        return response

    def _begin_calibrating(self) -> None:
        self._saw_calibrating = False
        self._tracking_state = ''
        self._calib_sent_at = time.monotonic()
        self._phase = _Phase.WAIT_TRACKING
        msg = String()
        msg.data = 'CALIBRATING'
        self._state_pub.publish(msg)
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in self._q_pref.reshape(N_DOF)]
        self._qcalib_pub.publish(js)
        self.get_logger().info(
            'sent CALIBRATING + q_calib=HOME; waiting for TRACKING on /kmpc/status ...')

    def _finish_grasp(self) -> None:
        if self._phase != _Phase.WAIT_TRACKING:
            return
        self._phase = _Phase.DONE
        self._grasp_start_pub.publish(Empty())
        self.get_logger().info(
            'calibration done → TRACKING; published /kmpc/grasp_start (red ball on)')

    def _on_status(self, msg: String) -> None:
        match = self._state_re.search(str(msg.data))
        if match:
            self._tracking_state = match.group(1)
        if self._phase != _Phase.WAIT_TRACKING:
            return
        if self._tracking_state == 'CALIBRATING':
            self._saw_calibrating = True
            return
        # Already at HOME: Planner may skip visible CALIBRATING. Wait a beat
        # so we do not treat the stale TRACKING line as "done".
        if self._tracking_state != 'TRACKING':
            return
        elapsed = 0.0 if self._calib_sent_at is None else (
            time.monotonic() - self._calib_sent_at)
        if self._saw_calibrating or elapsed >= 0.6:
            self._finish_grasp()

    def _check_timeout(self) -> None:
        if self._phase != _Phase.WAIT_TRACKING or self._calib_sent_at is None:
            return
        if time.monotonic() - self._calib_sent_at > self._timeout:
            self.get_logger().error(
                f'calibration timeout after {self._timeout:.0f}s '
                f'(last tracking_state={self._tracking_state}). '
                'Check /kmpc/status com_reject= or arm limits.')
            self._phase = _Phase.DONE


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = CalibrateThenGrasp()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            node._check_timeout()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
