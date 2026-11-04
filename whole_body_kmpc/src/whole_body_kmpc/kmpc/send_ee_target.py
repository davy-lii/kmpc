"""Publish a base_link PoseStamped as a fake perception target for one EE.

The grasp pose can be changed live (no launch restart):
  - RViz Interact tool: drag / rotate the handle (or axis arrows)
  - ros2 param set /send_ee_target x|y|z|qx|qy|qz|qw
  - ros2 topic pub --once /kmpc/grasp_xyz geometry_msgs/msg/Point "{x: .., y: .., z: ..}"
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

import math

import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, ColorRGBA, Empty, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker


def _axis_quat(x: float, y: float, z: float) -> Quaternion:
    # Same convention as interactive_marker_tutorials (w=1 + axis, then normalize).
    n = math.sqrt(1.0 + x * x + y * y + z * z)
    return Quaternion(w=1.0 / n, x=x / n, y=y / n, z=z / n)


def _pt(x: float, y: float, z: float) -> Point:
    return Point(x=float(x), y=float(y), z=float(z))


def _quad(points: list, a: Point, b: Point, c: Point, d: Point) -> None:
    points.extend((a, b, c, a, c, d))


def _tapered_handle_marker() -> Marker:
    """Car-door-handle bar in the palm frame: thick at -Z, thin at +Z."""
    marker = Marker()
    marker.type = Marker.TRIANGLE_LIST
    marker.color = ColorRGBA(r=1.0, g=0.12, b=0.08, a=0.78)
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    length = 0.18
    z0 = -0.5 * length
    z1 = 0.5 * length
    xt, yt = 0.030, 0.016
    xn, yn = 0.011, 0.008

    t00 = _pt(-xt, -yt, z0)
    t10 = _pt(xt, -yt, z0)
    t11 = _pt(xt, yt, z0)
    t01 = _pt(-xt, yt, z0)
    n00 = _pt(-xn, -yn, z1)
    n10 = _pt(xn, -yn, z1)
    n11 = _pt(xn, yn, z1)
    n01 = _pt(-xn, yn, z1)

    pts = marker.points
    _quad(pts, t00, t01, t11, t10)
    _quad(pts, n00, n10, n11, n01)
    _quad(pts, t00, t10, n10, n00)
    _quad(pts, t11, t01, n01, n11)
    _quad(pts, t01, t00, n00, n01)
    _quad(pts, t10, t11, n11, n10)
    return marker


class SendEeTarget(Node):
    def __init__(self) -> None:
        super().__init__('send_ee_target')
        self.declare_parameter('side', 'left')
        self.declare_parameter('x', 0.50)
        self.declare_parameter('y', 0.12)
        self.declare_parameter('z', 1.18)
        self.declare_parameter('qx', 0.0)
        self.declare_parameter('qy', 0.0)
        self.declare_parameter('qz', 0.0)
        self.declare_parameter('qw', 1.0)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('enable', True)
        self.declare_parameter('autostart', True)

        side = str(self.get_parameter('side').value).strip().lower()
        if side not in ('left', 'right'):
            raise RuntimeError("side must be 'left' or 'right'")
        topic = f'/kmpc/target_{side}'
        self._lock = Lock()
        self._updating_params = False
        self._x = float(self.get_parameter('x').value)
        self._y = float(self.get_parameter('y').value)
        self._z = float(self.get_parameter('z').value)
        self._qx = float(self.get_parameter('qx').value)
        self._qy = float(self.get_parameter('qy').value)
        self._qz = float(self.get_parameter('qz').value)
        self._qw = float(self.get_parameter('qw').value)
        self._frame = str(self.get_parameter('frame_id').value)

        self._pose_pub = self.create_publisher(PoseStamped, topic, 10)
        self._marker_pub = self.create_publisher(Marker, '/kmpc/debug/grasp_marker', 10)
        self._enable_pub = self.create_publisher(Bool, '/kmpc/enable', 10)
        self._mode_pub = self.create_publisher(String, '/kmpc/mode', 10)
        self._tf = TransformBroadcaster(self)
        self._enable = self._bool('enable')
        self._started = self._bool('autostart')
        self._tf_frame = f'kmpc_grasp_{side}'
        self._im_name = 'grasp'
        self._im_server = InteractiveMarkerServer(self, '/kmpc/grasp')
        self._im_server.insert(
            self._make_interactive_marker(),
            feedback_callback=self._on_im_feedback,
        )
        self._im_server.applyChanges()
        self.create_subscription(Point, '/kmpc/grasp_xyz', self._on_xyz_topic, 10)
        self.create_subscription(Pose, '/kmpc/grasp_pose', self._on_pose_topic, 10)
        self.create_subscription(Empty, '/kmpc/grasp_start', self._on_grasp_start, 10)
        self.create_subscription(Empty, '/kmpc/grasp_stop', self._on_grasp_stop, 10)
        self._clear_pub = self.create_publisher(Empty, '/kmpc/clear_targets', 10)
        self.add_on_set_parameters_callback(self._on_set_params)

        rate = max(float(self.get_parameter('rate').value), 1.0)
        self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(
            f'Fake perception → {topic}  xyz=({self._x:.3f}, {self._y:.3f}, '
            f'{self._z:.3f}) m  quat=({self._qx:.3f},{self._qy:.3f},'
            f'{self._qz:.3f},{self._qw:.3f})  frame={self._frame}'
        )
        if not self._started:
            self.get_logger().info(
                'autostart=false: waiting for /kmpc/grasp_start before publishing target.')
        self.get_logger().info(
            'Hot-move grasp: RViz Interact-drag, or '
            'ros2 topic pub --once /kmpc/grasp_xyz geometry_msgs/msg/Point '
            '"{x: 0.45, y: 0.20, z: 1.15}". Identity quat = position only.'
        )

    def _bool(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def _xyz(self) -> tuple[float, float, float]:
        with self._lock:
            return self._x, self._y, self._z

    def _quat(self) -> tuple[float, float, float, float]:
        with self._lock:
            return self._qx, self._qy, self._qz, self._qw

    def _fill_pose(self, pose: Pose) -> None:
        x, y, z = self._xyz()
        qx, qy, qz, qw = self._quat()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

    def _set_xyz(
        self,
        x: float,
        y: float,
        z: float,
        *,
        source: str,
        sync_im: bool,
        log: bool,
    ) -> None:
        with self._lock:
            self._x = float(x)
            self._y = float(y)
            self._z = float(z)
            nx, ny, nz = self._x, self._y, self._z
        if sync_im:
            self._sync_interactive_pose()
        if log:
            self.get_logger().info(
                f'Grasp xyz ← {source}  ({nx:.3f}, {ny:.3f}, {nz:.3f}) m'
            )

    def _sync_interactive_pose(self) -> None:
        pose = Pose()
        self._fill_pose(pose)
        if self._im_server.setPose(self._im_name, pose):
            self._im_server.applyChanges()

    def _set_quat(
        self,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        *,
        source: str,
        sync_im: bool,
        log: bool,
    ) -> None:
        with self._lock:
            self._qx = float(qx)
            self._qy = float(qy)
            self._qz = float(qz)
            self._qw = float(qw)
            nq = (self._qx, self._qy, self._qz, self._qw)
        if sync_im:
            self._sync_interactive_pose()
        if log:
            self.get_logger().info(
                f'Grasp quat ← {source}  ({nq[0]:.3f},{nq[1]:.3f},'
                f'{nq[2]:.3f},{nq[3]:.3f})'
            )

    def _sync_params(self) -> None:
        x, y, z = self._xyz()
        qx, qy, qz, qw = self._quat()
        self._updating_params = True
        try:
            self.set_parameters([
                Parameter('x', Parameter.Type.DOUBLE, x),
                Parameter('y', Parameter.Type.DOUBLE, y),
                Parameter('z', Parameter.Type.DOUBLE, z),
                Parameter('qx', Parameter.Type.DOUBLE, qx),
                Parameter('qy', Parameter.Type.DOUBLE, qy),
                Parameter('qz', Parameter.Type.DOUBLE, qz),
                Parameter('qw', Parameter.Type.DOUBLE, qw),
            ])
        finally:
            self._updating_params = False

    def _on_set_params(self, params: list[Parameter]) -> SetParametersResult:
        if self._updating_params:
            return SetParametersResult(successful=True)
        x, y, z = self._xyz()
        qx, qy, qz, qw = self._quat()
        changed = False
        quat_changed = False
        for p in params:
            if p.name == 'x':
                x = float(p.value)
                changed = True
            elif p.name == 'y':
                y = float(p.value)
                changed = True
            elif p.name == 'z':
                z = float(p.value)
                changed = True
            elif p.name == 'qx':
                qx = float(p.value)
                quat_changed = True
            elif p.name == 'qy':
                qy = float(p.value)
                quat_changed = True
            elif p.name == 'qz':
                qz = float(p.value)
                quat_changed = True
            elif p.name == 'qw':
                qw = float(p.value)
                quat_changed = True
        if changed:
            self._set_xyz(x, y, z, source='param', sync_im=True, log=True)
        if quat_changed:
            self._set_quat(qx, qy, qz, qw, source='param', sync_im=True, log=True)
        return SetParametersResult(successful=True)

    def _on_xyz_topic(self, msg: Point) -> None:
        self._set_xyz(msg.x, msg.y, msg.z, source='topic', sync_im=True, log=True)
        self._sync_params()

    def _on_pose_topic(self, msg: Pose) -> None:
        self._set_xyz(
            msg.position.x, msg.position.y, msg.position.z,
            source='pose', sync_im=False, log=True)
        self._set_quat(
            msg.orientation.x, msg.orientation.y, msg.orientation.z,
            msg.orientation.w, source='pose', sync_im=True, log=True)
        self._sync_params()

    def _on_im_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        if feedback.event_type not in (
            InteractiveMarkerFeedback.POSE_UPDATE,
            InteractiveMarkerFeedback.MOUSE_UP,
        ):
            return
        p = feedback.pose.position
        o = feedback.pose.orientation
        mouse_up = feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP
        self._set_xyz(
            p.x, p.y, p.z,
            source='drag',
            sync_im=False,
            log=mouse_up,
        )
        self._set_quat(
            o.x, o.y, o.z, o.w,
            source='drag',
            sync_im=False,
            log=mouse_up,
        )
        if mouse_up:
            self._sync_params()

    def _make_interactive_marker(self) -> InteractiveMarker:
        im = InteractiveMarker()
        im.header.frame_id = self._frame
        im.name = self._im_name
        im.description = 'drag / rotate (palm follows)'
        im.scale = 0.18
        self._fill_pose(im.pose)

        vis = InteractiveMarkerControl()
        vis.always_visible = True
        vis.interaction_mode = InteractiveMarkerControl.MOVE_3D
        vis.name = 'move_3d'
        vis.markers.append(_tapered_handle_marker())
        im.controls.append(vis)

        for name, quat in (
            ('move_x', _axis_quat(1.0, 0.0, 0.0)),
            ('move_y', _axis_quat(0.0, 0.0, 1.0)),
            ('move_z', _axis_quat(0.0, 1.0, 0.0)),
        ):
            axis = InteractiveMarkerControl()
            axis.name = name
            axis.orientation = quat
            axis.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            im.controls.append(axis)
        for name, quat in (
            ('rot_x', _axis_quat(1.0, 0.0, 0.0)),
            ('rot_y', _axis_quat(0.0, 0.0, 1.0)),
            ('rot_z', _axis_quat(0.0, 1.0, 0.0)),
        ):
            axis = InteractiveMarkerControl()
            axis.name = name
            axis.orientation = quat
            axis.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            im.controls.append(axis)
        return im

    def _on_grasp_start(self, _msg: Empty) -> None:
        self._started = True
        self.get_logger().info('grasp_start received → publishing target + RViz marker.')

    def _on_grasp_stop(self, _msg: Empty) -> None:
        if not self._started:
            self._clear_pub.publish(Empty())
            return
        self._started = False
        self._clear_pub.publish(Empty())
        self.get_logger().info('grasp_stop: paused target, cleared cartesian.')

    def _on_timer(self) -> None:
        if not self._started:
            return
        if self._enable:
            en = Bool()
            en.data = True
            self._enable_pub.publish(en)
            mode = String()
            mode.data = 'single_ee'
            self._mode_pub.publish(mode)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        self._fill_pose(msg.pose)
        self._pose_pub.publish(msg)
        self._publish_marker(msg)
        self._publish_tf(msg)

    def _publish_marker(self, pose: PoseStamped) -> None:
        handle = _tapered_handle_marker()
        handle.header = pose.header
        handle.ns = 'kmpc_grasp'
        handle.id = 0
        handle.action = Marker.ADD
        handle.pose = pose.pose
        handle.frame_locked = True
        self._marker_pub.publish(handle)

        label = Marker()
        label.header = pose.header
        label.ns = 'kmpc_grasp'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose = pose.pose
        label.pose.position.z = pose.pose.position.z + 0.10
        label.scale.z = 0.07
        label.color = ColorRGBA(r=1.0, g=0.85, b=0.2, a=1.0)
        label.text = 'grasp'
        label.frame_locked = True
        self._marker_pub.publish(label)

    def _publish_tf(self, pose: PoseStamped) -> None:
        tf = TransformStamped()
        tf.header = pose.header
        tf.child_frame_id = self._tf_frame
        tf.transform.translation.x = pose.pose.position.x
        tf.transform.translation.y = pose.pose.position.y
        tf.transform.translation.z = pose.pose.position.z
        tf.transform.rotation = pose.pose.orientation
        self._tf.sendTransform(tf)

    def close(self) -> None:
        self._im_server.shutdown()
        self.destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SendEeTarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
