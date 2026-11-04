"""Joint pose GUI: sliders + numeric entry, HOME, then grasp.

Layout follows robot_tf / joint_state_publisher_gui (label, edit, slider).
Units in the window are degrees; ROS topics stay radians.
"""

from __future__ import annotations

import math
import sys
import threading
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String

from kmpc.limits import JointLimits, load_q_pref
from kmpc.types import JOINT_NAMES, N_DOF

try:
    from python_qt_binding.QtCore import Qt, QTimer
    from python_qt_binding.QtGui import QFont
    from python_qt_binding.QtWidgets import (
        QApplication,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

_SLIDER_RANGE = 10000
_GROUPS = (
    ('腰', slice(0, 4)),
    ('右臂', slice(4, 11)),
    ('左臂', slice(11, 18)),
)
_LABELS = [
    '腰 J1 俯仰', '腰 J2 折叠', '腰 J3 胸', '腰 J4 旋转',
    '右臂 J1', '右臂 J2 肩', '右臂 J3', '右臂 J4 肘',
    '右臂 J5', '右臂 J6', '右臂 J7',
    '左臂 J1', '左臂 J2 肩', '左臂 J3', '左臂 J4 肘',
    '左臂 J5', '左臂 J6', '左臂 J7',
]


class PoseGuiNode(Node):
    def __init__(self) -> None:
        super().__init__('pose_gui')
        self.declare_parameter('q_pref_file', '')
        self.declare_parameter('q_pref_station', 'default')
        self.declare_parameter('joint_limits_file', '')
        q_file = str(self.get_parameter('q_pref_file').value)
        station = str(self.get_parameter('q_pref_station').value)
        lim_file = str(self.get_parameter('joint_limits_file').value)
        if not q_file or not lim_file:
            raise RuntimeError('pose_gui needs q_pref_file and joint_limits_file')
        self.q_home = load_q_pref(q_file, station)
        self.limits = JointLimits.from_yaml(lim_file)
        self._q = self.q_home.copy()
        self._lock = threading.Lock()
        self._dragging = False

        self._q_pub = self.create_publisher(JointState, '/kmpc/q_target', 10)
        self._state_pub = self.create_publisher(String, '/kmpc/tracking_state', 10)
        self._calib_pub = self.create_publisher(JointState, '/kmpc/q_calib', 10)
        self._grasp_start = self.create_publisher(Empty, '/kmpc/grasp_start', 10)
        self._grasp_stop = self.create_publisher(Empty, '/kmpc/grasp_stop', 10)
        self._enable_pub = self.create_publisher(Bool, '/kmpc/enable', 10)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._q.copy()

    def set_dragging(self, on: bool) -> None:
        with self._lock:
            self._dragging = bool(on)

    def apply_q(self, q: np.ndarray, *, jog: bool) -> None:
        q = np.clip(
            np.asarray(q, dtype=float).reshape(N_DOF),
            self.limits.q_min, self.limits.q_max)
        with self._lock:
            self._q = q.copy()
        if jog:
            self._grasp_stop.publish(Empty())
            msg = String()
            msg.data = 'TRACKING'
            self._state_pub.publish(msg)
            self._publish_q(q)
            en = Bool()
            en.data = True
            self._enable_pub.publish(en)

    def go_home(self) -> None:
        q = self.q_home.copy()
        with self._lock:
            self._q = q.copy()
        self._grasp_stop.publish(Empty())
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in q]
        self._calib_pub.publish(js)
        msg = String()
        msg.data = 'CALIBRATING'
        self._state_pub.publish(msg)
        self._publish_q(q)
        self.get_logger().info('HOME: CALIBRATING to q_pref')

    def start_grasp(self) -> None:
        msg = String()
        msg.data = 'TRACKING'
        self._state_pub.publish(msg)
        self._grasp_start.publish(Empty())
        self.get_logger().info('grasp: TRACKING + /kmpc/grasp_start')

    def _publish_q(self, q: np.ndarray) -> None:
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in q]
        self._q_pub.publish(js)

    def _on_js(self, msg: JointState) -> None:
        with self._lock:
            if self._dragging:
                return
        if not msg.name or not msg.position:
            return
        named = {str(n): float(p) for n, p in zip(msg.name, msg.position)}
        q = self.snapshot()
        for i, name in enumerate(JOINT_NAMES):
            if name in named:
                q[i] = named[name]
        with self._lock:
            if self._dragging:
                return
            self._q = q


class JointRow(QWidget):
    def __init__(self, index: int, lo: float, hi: float, parent: 'PoseGuiWindow') -> None:
        super().__init__()
        self.index = index
        self.lo = float(lo)
        self.hi = float(hi)
        self._win = parent
        self._updating = False
        font = QFont('Helvetica', 9)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        lab = QLabel(_LABELS[index])
        lab.setFont(QFont('Helvetica', 9, QFont.Bold))
        lab.setMinimumWidth(88)
        layout.addWidget(lab)
        self.edit = QLineEdit()
        self.edit.setFont(font)
        self.edit.setFixedWidth(64)
        self.edit.setAlignment(Qt.AlignRight)
        self.edit.editingFinished.connect(self._on_edit)
        layout.addWidget(self.edit)
        unit = QLabel('°')
        unit.setFont(font)
        layout.addWidget(unit)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, _SLIDER_RANGE)
        self.slider.setMinimumWidth(220)
        self.slider.sliderPressed.connect(lambda: parent.node.set_dragging(True))
        self.slider.sliderReleased.connect(self._on_release)
        self.slider.valueChanged.connect(self._on_slide)
        layout.addWidget(self.slider, 1)
        span = QLabel(f'{math.degrees(self.lo):.0f}~{math.degrees(self.hi):.0f}')
        span.setFont(QFont('Helvetica', 8))
        span.setMinimumWidth(72)
        layout.addWidget(span)
        self.setLayout(layout)

    def set_rad(self, rad: float, *, emit: bool = False) -> None:
        rad = float(np.clip(rad, self.lo, self.hi))
        self._updating = True
        try:
            self.edit.setText(f'{math.degrees(rad):.1f}')
            if self.hi > self.lo:
                frac = (rad - self.lo) / (self.hi - self.lo)
            else:
                frac = 0.5
            self.slider.setValue(int(round(frac * _SLIDER_RANGE)))
        finally:
            self._updating = False
        if emit:
            self._win.on_joint_changed(self.index, rad)

    def _slider_to_rad(self) -> float:
        frac = float(self.slider.value()) / float(_SLIDER_RANGE)
        return self.lo + frac * (self.hi - self.lo)

    def _on_slide(self, _value: int) -> None:
        if self._updating:
            return
        rad = self._slider_to_rad()
        self._updating = True
        self.edit.setText(f'{math.degrees(rad):.1f}')
        self._updating = False
        self._win.on_joint_changed(self.index, rad)

    def _on_release(self) -> None:
        self._win.node.set_dragging(False)
        self._win.on_joint_changed(self.index, self._slider_to_rad())

    def _on_edit(self) -> None:
        try:
            deg = float(self.edit.text().strip())
        except ValueError:
            return
        rad = float(np.clip(math.radians(deg), self.lo, self.hi))
        self.set_rad(rad, emit=True)


class PoseGuiWindow(QMainWindow):
    def __init__(self, node: PoseGuiNode) -> None:
        super().__init__()
        self.node = node
        self.setWindowTitle('KMPC 关节设置')
        self.rows: List[JointRow] = []
        root = QVBoxLayout()
        hint = QLabel(
            '拉滑条或输入角度（度）→ 机器人跟过去。'
            '先点「回 HOME」回到准备姿态，再点「抓目标点」。')
        hint.setWordWrap(True)
        root.addWidget(hint)
        btns = QHBoxLayout()
        home_btn = QPushButton('回 HOME')
        home_btn.clicked.connect(self._on_home)
        grasp_btn = QPushButton('抓目标点')
        grasp_btn.clicked.connect(self._on_grasp)
        btns.addWidget(home_btn)
        btns.addWidget(grasp_btn)
        root.addLayout(btns)
        self.status = QLabel('拖动关节即可改姿态')
        root.addWidget(self.status)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_l = QVBoxLayout()
        q0 = node.snapshot()
        for title, slc in _GROUPS:
            box = QGroupBox(title)
            box_l = QVBoxLayout()
            for i in range(*slc.indices(N_DOF)):
                row = JointRow(i, node.limits.q_min[i], node.limits.q_max[i], self)
                row.set_rad(q0[i])
                self.rows.append(row)
                box_l.addWidget(row)
            box.setLayout(box_l)
            inner_l.addWidget(box)
        inner.setLayout(inner_l)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(620, 760)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._sync_from_robot)
        self._timer.start(100)

    def on_joint_changed(self, index: int, rad: float) -> None:
        q = self.node.snapshot()
        q[index] = rad
        self.node.set_dragging(True)
        self.node.apply_q(q, jog=True)
        self.status.setText(
            f'关节 {_LABELS[index]} = {math.degrees(rad):.1f}°')

    def _on_home(self) -> None:
        self.node.set_dragging(False)
        self.node.go_home()
        q = self.node.q_home
        for row in self.rows:
            row.set_rad(q[row.index])
        self.status.setText('正在回 HOME（标定弹簧，不是瞬移）…')

    def _on_grasp(self) -> None:
        self.node.set_dragging(False)
        self.node.start_grasp()
        self.status.setText('已发抓目标点：红球出现后左手去够。')

    def _sync_from_robot(self) -> None:
        if self.node._dragging:
            return
        q = self.node.snapshot()
        for row in self.rows:
            row.set_rad(q[row.index])


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = PoseGuiNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    app = QApplication(sys.argv)
    win = PoseGuiWindow(node)
    win.show()
    code = app.exec() if hasattr(app, 'exec') else app.exec_()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
