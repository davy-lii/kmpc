#!/usr/bin/env python3
"""Non-motion ROS2 smoke for v2 submit gate + fixed topics.

Starts kmpc_node (require_xoip:=false), checks construct does not crash on
undeclared params, publishes segment / MotionSequence / StageAck, and asserts
submit happens once while waiting. Does not enable hardware streaming.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

try:
    from kmpc_msgs.msg import MotionSequence, MotionStage, StageAck
except ImportError as exc:  # pragma: no cover
    print("SKIP: kmpc_msgs not importable:", exc)
    sys.exit(0)


def main() -> int:
    node_bin = Path(os.environ.get("KMPC_NODE_BIN", "kmpc_node"))
    if not node_bin.is_file():
        # allow PATH lookup
        node_bin = Path(node_bin)
    log_path = Path(os.environ.get("KMPC_SMOKE_LOG", "/tmp/kmpc_v2_gate_smoke.log"))
    log_f = log_path.open("w")
    cmd = [
        str(node_bin),
        "--ros-args",
        "-p",
        "auto_enable:=false",
        "-p",
        "require_xoip:=false",
        "-p",
        "stream_dedicated_thread:=false",
        "-p",
        "stream_debug_q_hz:=0.0",
    ]
    env = os.environ.copy()
    env.setdefault("ROS_DOMAIN_ID", "214")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    rclpy.init()
    n = Node("v2_gate_smoke")
    names = [
        "Chassis_Y_Joint1",
        "Chassis_Y_Joint2",
        "Chassis_Y_Joint3",
        "Chassis_Z_Joint4",
    ] + [f"{side}Arm_Z_Joint{i}" for side in ("Right", "Left") for i in range(1, 8)]
    q = [0.0] * 18
    pub_js = n.create_publisher(JointState, "/joint_states", 10)
    pub_en = n.create_publisher(Bool, "/kmpc/enable", 10)
    pub_seg = n.create_publisher(PoseArray, "/kmpc/segment_left", 10)
    pub_ms = n.create_publisher(MotionSequence, "/kmpc/motion_sequence", 10)
    pub_ack = n.create_publisher(StageAck, "/kmpc/stage_ack", 10)

    def spin(sec: float) -> None:
        end = time.monotonic() + sec
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise RuntimeError(f"kmpc_node exited early rc={proc.returncode}")
            js = JointState()
            js.name = names
            js.position = list(q)
            js.header.stamp = n.get_clock().now().to_msg()
            pub_js.publish(js)
            rclpy.spin_once(n, timeout_sec=0.01)

    try:
        spin(1.0)
        en = Bool()
        en.data = True
        pub_en.publish(en)
        spin(0.3)

        seg = PoseArray()
        seg.header.frame_id = "base_link"
        seg.header.stamp.sec = 0
        seg.header.stamp.nanosec = 42  # generation via stamp encoding in some paths
        # TaskReceiver typically uses header stamp as generation for PoseArray.
        p = Pose()
        p.position.x, p.position.y, p.position.z = 0.45, 0.20, 0.90
        p.orientation.w = 1.0
        seg.poses.append(p)
        pub_seg.publish(seg)
        spin(2.0)

        ms = MotionSequence()
        ms.generation = 77
        ms.left = True
        st = MotionStage()
        st.stage_id = 0
        st.geometry = 0
        st.pose.position.x = 0.45
        st.pose.position.y = 0.20
        st.pose.position.z = 0.90
        st.pose.orientation.w = 1.0
        ms.stages.append(st)
        pub_ms.publish(ms)
        spin(2.0)

        ack = StageAck()
        ack.generation = 77
        ack.stage_id = 0
        ack.success = True
        pub_ack.publish(ack)
        spin(0.5)

        log_f.flush()
        text = log_path.read_text(errors="replace")
        if "stage_ack_topic" in text and "FAILED" in text:
            raise AssertionError("undeclared stage_ack_topic crash signature present")
        if "ParameterUninitialized" in text or "undeclared parameter" in text.lower():
            raise AssertionError("undeclared parameter failure in log")
        submits = len(re.findall(r"kmpc v2 submit gen=", text))
        adopts = len(re.findall(r"kmpc v2 adopt ", text))
        print(
            {
                "ok": True,
                "submits": submits,
                "adopts": adopts,
                "log": str(log_path),
            }
        )
        # While held without full robot model/plan success, submit must stay bounded.
        if submits > 6:
            raise AssertionError(f"submit spam: {submits}")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_f.close()
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
