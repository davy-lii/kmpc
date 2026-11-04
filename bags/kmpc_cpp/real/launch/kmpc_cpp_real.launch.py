#!/usr/bin/env python3
"""真机控制：无假植物体。默认不起 RViz。

默认 dq_scale=0.1、auto_enable=false、rviz=false。
插补后发 /xoip/robot/multi_joint_servo_cmd，250 Hz。
笔记本上看姿态：另开 kmpc_cpp_viz.launch.py（同一 ROS_DOMAIN_ID）。驱动若已发 TF，控制端 use_rsp:=false。
"""

import sys
from pathlib import Path


def _load_common() -> None:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parents[1] / 'ros2'):
        if (candidate / 'launch_common.py').exists():
            sys.path.insert(0, str(candidate))
            return
    raise ImportError('launch_common.py not found next to launch or in ros2/')


_load_common()
from launch_common import generate_real_launch  # noqa: E402


def generate_launch_description():
    return generate_real_launch()
