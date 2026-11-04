#!/usr/bin/env python3
"""只看姿态：把机器人 /joint_states 映到本机 RViz，不起 KMPC。"""

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
from launch_common import generate_viz_launch  # noqa: E402


def generate_launch_description():
    return generate_viz_launch()
