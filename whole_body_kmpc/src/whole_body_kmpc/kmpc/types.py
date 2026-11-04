"""18-DOF index convention, unit conversion, stream packing.

Internal order (must stay consistent across the package):

    q[0:4]   = waist  = robot_id 3 = Chassis_Y_Joint1..3, Chassis_Z_Joint4
    q[4:11]  = right  = robot_id 1 = RightArm_Z_Joint1..7
    q[11:18] = left   = robot_id 2 = LeftArm_Z_Joint1..7

I/O boundary: joint_stream_cmd is degrees; everything inside is radians.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import numpy as np

N_DOF = 18
N_WAIST = 4
N_ARM = 7

SLICE_WAIST = slice(0, 4)
SLICE_RIGHT = slice(4, 11)
SLICE_LEFT = slice(11, 18)

# Shoulder pitch (J2) and elbow (J4) in the 18-vector.
IDX_J2_R, IDX_J4_R = 5, 7
IDX_J2_L, IDX_J4_L = 12, 14

ROBOT_ID_RIGHT = 1
ROBOT_ID_LEFT = 2
ROBOT_ID_WAIST = 3
ROBOT_ID_HEAD = 4

# Publish order in the control loop: waist, right, left.
STREAM_ROBOT_IDS = (ROBOT_ID_WAIST, ROBOT_ID_RIGHT, ROBOT_ID_LEFT)

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

NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

ROBOT_ID_TO_SLICE: Dict[int, slice] = {
    ROBOT_ID_WAIST: SLICE_WAIST,
    ROBOT_ID_RIGHT: SLICE_RIGHT,
    ROBOT_ID_LEFT: SLICE_LEFT,
}

ROBOT_ID_TO_NAMES: Dict[int, List[str]] = {
    ROBOT_ID_WAIST: JOINT_NAMES[SLICE_WAIST],
    ROBOT_ID_RIGHT: JOINT_NAMES[SLICE_RIGHT],
    ROBOT_ID_LEFT: JOINT_NAMES[SLICE_LEFT],
}

MODE_SINGLE_EE = 'single_ee'
MODE_DUAL_CARRY = 'dual_carry'
MODE_HOLD = 'hold'
VALID_MODES = (MODE_SINGLE_EE, MODE_DUAL_CARRY, MODE_HOLD)

EE_FRAME_RIGHT = 'Link7_R'  # identity alias of RightArm_Z_Link7
EE_FRAME_LEFT = 'Link7_L'  # identity alias of LeftArm_Z_Link7
# URDF 掌心（mesh），跟踪不用这个点。欧拉角 / 抓放偏置由上层给出。
PALM_FRAME_RIGHT = 'r_hand_link'
PALM_FRAME_LEFT = 'l_hand_link'
# 跟踪点是法兰原点。保留为零，避免旧引用再叠 −Y 射线 / 夹爪 TCP。
EE_TCP_LOCAL = (0.0, 0.0, 0.0)
BASE_FRAME = 'base_link'

_PALM_BY_FLANGE = {
    EE_FRAME_LEFT: PALM_FRAME_LEFT,
    EE_FRAME_RIGHT: PALM_FRAME_RIGHT,
    'LeftArm_Z_Link7': PALM_FRAME_LEFT,
    'RightArm_Z_Link7': PALM_FRAME_RIGHT,
}


def palm_frame(ee_or_palm: str) -> str:
    """Map flange / Link7_* to the palm link; pass through l/r_hand_link."""
    name = str(ee_or_palm).strip()
    return _PALM_BY_FLANGE.get(name, name)


def deg_to_rad(degrees: float) -> float:
    return math.radians(degrees)


def rad_to_deg(radians: float) -> float:
    return math.degrees(radians)


def deg_array_to_rad(values: Sequence[float]) -> np.ndarray:
    return np.deg2rad(np.asarray(values, dtype=float))


def rad_array_to_deg_list(values: Sequence[float]) -> List[float]:
    """float32-friendly Python list for JointStreamCommand.joints."""
    arr = np.rad2deg(np.asarray(values, dtype=float))
    return [float(x) for x in arr]


def pack_stream_deg(q_rad: Sequence[float]) -> Dict[int, List[float]]:
    """Split 18-vector (rad) into robot_id → joints (deg). Head is not included."""
    q = np.asarray(q_rad, dtype=float).reshape(N_DOF)
    return {
        robot_id: rad_array_to_deg_list(q[slc])
        for robot_id, slc in ROBOT_ID_TO_SLICE.items()
    }


def q_from_name_map(positions: Mapping[str, float], default: float = 0.0) -> np.ndarray:
    q = np.full(N_DOF, default, dtype=float)
    for name, idx in NAME_TO_INDEX.items():
        if name in positions:
            q[idx] = float(positions[name])
    return q
