"""Load joint_map.yaml and sanity-check against the 18-DOF convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import yaml

from kmpc.types import (
    JOINT_NAMES,
    ROBOT_ID_HEAD,
    ROBOT_ID_LEFT,
    ROBOT_ID_RIGHT,
    ROBOT_ID_WAIST,
    ROBOT_ID_TO_NAMES,
)


class JointMapError(ValueError):
    """joint_map.yaml does not match the expected robot_id / names."""


@dataclass(frozen=True)
class RobotChain:
    robot_id: int
    name: str
    joints: List[str]


class JointMap:
    def __init__(self, robots: Mapping[int, RobotChain]) -> None:
        self._robots: Dict[int, RobotChain] = dict(robots)
        self._validate()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'JointMap':
        with open(yaml_path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        robots_raw = data.get('robots') or {}
        robots: Dict[int, RobotChain] = {}
        for key, spec in robots_raw.items():
            robot_id = int(key)
            joints = [str(n) for n in (spec.get('joints') or [])]
            if not joints:
                raise JointMapError(f'robot_id={robot_id} has empty joints')
            robots[robot_id] = RobotChain(
                robot_id=robot_id,
                name=str(spec.get('name', f'robot_{robot_id}')),
                joints=joints,
            )
        return cls(robots)

    def _validate(self) -> None:
        for robot_id, expected in ROBOT_ID_TO_NAMES.items():
            chain = self._robots.get(robot_id)
            if chain is None:
                raise JointMapError(f'missing robot_id={robot_id}')
            if chain.joints != expected:
                raise JointMapError(
                    f'robot_id={robot_id} joints {chain.joints} != {expected}'
                )
        head = self._robots.get(ROBOT_ID_HEAD)
        if head is not None and len(head.joints) != 2:
            raise JointMapError('robot_id=4 (head) must have 2 joints')

    @property
    def robot_ids(self) -> List[int]:
        return sorted(self._robots)

    def chain(self, robot_id: int) -> RobotChain:
        if robot_id not in self._robots:
            raise JointMapError(f'unknown robot_id={robot_id}')
        return self._robots[robot_id]

    def urdf_names(self, robot_id: int) -> List[str]:
        return list(self.chain(robot_id).joints)

    def dof(self, robot_id: int) -> int:
        return len(self.urdf_names(robot_id))

    def mpc_joint_names(self) -> List[str]:
        return list(JOINT_NAMES)
