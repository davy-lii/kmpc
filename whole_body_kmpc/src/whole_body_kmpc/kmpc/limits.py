"""Joint limits and q_pref loaders (radians)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import yaml

from kmpc.types import JOINT_NAMES, N_DOF


class LimitsError(ValueError):
    pass


@dataclass(frozen=True)
class JointLimits:
    names: List[str]
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'JointLimits':
        with open(yaml_path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        names = [str(n) for n in (data.get('names') or [])]
        if names != JOINT_NAMES:
            raise LimitsError(f'joint_limits names mismatch: {names}')
        q_min = _vec(data.get('q_min'), 'q_min')
        q_max = _vec(data.get('q_max'), 'q_max')
        dq_max = _vec(data.get('dq_max'), 'dq_max')
        if np.any(q_min >= q_max):
            raise LimitsError('q_min must be < q_max')
        if np.any(dq_max <= 0.0):
            raise LimitsError('dq_max must be > 0')
        return cls(names=names, q_min=q_min, q_max=q_max, dq_max=dq_max)


def load_q_pref(yaml_path: str, station: str = 'default') -> np.ndarray:
    with open(yaml_path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    stations = data.get('stations') or {}
    if station not in stations:
        raise LimitsError(f'unknown q_pref station={station}')
    return _vec(stations[station].get('q_pref'), 'q_pref')


def _vec(raw, field: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size != N_DOF:
        raise LimitsError(f'{field} must have {N_DOF} entries, got {arr.size}')
    return arr
