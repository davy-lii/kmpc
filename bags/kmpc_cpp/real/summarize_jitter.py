#!/usr/bin/env python3
"""Summarize a jitter.csv from record_jitter.py. Does not talk to the robot."""

from __future__ import annotations

import csv
import math
import os
import sys
from typing import List

JOINT_NAMES = [
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
RAD2DEG = 180.0 / math.pi


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python3 summarize_jitter.py /path/to/jitter.csv', file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    with open(path, newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        print('empty csv')
        sys.exit(1)
    t0 = _f(rows[0]['t_s'])
    t1 = _f(rows[-1]['t_s'])
    peak = 0.0
    peak_t = 0.0
    enabled = [r for r in rows if r.get('enable') == '1']
    for r in rows:
        v = _f(r.get('err_inf_deg', ''))
        if math.isfinite(v) and v > peak:
            peak = v
            peak_t = _f(r['t_s'])
    print(f'file {path}')
    print(f'duration {t1 - t0:.2f}s  rows {len(rows)}  enable_true_rows {len(enabled)}')
    print(f'peak |meas-cmd|_∞ {peak:.2f} deg at t={peak_t:.2f}s')
    print('per-joint peak |err| deg  (meas vs cmd):')
    worst: List[tuple] = []
    for name in JOINT_NAMES:
        key = f'{name}_err'
        m = 0.0
        for r in rows:
            v = abs(_f(r.get(key, '')))
            if math.isfinite(v):
                m = max(m, v)
        worst.append((m * RAD2DEG, name))
        print(f'  {name:20s}  {m * RAD2DEG:7.2f}')
    worst.sort(reverse=True)
    print('top 3:', ', '.join(f'{n} {d:.1f}°' for d, n in worst[:3]))
    last_flips = rows[-1].get('dq_sign_flips_cum', '')
    print(f'dq_sign_flips_cum last={last_flips}')
    print(f'notes: {os.path.dirname(path)}/README.txt')


if __name__ == '__main__':
    main()
