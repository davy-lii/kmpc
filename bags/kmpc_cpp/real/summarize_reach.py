#!/usr/bin/env python3
"""Summarize a reach recording. Does not talk to the robot.

usage: python3 summarize_reach.py /path/to/logs/reach/<stamp>
   or: python3 summarize_reach.py /path/to/reach.csv
"""

from __future__ import annotations

import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

WAIST = [
    'Chassis_Y_Joint1',
    'Chassis_Y_Joint2',
    'Chassis_Y_Joint3',
    'Chassis_Z_Joint4',
]
LEFT = [
    'LeftArm_Z_Joint1',
    'LeftArm_Z_Joint2',
    'LeftArm_Z_Joint3',
    'LeftArm_Z_Joint4',
    'LeftArm_Z_Joint5',
    'LeftArm_Z_Joint6',
    'LeftArm_Z_Joint7',
]
RIGHT = [
    'RightArm_Z_Joint1',
    'RightArm_Z_Joint2',
    'RightArm_Z_Joint3',
    'RightArm_Z_Joint4',
    'RightArm_Z_Joint5',
    'RightArm_Z_Joint6',
    'RightArm_Z_Joint7',
]


def _f(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float('nan')


def _i(x: object) -> Optional[int]:
    s = str(x).strip()
    if s in ('', 'None'):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as fp:
        return list(csv.DictReader(fp))


def _rms_group(row: Dict[str, str], names: List[str], suffix: str) -> float:
    acc = 0.0
    n = 0
    for name in names:
        v = _f(row.get(f'{name}{suffix}', ''))
        if math.isfinite(v):
            acc += v * v
            n += 1
    return math.sqrt(acc / n) if n else float('nan')


def _mean(xs: List[float]) -> float:
    ys = [x for x in xs if math.isfinite(x)]
    return sum(ys) / len(ys) if ys else float('nan')


def _edges(rows: List[Dict[str, str]], key: str) -> List[Tuple[float, str, str]]:
    out: List[Tuple[float, str, str]] = []
    prev: Optional[str] = None
    for r in rows:
        cur = str(r.get(key, '')).strip()
        if cur == '':
            continue
        if prev is None:
            prev = cur
            continue
        if cur != prev:
            out.append((_f(r['t_s']), prev, cur))
            prev = cur
    return out


def _window(rows: List[Dict[str, str]], t0: float, t1: float) -> List[Dict[str, str]]:
    return [r for r in rows if t0 <= _f(r['t_s']) <= t1]


def _pct(xs: List[float], p: float) -> float:
    ys = sorted(x for x in xs if math.isfinite(x))
    if not ys:
        return float('nan')
    i = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[i]


def _print_window(name: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f'  {name}: empty')
        return
    t0 = _f(rows[0]['t_s'])
    t1 = _f(rows[-1]['t_s'])
    hz_w = [_f(r.get('motion_hz_waist', '')) for r in rows]
    hz_l = [_f(r.get('motion_hz_left', '')) for r in rows]
    gap_w = [_f(r.get('motion_gap_ms_waist', '')) for r in rows]
    dq_w = [_rms_group(r, WAIST, '_dq') for r in rows]
    dq_l = [_rms_group(r, LEFT, '_dq') for r in rows]
    dq_r = [_rms_group(r, RIGHT, '_dq') for r in rows]
    xcorr = [_f(r.get('xcorr_lag_ms', '')) for r in rows]
    mage = [_f(r.get('meas_age_ms', '')) for r in rows]
    print(
        f'  {name}: {t1 - t0:.2f}s  rows={len(rows)}  '
        f'motion_hz waist={_mean(hz_w):.1f} left={_mean(hz_l):.1f}  '
        f'gap_ms_waist={_mean(gap_w):.1f}  '
        f'|dq|_rms waist={_mean(dq_w):.4f} left={_mean(dq_l):.4f} right={_mean(dq_r):.4f}'
    )
    print(
        f'    xcorr_lag_ms mean={_mean(xcorr):.1f} p50={_pct(xcorr, 50):.1f} '
        f'p95={_pct(xcorr, 95):.1f}  meas_age_ms mean={_mean(mage):.1f}'
    )


SETTLE_JOINTS = [
    'Chassis_Y_Joint2',
    'RightArm_Z_Joint4',
    'LeftArm_Z_Joint4',
    'RightArm_Z_Joint5',
    'LeftArm_Z_Joint5',
]
RAD2DEG = 180.0 / math.pi


def _err_inf(rows: List[Dict[str, str]]) -> List[float]:
    return [_f(r.get('err_inf_deg', '')) for r in rows]


def _joint_err_deg(row: Dict[str, str], name: str) -> float:
    m = _f(row.get(f'{name}_meas', ''))
    c = _f(row.get(f'{name}_cmd', ''))
    if math.isfinite(m) and math.isfinite(c):
        return (m - c) * RAD2DEG
    return float('nan')


def _event_cuts(events: List[Dict[str, str]]) -> List[Tuple[float, str]]:
    cuts: List[Tuple[float, str]] = []
    for e in events:
        kind = str(e.get('kind', ''))
        detail = str(e.get('detail', ''))
        t = _f(e.get('t_s'))
        if not math.isfinite(t):
            continue
        if kind == 'enable' and 'data=1' in detail:
            cuts.append((t, 'enable HOME'))
        elif kind == 'q_target':
            cuts.append((t, 'q_target'))
        elif kind == 'target_left':
            cuts.append((t, 'target_left'))
        elif kind == 'enable' and 'data=0' in detail:
            cuts.append((t, 'enable off'))
    return cuts


def _print_delay_settle(
    rows: List[Dict[str, str]], events: List[Dict[str, str]]
) -> None:
    print('delay vs settle (last 2 s of each command segment):')
    if not rows:
        print('  (empty reach.csv)')
        return
    t_end = _f(rows[-1]['t_s'])
    cuts = _event_cuts(events)
    starts = [(t, lab) for t, lab in cuts if lab != 'enable off']
    if not starts:
        en = next(
            (r for r in rows if _i(r.get('enable')) == 1),
            None,
        )
        t0 = _f(en['t_s']) if en else _f(rows[0]['t_s'])
        starts = [(t0, 'enable (inferred)')]
    off = next((t for t, lab in cuts if lab == 'enable off'), t_end)
    settle_means: List[float] = []
    motion_p95: List[float] = []
    for i, (t0, lab) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else off
        if t1 <= t0:
            continue
        seg = _window(rows, t0, t1)
        if len(seg) < 5:
            print(f'  {lab} [{t0:.2f},{t1:.2f}]: too short')
            continue
        settle = _window(rows, max(t0, t1 - 2.0), t1)
        motion = _window(rows, t0, max(t0, t1 - 2.0))
        e_m = _err_inf(motion)
        e_s = _err_inf(settle)
        settle_means.append(_mean(e_s))
        motion_p95.append(_pct(e_m, 95))
        print(
            f'  {lab} [{t0:.2f},{t1:.2f}]  '
            f'motion {t1 - t0 - min(2.0, t1 - t0):.2f}s  '
            f'settle {min(2.0, t1 - t0):.2f}s'
        )
        print(
            f'    err_inf_deg motion p50={_pct(e_m, 50):.3f} '
            f'p95={_pct(e_m, 95):.3f}  '
            f'settle mean={_mean(e_s):.3f} p95={_pct(e_s, 95):.3f}'
        )
        bits = []
        for name in SETTLE_JOINTS:
            xs = [_joint_err_deg(r, name) for r in settle]
            bits.append(f'{name}={_mean(xs):+.3f}°')
        print('    settle meas-cmd  ' + '  '.join(bits))
        paused = [
            _i(r.get('paused_waist')) for r in settle
            if str(r.get('paused_waist', '')).strip() != ''
        ]
        if paused:
            print(
                f'    paused_waist last2s  '
                f'1={sum(1 for x in paused if x == 1)}  '
                f'0={sum(1 for x in paused if x == 0)}'
            )
        xcorr = [_f(r.get('xcorr_lag_ms', '')) for r in motion]
        print(
            f'    xcorr_lag_ms motion mean={_mean(xcorr):.1f} '
            f'p50={_pct(xcorr, 50):.1f}'
        )
    stale = [
        r for r in rows
        if _i(r.get('stream_stale_waist')) == 1
        or _i(r.get('stream_stale_left')) == 1
        or _i(r.get('stream_stale_right')) == 1
    ]
    print(f'  stream_status stale rows: {len(stale)}')
    near86 = sum(1 for r in rows if _f(r.get('err_inf_deg')) >= 8.0)
    print(f'  err_inf_deg >= 8° rows: {near86}')
    sm = [x for x in settle_means if math.isfinite(x)]
    mp = [x for x in motion_p95 if math.isfinite(x)]
    if not sm:
        print('  verdict: no settle window — wait ≥2 s after visual stop')
        return
    mx = max(sm)
    if near86 > 20:
        print('  verdict: 流控/卡住 — 运动中经常 ≥8°，先查华成')
    elif mx < 0.2:
        print('  verdict: 延迟  K=0（停稳 err_inf mean < 0.2°）')
    elif mx > 0.5:
        print('  verdict: 稳态偏  考虑慢 hat d（停稳仍 > 0.5°）')
    else:
        print(
            f'  verdict: 停稳 mean={mx:.3f}° 介于 0.2–0.5，两口都录完再定'
        )
    if mp:
        print(f'    (motion p95 max={max(mp):.3f}°)')
    print(
        '    末 2 s 仍明显掉的话，那段不算停稳；肉眼停后再等 2 s 重录。'
    )


def _print_packet_csv(path: str, label: str) -> None:
    print(f'{label}:')
    if not os.path.isfile(path):
        print('  (missing)')
        return
    rows = _load_csv(path)
    if not rows:
        print('  (empty — expected if this interface was not used)')
        return
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(str(r.get('robot_id', '')), []).append(r)
    print(f'  rows={len(rows)}')
    for rid, lab in (('1', 'right'), ('2', 'left'), ('3', 'waist')):
        hit = by.get(rid, [])
        if not hit:
            continue
        dt = [_f(r.get('dt_ms')) for r in hit]
        acc = sorted({str(r.get('acc_time', '')).strip() for r in hit} - {''})
        hz = 1000.0 / _pct(dt, 50) if _pct(dt, 50) > 1e-9 else float('nan')
        extra = f'  acc_time={acc}' if acc else ''
        print(
            f'    {lab} n={len(hit)}  dt_ms p50={_pct(dt, 50):.1f} '
            f'p95={_pct(dt, 95):.1f}  ~{hz:.1f} Hz{extra}'
        )


def _print_stream_status(path: str) -> None:
    print('stream.csv (driver StreamStatus):')
    if not os.path.isfile(path):
        print('  (missing)')
        return
    rows = _load_csv(path)
    if not rows:
        print('  (empty)')
        return
    stale = sum(1 for r in rows if _i(r.get('stale')) == 1)
    age = [_f(r.get('command_age_ms')) for r in rows]
    modes = sorted({str(r.get('mode', '')).strip() for r in rows} - {''})
    print(
        f'  rows={len(rows)}  stale=1 count={stale}  '
        f'command_age_ms max={max((x for x in age if math.isfinite(x)), default=float("nan")):.0f}  '
        f'mode={modes}'
    )


def _print_latency(lats: List[Dict[str, str]], t0: float, t1: float, name: str) -> None:
    rows = [r for r in lats if t0 <= _f(r.get('t_pub')) <= t1]
    if not rows:
        print(f'  latency {name}: no packets')
        return
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(str(r.get('status', '')), []).append(r)
    n = len(rows)
    parts = ', '.join(f'{k}={len(v)}' for k, v in sorted(by.items()))
    print(f'  latency {name}: n={n}  {parts}')
    for rid, label in ((3, 'waist'), (2, 'left'), (1, 'right')):
        hit = [
            _f(r.get('latency_ms'))
            for r in rows
            if _i(r.get('robot_id')) == rid and r.get('status') == 'arrived'
        ]
        ros = [
            _f(r.get('latency_ros_ms'))
            for r in rows
            if _i(r.get('robot_id')) == rid and r.get('status') == 'arrived'
        ]
        if hit:
            print(
                f'    {label} arrived n={len(hit)}  '
                f'latency_ms p50={_pct(hit, 50):.1f} p95={_pct(hit, 95):.1f} '
                f'mean={_mean(hit):.1f}  '
                f'ros_ms p50={_pct(ros, 50):.1f}'
            )
        else:
            print(f'    {label} arrived n=0')


def _sample_reach(rows: List[Dict[str, str]], dt: float = 0.5) -> None:
    if not rows:
        return
    print('  t     eL     eL_z    eL_xy  com_hold wall pitch  |dq_w| |dq_L|')
    next_t = _f(rows[0]['t_s'])
    for r in rows:
        t = _f(r['t_s'])
        if t + 1e-6 < next_t:
            continue
        print(
            f'  {t:6.2f} {_f(r.get("eL")):6.3f} {_f(r.get("eL_z")):7.3f} '
            f'{_f(r.get("eL_xy")):6.3f}  {str(r.get("com_hold","")):^8} '
            f'{str(r.get("com_work_wall","")):^4} {_f(r.get("pitch_deg")):6.1f} '
            f'{_rms_group(r, WAIST, "_dq"):6.3f} {_rms_group(r, LEFT, "_dq"):6.3f}'
        )
        next_t += dt


def main() -> None:
    if len(sys.argv) < 2:
        print(
            'usage: python3 summarize_reach.py /path/to/logs/reach/<stamp>',
            file=sys.stderr,
        )
        sys.exit(2)
    arg = os.path.abspath(sys.argv[1])
    if os.path.isdir(arg):
        d = arg
        reach_path = os.path.join(d, 'reach.csv')
    else:
        d = os.path.dirname(arg)
        reach_path = arg
    if not os.path.isfile(reach_path):
        print(f'missing {reach_path}', file=sys.stderr)
        sys.exit(1)
    rows = _load_csv(reach_path)
    events_path = os.path.join(d, 'events.csv')
    events = _load_csv(events_path) if os.path.isfile(events_path) else []
    print(f'dir {d}')
    print(f'reach.csv rows={len(rows)} duration={_f(rows[-1]["t_s"]) - _f(rows[0]["t_s"]):.2f}s')
    print('events:')
    if not events:
        print('  (none — recorder started after the pubs, or no commands)')
    for e in events:
        print(f'  t={_f(e["t_s"]):7.3f}  {e.get("kind",""):14s}  {e.get("detail","")}')

    _print_delay_settle(rows, events)
    _print_packet_csv(os.path.join(d, 'motion.csv'), 'motion.csv (joint_motion_cmd)')
    _print_packet_csv(
        os.path.join(d, 'stream_cmd.csv'), 'stream_cmd.csv (joint_stream_cmd)')
    _print_stream_status(os.path.join(d, 'stream.csv'))

    t_home = None
    t_left = None
    t_end = None
    for e in events:
        kind = e.get('kind', '')
        detail = e.get('detail', '')
        t = _f(e['t_s'])
        if kind == 'q_target' and t_home is None:
            t_home = t
        if kind == 'enable' and 'data=1' in detail and t_home is None:
            t_home = t
        if kind == 'target_left' and t_left is None:
            t_left = t
        if kind == 'enable' and 'data=0' in detail:
            t_end = t
    if t_left is None:
        for r in rows:
            if _i(r.get('cart')) == 1 and _f(r.get('eL')) > 0.05:
                t_left = _f(r['t_s'])
                break
    if t_home is None:
        t_home = _f(rows[0]['t_s'])
    if t_end is None:
        t_end = _f(rows[-1]['t_s'])

    home_rows = _window(rows, t_home, t_left if t_left is not None else t_end)
    reach_rows = _window(rows, t_left, t_end) if t_left is not None else []
    print('windows:')
    print(f'  HOME inferred [{t_home:.2f}, {t_left if t_left is not None else t_end:.2f}]')
    _print_window('HOME (joint QP)', home_rows)
    if t_left is not None:
        print(f'  reach inferred [{t_left:.2f}, {t_end:.2f}]')
    _print_window('LEFT reach (cartesian)', reach_rows)

    print('com_hold edges (0=off, 1=on):')
    holds = _edges(reach_rows or rows, 'com_hold')
    if not holds:
        print('  none')
    for t, a, b in holds:
        print(f'  t={t:.3f}  {a} -> {b}')

    print('arrived edges:')
    arr = _edges(reach_rows or rows, 'arrived')
    if not arr:
        print('  none')
    for t, a, b in arr:
        print(f'  t={t:.3f}  {a} -> {b}')

    print('cart edges (0=joint, 1=cartesian):')
    carts = _edges(rows, 'cart')
    if not carts:
        print('  none')
    for t, a, b in carts:
        print(f'  t={t:.3f}  {a} -> {b}')

    if reach_rows:
        print('reach 0.5s samples:')
        _sample_reach(reach_rows)
        # Heuristic phases for the far left ball.
        z_cross = None
        hold_off = None
        for r in reach_rows:
            if z_cross is None and math.isfinite(_f(r.get('eL_z'))) and abs(_f(r.get('eL_z'))) < 0.03:
                z_cross = _f(r['t_s'])
            if hold_off is None and _i(r.get('com_hold')) == 0:
                hold_off = _f(r['t_s'])
        t0 = _f(reach_rows[0]['t_s'])
        print('heuristic phases (far-ball squat / arm / CoM):')
        print(f'  start cart t={t0:.2f}')
        print(f'  |eL_z| first < 3cm  t={z_cross}')
        print(f'  com_hold first 0     t={hold_off}')
        print(
            '  If HOME |dq| is smooth and reach samples show z dropping with '
            'com_hold=1, then eL_xy shrinking, then com_hold→0, the three '
            'visual segments are the cartesian scheduler, not motion_cmd.'
        )

    stale = [
        r for r in (reach_rows or rows)
        if _i(r.get('stream_stale_waist')) == 1
        or _i(r.get('stream_stale_left')) == 1
        or _i(r.get('stream_stale_right')) == 1
    ]
    print(f'stream_status stale rows: {len(stale)}')

    lat_path = os.path.join(d, 'latency.csv')
    print('latency (motion packet → meas first inside 0.15 deg):')
    if os.path.isfile(lat_path):
        lats = _load_csv(lat_path)
        t_reach0 = t_left if t_left is not None else t_end
        _print_latency(lats, t_home, t_reach0, 'HOME')
        if t_left is not None:
            _print_latency(lats, t_left, t_end, 'LEFT reach')
        opens = sum(1 for r in lats if r.get('status') == 'open')
        timeouts = sum(1 for r in lats if r.get('status') == 'timeout')
        print(
            f'  totals open={opens} timeout={timeouts}  '
            '(timeout = never within 0.15 deg in 1.5 s; expected if lag >> step)'
        )
    else:
        print('  (no latency.csv — old recorder)')

    print(f'notes: {os.path.join(d, "README.txt")}')


if __name__ == '__main__':
    main()
