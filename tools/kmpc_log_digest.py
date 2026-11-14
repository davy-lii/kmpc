#!/usr/bin/env python3
"""Read-only KMPC log triage. Evidence remains in the input files."""
import argparse
import csv
import math
import re
import sys
from collections import Counter, deque
from pathlib import Path

KEYS = ('enable','mode','line','goal_generation','segment_active','frozen','abort','reason','cascade','line_candidate','reverse_line','tracking_state','arrived','fail','com_reject','pitch_clip','stream_limited')
METRICS = ('e0','eL','eR','pitch','line_rate','stream_commit_scale','solve_ms','err_inf_deg','command_age_ms','sdk_duration_ms','arrival_interval_ms','sequence_gap')
KV = re.compile(r'([A-Za-z][\w]*)=([^\s]+)')
NUM = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
ALERT = re.compile(r'\b(error|fatal|exception|traceback|abort|refused|reject|fail|timeout|报警|错误|拒线|失败)\b', re.I)

def number(value):
    m = NUM.match(str(value or ''))
    if not m: return None
    try:
        v = float(m.group())
        return v if math.isfinite(v) else None
    except ValueError: return None

def nonzero(value):
    return str(value or '').strip().lower() not in ('','0','0.0','false','none','na','nan','ok','no','not_provided')

def summarize(path, limit, context):
    count = 0; errors = 0; changes = Counter(); extrema = {}; chosen = []; recent = deque(maxlen=max(1, limit * 2)); last = {}; first = None; tail = None; waist_last = None; waist_sign = [0, 0, 0]
    before = path.stat(); is_csv = path.suffix.lower() == '.csv'
    def add(kind, line, timestamp, detail):
        # Keep bounded memory and count dropped candidates explicitly.
        nonlocal dropped
        if len(chosen) < limit * 2: chosen.append((kind, line, timestamp, detail))
        else: recent.append((kind, line, timestamp, detail)); dropped += 1
    dropped = 0
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        if is_csv:
            reader = csv.DictReader(f, strict=True)
            if not reader.fieldnames: raise ValueError('缺少 CSV 表头')
            iterator = ((reader.line_num, row) for row in reader)
        else:
            iterator = ((i, {'text': line.rstrip('\n')}) for i, line in enumerate(f, 1))
        try:
            for lineno, row in iterator:
                count += 1
                if None in row or any(v is None for v in row.values()): errors += 1; continue
                t = row.get('t_s', row.get('timestamp', '?'))
                raw = row.get('status','') or row.get('text','')
                fields = dict(KV.findall(raw)); fields.update({k:v for k,v in row.items() if k not in ('status','text')})
                compact = ' '.join(f'{k}={fields[k]}' for k in KEYS if k in fields)
                compact += ' ' + ' '.join(f'{k}={fields[k]}' for k in METRICS if k in fields)
                if not compact.strip(): compact = raw or ' '.join(f'{k}={v}' for k,v in row.items() if v and k != 't_s')
                record = (lineno, t, compact.strip()[:700])
                if first is None: first = record
                tail = record
                changed = []
                for k in KEYS:
                    if k in fields:
                        if k in last and fields[k] != last[k]:
                            changes[k] += 1; changed.append(f'{k}:{last[k]}→{fields[k]}')
                        last[k] = fields[k]
                for k in ('faulted','reject_reason','alarm','sequence_gap','last_send_ok'):
                    if k in fields and k in last and fields[k] != last[k]:
                        changes[k] += 1; changed.append(f'{k}:{last[k]}→{fields[k]}')
                    if k in fields: last[k] = fields[k]
                alarm = any(nonzero(fields.get(k)) for k in ('alarm','reject_reason','faulted')) or (fields.get('last_send_ok') == '0') or (number(fields.get('sequence_gap')) or 0) > 0
                if changed: add('变化 '+','.join(changed), *record)
                elif alarm: add('异常', *record)
                elif not is_csv and ALERT.search(raw): add('关键词', *record)
                if row.get('robot_id') == '3' and row.get('joints_deg'):
                    vals = [number(x) for x in row['joints_deg'].split()]
                    if len(vals) >= 3 and all(v is not None for v in vals[:3]):
                        if waist_last is not None:
                            for j in range(3):
                                delta = vals[j] - waist_last[j]
                                sign = 1 if delta > 0.01 else -1 if delta < -0.01 else 0
                                if sign and waist_sign[j] and sign != waist_sign[j]:
                                    changes[f'腰J{j+1}_疑似方向变化'] += 1
                                    add(f'腰J{j+1}疑似方向变化(单步阈值0.01度)', *record)
                                if sign: waist_sign[j] = sign
                        waist_last = vals
                        compact_waist = vals[:3] + [sum(vals[:3])]
                        for j,v in enumerate(compact_waist):
                            k = ('腰J1指令度','腰J2指令度','腰J3指令度','腰J1+J2+J3指令度')[j]
                            if k not in extrema: extrema[k] = [(v,lineno,t),(v,lineno,t)]
                            else:
                                if v < extrema[k][0][0]: extrema[k][0] = (v,lineno,t)
                                if v > extrema[k][1][0]: extrema[k][1] = (v,lineno,t)
                for k in METRICS:
                    v = number(fields.get(k))
                    if v is None: continue
                    if k not in extrema: extrema[k] = [(v,lineno,t),(v,lineno,t)]
                    else:
                        if v < extrema[k][0][0]: extrema[k][0] = (v,lineno,t)
                        if v > extrema[k][1][0]: extrema[k][1] = (v,lineno,t)
        except csv.Error as exc:
            errors += 1; print(f'  CSV 解析中断: {exc}; line≈{reader.line_num}')
    after = path.stat()
    print(f'\n## {path}\n记录={count} 异常/残缺行={errors} 字节={before.st_size}→{after.st_size}'+(' 文件变化，快照可能不一致' if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) else ''))
    if first: print(f'首行 L{first[0]} t={first[1]} | {first[2]}')
    if tail and tail != first: print(f'末行 L{tail[0]} t={tail[1]} | {tail[2]}')
    if changes: print('变化次数: '+', '.join(f'{k}={v}' for k,v in changes.most_common()))
    for k, (lo,hi) in extrema.items(): print(f'极值 {k}: 最小={lo[0]:g}(L{lo[1]},t={lo[2]}) 最大={hi[0]:g}(L{hi[1]},t={hi[2]})')
    total_candidates = len(chosen) + dropped
    if recent: chosen.extend(recent)
    print(f'候选事件={total_candidates} 展示上限={limit}; 省略={max(0,total_candidates-min(len(chosen),limit))}（事件较多时只保留首尾候选）')
    # Spread limited examples across the run, include first and last candidate.
    indices = range(len(chosen)) if len(chosen) <= limit else sorted({round(i*(len(chosen)-1)/(limit-1)) for i in range(limit)}) if limit > 1 else [0]
    for i in indices:
        kind,line,t,detail = chosen[i]; print(f'  L{line} t={t} [{kind}] {detail}')
    if context: print('回查: 用 --window 文件:行号 --context N 查看事件附近的原始行。')

def main():
    p=argparse.ArgumentParser(description='只读精简 KMPC CSV / 文本日志；汇总不是完整证据。')
    p.add_argument('input', nargs='?', help='运行目录或单个文件')
    p.add_argument('--limit', type=int, default=30, help='每个文件展示事件数（默认30）')
    p.add_argument('--window', help='回查原文件:物理行号（例如 servo.csv:123）')
    p.add_argument('--context', type=int, default=3, help='回查前后行数')
    args=p.parse_args()
    if args.limit < 1 or args.context < 0: p.error('limit 须大于0，context 不得为负')
    if args.window:
        try: name,n = args.window.rsplit(':',1); n=int(n); assert n > 0
        except (ValueError,AssertionError): p.error('--window 格式为 文件:正整数行号')
        with open(name, encoding='utf-8-sig', errors='replace') as f:
            for i,line in enumerate(f,1):
                if n-args.context <= i <= n+args.context: print(f'{name}:L{i}: {line.rstrip()}')
                if i >= n+args.context: break
        return
    if not args.input: p.error('需要运行目录或文件')
    path=Path(args.input)
    if not path.exists(): p.error(f'路径不存在: {path}')
    files=sorted(x for x in path.iterdir() if x.is_file() and x.suffix.lower() in ('.csv','.log','.txt')) if path.is_dir() else [path]
    if not files: p.error('未发现 CSV / log / txt 文件')
    print('KMPC 日志初筛：时间基准/单位以原始记录为准；结果不证明连续过程合规，关键时段必须回查。')
    failed=0
    for f in files:
        try: summarize(f,args.limit,True)
        except (OSError,ValueError,UnicodeError) as exc: failed+=1; print(f'\n## {f}\n读取失败: {exc}', file=sys.stderr)
    return 1 if failed else 0

if __name__ == '__main__': sys.exit(main())
