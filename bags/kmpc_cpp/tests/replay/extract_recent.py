#!/usr/bin/env python3
"""Read-only extraction. No ROS imports, publishers, sockets or robot clients."""
import csv, re, math, json, argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('output');a=p.parse_args()
root=Path(a.data); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
pat=re.compile(r't=([\d.]+) target_(left|right) xyz=\(([^)]+)\) quat=\(([^)]+)\)')
allcases=[];summary=[]
for folder in sorted(root.iterdir(),key=lambda x:x.stat().st_mtime,reverse=True)[:5]:
 events=[]
 for m in pat.finditer((folder/'notes.txt').read_text()):
  events.append(dict(t=float(m[1]),left=m[2]=='left',xyz=list(map(float,m[3].split(','))),quat=list(map(float,m[4].split(',')))))
 if not events:continue
 with (folder/'jitter.csv').open() as f:
  reader=csv.DictReader(f);names=[x for x in reader.fieldnames if x.endswith('_meas')]
  assert len(names)==18
  idx=0;last=None
  for row in reader:
   if not row['t_s']:continue
   t=float(row['t_s']);q=[float(row[n]) for n in names]
   if not all(map(math.isfinite,q)):continue
   while idx<len(events) and events[idx]['t']<=t:
    events[idx]['q']=last[1] if last else q
    events[idx]['seed_t']=last[0] if last else t
    idx+=1
   last=(t,q)
   if idx==len(events):break
 # The fixed point task's named preparation poses, confirmed against the recorded
 # sequence. The next pose itself is copied from the following recorded command.
 prep={True:[(0.5161,0.2728,0.8150),(0.2458,0.3435,0.6900)],False:[(0.4984,-0.2867,0.8100),(0.2458,-0.3915,0.6900)]}
 for i,e in enumerate(events):
  if 'q' not in e:raise RuntimeError('missing seed')
  e['record']=folder.name;e['index']=i;e['tip']=None
  if i+1<len(events) and events[i+1]['left']==e['left']:
   nxt=events[i+1]
   is_grasp=tuple(e['xyz'])==prep[e['left']][0] and abs(nxt['xyz'][2]-(e['xyz'][2]-.15))<1e-4
   is_place=tuple(e['xyz'])==prep[e['left']][1] and abs(nxt['xyz'][0]-(e['xyz'][0]+.24))<1e-4
   if is_grasp or is_place:e['tip']={k:nxt[k] for k in ['xyz','quat']}
  allcases.append(e)
 summary.append(dict(record=folder.name,segments=len(events),seed_max_age_s=max(abs(e['t']-e['seed_t']) for e in events)))
(out/'cases.json').write_text(json.dumps(allcases,ensure_ascii=False,indent=2))
with (out/'cases.tsv').open('w') as f:
 for e in allcases:
  tip=e['tip'];vals=[e['record'],e['index'],int(e['left']),e['t'],*e['q'],*e['xyz'],*e['quat'],int(tip is not None)]
  vals+=tip['xyz']+tip['quat'] if tip else [0,0,0,0,0,0,1]
  f.write(' '.join(map(str,vals))+'\n')
print(json.dumps(summary,indent=2))
