#!/usr/bin/env python3
import re
import json
from pathlib import Path
import pickle

results_root = Path('results')
seqs = []
for p in sorted(results_root.iterdir()):
    if not p.is_dir():
        continue
    txt = p / 'eval_with_pred_pose.txt'
    pkl = p / 'eval_with_pred_pose.pkl'
    ade = None
    fde = None
    if txt.exists():
        s = txt.read_text()
        m = re.search(r'Trajectory: mean_ADE=([0-9.+-eE]+)\s+mean_FDE=([0-9.+-eE]+)', s)
        if m:
            ade = float(m.group(1))
            fde = float(m.group(2))
    if ade is None and pkl.exists():
        try:
            with open(pkl,'rb') as f:
                obj = pickle.load(f)
            # try known places
            traj = None
            if hasattr(obj,'overall_metrics'):
                traj = obj.overall_metrics.get('trajectory')
            elif isinstance(obj, dict):
                traj = obj.get('overall_metrics', {}).get('trajectory') or obj.get('trajectory')
            if traj:
                ade = float(traj.get('mean_ADE'))
                fde = float(traj.get('mean_FDE'))
        except Exception:
            pass
    if ade is not None and fde is not None:
        seqs.append({'seq': p.name, 'mean_ADE': ade, 'mean_FDE': fde})

# filter those with both < 3.0
sel = [x for x in seqs if x['mean_ADE'] < 3.0 and x['mean_FDE'] < 3.0]
if sel:
    mean_ade = sum(x['mean_ADE'] for x in sel) / len(sel)
    mean_fde = sum(x['mean_FDE'] for x in sel) / len(sel)
else:
    mean_ade = None
    mean_fde = None

out = {
    'count_sequences_considered': len(sel),
    'sequences': sel,
    'average_mean_ADE': mean_ade,
    'average_mean_FDE': mean_fde,
}
with open(results_root / 'summary_mean_under3.json','w') as f:
    json.dump(out,f,indent=2)
print('Wrote results/summary_mean_under3.json')
print('Count:', out['count_sequences_considered'])
print('Average mean_ADE:', out['average_mean_ADE'])
print('Average mean_FDE:', out['average_mean_FDE'])
print('\nSequences considered:')
for x in sel:
    print(f"{x['seq']}: mean_ADE={x['mean_ADE']}, mean_FDE={x['mean_FDE']}")
