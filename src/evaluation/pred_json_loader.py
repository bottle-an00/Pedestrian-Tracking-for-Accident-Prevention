import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_prediction_json(file_path: str) -> Optional[Dict]:
    p = Path(file_path)
    if not p.exists():
        return None

    with open(p, 'r') as jf:
        data = json.load(jf)

    # normalize to: { 'frame': int, 'objects': [ {id:int, pos:(x,y), obs_count:int, future:[(x,y), ...], pos_bev:{'u':..,'v':..}, future_bev:[...] }, ... ], 'meta': {...} }
    out = {'frame': int(data.get('frame', 0)), 'objects': [], 'meta': data.get('meta', None)}
    for obj in data.get('objects', []):
        oid = int(obj.get('id'))
        pos = obj.get('pos', {})
        x = float(pos.get('x', 0.0))
        y = float(pos.get('y', 0.0))
        obs = int(obj.get('obs_count', 0))
        future = []
        for f in obj.get('future', []):
            future.append((float(f.get('x', 0.0)), float(f.get('y', 0.0))))
        # optionally include BEV pixel coordinates if present
        pos_bev = None
        if isinstance(obj.get('pos_bev', None), dict):
            try:
                pos_bev = {'u': int(obj['pos_bev'].get('u', -1)), 'v': int(obj['pos_bev'].get('v', -1))}
            except Exception:
                pos_bev = None

        future_bev = []
        for fb in obj.get('future_bev', []):
            try:
                future_bev.append({'u': int(fb.get('u', -1)), 'v': int(fb.get('v', -1))})
            except Exception:
                continue

        out['objects'].append({'id': oid, 'pos': (x, y), 'obs_count': obs, 'future': future, 'pos_bev': pos_bev, 'future_bev': future_bev})

    return out
