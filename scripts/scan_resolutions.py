#!/usr/bin/env python3
"""Scan data/raw/*/image0 and record image resolutions per sequence.
Produces results/resolution_map.json with:
  { "by_resolution": {"WxH": [seq,...]}, "by_sequence": {seq: {width, height, sample_image}} }
"""
import json
from pathlib import Path
try:
    import cv2
except Exception:
    cv2 = None

root = Path('data/raw')
res_map = {}
seq_map = {}
for p in sorted([d for d in root.iterdir() if d.is_dir()]):
    seq = p.name
    img_dir = p / 'image0'
    if not img_dir.exists():
        continue
    imgs = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tiff']])
    if not imgs:
        continue
    sample = imgs[0]
    try:
        if cv2:
            im = cv2.imread(str(sample))
            if im is None:
                raise RuntimeError('cv2.imread returned None')
            h, w = im.shape[:2]
        else:
            from PIL import Image
            im = Image.open(sample)
            w, h = im.size
    except Exception:
        from PIL import Image
        im = Image.open(sample)
        w, h = im.size
    key = f"{w}x{h}"
    res_map.setdefault(key, []).append(seq)
    seq_map[seq] = {'width': w, 'height': h, 'sample_image': str(sample)}

out = {'by_resolution': res_map, 'by_sequence': seq_map}
Path('results').mkdir(exist_ok=True)
with open('results/resolution_map.json', 'w') as f:
    json.dump(out, f, indent=2)

print('Wrote results/resolution_map.json')
for k, v in sorted(res_map.items(), key=lambda x: (-len(x[1]), x[0])):
    print(f"{k}: {len(v)} sequences -> {v}")
