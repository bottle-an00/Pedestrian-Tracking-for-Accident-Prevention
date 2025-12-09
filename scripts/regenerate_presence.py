#!/usr/bin/env python3
import json
from pathlib import Path

root = Path('data/raw')
out = {}
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    seq = d.name
    p = d / 'outputs' / 'full_pipeline_test'
    info = {'img_mp4': False, 'bev_mp4': False, 'img_frames_present': False, 'bev_frames_present': False}
    if p.exists():
        img_mp4 = (p / f'ekf_img_{seq}.mp4')
        bev_mp4 = (p / f'ekf_bev_{seq}.mp4')
        info['img_mp4'] = img_mp4.exists()
        info['bev_mp4'] = bev_mp4.exists()
        img_dir = p / 'ekf_img'
        bev_dir = p / 'ekf_bev'
        info['img_frames_present'] = any(img_dir.glob('*.jpg')) if img_dir.exists() else False
        info['bev_frames_present'] = any(bev_dir.glob('*.jpg')) if bev_dir.exists() else False
    out[seq] = info

Path('results').mkdir(exist_ok=True)
with open('results/ekf_videos_presence.json', 'w') as fh:
    json.dump(out, fh, indent=2)

print('wrote results/ekf_videos_presence.json')
