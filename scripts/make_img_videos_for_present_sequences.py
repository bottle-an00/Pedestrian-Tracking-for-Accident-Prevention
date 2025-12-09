#!/usr/bin/env python3
"""Create ekf_img_{seq}.mp4 only for sequences that have ekf_img frames but no ekf_img_{seq}.mp4
Uses same numeric-pattern approach as fix_make_bev_videos.py to be robust to naming.
"""
import re
import subprocess
from pathlib import Path
import json

presence_path = Path('results/ekf_videos_presence.json')
if presence_path.exists():
    presence = json.load(open(presence_path))
else:
    presence = {}

created = []
failed = []
for seq, info in presence.items():
    # Only target sequences with img frames present but missing img mp4
    if not (info.get('img_frames_present') and not info.get('img_mp4')):
        continue
    img_dir = Path('data/raw')/seq/'outputs'/'full_pipeline_test'/'ekf_img'
    if not img_dir.exists():
        failed.append(f'no_img_dir:{seq}')
        continue
    files = sorted([p.name for p in img_dir.iterdir() if p.name.lower().endswith('.jpg')])
    if not files:
        failed.append(f'no_jpgs:{seq}')
        continue
    first = files[0]
    m = re.search(r"(.*?)(\d+)\.jpe?g$", first, flags=re.IGNORECASE)
    if not m:
        pattern = 'ekf_img_*.jpg'
    else:
        prefix = m.group(1)
        digits = len(m.group(2))
        pattern = f"{prefix}%0{digits}d.jpg"
    out = img_dir.parent / f'ekf_img_{seq}.mp4'
    out_abs = str(out.resolve())
    print(f"Creating {out} using pattern {pattern} in {img_dir}")
    cmd = ['ffmpeg','-y','-framerate','10','-i', pattern, '-c:v','libx264','-pix_fmt','yuv420p', out_abs]
    try:
        r = subprocess.run(cmd, cwd=str(img_dir))
        if r.returncode == 0:
            created.append(str(out))
        else:
            failed.append(str(out))
    except Exception as e:
        failed.append(f"exc:{seq}:{e}")

print('Done. created', len(created), 'failed', len(failed))
for p in created:
    print('CREATED', p)
for p in failed:
    print('FAILED', p)
