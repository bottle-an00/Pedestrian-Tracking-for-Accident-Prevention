#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess

jpath = Path('results/ekf_videos_presence.json')
if not jpath.exists():
    raise SystemExit('results/ekf_videos_presence.json not found')

j = json.load(open(jpath))
created = []
failed = []
for s, v in j.items():
    d = Path('data/raw') / s / 'outputs' / 'full_pipeline_test'
    # ekf_img
    if v.get('img_frames_present') and not v.get('ekf_img_mp4'):
        img_dir = d / 'ekf_img'
        out = d / f'ekf_img_{s}.mp4'
        if img_dir.exists():
            print('Creating', out)
            r = subprocess.run(['ffmpeg', '-y', '-framerate', '10', '-pattern_type', 'glob', '-i', 'ekf_img_*.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], cwd=str(img_dir))
            if r.returncode == 0:
                created.append(str(out))
            else:
                failed.append(str(out))
    # ekf_bev
    if v.get('bev_frames_present') and not v.get('ekf_bev_mp4'):
        bev_dir = d / 'ekf_bev'
        out = d / f'ekf_bev_{s}.mp4'
        if bev_dir.exists():
            print('Creating', out)
            r = subprocess.run(['ffmpeg', '-y', '-framerate', '10', '-pattern_type', 'glob', '-i', 'ekf_bev_*', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], cwd=str(bev_dir))
            if r.returncode == 0:
                created.append(str(out))
            else:
                failed.append(str(out))

print('Done. created:', len(created), 'failed:', len(failed))
for p in created:
    print('CREATED', p)
for p in failed:
    print('FAILED', p)
