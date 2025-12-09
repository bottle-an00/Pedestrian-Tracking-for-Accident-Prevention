#!/usr/bin/env python3
"""Create missing ekf_bev MP4s by deriving an indexed pattern from filenames.
This is more robust than glob patterns when filenames are like ekf_bev_gt_13_006_000.jpg
"""
import re
import subprocess
from pathlib import Path

jpath = Path('results/ekf_videos_presence.json')
if not jpath.exists():
    raise SystemExit('results/ekf_videos_presence.json not found')
import json
j = json.load(open(jpath))
created = []
failed = []
for s,v in j.items():
    if not (v.get('bev_frames_present') and not v.get('ekf_bev_mp4')):
        continue
    bev_dir = Path('data/raw')/s/'outputs'/'full_pipeline_test'/'ekf_bev'
    if not bev_dir.exists():
        failed.append(f"no_bev_dir:{s}")
        continue
    # pick first file
    files = sorted([p.name for p in bev_dir.iterdir() if p.name.lower().endswith('.jpg')])
    if not files:
        failed.append(f"no_jpgs:{s}")
        continue
    first = files[0]
    # find trailing digits before .jpg
    m = re.search(r"(.*?)(\d+)\.jpe?g$", first, flags=re.IGNORECASE)
    if not m:
        # fallback to glob
        pattern = 'ekf_bev_*.jpg'
    else:
        prefix = m.group(1)
        digits = len(m.group(2))
        pattern = f"{prefix}%0{digits}d.jpg"
    out = bev_dir.parent / f'ekf_bev_{s}.mp4'
    print(f"Creating {out} using pattern {pattern} in {bev_dir}")
    # Ensure ffmpeg writes to an absolute path (we run with cwd=bev_dir)
    out_abs = str(out.resolve())
    cmd = ['ffmpeg','-y','-framerate','10','-i', pattern, '-c:v','libx264','-pix_fmt','yuv420p', out_abs]
    try:
        r = subprocess.run(cmd, cwd=str(bev_dir))
        if r.returncode == 0:
            created.append(str(out))
        else:
            failed.append(str(out))
    except Exception as e:
        failed.append(f"exc:{s}:{e}")

print('Done. created', len(created), 'failed', len(failed))
for p in created:
    print('CREATED', p)
for p in failed:
    print('FAILED', p)
