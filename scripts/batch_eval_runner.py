#!/usr/bin/env python3
"""Batch runner: for each sequence, set dataset_id, run full pipeline test, normalize pred JSONs, run evaluator.
Usage: python3 scripts/batch_eval_runner.py 006 009 017 ...
"""
import sys
import subprocess
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'configs' / 'system.yaml'
BACKUP = ROOT / 'configs' / 'system.yaml.bak'

SEQ_LIST = sys.argv[1:]
if not SEQ_LIST:
    print('Usage: python3 scripts/batch_eval_runner.py <seq1> <seq2> ...')
    sys.exit(1)

# require existing backup (do not create a new one)
if not BACKUP.exists():
    print('backup missing:', BACKUP)
    print('Please create an existing backup at configs/system.yaml.bak before running this script.')
    sys.exit(1)
else:
    print('using existing backup', BACKUP)

results = {}

for seq in SEQ_LIST:
    print('\n==== Processing sequence', seq, '====')
    # replace dataset_id line in config
    text = CONFIG.read_text()
    new_text = []
    replaced = False
    for line in text.splitlines(True):
        if line.strip().startswith('dataset_id:') and not replaced:
            new_text.append(f"dataset_id: '{seq}'\n")
            replaced = True
        else:
            new_text.append(line)
    CONFIG.write_text(''.join(new_text))
    print('wrote dataset_id ->', seq)

    # run pytest single test
    print('running pipeline test (this may take a while)...')
    r = subprocess.run(['pytest', '-q', 'tests/test_final_full_pipeline.py::test_full_pipeline'], cwd=ROOT)
    if r.returncode != 0:
        print('pytest failed for', seq, 'skipping evaluation')
        results[seq] = {'status': 'pytest_failed'}
        continue

    # locate pred_json dir
    pred_dir = ROOT / f'data/raw/{seq}/outputs/full_pipeline_test/pred_json'
    if not pred_dir.exists():
        print('pred json dir not found for', seq, 'expected at', pred_dir)
        results[seq] = {'status': 'no_pred_dir'}
        continue

    # normalize JSONs
    print('normalizing pred jsons in', pred_dir)
    cnt = 0
    for f in sorted(pred_dir.glob('frame_*.json')):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        objs = j.get('objects', [])
        changed = False
        for o in objs:
            pos = o.get('pos')
            if isinstance(pos, list):
                o['pos'] = {'x': float(pos[0]), 'y': float(pos[1])}
                changed = True
            fut = o.get('future')
            if isinstance(fut, list) and fut and isinstance(fut[0], list):
                o['future'] = [{'x': float(p[0]), 'y': float(p[1])} for p in fut]
                changed = True
        if changed:
            json.dump(j, open(f, 'w'), indent=2)
        cnt += 1
    print('normalized', cnt, 'files')

    # run evaluator
    outpkl = ROOT / f'results/{seq}/eval_with_pred_pose.pkl'
    outpkl.parent.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / f'data/raw/{seq}/image0'
    gt_dir = ROOT / f'data/gt/{seq}/image0'
    if not img_dir.exists() or not gt_dir.exists():
        print('missing image or gt dir, skipping eval for', seq)
        results[seq] = {'status': 'missing_image_or_gt'}
        continue

    cmd = ['python3', 'scripts/run_evaluation.py', '--model', 'models/yolo/yolo11s-pose.pt', '--image-dir', str(img_dir), '--image-label-dir', str(gt_dir), '--pred-json-dir', str(pred_dir), '--output', str(outpkl)]
    print('running evaluator...')
    r2 = subprocess.run(cmd, cwd=ROOT)
    if r2.returncode != 0:
        print('evaluator failed for', seq)
        results[seq] = {'status': 'eval_failed'}
        continue

    print('done eval for', seq, '->', outpkl)
    results[seq] = {'status': 'ok', 'out': str(outpkl)}

# restore config
if BACKUP.exists():
    shutil.copy(BACKUP, CONFIG)
    print('\nrestored original config from backup')

print('\nSUMMARY:')
for s, v in results.items():
    print(s, v)

print('\ndone')
