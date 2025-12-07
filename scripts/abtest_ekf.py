#!/usr/bin/env python3
"""AB test for EKF parameterization and model (CA vs CV).

Generates predictions using GT detections as inputs (so detection errors are not confounding)
and runs the existing prediction evaluator to compute ADE/FDE.

Outputs CSV at results/ekf_abtest.csv
"""
import sys
from pathlib import Path
import csv
import argparse

# ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.evaluation.runner import EvaluationResult
from src.evaluation.prediction_evaluator import evaluate_prediction_sequence as eval_pred_seq
from src.evaluation.gt_loader import GTLoader
from src.trajectory.pedestrian_state_manager import PedestrianStateManager
from src.trajectory.ekf_tracker import EKFTracker
from src.trajectory.ekf_tracker_cv import EKFTrackerCV


def generate_predictions_from_gt(image_label_dir: Path, out_root: Path, ekf_constructor, obs_len=10, T=20):
    image_label_dir = Path(image_label_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pred_json_dir = out_root / 'pred_json'
    pred_json_dir.mkdir(parents=True, exist_ok=True)

    gt_loader = GTLoader(target_classes=['pedestrian'])

    # homography not needed for GT labels here: image label has foot_uv -> convert later if required by pipeline
    # But prediction saving expects pos in local BEV meters; we will use pixel->world conversion via Homography later in evaluator

    state_manager = PedestrianStateManager(obs_len=obs_len, max_missing=5, ekf_constructor=ekf_constructor)

    # iterate sorted label files
    files = sorted([p for p in image_label_dir.glob('*.json')])
    for pf in files:
        frame_idx = int(pf.stem.split('_')[-1]) if '_' in pf.stem else int(pf.stem)
        try:
            data = gt_loader.load_image_label(pf)
            img_objs, _ = gt_loader.parse_image_label(data)
        except Exception:
            img_objs = []

        detections = []
        for obj in img_objs:
            if obj.foot_uv is None:
                continue
            # Use track_id if available otherwise instance_id
            tid = obj.track_id if obj.track_id is not None else obj.instance_id if obj.instance_id is not None else -1
            # The PedestrianStateManager expects local BEV meters; but without homography we will place the {
            # pos value as (u,v) placeholder; later evaluator will convert when pred_in_bev_pixels is set.
            # Simpler: store pixel coords and mark pred_in_bev_pixels=True in evaluator.
            # However prediction_evaluator expects world meters by default; to avoid complexity, we'll convert using Homography.
            detections.append((int(tid), obj.foot_uv))

        # If detections are in image uv, need to convert to world meters using Homography. We'll leave this to the caller by providing pixel data
        # For this generator, we'll convert to dummy world coords by using the uv directly as (x,y) — but better to require user pass calibration_dir.
        # To keep this script simple, require calibration_dir argument in main.

        # Save: we need to provide world positions; so main will do conversion before calling this function.
    state_manager.update(detections, frame_idx)
    state_manager.generate_future_predictions(T=T)
    state_manager.save_predictions_json(frame_idx, str(out_root), eval_mode=True, bev_transformer=None)

    return pred_json_dir


def run_abtest(sequence_root: Path, image_label_subdir: str, calib_dir: Path, out_base: Path, grid, obs_lens, T=20, dt: float = 1.0):
    image_label_dir = sequence_root / image_label_subdir
    # build homography
    loader = CalibrationInfoLoader()
    cam_file = Path(calib_dir) / 'calib_Camera0.txt'
    ext_file = Path(calib_dir) / 'calib_CameraToLidar0.txt'
    bev = None
    if cam_file.exists() and ext_file.exists():
        cam = loader.load_camera_calibration(str(cam_file))
        ext = loader.load_camera_extrinsics(str(ext_file))
        bev = Homography(intrinsics=cam, extrinsics=ext)

    results = []

    for model in ['CA','CV']:
        for q in grid['q']:
            for R in grid['R']:
                for obs_len in obs_lens:
                    tag = f"{model}_q{q}_R{R}_obs{obs_len}"
                    out_root = out_base / tag
                    out_root.mkdir(parents=True, exist_ok=True)

                    # choose ekf_constructor
                    if model == 'CA':
                        ekf_constructor = lambda q=q, R=R: EKFTracker((0.0,0.0), dt=dt, q=q, R_scale=R)
                    else:
                        ekf_constructor = lambda q=q, R=R: EKFTrackerCV((0.0,0.0), dt=dt, q=q, R_scale=R)

                    # generate predictions: but we need to convert image uv -> world meters using bev
                    # simpler approach: generate pred jsons directly using gt world positions computed here
                    # load GT labels and convert foot_uv -> world meters
                    gt_loader = GTLoader(target_classes=['pedestrian'])
                    files = sorted([p for p in image_label_dir.glob('*.json')])
                    state_manager = PedestrianStateManager(obs_len=obs_len, max_missing=5, ekf_constructor=ekf_constructor, dt=dt)

                    for pf in files:
                        frame_idx = int(pf.stem.split('_')[-1]) if '_' in pf.stem else int(pf.stem)
                        try:
                            data = gt_loader.load_image_label(pf)
                            img_objs, _ = gt_loader.parse_image_label(data)
                        except Exception:
                            img_objs = []

                        detections_world = []
                        for obj in img_objs:
                            if obj.foot_uv is None:
                                continue
                            u,v = obj.foot_uv
                            if bev is None:
                                continue
                            try:
                                bx, by = bev.pixel_to_bev_warp(u, v)
                                x_m, y_m = bev.pixel_to_world(float(by), float(bx), flipped=True)
                            except Exception:
                                continue
                            tid = obj.track_id if obj.track_id is not None else obj.instance_id if obj.instance_id is not None else -1
                            detections_world.append((int(tid), (x_m, y_m)))

                        state_manager.update(detections_world, frame_idx)
                        state_manager.generate_future_predictions(T=T)
                        state_manager.save_predictions_json(frame_idx, str(out_root), eval_mode=True, bev_transformer=bev)

                    # run evaluation on generated pred_json using prediction_evaluator directly
                    traj = eval_pred_seq(
                        image_label_dir=image_label_dir,
                        lidar_label_dir=None,
                        pred_json_dir=out_root / 'pred_json',
                        T=T,
                        distance_threshold=2.0,
                        use_hungarian=False,
                        bev_transformer=bev,
                    )

                    mean_ADE = traj.get('mean_ADE', float('nan'))
                    mean_FDE = traj.get('mean_FDE', float('nan'))
                    n = traj.get('n_samples', 0)

                    results.append({'model': model, 'q': q, 'R': R, 'obs_len': obs_len, 'mean_ADE': mean_ADE, 'mean_FDE': mean_FDE, 'n': n, 'tag': tag})
                    print('Done', tag, mean_ADE, mean_FDE, n)

    # write CSV
    out_csv = out_base / 'ekf_abtest.csv'
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['tag','model','q','R','obs_len','mean_ADE','mean_FDE','n'])
        writer.writeheader()
        for r in results:
            writer.writerow({'tag': r['tag'], 'model': r['model'], 'q': r['q'], 'R': r['R'], 'obs_len': r['obs_len'], 'mean_ADE': r['mean_ADE'], 'mean_FDE': r['mean_FDE'], 'n': r['n']})

    print('AB test finished. Results:', out_csv)
    return out_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-root', type=str, required=False, default='data/raw/053')
    parser.add_argument('--image-label-subdir', type=str, default='image0')
    parser.add_argument('--calib-dir', type=str, required=False, default='data/raw/053')
    parser.add_argument('--out', type=str, default='results/ekf_abtest')
    parser.add_argument('--dt', type=float, default=1.0, help='time step dt to pass to EKF')
    parser.add_argument('--q', type=str, default=None, help='comma-separated q grid override')
    parser.add_argument('--R', type=str, default=None, help='comma-separated R grid override')
    parser.add_argument('--obs', type=str, default=None, help='comma-separated obs_len override')
    args = parser.parse_args()

    # finer default grid; can be overridden by --q and --R args (comma-separated)
    grid = {'q': [0.05, 0.1, 0.2, 0.5, 1.0], 'R': [0.5, 1.0, 2.0, 3.0]}
    obs_lens = [5, 10]

    # allow override via CLI args
    if args.q:
        grid['q'] = [float(x) for x in args.q.split(',')]
    if args.R:
        grid['R'] = [float(x) for x in args.R.split(',')]
    if args.obs:
        obs_lens = [int(x) for x in args.obs.split(',')]

    dt = float(args.dt)

    # pass dt into run_abtest via lambdas that capture dt
    # update run_abtest call to include dt by partial application using default arg
    run_abtest(Path(args.sequence_root), args.image_label_subdir, Path(args.calib_dir), Path(args.out), grid, obs_lens, T=20, dt=dt)
