"""Trajectory prediction evaluation (ADE/FDE) utilities.

This module provides a single function `evaluate_prediction_sequence` which
consumes prediction JSONs, GT lidar labels, and produces aggregated ADE/FDE
metrics. It relies on `matcher`, `pred_json_loader`, `gt_loader`, and
`trajectory_evaluator` modules under `src.evaluation`.
"""
from pathlib import Path
from typing import Optional, List, Dict
import json

from .pred_json_loader import load_prediction_json
from .gt_loader import GTLoader
from .matcher import match_gt_preds
from .trajectory_evaluator import TrajectoryEvaluator


def evaluate_prediction_sequence(
    image_label_dir: str | Path,
    lidar_label_dir: Optional[str | Path],
    pred_json_dir: str | Path,
    T: int = 20,
    distance_threshold: float = 2.0,
    use_hungarian: bool = False,
    mapping_out: Optional[str | Path] = None,
    bev_transformer=None,
    pred_in_bev_pixels: bool = False,
    id_match_first: bool = False,
    pred_frame_start: int = None,
    pred_frame_end: int = None,
) -> Dict:
    """Evaluate prediction JSONs against GT lidar labels for ADE/FDE.

    Args:
        image_label_dir: directory with image label JSONs (not required for lidar-only matching)
        lidar_label_dir: directory with lidar label JSONs (used to find GT by frame index)
        pred_json_dir: directory with prediction JSONs named `frame_XXXX.json`
        T: prediction horizon (number of timesteps)
        distance_threshold: maximum distance (meters) to accept a match
        use_hungarian: if True, try to use Hungarian assignment (matcher autodetects SciPy)
        mapping_out: optional directory to save mapping JSONs; if None, saved next to pred files

    Returns:
        dict: aggregated results from TrajectoryEvaluator.finalize_results()
    """
    pred_json_dir = Path(pred_json_dir)
    lidar_label_dir = Path(lidar_label_dir) if lidar_label_dir else None
    image_label_dir = Path(image_label_dir) if image_label_dir else None

    gt_loader = GTLoader(target_classes=["pedestrian"])  # used to parse image JSONs
    # decide which object provides homography/world<->bev conversions
    homography = None
    if bev_transformer is not None:
        # bev_transformer might be a Homography or a wrapper that exposes .homography
        if hasattr(bev_transformer, 'world_to_bev_img_pixel'):
            homography = bev_transformer
        else:
            try:
                homography = bev_transformer.homography
            except Exception:
                homography = None

    traj_eval = TrajectoryEvaluator(T=T, match_threshold=float(distance_threshold), bev_transformer=homography, flipped=True)

    if not pred_json_dir.exists():
        return traj_eval.finalize_results()

    # only include files exactly matching frame_<digits>.json (exclude frame_XXXX.match.json)
    import re
    pred_files_all = sorted([p for p in pred_json_dir.glob("frame_*.json") if re.match(r"^frame_\d+\.json$", p.name)])

    # If frame range provided, filter files to that inclusive range
    if pred_frame_start is not None or pred_frame_end is not None:
        pred_files = []
        for p in pred_files_all:
            idx = int(p.stem.split('_')[-1])
            if pred_frame_start is not None and idx < int(pred_frame_start):
                continue
            if pred_frame_end is not None and idx > int(pred_frame_end):
                continue
            pred_files.append(p)
    else:
        pred_files = pred_files_all

    # First pass: load all predictions and GTs per frame and store them
    frames = []
    preds_per_frame = []
    gts_per_frame = []

    for pf in pred_files:
        frame_idx = int(pf.stem.split("_")[-1])
        frames.append(frame_idx)

        # load pred JSON
        pred_data = load_prediction_json(str(pf))
        pred_objects = []
        if pred_data is not None:
            for obj in pred_data.get('objects', []):
                pos_val = obj.get('pos')
                if isinstance(pos_val, dict):
                    x = float(pos_val.get('x', 0.0))
                    y = float(pos_val.get('y', 0.0))
                else:
                    try:
                        x = float(pos_val[0]); y = float(pos_val[1])
                    except Exception:
                        x = 0.0; y = 0.0

                future_list = []
                for f in obj.get('future', []):
                    if isinstance(f, dict):
                        fx = float(f.get('x', 0.0)); fy = float(f.get('y', 0.0))
                    else:
                        try:
                            fx = float(f[0]); fy = float(f[1])
                        except Exception:
                            fx = 0.0; fy = 0.0
                    future_list.append((fx, fy))

                pred_objects.append({'id': int(obj.get('id', -1)), 'pos': (x, y), 'future': future_list})

        # convert pred BEV pixels if necessary (only when pred_in_bev_pixels=True)
        if pred_in_bev_pixels and homography is not None:
            converted_preds = []
            for p in pred_objects:
                try:
                    bx = float(p['pos'][0]); by = float(p['pos'][1])
                    x_m, y_m = homography.pixel_to_world(float(by), float(bx), flipped=True)
                except Exception:
                    x_m, y_m = 0.0, 0.0
                future_world = []
                for f in p.get('future', []):
                    try:
                        fbx = float(f[0]); fby = float(f[1])
                        fx_m, fy_m = homography.pixel_to_world(float(fby), float(fbx), flipped=True)
                    except Exception:
                        fx_m, fy_m = 0.0, 0.0
                    future_world.append((fx_m, fy_m))
                converted_preds.append({'id': p['id'], 'pos': (x_m, y_m), 'future': future_world})
            pred_objects = converted_preds

        # load GT from image labels (convert image bbox bottom-center -> BEV local meters via bev warp)
        gt_objects = []
        if image_label_dir and Path(image_label_dir).exists() and homography is not None:
            img_label_path = Path(image_label_dir) / f"{frame_idx}.json"
            if not img_label_path.exists():
                pattern = f"*_{frame_idx:03d}.json"
                matches = sorted(Path(image_label_dir).glob(pattern))
                img_label_path = matches[0] if matches else None

            if img_label_path and Path(img_label_path).exists():
                try:
                    image_data = gt_loader.load_image_label(img_label_path)
                    image_objs, _ = gt_loader.parse_image_label(image_data)
                    for img_obj in image_objs:
                        if img_obj.foot_uv is None:
                            continue
                        u, v = img_obj.foot_uv
                        try:
                            bev_px, bev_py = homography.pixel_to_bev_warp(u, v)
                            if bev_px < 0 or bev_py < 0:
                                continue
                            x_m, y_m = homography.pixel_to_world(float(bev_py), float(bev_px), flipped=True)
                        except Exception:
                            continue
                        obj_id = img_obj.track_id if img_obj.track_id is not None else img_obj.instance_id
                        gt_objects.append({'id': int(obj_id) if obj_id is not None else -1, 'pos': (float(x_m), float(y_m))})
                except Exception:
                    gt_objects = []
        else:
            gt_objects = []

        preds_per_frame.append(pred_objects)
        gts_per_frame.append(gt_objects)

    # Build GT history map and frame-index map so we can slice GT futures
    gt_history_map = {}  # tid -> list of (x,y)
    gt_frame_map = {}    # tid -> list of frame indices corresponding to history entries
    for fi, gt_objs in zip(frames, gts_per_frame):
        for g in gt_objs:
            tid = int(g['id'])
            pos = (float(g['pos'][0]), float(g['pos'][1]))
            gt_history_map.setdefault(tid, []).append(pos)
            gt_frame_map.setdefault(tid, []).append(fi)

    # Inject built history into traj_eval for consistency
    traj_eval.gt_history = gt_history_map

    # Second pass: perform matching per frame and compute ADE/FDE using the pre-built GT history
    # Diagnostics containers
    mapping_counts = {}  # frame_idx -> number of mappings
    per_track_time_series = {}  # tid -> list of {'frame':frame_idx,'ADE':..., 'FDE':...}

    for fi, pf, gf, pf_path in zip(frames, preds_per_frame, gts_per_frame, pred_files):
        frame_idx = fi
        pred_objects = pf
        gt_objects = gf

        # distance-only matching
        pairs, mapping = match_gt_preds(gt_objects, pred_objects, thresh=float(distance_threshold))

        # write mapping
        out_map = Path(mapping_out) if mapping_out else pf_path.with_suffix('.match.json')
        try:
            with open(out_map, 'w') as mf:
                json.dump({'frame': frame_idx, 'mapping': mapping}, mf, indent=2)
        except Exception:
            pass

        # record mapping count for this frame
        mapping_counts[frame_idx] = len(mapping)

        # For each match, compute ADE/FDE using the pre-built GT history
        for r, c, dist in pairs:
            g = gt_objects[r]
            p = pred_objects[c]

            tid = int(g['id'])
            # find the index of this frame in the GT frame map
            if tid not in gt_frame_map:
                continue
            try:
                k = gt_frame_map[tid].index(frame_idx)
            except ValueError:
                continue

            start = k + 1
            end = start + int(T)
            history = gt_history_map.get(tid, [])
            if end > len(history):
                # not enough GT future frames
                continue

            gt_future = history[start:end]

            pred_future = p.get('future', [])
            pred_future_list = [(float(pp[0]), float(pp[1])) for pp in pred_future]
            if len(pred_future_list) != int(T):
                continue

            try:
                ade, fde = traj_eval.compute_ade_fde(gt_future, pred_future_list)
            except Exception:
                continue

            if tid not in traj_eval.track_metrics:
                traj_eval.track_metrics[tid] = {'ADE': [], 'FDE': []}
            traj_eval.track_metrics[tid]['ADE'].append(ade)
            traj_eval.track_metrics[tid]['FDE'].append(fde)
            traj_eval.all_ADE.append(ade)
            traj_eval.all_FDE.append(fde)

            # append to per-track time series
            per_track_time_series.setdefault(tid, []).append({'frame': frame_idx, 'ADE': ade, 'FDE': fde})

    results = traj_eval.finalize_results()

    # Output diagnostics: per-track CSVs and mapping summary JSON
    try:
        out_root = Path('results')
        out_root.mkdir(exist_ok=True)

        tracks_dir = out_root / 'trajectory_tracks'
        tracks_dir.mkdir(parents=True, exist_ok=True)
        # per-track CSVs and summary
        summary_rows = []
        for tid, series in per_track_time_series.items():
            csv_path = tracks_dir / f'track_{tid}.csv'
            with open(csv_path, 'w') as cf:
                cf.write('frame,ADE,FDE\n')
                for row in series:
                    cf.write(f"{row['frame']},{row['ADE']},{row['FDE']}\n")
            ade_list = [r['ADE'] for r in series]
            fde_list = [r['FDE'] for r in series]
            summary_rows.append({'tid': tid, 'n': len(series), 'mean_ADE': float(sum(ade_list)/len(ade_list)), 'mean_FDE': float(sum(fde_list)/len(fde_list))})

        # write summary CSV
        if summary_rows:
            summary_path = tracks_dir / 'track_summary.csv'
            with open(summary_path, 'w') as sf:
                sf.write('tid,n,mean_ADE,mean_FDE\n')
                for r in summary_rows:
                    sf.write(f"{r['tid']},{r['n']},{r['mean_ADE']},{r['mean_FDE']}\n")

        # mapping summary JSON
        mapping_dir = out_root / 'mapping_summary'
        mapping_dir.mkdir(parents=True, exist_ok=True)
        mapping_summary = {
            'total_frames': len(frames),
            'frames_with_mappings': sum(1 for v in mapping_counts.values() if v>0),
            'total_mappings': sum(mapping_counts.values()),
            'mapping_counts_per_frame': mapping_counts,
        }
        with open(mapping_dir / 'mapping_summary.json', 'w') as mf:
            json.dump(mapping_summary, mf, indent=2)
    except Exception:
        pass

    return results
