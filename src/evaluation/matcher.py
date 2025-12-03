"""GT <-> Prediction matching utilities

Provides a one-to-one distance-based matcher with optional Hungarian assignment.
"""
from typing import List, Dict, Tuple
import math
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_HUNGARIAN = True
except Exception:
    _HAS_HUNGARIAN = False


def match_gt_preds(
    gt_objects: List[Dict],
    pred_objects: List[Dict],
    thresh: float = 2.0,
) -> Tuple[List[Tuple[int, int, float]], List[Dict]]:
    """Match GT and predicted objects one-to-one by 2D Euclidean distance.

    Args:
        gt_objects: [{'id': int, 'pos': (x,y)}]
        pred_objects: [{'id': int, 'pos': (x,y), 'future': [...]}]
        thresh: maximum distance (meters) for a valid match

    Returns:
        pairs: list of tuples (gt_index, pred_index, distance)
        mapping: list of dicts with keys gt_index, gt_id, pred_index, pred_id, distance
    """
    pairs = []
    mapping = []

    if not gt_objects or not pred_objects:
        return pairs, mapping

    gt_pos = np.array([g['pos'] for g in gt_objects], dtype=float)
    pred_pos = np.array([p['pos'] for p in pred_objects], dtype=float)
    dists = np.linalg.norm(gt_pos[:, None, :] - pred_pos[None, :, :], axis=2)

    # mask distances beyond threshold
    dists_masked = dists.copy()
    dists_masked[dists_masked > thresh] = np.inf

    if _HAS_HUNGARIAN:
        try:
            row_ind, col_ind = linear_sum_assignment(dists_masked)
            for r, c in zip(row_ind, col_ind):
                if math.isfinite(dists_masked[r, c]):
                    pairs.append((int(r), int(c), float(dists[r, c])))
        except Exception:
            # If Hungarian fails (e.g. infeasible cost matrix), fall back to greedy matching
            ds = dists_masked.copy()
            while True:
                idx = np.unravel_index(np.argmin(ds), ds.shape)
                r, c = int(idx[0]), int(idx[1])
                if not math.isfinite(ds[r, c]):
                    break
                pairs.append((r, c, float(dists[r, c])))
                ds[r, :] = np.inf
                ds[:, c] = np.inf
    else:
        # greedy minimal matching
        ds = dists_masked.copy()
        while True:
            idx = np.unravel_index(np.argmin(ds), ds.shape)
            r, c = int(idx[0]), int(idx[1])
            if not math.isfinite(ds[r, c]):
                break
            pairs.append((r, c, float(dists[r, c])))
            ds[r, :] = np.inf
            ds[:, c] = np.inf

    for r, c, dist in pairs:
        mapping.append({
            'gt_index': int(r),
            'gt_id': int(gt_objects[r]['id']),
            'pred_index': int(c),
            'pred_id': int(pred_objects[c]['id']),
            'distance': float(dist),
        })

    return pairs, mapping
