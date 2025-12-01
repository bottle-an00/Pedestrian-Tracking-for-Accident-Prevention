"""Pedestrian prediction evaluation utilities: ADE/FDE and simple risk-area entry check.

Expected data shapes:
- predictions: dict mapping track_id -> np.ndarray shape (T_pred, 2) in BEV coords (x,y)
- ground_truth: dict mapping track_id -> np.ndarray shape (T_gt, 2)

Functions return scalar metrics or per-track arrays for further analysis.
"""

import numpy as np
from typing import Dict, Tuple


def ade_fde(predictions: Dict[int, np.ndarray], ground_truth: Dict[int, np.ndarray]) -> Tuple[float, float]:
    """Compute mean ADE and FDE over tracks that appear in both dicts.

    ADE: average displacement error over all predicted timesteps
    FDE: final displacement error at last predicted timestep
    """
    ade_list = []
    fde_list = []

    for tid, pred in predictions.items():
        gt = ground_truth.get(tid)
        if gt is None:
            continue
        # align lengths: use min length
        T = min(pred.shape[0], gt.shape[0])
        if T == 0:
            continue
        diff = pred[:T] - gt[:T]
        dists = np.linalg.norm(diff, axis=1)
        ade_list.append(dists.mean())
        fde_list.append(dists[-1])

    if len(ade_list) == 0:
        return float('nan'), float('nan')

    return float(np.mean(ade_list)), float(np.mean(fde_list))


def risk_zone_entry(predictions: Dict[int, np.ndarray], zone_polygon: np.ndarray) -> Dict[int, bool]:
    """Check whether predicted trajectories enter a polygonal risk zone.

    zone_polygon: (N,2) polygon in same BEV coordinates. Uses winding rule via shapely if available,
    otherwise uses a simple ray casting implementation.

    Returns dict: track_id -> True/False
    """
    try:
        from shapely.geometry import Point, Polygon
        poly = Polygon(zone_polygon)
        out = {}
        for tid, pred in predictions.items():
            entered = False
            for p in pred:
                if poly.contains(Point(p[0], p[1])):
                    entered = True
                    break
            out[tid] = entered
        return out
    except Exception:
        # fallback: simple point-in-polygon (ray casting)
        def point_in_poly(x, y, poly):
            inside = False
            n = poly.shape[0]
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
                if intersect:
                    inside = not inside
                j = i
            return inside

        out = {}
        for tid, pred in predictions.items():
            entered = False
            for p in pred:
                if point_in_poly(p[0], p[1], zone_polygon):
                    entered = True
                    break
            out[tid] = entered
        return out
