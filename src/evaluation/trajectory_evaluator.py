from typing import List, Dict, Tuple, Optional
import numpy as np


class TrajectoryEvaluator:
    """Offline trajectory evaluator for ADE/FDE using GT loader and pred_json format.

    All coordinates must be in world meters (LiDAR frame).
    """

    def __init__(self, T: int, match_threshold: float = 2.0, bev_transformer=None, flipped: bool = True):
        self.T = int(T)
        self.match_threshold = float(match_threshold)
        # Optional BEV transformer retained for potential future use (not used for metrics)
        self.bev_transformer = bev_transformer
        # flipped kept for compatibility but not used
        self.flipped = bool(flipped)

        # GT history: track_id -> list of (x,y) ordered by frame index
        self.gt_history: Dict[int, List[Tuple[float, float]]] = {}

        # Metrics
        self.track_metrics: Dict[int, Dict[str, List[float]]] = {}
        self.all_ADE: List[float] = []
        self.all_FDE: List[float] = []

        # Keep track of last processed frame index for GT histories length management
        self.last_frame_idx: Optional[int] = None

    def match(self, gt_objects: List[Dict], pred_objects: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match GT and predictions using greedy nearest-neighbor within threshold.

        gt_objects: list of {'id': int, 'pos': (x,y)}
        pred_objects: list of {'id': int, 'pos': (x,y), 'future': [(x,y), ...]}
        Returns list of (gt_obj, pred_obj) matched pairs.
        """
        matches: List[Tuple[Dict, Dict]] = []

        if not gt_objects or not pred_objects:
            return matches

        # Build arrays
        gt_pos = np.array([g['pos'] for g in gt_objects], dtype=np.float32)
        pred_pos = np.array([p['pos'] for p in pred_objects], dtype=np.float32)

        # Compute distance matrix
        dists = np.linalg.norm(gt_pos[:, None, :] - pred_pos[None, :, :], axis=2)

        # Greedy matching: find minimal pair repeatedly
        gt_indices = set(range(len(gt_objects)))
        pred_indices = set(range(len(pred_objects)))

        while gt_indices and pred_indices:
            # find global min
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            g_idx, p_idx = int(min_idx[0]), int(min_idx[1])
            min_dist = float(dists[g_idx, p_idx])
            if min_dist > self.match_threshold:
                break

            matches.append((gt_objects[g_idx], pred_objects[p_idx]))

            # remove matched rows/cols by setting to large value
            dists[g_idx, :] = np.inf
            dists[:, p_idx] = np.inf
            gt_indices.discard(g_idx)
            pred_indices.discard(p_idx)

        return matches

    def compute_ade_fde(self, gt_future: List[Tuple[float, float]], pred_future: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compute ADE and FDE between GT and prediction futures. Assumes equal length T."""
        if gt_future is None or pred_future is None:
            raise ValueError('gt_future and pred_future must be provided')

        if len(gt_future) != len(pred_future):
            raise ValueError('Mismatched future lengths')

        arr_gt = np.array(gt_future, dtype=np.float32)
        arr_pred = np.array(pred_future, dtype=np.float32)

        dists = np.linalg.norm(arr_pred - arr_gt, axis=1)
        ade = float(dists.mean())
        fde = float(dists[-1])
        return ade, fde

    def update(self, frame_idx: int, gt_objects: List[Dict], pred_objects: List[Dict]):
        """Process a single frame.

        gt_objects: [{'id': int, 'pos': (x,y)}]
        pred_objects: [{'id': int, 'pos': (x,y), 'future': [(x,y), ...] }]
        """
        # Append GT history
        for g in gt_objects:
            tid = int(g['id'])
            pos = (float(g['pos'][0]), float(g['pos'][1]))
            if tid not in self.gt_history:
                self.gt_history[tid] = []
            self.gt_history[tid].append(pos)

        # Matching
        matches = self.match(gt_objects, pred_objects)

        # For each match, perform anchor evaluation only when pred.future exists
        for gt_obj, pred_obj in matches:
            pred_future = pred_obj.get('future', None)
            if not pred_future:
                continue

            # Build GT future: need to find the GT track's future positions following current frame
            tid = int(gt_obj['id'])
            # Ensure track exists in history
            if tid not in self.gt_history:
                continue

            # Determine position index of current frame within GT history
            # We assume caller appends GT each frame in order, so last appended corresponds to this frame
            history = self.gt_history[tid]
            # Find where the last occurrence corresponds to this frame
            # Assuming update is called once per frame and GTs appended in order, current frame index maps to len(history)-1
            cur_idx = len(history) - 1

            # Construct GT future slices
            start = cur_idx + 1
            end = start + self.T
            if end > len(history):
                # Not enough GT future available -> skip
                continue

            gt_future = history[start:end]

            # Normalize pred_future into list of tuples and check length
            pred_future_list = [(float(p[0]), float(p[1])) for p in pred_future]
            if len(pred_future_list) != self.T:
                continue

            # compute metrics
            try:
                ade, fde = self.compute_ade_fde(gt_future, pred_future_list)
            except Exception:
                continue

            # record
            if tid not in self.track_metrics:
                self.track_metrics[tid] = {'ADE': [], 'FDE': []}
            self.track_metrics[tid]['ADE'].append(ade)
            self.track_metrics[tid]['FDE'].append(fde)

            self.all_ADE.append(ade)
            self.all_FDE.append(fde)

    def finalize_results(self) -> Dict:
        """Aggregate and return final metrics."""
        mean_ADE = float(np.mean(self.all_ADE)) if self.all_ADE else float('nan')
        mean_FDE = float(np.mean(self.all_FDE)) if self.all_FDE else float('nan')
        min_ADE = float(np.min(self.all_ADE)) if self.all_ADE else float('nan')
        max_ADE = float(np.max(self.all_ADE)) if self.all_ADE else float('nan')
        min_FDE = float(np.min(self.all_FDE)) if self.all_FDE else float('nan')
        max_FDE = float(np.max(self.all_FDE)) if self.all_FDE else float('nan')

        per_track = {}
        for tid, metrics in self.track_metrics.items():
            ade_list = metrics.get('ADE', [])
            fde_list = metrics.get('FDE', [])
            per_track[tid] = {
                'ADE': ade_list,
                'FDE': fde_list,
                'mean_ADE': float(np.mean(ade_list)) if ade_list else float('nan'),
                'mean_FDE': float(np.mean(fde_list)) if fde_list else float('nan'),
                'min_ADE': float(np.min(ade_list)) if ade_list else float('nan'),
                'max_ADE': float(np.max(ade_list)) if ade_list else float('nan'),
                'min_FDE': float(np.min(fde_list)) if fde_list else float('nan'),
                'max_FDE': float(np.max(fde_list)) if fde_list else float('nan'),
            }

        return {
            'mean_ADE': mean_ADE,
            'mean_FDE': mean_FDE,
            'min_ADE': min_ADE,
            'max_ADE': max_ADE,
            'min_FDE': min_FDE,
            'max_FDE': max_FDE,
            'n_samples': len(self.all_ADE),
            'per_track': per_track
        }

