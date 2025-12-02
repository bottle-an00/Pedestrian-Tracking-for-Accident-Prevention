# 평가 모듈: Detection, Tracking Evaluator
# - Detection: AP@50, AP@75, mAP@50:95 (COCO 방식)
# - Tracking: MOTA, MOTP, IDF1, HOTA (MOT 벤치마크)

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from .matching import compute_iou_matrix, match_detections, compute_euclidean_distance
from .gt_loader import GTObject


@dataclass
class DetectionMatch:
    """단일 GT-Pred 매칭 결과 (프레임 내)"""
    frame_id: str
    gt_idx: int
    pred_idx: int
    gt_track_id: int = None
    pred_track_id: int = None
    iou: float = 0.0
    # GT
    gt_bbox: Tuple = None
    gt_foot_uv: Tuple = None
    gt_location_3d: Tuple = None
    # Pred
    pred_bbox: Tuple = None
    pred_foot_uv: Tuple = None
    pred_foot_bev: Tuple = None
    pred_score: float = 0.0
    foot_uv_type: str = "detected"
    # Error
    foot_uv_error: float = None
    bev_error: float = None

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "gt_idx": self.gt_idx,
            "pred_idx": self.pred_idx,
            "gt_track_id": self.gt_track_id,
            "pred_track_id": self.pred_track_id,
            "iou": self.iou,
            "gt_bbox": self.gt_bbox,
            "gt_foot_uv": self.gt_foot_uv,
            "gt_location_3d": self.gt_location_3d,
            "pred_bbox": self.pred_bbox,
            "pred_foot_uv": self.pred_foot_uv,
            "pred_foot_bev": self.pred_foot_bev,
            "pred_score": self.pred_score,
            "foot_uv_type": self.foot_uv_type,
            "foot_uv_error": self.foot_uv_error,
            "bev_error": self.bev_error,
        }


class BaseEvaluator(ABC):
    """Evaluator 베이스 클래스"""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, gt_objects: List[GTObject], pred_objects: List[Dict], frame_id: str = ""):
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        pass

    def _convert_bbox(self, bbox) -> Tuple:
        """[x,y,w,h] -> (x1,y1,x2,y2)"""
        if bbox is None:
            return None
        if len(bbox) == 4:
            x, y, w, h = bbox
            return (x, y, x + w, y + h)
        return tuple(bbox)


class DetectionEvaluator(BaseEvaluator):
    """
    Detection 평가 (COCO 방식)
    - AP@50, AP@75, mAP@50:95
    - Precision, Recall, F1
    """
    # 부동소수점 비교 문제 방지를 위해 round
    IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    def reset(self):
        self.all_detections: List[Tuple[float, bool, float]] = []
        self.total_gt = 0
        self.stats_by_iou = {iou: {"tp": 0, "fp": 0, "fn": 0} for iou in self.IOU_THRESHOLDS}

    def update(self, gt_objects: List[GTObject], pred_objects: List[Dict], frame_id: str = ""):
        gt_boxes = [obj.bbox for obj in gt_objects if obj.bbox is not None]
        pred_data = [(self._convert_bbox(obj.get("bbox")), obj.get("score", 1.0))
                     for obj in pred_objects if obj.get("bbox") is not None]

        n_gt, n_pred = len(gt_boxes), len(pred_data)
        self.total_gt += n_gt

        if n_gt == 0 or n_pred == 0:
            for iou_th in self.IOU_THRESHOLDS:
                self.stats_by_iou[iou_th]["fp"] += n_pred
                self.stats_by_iou[iou_th]["fn"] += n_gt
            return

        pred_boxes = [p[0] for p in pred_data]
        pred_scores = [p[1] for p in pred_data]
        iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)

        for iou_th in self.IOU_THRESHOLDS:
            matches, unmatched_gt, unmatched_pred = self._greedy_match(iou_matrix, iou_th)
            self.stats_by_iou[iou_th]["tp"] += len(matches)
            self.stats_by_iou[iou_th]["fp"] += len(unmatched_pred)
            self.stats_by_iou[iou_th]["fn"] += len(unmatched_gt)

        matches_05, _, _ = self._greedy_match(iou_matrix, 0.5)
        matched_pred = {m[1] for m in matches_05}

        for pred_idx, score in enumerate(pred_scores):
            if pred_idx in matched_pred:
                gt_idx = next(m[0] for m in matches_05 if m[1] == pred_idx)
                self.all_detections.append((score, True, iou_matrix[gt_idx, pred_idx]))
            else:
                self.all_detections.append((score, False, 0.0))

    def _greedy_match(self, iou_matrix: np.ndarray, threshold: float):
        n_gt, n_pred = iou_matrix.shape
        matrix = iou_matrix.copy()
        matches, matched_gt, matched_pred = [], set(), set()

        while True:
            max_iou = matrix.max()
            if max_iou < threshold:
                break
            gt_idx, pred_idx = np.unravel_index(matrix.argmax(), matrix.shape)
            matches.append((gt_idx, pred_idx, max_iou))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matrix[gt_idx, :] = -1
            matrix[:, pred_idx] = -1

        return matches, [i for i in range(n_gt) if i not in matched_gt], \
               [i for i in range(n_pred) if i not in matched_pred]

    def _compute_ap(self, precisions: List[float], recalls: List[float]) -> float:
        if not precisions:
            return 0.0
        sorted_idx = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_idx]
        precisions = np.array(precisions)[sorted_idx]

        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            prec = precisions[recalls >= t]
            ap += prec.max() if len(prec) > 0 else 0.0
        return ap / 11.0

    def compute(self) -> Dict[str, float]:
        metrics = {}

        for iou_th in [0.5, 0.75]:
            s = self.stats_by_iou[iou_th]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            th = int(iou_th * 100)
            metrics[f"Precision@{th}"] = prec
            metrics[f"Recall@{th}"] = rec
            metrics[f"F1@{th}"] = f1

        if self.all_detections:
            sorted_dets = sorted(self.all_detections, key=lambda x: -x[0])
            for th, th_name in [(0.5, "AP@50"), (0.75, "AP@75")]:
                tp_cum, precs, recs = 0, [], []
                for _, is_tp, iou in sorted_dets:
                    if is_tp and iou >= th:
                        tp_cum += 1
                    precs.append(tp_cum / (len(precs) + 1))
                    recs.append(tp_cum / self.total_gt if self.total_gt > 0 else 0)
                metrics[th_name] = self._compute_ap(precs, recs)

        aps = []
        for iou_th in self.IOU_THRESHOLDS:
            s = self.stats_by_iou[iou_th]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            aps.append(2 * prec * rec / (prec + rec + 1e-6) if prec + rec > 0 else 0.0)
        metrics["mAP@50:95"] = np.mean(aps)

        s = self.stats_by_iou[0.5]
        metrics["TP"] = s["tp"]
        metrics["FP"] = s["fp"]
        metrics["FN"] = s["fn"]
        metrics["Total_GT"] = self.total_gt

        return metrics


class TrackingEvaluator(BaseEvaluator):
    """
    Tracking 평가 (MOT 벤치마크)
    - MOTA, MOTP, IDF1, HOTA
    """

    def reset(self):
        self.tp = self.fp = self.fn = self.id_switches = 0
        self.total_iou = 0.0
        self.total_gt = 0
        self.gt_id_counts: Dict[int, int] = {}
        self.pred_id_counts: Dict[int, int] = {}
        self.associations: List[Tuple[int, int]] = []
        self.prev_matches: Dict[int, int] = {}

    def update(self, gt_objects: List[GTObject], pred_objects: List[Dict], frame_id: str = ""):
        gt_boxes = [obj.bbox for obj in gt_objects if obj.bbox is not None]
        gt_track_ids = [obj.track_id for obj in gt_objects if obj.bbox is not None]

        pred_boxes, pred_track_ids = [], []
        for obj in pred_objects:
            if obj.get("bbox") is not None:
                pred_boxes.append(self._convert_bbox(obj["bbox"]))
                pred_track_ids.append(obj.get("track_id", obj.get("id")))

        n_gt, n_pred = len(gt_boxes), len(pred_boxes)
        self.total_gt += n_gt

        for gid in gt_track_ids:
            if gid is not None:
                self.gt_id_counts[gid] = self.gt_id_counts.get(gid, 0) + 1
        for pid in pred_track_ids:
            if pid is not None:
                self.pred_id_counts[pid] = self.pred_id_counts.get(pid, 0) + 1

        if n_gt == 0 or n_pred == 0:
            self.fn += n_gt
            self.fp += n_pred
            return

        matches, unmatched_gt, unmatched_pred = match_detections(gt_boxes, pred_boxes, self.iou_threshold)
        self.tp += len(matches)
        self.fn += len(unmatched_gt)
        self.fp += len(unmatched_pred)

        current_matches = {}
        for gt_idx, pred_idx, iou in matches:
            self.total_iou += iou
            gt_tid, pred_tid = gt_track_ids[gt_idx], pred_track_ids[pred_idx]

            if gt_tid is not None and pred_tid is not None:
                current_matches[gt_tid] = pred_tid
                self.associations.append((gt_tid, pred_tid))

                if gt_tid in self.prev_matches and self.prev_matches[gt_tid] != pred_tid:
                    self.id_switches += 1

        self.prev_matches = current_matches

    def _compute_hota(self) -> float:
        if not self.associations:
            return 0.0

        det_a = self.tp / (self.tp + self.fp + self.fn) if self.tp + self.fp + self.fn > 0 else 0.0

        gt_pred_counts: Dict[int, Dict[int, int]] = {}
        for gt_id, pred_id in self.associations:
            if gt_id not in gt_pred_counts:
                gt_pred_counts[gt_id] = {}
            gt_pred_counts[gt_id][pred_id] = gt_pred_counts[gt_id].get(pred_id, 0) + 1

        ass_scores = []
        for gt_id, pred_counts in gt_pred_counts.items():
            best_pred = max(pred_counts, key=pred_counts.get)
            tpa = pred_counts[best_pred]
            fna = self.gt_id_counts.get(gt_id, 0) - tpa
            fpa = self.pred_id_counts.get(best_pred, 0) - tpa
            ass_scores.append(tpa / (tpa + fna + fpa) if tpa + fna + fpa > 0 else 0.0)

        ass_a = np.mean(ass_scores) if ass_scores else 0.0
        return np.sqrt(det_a * ass_a)

    def compute(self) -> Dict[str, float]:
        mota = 1 - (self.fn + self.fp + self.id_switches) / self.total_gt if self.total_gt > 0 else 0.0
        motp = self.total_iou / self.tp if self.tp > 0 else 0.0
        idf1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn) if 2 * self.tp + self.fp + self.fn > 0 else 0.0

        return {
            "HOTA": self._compute_hota(),
            "MOTA": mota,
            "MOTP": motp,
            "IDF1": idf1,
            "ID_Switch": self.id_switches,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "Total_GT": self.total_gt,
        }


class LocalizationEvaluator:
    """
    Localization 평가 (Foot UV, BEV 오차)
    - GT-Pred 매칭 결과 수집
    - Foot UV 오차 (픽셀)
    - BEV 오차 (미터)
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.matches: List[DetectionMatch] = []
        self.false_negatives: List[Dict] = []  # 놓친 GT
        self.false_positives: List[Dict] = []  # 잘못된 Pred

    def update(self, gt_objects: List[GTObject], pred_objects: List[Dict], frame_id: str = ""):
        gt_boxes = [obj.bbox for obj in gt_objects if obj.bbox is not None]
        pred_boxes = [self._convert_bbox(obj.get("bbox")) for obj in pred_objects if obj.get("bbox") is not None]

        if not gt_boxes or not pred_boxes:
            for i, obj in enumerate(gt_objects):
                if obj.bbox:
                    self.false_negatives.append({
                        "frame_id": frame_id, "gt_idx": i, "gt_track_id": obj.track_id,
                        "gt_bbox": obj.bbox, "gt_foot_uv": obj.foot_uv
                    })
            for i, obj in enumerate(pred_objects):
                if obj.get("bbox"):
                    self.false_positives.append({
                        "frame_id": frame_id, "pred_idx": i,
                        "pred_bbox": tuple(obj["bbox"]), "pred_score": obj.get("score", 0)
                    })
            return

        matched, unmatched_gt, unmatched_pred = match_detections(gt_boxes, pred_boxes, self.iou_threshold)

        for gt_idx, pred_idx, iou in matched:
            gt_obj = gt_objects[gt_idx]
            pred_obj = pred_objects[pred_idx]

            pred_foot_uv = pred_obj.get("foot_uv")
            pred_foot_bev = pred_obj.get("foot_bev")

            foot_uv_error = None
            if gt_obj.foot_uv and pred_foot_uv:
                foot_uv_error = compute_euclidean_distance(gt_obj.foot_uv, pred_foot_uv)

            bev_error = None
            if gt_obj.location_3d and pred_foot_bev and pred_foot_bev[0] >= 0:
                gt_bev = (gt_obj.location_3d[0], gt_obj.location_3d[1])
                bev_error = compute_euclidean_distance(gt_bev, pred_foot_bev)

            match = DetectionMatch(
                frame_id=frame_id,
                gt_idx=gt_idx,
                pred_idx=pred_idx,
                gt_track_id=gt_obj.track_id,
                pred_track_id=pred_obj.get("track_id", pred_obj.get("id")),
                iou=iou,
                gt_bbox=gt_obj.bbox,
                gt_foot_uv=gt_obj.foot_uv,
                gt_location_3d=gt_obj.location_3d,
                pred_bbox=tuple(pred_obj["bbox"]) if pred_obj.get("bbox") else None,
                pred_foot_uv=tuple(pred_foot_uv) if pred_foot_uv else None,
                pred_foot_bev=pred_foot_bev,
                pred_score=pred_obj.get("score", 0.0),
                foot_uv_type=pred_obj.get("foot_uv_type", "detected"),
                foot_uv_error=foot_uv_error,
                bev_error=bev_error,
            )
            self.matches.append(match)

        for idx in unmatched_gt:
            obj = gt_objects[idx]
            self.false_negatives.append({
                "frame_id": frame_id, "gt_idx": idx, "gt_track_id": obj.track_id,
                "gt_bbox": obj.bbox, "gt_foot_uv": obj.foot_uv
            })

        for idx in unmatched_pred:
            obj = pred_objects[idx]
            self.false_positives.append({
                "frame_id": frame_id, "pred_idx": idx,
                "pred_bbox": tuple(obj["bbox"]) if obj.get("bbox") else None,
                "pred_score": obj.get("score", 0)
            })

    def _convert_bbox(self, bbox) -> Tuple:
        if bbox is None:
            return None
        if len(bbox) == 4:
            x, y, w, h = bbox
            return (x, y, x + w, y + h)
        return tuple(bbox)

    def get_matches(self) -> List[Dict]:
        return [m.to_dict() for m in self.matches]

    def get_false_negatives(self) -> List[Dict]:
        return self.false_negatives

    def get_false_positives(self) -> List[Dict]:
        return self.false_positives

    def compute(self) -> Dict[str, float]:
        foot_errors = [m.foot_uv_error for m in self.matches if m.foot_uv_error is not None]
        bev_errors = [m.bev_error for m in self.matches if m.bev_error is not None]

        foot_detected = [m.foot_uv_error for m in self.matches
                         if m.foot_uv_error is not None and m.foot_uv_type == "detected"]
        foot_out_of_fov = [m.foot_uv_error for m in self.matches
                           if m.foot_uv_error is not None and m.foot_uv_type != "detected"]

        return {
            "FootUV_MAE": float(np.mean(foot_errors)) if foot_errors else 0.0,
            "FootUV_STD": float(np.std(foot_errors)) if foot_errors else 0.0,
            "FootUV_Median": float(np.median(foot_errors)) if foot_errors else 0.0,
            "FootUV_MAE_detected": float(np.mean(foot_detected)) if foot_detected else 0.0,
            "FootUV_MAE_out_of_fov": float(np.mean(foot_out_of_fov)) if foot_out_of_fov else 0.0,
            "BEV_MAE": float(np.mean(bev_errors)) if bev_errors else 0.0,
            "BEV_STD": float(np.std(bev_errors)) if bev_errors else 0.0,
            "BEV_Median": float(np.median(bev_errors)) if bev_errors else 0.0,
            "N_matches": len(self.matches),
            "N_FN": len(self.false_negatives),
            "N_FP": len(self.false_positives),
        }
