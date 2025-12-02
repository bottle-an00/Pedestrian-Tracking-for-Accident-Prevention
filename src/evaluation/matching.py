# GT-Pred 매칭 유틸리티

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    두 bbox의 IoU 계산

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 교집합 좌표
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # 교집합 넓이
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # 각 박스 넓이
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 합집합 넓이
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(gt_boxes: List[Tuple], pred_boxes: List[Tuple]) -> np.ndarray:
    """
    GT-Pred bbox 간 IoU 행렬 계산

    Args:
        gt_boxes: GT bbox 리스트 [(x1,y1,x2,y2), ...]
        pred_boxes: Pred bbox 리스트

    Returns:
        IoU 행렬 (N_gt x N_pred)
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred))

    iou_matrix = np.zeros((n_gt, n_pred))

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)

    return iou_matrix


def match_detections(
    gt_boxes: List[Tuple],
    pred_boxes: List[Tuple],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    GT-Pred 1:1 매칭 (Greedy 방식)

    Args:
        gt_boxes: GT bbox 리스트
        pred_boxes: Pred bbox 리스트
        iou_threshold: 매칭 임계값

    Returns:
        matches: [(gt_idx, pred_idx, iou), ...] 매칭된 쌍
        unmatched_gt: 매칭 안 된 GT 인덱스
        unmatched_pred: 매칭 안 된 Pred 인덱스
    """
    iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)

    n_gt, n_pred = iou_matrix.shape

    matched_gt = set()
    matched_pred = set()
    matches = []

    # Greedy 매칭: IoU 높은 순서대로
    while True:
        if iou_matrix.size == 0:
            break

        # 최대 IoU 찾기
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break

        # 최대 IoU 위치
        gt_idx, pred_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

        matches.append((gt_idx, pred_idx, max_iou))
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

        # 매칭된 행/열 제거 (무효화)
        iou_matrix[gt_idx, :] = -1
        iou_matrix[:, pred_idx] = -1

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def match_by_track_id(
    gt_track_ids: List[int],
    pred_track_ids: List[int],
) -> Dict[int, Tuple[int, int]]:
    """
    Track ID 기반 매칭

    Args:
        gt_track_ids: GT track ID 리스트
        pred_track_ids: Pred track ID 리스트

    Returns:
        매칭 결과: {track_id: (gt_idx, pred_idx)}
    """
    gt_id_to_idx = {tid: idx for idx, tid in enumerate(gt_track_ids) if tid is not None}
    pred_id_to_idx = {tid: idx for idx, tid in enumerate(pred_track_ids) if tid is not None}

    matches = {}
    for track_id in gt_id_to_idx:
        if track_id in pred_id_to_idx:
            matches[track_id] = (gt_id_to_idx[track_id], pred_id_to_idx[track_id])

    return matches


def compute_euclidean_distance(p1: Tuple, p2: Tuple) -> float:
    """
    두 점 사이의 유클리드 거리

    Args:
        p1: (x, y) 또는 (x, y, z)
        p2: (x, y) 또는 (x, y, z)

    Returns:
        거리
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))
