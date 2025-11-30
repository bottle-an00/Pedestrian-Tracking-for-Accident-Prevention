# Evaluation 모듈 단위 테스트
import pytest
import numpy as np

from src.evaluation.matching import (
    compute_iou,
    compute_iou_matrix,
    match_detections,
    compute_euclidean_distance,
)
from src.evaluation.gt_loader import GTObject, GTFrame
from src.evaluation.evaluators import (
    DetectionEvaluator,
    TrackingEvaluator,
    LocalizationEvaluator,
    DetectionMatch,
)


# =============================================================================
# matching.py 테스트
# =============================================================================

class TestComputeIoU:
    """compute_iou 함수 테스트"""

    def test_perfect_overlap(self):
        """완전 일치 -> IoU = 1.0"""
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        """겹침 없음 -> IoU = 0.0"""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert compute_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        """부분 겹침 -> 0 < IoU < 1"""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1
        # 교집합: 50x50=2500, 합집합: 100x100 + 100x100 - 2500 = 17500
        expected = 2500 / 17500
        assert abs(iou - expected) < 1e-6

    def test_one_inside_other(self):
        """하나가 다른 하나 안에 포함"""
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        iou = compute_iou(outer, inner)
        # 교집합 = inner 넓이 = 50x50 = 2500
        # 합집합 = outer 넓이 = 100x100 = 10000
        expected = 2500 / 10000
        assert abs(iou - expected) < 1e-6


class TestMatchDetections:
    """match_detections 함수 테스트"""

    def test_perfect_match(self):
        """1:1 완벽 매칭"""
        gt = [(0, 0, 100, 100), (200, 200, 300, 300)]
        pred = [(0, 0, 100, 100), (200, 200, 300, 300)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, 0.5)
        assert len(matches) == 2
        assert len(unmatched_gt) == 0
        assert len(unmatched_pred) == 0

    def test_no_match_low_iou(self):
        """IoU 임계값 미달 -> 매칭 없음"""
        gt = [(0, 0, 50, 50)]
        pred = [(100, 100, 150, 150)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, 0.5)
        assert len(matches) == 0
        assert len(unmatched_gt) == 1
        assert len(unmatched_pred) == 1

    def test_more_pred_than_gt(self):
        """Pred > GT -> FP 발생"""
        gt = [(0, 0, 100, 100)]
        pred = [(0, 0, 100, 100), (200, 200, 300, 300)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, 0.5)
        assert len(matches) == 1
        assert len(unmatched_gt) == 0
        assert len(unmatched_pred) == 1

    def test_more_gt_than_pred(self):
        """GT > Pred -> FN 발생"""
        gt = [(0, 0, 100, 100), (200, 200, 300, 300)]
        pred = [(0, 0, 100, 100)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, 0.5)
        assert len(matches) == 1
        assert len(unmatched_gt) == 1
        assert len(unmatched_pred) == 0

    def test_empty_gt(self):
        """GT 없음"""
        gt = []
        pred = [(0, 0, 100, 100)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, 0.5)
        assert len(matches) == 0
        assert len(unmatched_pred) == 1

    def test_empty_pred(self):
        """Pred 없음"""
        gt = [(0, 0, 100, 100)]
        pred = []
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, 0.5)
        assert len(matches) == 0
        assert len(unmatched_gt) == 1


class TestComputeEuclideanDistance:
    """compute_euclidean_distance 함수 테스트"""

    def test_same_point(self):
        """같은 점 -> 거리 0"""
        assert compute_euclidean_distance((0, 0), (0, 0)) == 0.0

    def test_2d_distance(self):
        """2D 유클리드 거리"""
        dist = compute_euclidean_distance((0, 0), (3, 4))
        assert abs(dist - 5.0) < 1e-6

    def test_3d_distance(self):
        """3D 유클리드 거리"""
        dist = compute_euclidean_distance((0, 0, 0), (1, 2, 2))
        assert abs(dist - 3.0) < 1e-6


# =============================================================================
# DetectionEvaluator 테스트
# =============================================================================

class TestDetectionEvaluator:
    """DetectionEvaluator 클래스 테스트"""

    def test_perfect_detection(self):
        """완벽한 Detection -> 높은 Precision/Recall"""
        evaluator = DetectionEvaluator(iou_threshold=0.5)

        # GT bbox: (x1, y1, x2, y2) 형식
        # Pred bbox: [x, y, w, h] 형식 (DetectionEvaluator._convert_bbox에서 변환됨)
        gt_objects = [
            GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100)),
            GTObject(class_name="pedestrian", instance_id=2, bbox=(200, 200, 300, 300)),
        ]
        pred_objects = [
            {"bbox": [0, 0, 100, 100], "score": 0.9},  # [x,y,w,h] -> (0,0,100,100)
            {"bbox": [200, 200, 100, 100], "score": 0.8},  # [x,y,w,h] -> (200,200,300,300)
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        metrics = evaluator.compute()

        assert metrics["TP"] == 2
        assert metrics["FP"] == 0
        assert metrics["FN"] == 0
        assert metrics["Precision@50"] == 1.0
        assert metrics["Recall@50"] == 1.0

    def test_all_false_positive(self):
        """GT 없음, Pred만 있음 -> FP만"""
        evaluator = DetectionEvaluator(iou_threshold=0.5)

        gt_objects = []
        pred_objects = [
            {"bbox": [0, 0, 100, 100], "score": 0.9},
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        metrics = evaluator.compute()

        assert metrics["TP"] == 0
        assert metrics["FP"] == 1
        assert metrics["Precision@50"] == 0.0

    def test_all_false_negative(self):
        """Pred 없음, GT만 있음 -> FN만"""
        evaluator = DetectionEvaluator(iou_threshold=0.5)

        gt_objects = [
            GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100)),
        ]
        pred_objects = []

        evaluator.update(gt_objects, pred_objects, "frame_001")
        metrics = evaluator.compute()

        assert metrics["TP"] == 0
        assert metrics["FN"] == 1
        assert metrics["Recall@50"] == 0.0

    def test_multi_frame_accumulation(self):
        """여러 프레임 누적"""
        evaluator = DetectionEvaluator(iou_threshold=0.5)

        for i in range(3):
            gt_objects = [
                GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100)),
            ]
            pred_objects = [
                {"bbox": [0, 0, 100, 100], "score": 0.9},
            ]
            evaluator.update(gt_objects, pred_objects, f"frame_{i:03d}")

        metrics = evaluator.compute()
        assert metrics["TP"] == 3
        assert metrics["Total_GT"] == 3

    def test_reset(self):
        """reset() 호출 후 초기화 확인"""
        evaluator = DetectionEvaluator(iou_threshold=0.5)

        gt_objects = [GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100))]
        pred_objects = [{"bbox": [0, 0, 100, 100], "score": 0.9}]
        evaluator.update(gt_objects, pred_objects, "frame_001")

        evaluator.reset()
        metrics = evaluator.compute()

        assert metrics["TP"] == 0
        assert metrics["Total_GT"] == 0


# =============================================================================
# TrackingEvaluator 테스트
# =============================================================================

class TestTrackingEvaluator:
    """TrackingEvaluator 클래스 테스트"""

    def test_no_id_switch(self):
        """ID 유지 -> ID Switch 0"""
        evaluator = TrackingEvaluator(iou_threshold=0.5)

        # 프레임 1
        gt1 = [GTObject(class_name="pedestrian", instance_id=1, track_id=1, bbox=(0, 0, 100, 100))]
        pred1 = [{"bbox": [0, 0, 100, 100], "track_id": 10}]
        evaluator.update(gt1, pred1, "frame_001")

        # 프레임 2 - 같은 매칭 유지
        gt2 = [GTObject(class_name="pedestrian", instance_id=1, track_id=1, bbox=(10, 10, 110, 110))]
        pred2 = [{"bbox": [10, 10, 110, 110], "track_id": 10}]
        evaluator.update(gt2, pred2, "frame_002")

        metrics = evaluator.compute()
        assert metrics["ID_Switch"] == 0

    def test_id_switch_detected(self):
        """ID 변경 -> ID Switch 감지"""
        evaluator = TrackingEvaluator(iou_threshold=0.5)

        # 프레임 1
        gt1 = [GTObject(class_name="pedestrian", instance_id=1, track_id=1, bbox=(0, 0, 100, 100))]
        pred1 = [{"bbox": [0, 0, 100, 100], "track_id": 10}]
        evaluator.update(gt1, pred1, "frame_001")

        # 프레임 2 - 같은 GT에 다른 Pred ID가 매칭됨
        gt2 = [GTObject(class_name="pedestrian", instance_id=1, track_id=1, bbox=(10, 10, 110, 110))]
        pred2 = [{"bbox": [10, 10, 110, 110], "track_id": 20}]  # ID 변경!
        evaluator.update(gt2, pred2, "frame_002")

        metrics = evaluator.compute()
        assert metrics["ID_Switch"] == 1

    def test_mota_calculation(self):
        """MOTA 계산 테스트"""
        evaluator = TrackingEvaluator(iou_threshold=0.5)

        # TP=2, FP=1, FN=1, ID_Switch=0, Total_GT=3
        # Pred bbox: [x, y, w, h] 형식
        gt_objects = [
            GTObject(class_name="pedestrian", instance_id=1, track_id=1, bbox=(0, 0, 100, 100)),
            GTObject(class_name="pedestrian", instance_id=2, track_id=2, bbox=(200, 200, 300, 300)),
            GTObject(class_name="pedestrian", instance_id=3, track_id=3, bbox=(400, 400, 500, 500)),
        ]
        pred_objects = [
            {"bbox": [0, 0, 100, 100], "track_id": 10},  # TP [x,y,w,h] -> (0,0,100,100)
            {"bbox": [200, 200, 100, 100], "track_id": 20},  # TP [x,y,w,h] -> (200,200,300,300)
            {"bbox": [600, 600, 100, 100], "track_id": 30},  # FP (매칭 안됨)
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        metrics = evaluator.compute()

        # MOTA = 1 - (FN + FP + ID_Switch) / Total_GT = 1 - (1 + 1 + 0) / 3 = 1/3
        expected_mota = 1 - (1 + 1 + 0) / 3
        assert abs(metrics["MOTA"] - expected_mota) < 1e-6


# =============================================================================
# LocalizationEvaluator 테스트
# =============================================================================

class TestLocalizationEvaluator:
    """LocalizationEvaluator 클래스 테스트"""

    def test_foot_uv_error_calculation(self):
        """Foot UV 오차 계산"""
        evaluator = LocalizationEvaluator(iou_threshold=0.5)

        gt_objects = [
            GTObject(
                class_name="pedestrian",
                instance_id=1,
                bbox=(0, 0, 100, 100),
                foot_uv=(50.0, 100.0),  # bbox 하단 중앙
            ),
        ]
        pred_objects = [
            {
                "bbox": [0, 0, 100, 100],
                "foot_uv": (53.0, 104.0),  # 3, 4 픽셀 오차 -> 5 픽셀 거리
                "score": 0.9,
            },
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        metrics = evaluator.compute()

        assert abs(metrics["FootUV_MAE"] - 5.0) < 1e-6

    def test_bev_error_calculation(self):
        """BEV 오차 계산"""
        evaluator = LocalizationEvaluator(iou_threshold=0.5)

        gt_objects = [
            GTObject(
                class_name="pedestrian",
                instance_id=1,
                bbox=(0, 0, 100, 100),
                foot_uv=(50.0, 100.0),
                location_3d=(5.0, 10.0, 0.0),  # GT BEV = (5, 10)
            ),
        ]
        pred_objects = [
            {
                "bbox": [0, 0, 100, 100],
                "foot_uv": (50.0, 100.0),
                "foot_bev": (5.0, 11.0),  # Pred BEV - 1m 오차
                "score": 0.9,
            },
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        metrics = evaluator.compute()

        assert abs(metrics["BEV_MAE"] - 1.0) < 1e-6

    def test_get_matches(self):
        """매칭 결과 반환"""
        evaluator = LocalizationEvaluator(iou_threshold=0.5)

        gt_objects = [
            GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100), track_id=1),
        ]
        pred_objects = [
            {"bbox": [0, 0, 100, 100], "track_id": 10, "score": 0.9},
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        matches = evaluator.get_matches()

        assert len(matches) == 1
        assert matches[0]["frame_id"] == "frame_001"
        assert matches[0]["gt_track_id"] == 1
        assert matches[0]["pred_track_id"] == 10

    def test_false_negatives(self):
        """FN (놓친 GT) 수집"""
        evaluator = LocalizationEvaluator(iou_threshold=0.5)

        gt_objects = [
            GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100), track_id=1),
            GTObject(class_name="pedestrian", instance_id=2, bbox=(500, 500, 600, 600), track_id=2),
        ]
        pred_objects = [
            {"bbox": [0, 0, 100, 100], "track_id": 10, "score": 0.9},
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        fn = evaluator.get_false_negatives()

        assert len(fn) == 1
        assert fn[0]["gt_track_id"] == 2

    def test_false_positives(self):
        """FP (잘못된 Pred) 수집"""
        evaluator = LocalizationEvaluator(iou_threshold=0.5)

        gt_objects = [
            GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100)),
        ]
        pred_objects = [
            {"bbox": [0, 0, 100, 100], "score": 0.9},
            {"bbox": [500, 500, 600, 600], "score": 0.7},  # FP
        ]

        evaluator.update(gt_objects, pred_objects, "frame_001")
        fp = evaluator.get_false_positives()

        assert len(fp) == 1
        assert fp[0]["pred_score"] == 0.7


# =============================================================================
# DetectionMatch 테스트
# =============================================================================

class TestDetectionMatch:
    """DetectionMatch dataclass 테스트"""

    def test_to_dict(self):
        """to_dict() 변환"""
        match = DetectionMatch(
            frame_id="frame_001",
            gt_idx=0,
            pred_idx=0,
            gt_track_id=1,
            pred_track_id=10,
            iou=0.85,
            gt_bbox=(0, 0, 100, 100),
            pred_bbox=(5, 5, 105, 105),
            foot_uv_error=5.0,
            bev_error=1.5,
        )

        d = match.to_dict()

        assert d["frame_id"] == "frame_001"
        assert d["iou"] == 0.85
        assert d["foot_uv_error"] == 5.0
        assert d["bev_error"] == 1.5


# =============================================================================
# GTFrame 테스트
# =============================================================================

class TestGTFrame:
    """GTFrame 클래스 테스트"""

    def test_get_pedestrians(self):
        """보행자 필터링"""
        frame = GTFrame(
            frame_id="frame_001",
            objects=[
                GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100)),
                GTObject(class_name="vehicle", instance_id=2, bbox=(200, 200, 400, 400)),
                GTObject(class_name="pedestrian", instance_id=3, bbox=(500, 500, 600, 600)),
            ],
        )

        pedestrians = frame.get_pedestrians()
        assert len(pedestrians) == 2
        assert all(p.class_name == "pedestrian" for p in pedestrians)

    def test_get_vehicles(self):
        """차량 필터링"""
        frame = GTFrame(
            frame_id="frame_001",
            objects=[
                GTObject(class_name="pedestrian", instance_id=1, bbox=(0, 0, 100, 100)),
                GTObject(class_name="vehicle", instance_id=2, bbox=(200, 200, 400, 400)),
            ],
        )

        vehicles = frame.get_vehicles()
        assert len(vehicles) == 1
        assert vehicles[0].class_name == "vehicle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
