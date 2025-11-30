# YoloDetector 단위 테스트
from pathlib import Path
import numpy as np
import cv2

from src.core.config import load_yaml
from src.detection.yolo_detector import YoloDetector
from src.io.image_loader import ImageLoader


def test_yolo_detector_detect():
    """YoloDetector.detect() 테스트 - 트래킹 없이 detection만"""
    cfg = load_yaml("configs/system.yaml")
    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        print(f"[SKIP] 테스트 이미지 폴더 없음: {image_dir}")
        return

    # YoloDetector 생성
    detector = YoloDetector(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 640)
    )

    image_loader = ImageLoader()

    for img_path, image in image_loader.iter_imgs_cv2(image_dir):
        assert image is not None
        assert len(image.shape) == 3  # (H, W, C)

        detections = detector.detect(image)

        # 반환값 타입 확인
        assert isinstance(detections, list)

        for det in detections:
            # 필수 키 확인
            assert "bbox" in det
            assert "score" in det
            assert "class" in det
            assert "foot_uv" in det
            assert "keypoints" in det
            assert "foot_uv_type" in det

            # bbox 형식 확인: [x, y, w, h]
            assert len(det["bbox"]) == 4

            # foot_uv 형식 확인: [u, v]
            assert len(det["foot_uv"]) == 2

            # score 범위 확인
            assert 0.0 <= det["score"] <= 1.0

            # foot_uv_type 값 확인
            assert det["foot_uv_type"] in ["detected", "out_of_fov"]

        print(f"[TEST] {img_path.name}: {len(detections)} detections")
        break  # 첫 이미지만 테스트

    print("[PASS] test_yolo_detector_detect")


def test_yolo_detector_foot_estimation():
    """foot_uv 추정 로직 테스트"""
    cfg = load_yaml("configs/system.yaml")
    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        print(f"[SKIP] 테스트 이미지 폴더 없음: {image_dir}")
        return

    detector = YoloDetector(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 640)
    )

    image_loader = ImageLoader()

    for img_path, image in image_loader.iter_imgs_cv2(image_dir):
        img_h, img_w = image.shape[:2]
        detections = detector.detect(image)

        for det in detections:
            foot_u, foot_v = det["foot_uv"]
            bbox = det["bbox"]
            x, y, w, h = bbox

            # foot_uv가 bbox 근처에 있는지 확인 (대략적으로)
            # foot_u는 bbox x 범위 근처
            assert x - w <= foot_u <= x + 2 * w, f"foot_u={foot_u} out of bbox range"

            # foot_v는 bbox 하단 근처 (위로 튀지 않음)
            assert foot_v >= y, f"foot_v={foot_v} above bbox top"

        print(f"[TEST] {img_path.name}: foot estimation OK")
        break

    print("[PASS] test_yolo_detector_foot_estimation")
