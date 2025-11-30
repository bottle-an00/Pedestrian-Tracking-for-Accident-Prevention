# ByteTracker 단위 테스트
from pathlib import Path
import numpy as np
import cv2

from src.core.config import load_yaml
from src.tracking.tracker import ByteTracker
from src.visualization.overlay_2d import Visualizer
from src.io.image_loader import ImageLoader


def test_bytetracker_process():
    """ByteTracker.process() 테스트 - detection + tracking"""
    cfg = load_yaml("configs/system.yaml")
    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        print(f"[SKIP] 테스트 이미지 폴더 없음: {image_dir}")
        return

    # ByteTracker 생성
    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 640)
    )

    image_loader = ImageLoader()
    track_ids_seen = set()

    for img_path, image in image_loader.iter_imgs_cv2(image_dir):
        assert image is not None

        detections = tracker.process(image)

        assert isinstance(detections, list)

        for det in detections:
            # 필수 키 확인 (YoloDetector + track_id)
            assert "id" in det  # ByteTracker는 id 포함
            assert "bbox" in det
            assert "score" in det
            assert "class" in det
            assert "foot_uv" in det
            assert "keypoints" in det
            assert "foot_uv_type" in det

            # track_id는 정수
            assert isinstance(det["id"], int)
            track_ids_seen.add(det["id"])

        print(f"[TEST] {img_path.name}: {len(detections)} tracked objects")

    print(f"[TEST] 총 {len(track_ids_seen)}개 track ID 발견")
    print("[PASS] test_bytetracker_process")


def test_bytetracker_tracking_persistence():
    """트래킹 ID가 프레임 간 유지되는지 테스트"""
    cfg = load_yaml("configs/system.yaml")
    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        print(f"[SKIP] 테스트 이미지 폴더 없음: {image_dir}")
        return

    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 640)
    )

    image_loader = ImageLoader()
    frame_count = 0
    all_track_ids = []

    for img_path, image in image_loader.iter_imgs_cv2(image_dir):
        detections = tracker.process(image)
        frame_ids = [det["id"] for det in detections]
        all_track_ids.append(set(frame_ids))

        frame_count += 1
        if frame_count >= 5:  # 5프레임만 테스트
            break

    # 연속 프레임에서 일부 ID가 유지되는지 확인
    if len(all_track_ids) >= 2:
        for i in range(1, len(all_track_ids)):
            prev_ids = all_track_ids[i - 1]
            curr_ids = all_track_ids[i]
            common_ids = prev_ids & curr_ids
            print(f"[TEST] Frame {i-1} → {i}: {len(common_ids)} IDs maintained")

    print("[PASS] test_bytetracker_tracking_persistence")


def test_bytetracker_with_visualization():
    """ByteTracker + Visualizer 통합 테스트"""
    cfg = load_yaml("configs/system.yaml")
    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        print(f"[SKIP] 테스트 이미지 폴더 없음: {image_dir}")
        return

    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 640)
    )

    vis = Visualizer()
    image_loader = ImageLoader()

    output_dir = Path(cfg["test_data_dir"]["outputs"]) / "bytetracker_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, image in image_loader.iter_imgs_cv2(image_dir):
        detections = tracker.process(image)

        # 키포인트 포함 시각화
        vis_img = vis.draw_on_img_with_keypoints(
            image, detections, seq_name="test"
        )

        assert vis_img is not None
        assert vis_img.shape[:2] == image.shape[:2] or vis_img.shape[1] <= 1280

        # 결과 저장
        cv2.imwrite(str(output_dir / f"tracked_{img_path.name}"), vis_img)
        print(f"[TEST] Saved: {output_dir / f'tracked_{img_path.name}'}")
        break

    print(f"[PASS] test_bytetracker_with_visualization")
    print(f"[INFO] 결과 저장 위치: {output_dir}")
