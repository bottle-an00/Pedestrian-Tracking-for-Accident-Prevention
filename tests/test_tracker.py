# ByteTracker + BEV 변환 통합 테스트
from pathlib import Path
import numpy as np
import cv2

from src.core.config import load_yaml
from src.visualization.overlay_2d import Visualizer
from src.io.image_loader import ImageLoader
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.tracking.tracker import ByteTracker


def test_tracker():
    cfg = load_yaml("configs/system.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        print(f"[SKIP] 테스트 이미지 폴더 없음: {image_dir}")
        return

    intrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_Camera"])
    extrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Camera"])

    calib_loader = CalibrationInfoLoader()
    intrinsics = calib_loader.load_camera_calibration(intrinsics_path)
    extrinsics = calib_loader.load_camera_extrinsics(extrinsics_path)

    assert intrinsics is not None
    assert extrinsics is not None

    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)
    image_loader = ImageLoader()
    vis = Visualizer()

    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    # ByteTracker 사용 
    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 640)
    )

    output_root = Path(cfg["test_data_dir"]["outputs"]) / "tracker_results"
    suboutput_root1 = output_root / "detection_result"
    suboutput_root2 = output_root / "bev_result"
    suboutput_root3 = output_root / "foot_uv_bev_result"
    output_root.mkdir(parents=True, exist_ok=True)
    suboutput_root1.mkdir(parents=True, exist_ok=True)
    suboutput_root2.mkdir(parents=True, exist_ok=True)
    suboutput_root3.mkdir(parents=True, exist_ok=True)

    for img_path, image in image_loader.iter_imgs_cv2(image_dir):

        assert image is not None
        assert np.any(image)

        detections = tracker.process(image)

        assert isinstance(detections, list)

        vis_img = vis.draw_on_img(image, detections)
        cv2.imwrite(str(suboutput_root1 / f"2d_overlay_{img_path.name}"), vis_img)

        bev_img = H.warp(image)
        assert bev_img is not None
        cv2.imwrite(str(suboutput_root2 / f"bev_{img_path.name}"), bev_img)

        foot_bevs = []
        for det in detections:
            u, v = det["foot_uv"]
            bev_coords = H.pixel_to_bev_warp(u, v)
            if bev_coords is not None:
                foot_bevs.append(bev_coords)

        if foot_bevs:
            bev_overlay = vis.draw_on_BEV(bev_img, 0, foot_bevs)
            cv2.imwrite(str(suboutput_root3 / f"bev_overlay_{img_path.name}"), bev_overlay)

    print(f"[PASS] test_tracker")
    print(f"[INFO] 결과 저장 위치: {output_root}")
