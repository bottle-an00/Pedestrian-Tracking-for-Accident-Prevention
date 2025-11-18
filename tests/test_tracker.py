from pathlib import Path
import numpy as np
import cv2
import os

from src.core.config import load_yaml
from src.visualization.overlay_2d import Visualizer
from src.io.image_loader import ImageLoader
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.tracking.tracker import UltralyticsTracker

def test_tracker():
    cfg = load_yaml("configs/system.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        raise ValueError("Image_directory does not exist.")

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

    yolo_cgf = load_yaml("configs/detector/yolo_detector.yaml")

    yolo_model_path = yolo_cgf["yolo_model_path"]
    conf_threshold  = yolo_cgf["conf_threshold"]
    target_classes  = yolo_cgf["target_classes"]
    tracker_type    = yolo_cgf["tracker"]


    tracker = UltralyticsTracker(
        model_path=yolo_model_path,
        conf_thres_config=conf_threshold,
        target_class_names=target_classes,
        tracker_type=tracker_type
    )

    output_root = Path(cfg["test_data_dir"]["outputs"]) / "tracker_results"
    suboutput_root1 = Path(output_root/"detection_result")
    suboutput_root2 = Path(output_root/"bev_result")
    suboutput_root3 = Path(output_root/"foot_uv_bev_result")
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

        assert len(foot_bevs) > 0

        bev_overlay = vis.draw_on_BEV(bev_img, foot_bevs)
        
        cv2.imwrite(str(suboutput_root3 / f"bev_overlay_{img_path.name}"), bev_overlay)

    print(f"[TEST] Visualizer + YOLO tracker results saved at: {output_root}")

    assert True