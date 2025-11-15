from pathlib import Path
import numpy as np
import cv2
import sys, os

from src.core.config import load_yaml
from src.io.image_loader import ImageLoader
from src.io.pcd_loader import PcdLoader

from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.calibration_utils import CameraLidarCalibrator

def test_camera_LiDAR_calibration():
    # Load test dataset
    cfg = load_yaml("configs/system.yaml")
    image_dir = Path(cfg["test_data_dir"]["images"])
    pcd_dir = Path(cfg["test_data_dir"]["lidar"])

    intrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_Camera"])
    extrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Camera"])

    if not image_dir.exists() or not pcd_dir.exists():
        raise ValueError("Image or LiDAR directory does not exist.")

    if not intrinsics_path.exists() or not extrinsics_path.exists():
        raise ValueError("Camera calibration files do not exist.")

    image_loader = ImageLoader()
    pcd_loader = PcdLoader()
    calib_loader = CalibrationInfoLoader()

    img_items = image_loader.iter_imgs_cv2(image_dir)
    pcd_items = pcd_loader.iter_pcds(pcd_dir, as_numpy=True)
    intrinsics = calib_loader.load_camera_calibration(intrinsics_path)
    extrinsics = calib_loader.load_camera_extrinsics(extrinsics_path)

    if not np.any(intrinsics):
        raise ValueError("Failed to load intrinsic data.")

    if not np.any(extrinsics):
        raise ValueError("Failed to load extrinsic data.")

    calibrator = CameraLidarCalibrator(intrinsics, extrinsics)
    calibration_results = []

    for img_item, pcd_item in zip(img_items, pcd_items):

        img_path, image = img_item
        pcd_path, point_cloud = pcd_item

        if not np.any(image) or not np.any(point_cloud):
            raise ValueError("Failed to load test data.")

        print(f"Processing Image: {img_path.name}, PointCloud: {pcd_path.name}")
        print(f"Image shape: {image.shape}, PointCloud shape: {point_cloud.shape}")
        print(f"Intrinsics:{intrinsics.shape}, Extrinsics:{extrinsics.shape}")

        calibration_result = calibrator.draw_lidar_on_image(image, point_cloud)

        calibration_results.append(calibration_result)

        assert calibration_results[-1] is not None

    output_dir = Path(cfg["test_data_dir"]["outputs"]) / "calibration_results"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    cnt = 0
    for result in calibration_results:
        output_path = output_dir / f"calibration_result_{cnt}.png"
        cv2.imwrite(str(output_path), result)
        cnt += 1
