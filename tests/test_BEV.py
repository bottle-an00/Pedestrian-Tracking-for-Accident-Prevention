from pathlib import Path
import numpy as np
import cv2
import os

from src.core.config import load_yaml
from src.calibration import CalibrationInfoLoader, CalibrationUtils
from src.calibration.homography import Homography
from src.io.image_loader import ImageLoader

def test_BEV_homography():
    # Load test dataset
    cfg = load_yaml("configs/system.yaml")
    calib_utils = CalibrationUtils()

    image_dir = Path(cfg["test_data_dir"]["images"])
    if not image_dir.exists():
        raise ValueError("Image directory does not exist.")

    intrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_Camera"])
    extrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Camera"])

    if not intrinsics_path.exists() or not extrinsics_path.exists():
        raise ValueError("Camera calibration files do not exist.")

    image_loader = ImageLoader()
    calib_loader = CalibrationInfoLoader()

    intrinsics = calib_loader.load_camera_calibration(intrinsics_path)
    extrinsics = calib_loader.load_camera_extrinsics(extrinsics_path)

    if intrinsics is None or not np.any(intrinsics):
        raise ValueError("Failed to load intrinsic data.")

    if extrinsics is None or not np.any(extrinsics):
        raise ValueError("Failed to load extrinsic data.")

    print(f"Loaded intrinsics shape: {intrinsics.shape}")
    print(f"Loaded extrinsics shape: {extrinsics.shape}")

    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)

    bev_results = []
    for img_path, image in image_loader.iter_imgs_cv2(image_dir):
        if image is None or not np.any(image):
            raise ValueError(f"Failed to load image: {img_path}")

        print(f"[BEV] Processing Image: {img_path.name}")
        print(f"Image shape: {image.shape}")
        print(f"BEV params: front={H.bev_front}, back={H.bev_back}, "
              f"left={H.bev_left}, right={H.bev_right}, res={H.bev_resolution}")

        bev_img = H.warp(image)

        assert bev_img is not None
        assert isinstance(bev_img, np.ndarray)
        assert bev_img.ndim == 3  # H, W, C
        assert bev_img.shape[2] == 3

        bev_results.append((img_path.name, bev_img))

    output_dir = Path(cfg["test_data_dir"]["outputs"]) / "bev_results"
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    for idx, (name, bev_img) in enumerate(bev_results):
        out_name = f"bev_{idx:03d}_{name}"
        output_path = output_dir / out_name
        cv2.imwrite(str(output_path), bev_img)
        print(f"[BEV] Saved: {output_path}")

    assert len(bev_results) > 0