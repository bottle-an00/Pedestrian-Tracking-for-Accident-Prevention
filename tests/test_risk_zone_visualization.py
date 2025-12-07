from pathlib import Path
import cv2
import os
import numpy as np

from src.core.config import load_yaml
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.bev.bev_transformer import BevTransformer
from src.visualization.overlay_2d import Visualizer
from src.trajectory.bev_zone_risk_manager import generate_risk_zones, PedestrianStateManager
from src.io.gps_loader import GpsImuLoader


def test_risk_zone_visualization():
    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"

    image_dir = Path(cfg[root_dir]["images"])
    out_root = Path(cfg[root_dir]["outputs"]) / "risk_viz"
    out_root.mkdir(parents=True, exist_ok=True)

    # load calibration and homography
    loader = CalibrationInfoLoader()
    intrinsics = loader.load_camera_calibration(Path(cfg[root_dir]["calibration"]["calib_Camera"]))
    extrinsics = loader.load_camera_extrinsics(Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"]))
    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)
    bev_transformer = BevTransformer(homography=H)

    vis = Visualizer()

    # prepare GPS loader and manager
    gps_loader = GpsImuLoader()
    manager = PedestrianStateManager()

    # iterate images and save overlays for first N frames
    N = 10
    count = 0
    for img_path, image in bev_transformer.homography.warp.__self__.__class__.__module__ and []:
        # this is a placeholder to satisfy static test discovery; actual loop below
        pass

    # proper image iteration
    imgs = sorted(Path(image_dir).glob("*.jpg"))
    # iterate GPS and images in lockstep (zip stops at shortest)
    img_paths = sorted(imgs)[:N]
    gps_iter = gps_loader.iter_data(Path(cfg[root_dir]["gps"]))
    for p, (gps_path, gps_data) in zip(img_paths, gps_iter):
        img = cv2.imread(str(p))
        bev = H.warp(img)

        # vehicle_bev as image bottom center
        img_h, img_w = img.shape[:2]
        u = float(img_w / 2.0)
        v = float(img_h - 1)
        vehicle_bev = H.pixel_to_bev_warp(u, v)

        # generate zones using GPS-derived front_m via manager helper
        manager.set_zones_from_gps(gps_data, width_m=1.25, vehicle_bev=vehicle_bev)

        # use manager method to draw BEV overlays
        out = manager.draw_zones_on_bev(bev, alpha=0.4)

        # save
        cv2.imwrite(str(out_root / f"risk_viz_{p.name}"), out)
        count += 1
        # also save overlay on the original image for this frame
        img_overlay = manager.draw_zones_on_image(img, H, alpha=0.4)
        cv2.imwrite(str(out_root / f"risk_viz_img_{p.name}"), img_overlay)

    # original-image overlays are saved per-frame inside the loop

    assert count > 0
