from pathlib import Path
import numpy as np
import cv2
import os
import json

from src.core.config import load_yaml
from src.io.image_loader import ImageLoader
from src.io.gps_loader import GpsImuLoader
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.tracking.tracker import ByteTracker
from src.bev.bev_transformer import BevTransformer, Detections_bev
from src.trajectory.trajectory_manager import TrajectoryBuffer
from src.visualization.overlay_2d import Visualizer
from src.trajectory.pedestrian_state_manager import PedestrianStateManager
from src.trajectory.bev_zone_risk_manager import PedestrianStateManager as BevZoneManager
from src.trajectory.ekf_manager import EKFManager


def test_full_pipeline():
    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"

    image_dir = Path(cfg[root_dir]["images"])
    gps_dir = Path(cfg[root_dir]["gps"])

    out_root = Path(cfg[root_dir]["outputs"]) / "full_pipeline_test"
    out_root.mkdir(parents=True, exist_ok=True)

    out_ekf_bev = out_root / "ekf_bev"
    out_ekf_img = out_root / "ekf_img"

    for p in [out_ekf_bev, out_ekf_img]:
        p.mkdir(parents=True, exist_ok=True)

    frame_idx = 0

    calib_loader = CalibrationInfoLoader()
    intrinsics = calib_loader.load_camera_calibration(Path(cfg[root_dir]["calibration"]["calib_Camera"]))
    extrinsics = calib_loader.load_camera_extrinsics(Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"]))

    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)

    image_loader = ImageLoader()
    gps_loader = GpsImuLoader()
    vis = Visualizer()

    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 1280)
    )

    bev_conv = BevTransformer(homography=H)
    trajectory_buffer = TrajectoryBuffer(max_length=100)
    eval_mode = True

    ekf_manager = EKFManager()

    #load DataSet
    img_iter = image_loader.iter_imgs_cv2(image_dir)
    gps_iter = gps_loader.iter_data(gps_dir)

    bev_risk_manager = BevZoneManager()

    # Main loop
    for (img_path, image), (gps_path, gps_data) in zip(img_iter, gps_iter):
        matched_count = 0
        print(f"[Full Pipeline] Processing {img_path.name}")

        detections = tracker.process(image)

        bev_img = H.warp(image)
        foot_bevs = bev_conv.foot_uv_to_foot_bev(detections)

        foot_bevs_m = []
        for fb in foot_bevs:
            bx, by = fb.foot_bev
            x_m, y_m = H.pixel_to_world(float(by), float(bx), flipped=True)
            foot_bevs_m.append(Detections_bev(id=fb.id, foot_bev=(x_m, y_m)))

        # vehicale_bev setting
        vehicle_bev = bev_risk_manager.compute_vehicle_bev(image,H)
        bev_risk_manager.set_zones_from_gps(gps_data, vehicle_bev=vehicle_bev)

        ekf_now = {}
        ekf_future = {}

        # Use external ekf_manager for current estimates and future predictions
        for d in foot_bevs_m:
            try:
                tid = int(d.id)
            except Exception:
                tid = d.id
            xy = ekf_manager.update(tid, d.foot_bev, gps_data.time)
            ekf_now[tid] = xy
            ekf_future[tid] = ekf_manager.predict_future(tid, steps=20)

        ekf_bev_img = bev_img.copy()
        ekf_img = image.copy()

        for tid, xy in ekf_now.items():
            if xy is None:
                continue
            lx, ly = float(xy[0]), float(xy[1])
            u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
            ekf_bev_img = vis.draw_points(ekf_bev_img, tid, [(v_bev, u_bev)], radius=6)
            u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
            if u_img != -1 and v_img != -1:
                ekf_img = vis.draw_on_img(ekf_img, [{"bbox": (0,0,0,0), "id": tid, "class": "ekf", "score": 1.0, "foot_uv": (u_img, v_img)}])

        for tid, future in ekf_future.items():
            if future is None:
                continue
            bev_future_pts = []
            img_uvs = []
            for p in future:
                lx, ly = float(p[0]), float(p[1])
                u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
                bev_future_pts.append((v_bev, u_bev))
                u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
                if u_img != -1 and v_img != -1:
                    img_uvs.append((u_img, v_img))

            if len(bev_future_pts) == 1:
                ekf_bev_img = vis.draw_points(ekf_bev_img, tid, bev_future_pts, radius=4)
            elif bev_future_pts:
                ekf_bev_img = vis.draw_polyline(ekf_bev_img, tid, bev_future_pts, color=(255, 0, 0), thickness=2)

            if img_uvs:
                ekf_img = vis.draw_on_img(ekf_img, [{"bbox": (0,0,0,0), "id": tid, "class": "ekf_future", "score": 1.0, "foot_uv": uv} for uv in img_uvs])

        # Update risk states
        obj_states = {}
        for tid, future in ekf_future.items():
            state = "SAFE"
            if future and len(future) > 0:
                updated_state = bev_risk_manager.update_state(tid, future)
                state = updated_state
            obj_states[tid] = state

        frame_idx += 1

        # Visualizations
        gt_vis = ekf_bev_img.copy()
        gt_vis = bev_risk_manager.draw_zones_on_bev(gt_vis, alpha=0.25)
        ekf_img = bev_risk_manager.draw_zones_on_image(ekf_img, H, alpha=0.25)
        ekf_img = bev_risk_manager.draw_detections_on_image(ekf_img, detections)

        cv2.imwrite(str(out_ekf_bev / f"ekf_bev_gt_{img_path.name}"), gt_vis)
        cv2.imwrite(str(out_ekf_img / f"ekf_img_{img_path.name}"), ekf_img)

