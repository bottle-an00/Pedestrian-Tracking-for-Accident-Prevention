from pathlib import Path
import numpy as np
import cv2
import os

from src.core.config import load_yaml

from src.io.image_loader import ImageLoader
from src.io.gps_loader import GpsImuLoader

from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography

from src.tracking.tracker import ByteTracker

from src.bev.bev_transformer import BevTransformer
from src.trajectory.ego_motion import EgoMotionCompensator
from src.trajectory.trajectory_manager import TrajectoryBuffer

from src.trajectory.ekf_manager import EKFManager
from src.visualization.overlay_2d import Visualizer


def test_full_pipeline():
    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"
    # ---------------------------------------------------------
    # 1) Load Directories
    # ---------------------------------------------------------
    image_dir = Path(cfg[root_dir]["images"])
    gps_dir = Path(cfg[root_dir]["gps"])
    out_root = Path(cfg[root_dir]["outputs"]) / "full_pipeline"
    out_root.mkdir(parents=True, exist_ok=True)

    # Subfolders
    out_bev_traj = out_root / "bev_trajectory"
    out_bev_result = out_root / "bev_result"
    out_img_traj = out_root / "img_trajectory"
    out_tracking = out_root / "tracking"
    out_ekf_bev = out_root / "ekf_bev"
    out_ekf_img = out_root / "ekf_img"

    for p in [out_bev_traj, out_bev_result, out_img_traj, out_tracking, out_ekf_bev, out_ekf_img]:
        p.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2) Calibration Load
    # ---------------------------------------------------------
    calib_loader = CalibrationInfoLoader()

    intrinsics = calib_loader.load_camera_calibration(
        Path(cfg[root_dir]["calibration"]["calib_Camera"])
    )
    extrinsics = calib_loader.load_camera_extrinsics(
        Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"])
    )

    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)

    # ---------------------------------------------------------
    # 3) Load Modules
    # ---------------------------------------------------------
    image_loader = ImageLoader()
    gps_loader = GpsImuLoader()
    vis = Visualizer()

    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"]
    )

    bev_conv = BevTransformer()
    ego_comp = EgoMotionCompensator()
    trajectory_buffer = TrajectoryBuffer(max_length=100)
    ekf_manager = EKFManager()

    # ---------------------------------------------------------
    # 4) Run Pipeline
    # ---------------------------------------------------------
    img_iter = image_loader.iter_imgs_cv2(image_dir)
    gps_iter = gps_loader.iter_data(gps_dir)

    for (img_path, image), (gps_path, gps_data) in zip(img_iter, gps_iter):
        print(f"[Full Pipeline] Processing {img_path.name}")

        # -----------------------------
        # A. Tracking (YOLO + Tracker)
        # -----------------------------
        detections = tracker.process(image)

        # 🎯 Output (3) Tracking 결과 : Original 이미지
        img_tracking = vis.draw_on_img(image.copy(), detections)
        cv2.imwrite(str(out_tracking / f"track_{img_path.name}"), img_tracking)

        # -----------------------------
        # B. Pixel → BEV 변환
        # -----------------------------
        bev_img = H.warp(image)

        foot_bevs = bev_conv.foot_uv_to_foot_bev(detections)
        bev_result = vis.draw_on_BEV(bev_img.copy(), 0, [p.foot_bev for p in foot_bevs])
        cv2.imwrite(str(out_bev_result / f"bev_{img_path.name}"), bev_result)
        # -----------------------------
        # C. Ego-motion 보정 + Trajectory 저장
        # -----------------------------
        compensated = ego_comp.compensate(foot_bevs, gps_data)
        trajectory_buffer.add(compensated, gps_data.time)

        # ===================================================================
        # (1) Trajectory → BEV Image
        # ===================================================================
        bev_traj_img = bev_img.copy()
        for tid, traj in trajectory_buffer.get_all().items():
            world_traj = ego_comp.inv_compensate_all(traj, gps_data)
            bev_traj_img = vis.draw_on_BEV(
                bev_traj_img,
                tid,
                [p.foot_bev for p in world_traj]
            )

        cv2.imwrite(str(out_bev_traj / f"bev_traj_{img_path.name}"), bev_traj_img)

        # ===================================================================
        # (2) Trajectory → Original Image
        # ===================================================================
        img_traj = image.copy()
        for tid, traj in trajectory_buffer.get_all().items():
            world_traj = ego_comp.inv_compensate_all(traj, gps_data)
            uv_list = bev_conv.foot_bev_to_foot_uv([p.foot_bev for p in world_traj])

            img_traj = vis.draw_on_img(
                img_traj,
                [
                    {"bbox": (0,0,0,0), "id": tid,
                     "class": "traj", "score": 1.0, "foot_uv": uv}
                    for uv in uv_list
                ]
            )

        cv2.imwrite(str(out_img_traj / f"img_traj_{img_path.name}"), img_traj)

        # -----------------------------
        # D. EKF 업데이트 + 미래 예측
        # -----------------------------
        ekf_now = {}
        ekf_future = {}

        for d in compensated:
            tid = d.id
            xy = ekf_manager.update(tid, d.foot_bev, gps_data.time)
            ekf_now[tid] = xy
            ekf_future[tid] = ekf_manager.predict_future(tid, steps=20)

        # ===================================================================
        # (4) EKF 결과 → BEV & Original
        # ===================================================================

        # BEV 이미지
        ekf_bev_img = bev_img.copy()

        # Original 이미지
        ekf_img = image.copy()

        for tid, xy in ekf_now.items():
            # 현재 위치 보정
            world_xy = ego_comp.inv_compensate_list(xy, gps_data)

            # --- BEV ---
            ekf_bev_img = vis.draw_points(ekf_bev_img, tid, [world_xy], radius=6)

            # --- Original ---
            uv = bev_conv.foot_bev_to_foot_uv([world_xy])[0]
            ekf_img = vis.draw_on_img(
                ekf_img,
                [{"bbox": (0,0,0,0), "id": tid,
                  "class": "ekf", "score": 1.0, "foot_uv": uv}]
            )

        for tid, future in ekf_future.items():
            if future is None:
                continue

            world_future = [ego_comp.inv_compensate_list(p, gps_data) for p in future]

            # BEV polyline
            ekf_bev_img = vis.draw_polyline(
                ekf_bev_img, tid, world_future,
                color=(255, 0, 0), thickness=2
            )

            # Original
            uv_future = bev_conv.foot_bev_to_foot_uv(world_future)
            ekf_img = vis.draw_on_img(
                ekf_img,
                [{"bbox": (0,0,0,0), "id": tid, "class": "ekf_future",
                  "score": 1.0, "foot_uv": uv}
                 for uv in uv_future]
            )

        cv2.imwrite(str(out_ekf_bev / f"ekf_bev_{img_path.name}"), ekf_bev_img)
        cv2.imwrite(str(out_ekf_img / f"ekf_img_{img_path.name}"), ekf_img)

    assert True
