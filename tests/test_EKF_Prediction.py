from pathlib import Path
import numpy as np
import cv2
import os

from src.core.config import load_yaml

from src.io.image_loader import ImageLoader
from src.io.gps_loader import GpsImuLoader, GpsImuSample

from src.calibration.load_calibration_info import CalibrationInfoLoader

from src.calibration.homography import Homography

from src.tracking.tracker import ByteTracker

from src.bev.bev_transformer import BevTransformer
from src.trajectory.ego_motion import EgoMotionCompensator
from src.trajectory.trajectory_manager import TrajectoryBuffer

from src.trajectory.ekf_manager import EKFManager

from src.visualization.overlay_2d import Visualizer

def test_tracker():
    cfg = load_yaml("configs/system.yaml")
    root_dir = "test_data_dir"
    image_dir = Path(cfg[root_dir]["images"])
    gps_dir = Path(cfg[root_dir]["gps"])

    if not image_dir.exists():
        print(f"Image directory: {image_dir}")
        raise ValueError("Image_directory does not exist.")
    if not gps_dir.exists():
        raise ValueError("GPS directory does not exist.")

    intrinsics_path = Path(cfg[root_dir]["calibration"]["calib_Camera"])
    extrinsics_path = Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"])

    calib_loader = CalibrationInfoLoader()
    intrinsics = calib_loader.load_camera_calibration(intrinsics_path)
    extrinsics = calib_loader.load_camera_extrinsics(extrinsics_path)

    assert intrinsics is not None
    assert extrinsics is not None

    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)
    image_loader = ImageLoader()
    gps_loader = GpsImuLoader()
    vis = Visualizer()

    yolo_cgf = load_yaml("configs/detector/yolo_detector.yaml")

    yolo_model_path = yolo_cgf["yolo_model_path"]
    conf_threshold  = yolo_cgf["conf_threshold"]
    target_classes  = yolo_cgf["target_classes"]

    tracker = ByteTracker(
        model_path=yolo_model_path,
        conf_thres_config=conf_threshold,
        target_class_names=target_classes
    )

    bev_converter = BevTransformer()
    ego_compensator = EgoMotionCompensator()
    trajectory_buffer = TrajectoryBuffer(max_length=100)
    ekf_manager = EKFManager()

    output_root = Path(cfg[root_dir]["outputs"]) / "EKF_results"
    output_root.mkdir(parents=True, exist_ok=True)

    sub_output1 = output_root / "bev_overlay"
    sub_output2 = output_root / "img_overlay"

    sub_outputs = [sub_output1, sub_output2]
    for sub_output in sub_outputs:
        sub_output.mkdir(parents=True, exist_ok=True)

    img_items = image_loader.iter_imgs_cv2(image_dir)
    gps_items = gps_loader.iter_data(gps_dir)

    for img_item, gps_item in zip(img_items, gps_items):

        img_path, image = img_item
        gps_path, gps_data = gps_item
        print(f"[Trajectory Test] Processing Image: {img_path} with GPS: {gps_path}")
        assert image is not None
        assert np.any(image)

        detections = tracker.process(image)

        assert isinstance(detections, list)

        bev_img = H.warp(image)
        assert bev_img is not None

        foot_bevs = bev_converter.foot_uv_to_foot_bev(detections)

        compensated_bevs = ego_compensator.compensate(foot_bevs, gps_data)

        trajectory_buffer.add(compensated_bevs, gps_data.time)

        # EKF 업데이트 + 미래 예측
        ekf_results = {}          # 현재 EKF 추정값
        ekf_future_results = {}   # 미래 예측 trajectory

        for d in compensated_bevs:
            track_id = d.id
            foot_bev = d.foot_bev

            # (A) EKF 업데이트 (현재 추정)
            ekf_xy = ekf_manager.update(track_id, foot_bev,gps_data.time)
            ekf_results[track_id] = ekf_xy

            # (B) 미래 20스텝 예측
            future_traj = ekf_manager.predict_future(track_id, steps=20)
            ekf_future_results[track_id] = future_traj

        bev_overlay = bev_img.copy()
        dst = image.copy()

        # (1) trajectory_buffer 기반 과거 traj 시각화
        for traj_id, traj in trajectory_buffer.get_all().items():
            # bev에 그리기
            bev_traj = ego_compensator.inv_compensate_all(traj, gps_data)
            bev_overlay = vis.draw_on_BEV(
                bev_overlay,
                track_id,
                [point.foot_bev for point in bev_traj]
            )
            # 원본 이미지에 그리기
            uv_traj = bev_converter.foot_bev_to_foot_uv([point.foot_bev for point in bev_traj])
            dst = vis.draw_on_img(
                dst,
                [{
                    "bbox": (0,0,0,0),   # bbox는 그리지 않음
                    "id": traj_id,
                    "class": "trajectory",
                    "score": 1.0,
                    "foot_uv": uv
                } for uv in uv_traj]
            )

        # (2) EKF 현재 위치 (빨간 점)
        for track_id, xy in ekf_results.items():
            # bev img에 그리기
            new_xy = ego_compensator.inv_compensate_list(xy, gps_data)
            bev_overlay = vis.draw_points(bev_overlay, track_id, [new_xy], radius=6)
            # 원본 이미지에 그리기
            uv = bev_converter.foot_bev_to_foot_uv([new_xy])
            dst = vis.draw_on_img(
                dst,
                [{
                    "bbox": (0,0,0,0),   # bbox는 그리지 않음
                    "id": track_id,
                    "class": "ekf_position",
                    "score": 1.0,
                    "foot_uv": uv[0]
                }]
            )

        # (3) EKF 미래 trajectory (파란 선)
        for track_id, future in ekf_future_results.items():
            if future is not None:

                # bev img에 그리기
                new_future = []
                for future_point in future:
                    new_future.append(ego_compensator.inv_compensate_list(future_point, gps_data))
                bev_overlay = vis.draw_polyline(
                    bev_overlay,
                    track_id,
                    new_future,
                    color=(255, 0, 0),
                    thickness=2
                )

                # 원본 이미지에 그리기
                uv_future = bev_converter.foot_bev_to_foot_uv(new_future)
                dst = vis.draw_on_img(
                   dst,
                   [{
                       "bbox": (0,0,0,0),   # bbox는 그리지 않음
                       "id": track_id,
                       "class": "ekf_future",
                       "score": 1.0,
                       "foot_uv": uv
                   } for uv in uv_future]
                )

        cv2.imwrite(str(sub_output1 / f"bev_overlay_{img_path.name}"), bev_overlay)
        cv2.imwrite(str(sub_output2 / f"img_overlay_{img_path.name}"), dst)
    assert True