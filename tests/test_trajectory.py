from pathlib import Path
import numpy as np
import cv2
import os

from src.core.config import load_yaml

from src.io.image_loader import ImageLoader
from src.io.gps_loader import GpsImuLoader, GpsImuSample

from src.calibration.load_calibration_info import CalibrationInfoLoader

from src.calibration.homography import Homography

from src.tracking.tracker import UltralyticsTracker

from src.bev.bev_transformer import BevTransformer
from src.trajectory.ego_motion import EgoMotionCompensator
from src.trajectory.trajectory_manager import TrajectoryBuffer

from src.visualization.overlay_2d import Visualizer

def test_tracker():
    cfg = load_yaml("configs/system.yaml")

    image_dir = Path(cfg["test_data_dir"]["images"])
    gps_dir = Path(cfg["test_data_dir"]["gps"])
    
    if not image_dir.exists():
        raise ValueError("Image_directory does not exist.") 
    if not gps_dir.exists():
        raise ValueError("GPS directory does not exist.")
    
    intrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_Camera"])
    extrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Camera"])

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
    tracker_type    = yolo_cgf["tracker"]


    tracker = UltralyticsTracker(
        model_path=yolo_model_path,
        conf_thres_config=conf_threshold,
        target_class_names=target_classes,
        tracker_type=tracker_type
    )

    bev_converter = BevTransformer()
    ego_compensator = EgoMotionCompensator()
    trajectory_buffer = TrajectoryBuffer(max_length=100)

    output_root = Path(cfg["test_data_dir"]["outputs"]) / "trajectory_results"
    output_root.mkdir(parents=True, exist_ok=True)

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
        
        # 현재 만들어진 traj 확인 -> traj_id 기준으로 latest point 확인
        bev_overlay = bev_img.copy()
        for traj_id, traj in trajectory_buffer.get_all().items():
            
            bev_traj = ego_compensator.inv_compensate_all(traj, gps_data)
            bev_overlay = vis.draw_on_BEV(bev_overlay, [point.foot_bev for point in bev_traj], color=(0,(100*(traj_id+1))%255,0))
        
        cv2.imwrite(str(output_root / f"bev_overlay_{img_path.name}"), bev_overlay)

    assert True