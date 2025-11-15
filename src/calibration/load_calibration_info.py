# intrinsic/extrinsic/LiDARToImu 로더

import numpy as np
from pathlib import Path

from src.core.config import load_yaml

class CalibrationInfoLoader:
    def __init__(self):
        calib_data = load_yaml("configs/system.yaml")

        self.lidar_to_imu = np.array({})
        self.camera_intrinsics = np.array({})
        self.camera_extrinsics = np.array({})


    def load_lidar_to_imu(self, path: str):
        calib = Path(path)

        if not Path(calib).exists():
            raise FileNotFoundError(f"[load_lidar_to_imu] File not found: {calib}")

        T = np.loadtxt(calib)

        if T.shape != (3, 4):
            raise ValueError(f"[load_lidar_to_imu] Invalid LiDAR to IMU extrinsic shape: {T.shape}")

        self.lidar_to_imu = T

    def load_camera_calibration(self, path: str):
        calib = Path(path)

        if not Path(calib).exists():
            raise FileNotFoundError(f"[load_camera_calibration] File not found: {calib}")

        K = np.loadtxt(calib)

        if K.shape != (3, 3):
            raise ValueError(f"[load_camera_calibration] Invalid camera intrinsic shape: {K.shape}")

        self.camera_intrinsics = K

    def load_camera_extrinsics(self, path: str):
        calib = Path(path)

        if not Path(calib).exists():
            raise FileNotFoundError(f"[load_camera_extrinsics] File not found: {calib}")

        T = np.loadtxt(calib)

        if T.shape != (3, 4):
            raise ValueError(f"[load_camera_extrinsics] Invalid camera extrinsic shape: {T.shape}")

        self.camera_extrinsics = T
