from pathlib import Path
import numpy as np 

from src.core.config import load_yaml
from src.calibration.load_calibration_info import CalibrationInfoLoader

def test_calibration_info_loader():
    loader = CalibrationInfoLoader()

    cfg = load_yaml("configs/system.yaml")

    intrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_Camera"])
    extrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Camera"])
    lidar_to_imu_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Imu"])

    if not intrinsics_path.exists():
        return
    if not extrinsics_path.exists():
        return
    if not lidar_to_imu_path.exists():
        return

    loader.load_camera_calibration(str(intrinsics_path))
    loader.load_camera_extrinsics(str(extrinsics_path))
    loader.load_lidar_to_imu(str(lidar_to_imu_path))

    assert loader.camera_intrinsics.shape == (3, 3)
    assert loader.camera_extrinsics.shape == (3, 4)
    assert loader.lidar_to_imu.shape == (3, 4)

    output_dir = Path(cfg["test_data_dir"]["outputs"]) / "calibration_info"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the calibration matrices as .npy files
    np.savetxt(output_dir / "camera_intrinsics.txt", loader.camera_intrinsics)
    np.savetxt(output_dir / "camera_extrinsics.txt", loader.camera_extrinsics)
    np.savetxt(output_dir / "lidar_to_imu.txt", loader.lidar_to_imu)
