# foot_uv → foot_bev 변환

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

from src.calibration.homography import Homography
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.core.config import load_yaml

@dataclass
class Detections_bev:
    id: int
    foot_bev: tuple

class BevTransformer:
    def __init__(self):

        cfg = load_yaml("configs/system.yaml")

        intrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_Camera"])
        extrinsics_path = Path(cfg["test_data_dir"]["calibration"]["calib_LiDAR_Camera"])

        calib_loader = CalibrationInfoLoader()

        intrinsics = calib_loader.load_camera_calibration(intrinsics_path)
        extrinsics = calib_loader.load_camera_extrinsics(extrinsics_path)

        self.homography = Homography(intrinsics=intrinsics, extrinsics=extrinsics)

    def foot_uv_to_foot_bev(self, detections:List[Dict]) -> List[Detections_bev]:
        bev_detections = []
        for detection in detections:
            foot_uv = detection["foot_uv"]
            foot_bev_x, foot_bev_y = self.homography.pixel_to_bev_warp(foot_uv[0],foot_uv[1])
            if foot_bev_x == -1 and foot_bev_y == -1:
                continue
            bev_detections.append(
                Detections_bev(
                    id = detection["id"],
                    foot_bev = (foot_bev_x, foot_bev_y)
                )
            )
        return bev_detections

    def foot_bev_to_foot_uv(self, bevs:List) -> List:
        results = []
        for bev in bevs:
            foot_uv_x, foot_uv_y = self.homography.bev_to_pixel(bev[0], bev[1])
            if foot_uv_x == -1 and foot_uv_y == -1:
                continue
            if foot_uv_x is not None and foot_uv_y is not None:
                results.append((foot_uv_x, foot_uv_y))
        return results