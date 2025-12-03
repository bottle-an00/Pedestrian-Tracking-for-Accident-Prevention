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
    def __init__(self, homography=None):
        if homography is None:
            raise ValueError("Homography must be provided during BevTransformer initialization")

        self.homography = homography


    def foot_uv_to_foot_bev(self, detections: List[Dict]) -> List[Detections_bev]:
        bev_detections = []
        for detection in detections:
            foot_uv = detection["foot_uv"]
            bx, by = self.homography.pixel_to_bev_warp(foot_uv[0], foot_uv[1])
            if bx == -1 and by == -1:
                continue
            bev_detections.append(
                Detections_bev(
                    id=detection["id"],
                    foot_bev=(bx, by)
                )
            )
        return bev_detections

    def foot_bev_to_foot_uv(self, bevs: List) -> List:
        results = []
        for bev in bevs:
            ux, uy = self.homography.bev_to_pixel(bev[0], bev[1])
            if ux == -1 and uy == -1:
                continue
            if ux is not None and uy is not None:
                results.append((ux, uy))
        return results

    def warp_image(self, image_bgr: np.ndarray, border_value=(0, 0, 0)) -> np.ndarray:
        """Warp 원본 이미지를 BEV 이미지로 변환"""
        return self.homography.warp(image_bgr, border_value=border_value)
