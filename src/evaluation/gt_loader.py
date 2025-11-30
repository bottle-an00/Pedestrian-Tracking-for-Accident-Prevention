# GT 라벨 파서 (image + lidar)

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class GTObject:
    """단일 GT 객체"""
    class_name: str
    instance_id: int
    track_id: Optional[int] = None

    # Image label (bbox)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

    # Derived from bbox: foot_uv = (cx, y2)
    foot_uv: Optional[Tuple[float, float]] = None

    # LiDAR label (3D)
    location_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z)
    dimension: Optional[Tuple[float, float, float]] = None    # (w, h, l)
    yaw: Optional[float] = None


@dataclass
class GTFrame:
    """단일 프레임의 GT 라벨"""
    frame_id: str
    image_resolution: Optional[Tuple[int, int]] = None  # (width, height)
    objects: List[GTObject] = field(default_factory=list)

    def get_pedestrians(self) -> List[GTObject]:
        """보행자 객체만 반환"""
        return [obj for obj in self.objects if obj.class_name == "pedestrian"]

    def get_vehicles(self) -> List[GTObject]:
        """차량 객체만 반환"""
        return [obj for obj in self.objects if obj.class_name == "vehicle"]


class GTLoader:
    """
    GT 라벨 로더
    - image label: bbox, track_id
    - lidar label: 3D location, dimension, yaw
    """

    def __init__(self, target_classes: List[str] = None):
        """
        Args:
            target_classes: 로드할 클래스 목록 (None이면 전체)
        """
        self.target_classes = target_classes or ["pedestrian"]

    def load_image_label(self, json_path: str | Path) -> Dict:
        """Image 라벨 JSON 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_lidar_label(self, json_path: str | Path) -> Dict:
        """LiDAR 라벨 JSON 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse_image_label(self, data: Dict) -> Tuple[List[GTObject], Tuple[int, int]]:
        """
        Image 라벨 파싱

        Returns:
            objects: GTObject 리스트 (bbox, foot_uv 포함)
            resolution: (width, height)
        """
        objects = []
        resolution = tuple(data.get("information", {}).get("resolution", [0, 0]))

        for ann in data.get("annotations", []):
            class_name = ann.get("class", "")

            # 타겟 클래스 필터링
            if class_name not in self.target_classes:
                continue

            # bbox 필수
            bbox = ann.get("bbox")
            if bbox is None:
                continue

            attr = ann.get("attribute", {})
            instance_id = attr.get("instance_id", -1)
            track_id = attr.get("track_id")

            # bbox: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            # foot_uv: bbox 하단 중앙
            foot_u = (x1 + x2) / 2.0
            foot_v = float(y2)

            obj = GTObject(
                class_name=class_name,
                instance_id=instance_id,
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                foot_uv=(foot_u, foot_v),
            )
            objects.append(obj)

        return objects, resolution

    def parse_lidar_label(self, data: Dict) -> List[GTObject]:
        """
        LiDAR 라벨 파싱

        Returns:
            objects: GTObject 리스트 (3D location 포함)
        """
        objects = []

        for ann in data.get("annotations", []):
            class_name = ann.get("class", "")

            # 타겟 클래스 필터링
            if class_name not in self.target_classes:
                continue

            attr = ann.get("attribute", {})
            instance_id = attr.get("instance_id", -1)
            track_id = attr.get("track_id")
            location = attr.get("location")
            dimension = attr.get("dimension")
            yaw = attr.get("yaw")

            if location is None:
                continue

            obj = GTObject(
                class_name=class_name,
                instance_id=instance_id,
                track_id=track_id,
                location_3d=tuple(location),
                dimension=tuple(dimension) if dimension else None,
                yaw=yaw,
            )
            objects.append(obj)

        return objects

    def load_frame(
        self,
        image_label_path: str | Path,
        lidar_label_path: str | Path = None,
        bev_transformer=None,
    ) -> GTFrame:
        """
        단일 프레임 GT 로드 (image + lidar 병합)

        Args:
            image_label_path: 이미지 라벨 JSON 경로
            lidar_label_path: LiDAR 라벨 JSON 경로 (optional)
            bev_transformer: BevTransformer (3D projection 기반 매칭용)

        Returns:
            GTFrame: 병합된 GT 프레임
        """
        image_label_path = Path(image_label_path)
        frame_id = image_label_path.stem

        # Image 라벨 파싱
        image_data = self.load_image_label(image_label_path)
        image_objects, resolution = self.parse_image_label(image_data)

        # LiDAR 라벨 병합 (있는 경우)
        if lidar_label_path and Path(lidar_label_path).exists():
            lidar_data = self.load_lidar_label(lidar_label_path)
            lidar_objects = self.parse_lidar_label(lidar_data)

            # 매칭 방법 결정
            if bev_transformer is not None:
                # 3D projection 기반 매칭
                self._match_by_projection(image_objects, lidar_objects, bev_transformer, resolution)
            else:
                # track_id 기반 매칭 (fallback)
                track_id_to_obj: Dict[int, GTObject] = {}
                for obj in image_objects:
                    if obj.track_id is not None:
                        track_id_to_obj[obj.track_id] = obj

                for lidar_obj in lidar_objects:
                    if lidar_obj.track_id in track_id_to_obj:
                        img_obj = track_id_to_obj[lidar_obj.track_id]
                        img_obj.location_3d = lidar_obj.location_3d
                        img_obj.dimension = lidar_obj.dimension
                        img_obj.yaw = lidar_obj.yaw

        return GTFrame(
            frame_id=frame_id,
            image_resolution=resolution,
            objects=image_objects,
        )

    def _match_by_projection(
        self,
        image_objects: List[GTObject],
        lidar_objects: List[GTObject],
        bev_transformer,
        resolution: Tuple[int, int],
    ):
        """
        BEV 픽셀 좌표 기반으로 이미지 GT와 LiDAR GT 매칭

        - 이미지 GT의 foot_uv → BEV 픽셀 좌표
        - LiDAR GT의 location_3d (x, y) → BEV 픽셀 좌표
        - 두 BEV 픽셀이 가까우면 매칭
        """
        if not image_objects or not lidar_objects:
            return

        homography = bev_transformer.homography

        # 이미지 GT의 foot_uv → BEV 픽셀
        image_bev_pixels = []
        for img_obj in image_objects:
            if img_obj.foot_uv is None:
                image_bev_pixels.append(None)
                continue

            u, v = img_obj.foot_uv
            try:
                bev_px, bev_py = homography.pixel_to_bev_warp(u, v)
                if bev_px >= 0 and bev_py >= 0:
                    image_bev_pixels.append((bev_px, bev_py))
                else:
                    image_bev_pixels.append(None)
            except Exception:
                image_bev_pixels.append(None)

        # LiDAR GT의 location_3d (x, y) → BEV 픽셀
        # BEV 설정 가져오기
        res = homography.bev_resolution
        x_min = homography.bev_x_min
        y_min = homography.bev_y_min
        bev_h = homography._bev_h
        bev_w = homography._bev_w

        lidar_bev_pixels = []
        for lidar_obj in lidar_objects:
            if lidar_obj.location_3d is None:
                lidar_bev_pixels.append(None)
                continue

            # LiDAR 좌표 (미터) → BEV 픽셀
            x_m, y_m = lidar_obj.location_3d[0], lidar_obj.location_3d[1]

            # world 좌표 → BEV 픽셀 (homography 좌표계 고려)
            # Xw = -bev_y, Yw = -bev_x (homography._build_bev_remap 참조)
            bev_px = int((bev_h - 1) - (-y_m - x_min) / res)
            bev_py = int((-x_m - y_min) / res)

            if 0 <= bev_px < bev_h and 0 <= bev_py < bev_w:
                lidar_bev_pixels.append((bev_px, bev_py))
            else:
                lidar_bev_pixels.append(None)

        # BEV 픽셀 좌표 기준 매칭
        matched_lidar = set()
        for i, img_obj in enumerate(image_objects):
            img_bev = image_bev_pixels[i]
            if img_bev is None:
                continue

            best_dist = float('inf')
            best_idx = -1

            for j, lidar_bev in enumerate(lidar_bev_pixels):
                if lidar_bev is None or j in matched_lidar:
                    continue

                # BEV 픽셀 거리
                dist = ((img_bev[0] - lidar_bev[0]) ** 2 + (img_bev[1] - lidar_bev[1]) ** 2) ** 0.5

                # 매칭 threshold: 300 픽셀 이내 (resolution=0.01이면 3m)
                if dist < best_dist and dist < 300:
                    best_dist = dist
                    best_idx = j

            if best_idx >= 0:
                matched_lidar.add(best_idx)
                lidar_obj = lidar_objects[best_idx]
                img_obj.location_3d = lidar_obj.location_3d
                img_obj.dimension = lidar_obj.dimension
                img_obj.yaw = lidar_obj.yaw

    def iter_frames(
        self,
        image_label_dir: str | Path,
        lidar_label_dir: str | Path = None,
    ):
        """
        디렉토리 내 전체 프레임 순회

        Args:
            image_label_dir: 이미지 라벨 디렉토리
            lidar_label_dir: LiDAR 라벨 디렉토리 (optional)

        Yields:
            GTFrame
        """
        image_label_dir = Path(image_label_dir)
        lidar_label_dir = Path(lidar_label_dir) if lidar_label_dir else None

        json_files = sorted(image_label_dir.glob("*.json"))

        for img_json in json_files:
            lidar_json = None
            if lidar_label_dir:
                lidar_json = lidar_label_dir / img_json.name

            yield self.load_frame(img_json, lidar_json)
