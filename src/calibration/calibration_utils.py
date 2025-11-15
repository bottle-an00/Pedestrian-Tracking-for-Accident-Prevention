from typing import Optional, Tuple

import numpy as np
import cv2
import open3d as o3d
Pcd = o3d.geometry.PointCloud


class CameraLidarCalibrator:
    def __init__(self, intrinsic: np.ndarray, extrinsic: np.ndarray):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

    def project_lidar_to_image(self, lidar_points) -> Optional[np.ndarray]:

        if hasattr(lidar_points, "points"):
            lidar_points = np.asarray(lidar_points.points)

        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"[project_lidar_to_image] Invalid camera intrinsic shape: {self.intrinsic.shape}")

        if self.extrinsic.shape != (3, 4):
            raise ValueError(f"[project_lidar_to_image] Invalid camera extrinsic shape: {self.extrinsic.shape}")


        num_points = lidar_points.shape[0]
        lidar_homogeneous = np.hstack((lidar_points, np.ones((num_points, 1))))

        cam_coords = (self.extrinsic @ lidar_homogeneous.T).T

        in_front_indices = cam_coords[:, 2] > 0
        if not np.any(in_front_indices):
            return None

        cam_coords = cam_coords[in_front_indices]

        img_points_homogeneous = (self.intrinsic @ cam_coords[:, :3].T).T

        img_points = img_points_homogeneous[:, :2] / img_points_homogeneous[:, 2:3]

        return img_points
    def extract_valid_points(self,
                             points_cloud: np.ndarray,
                             distance_threshold: float = 40.0) -> np.ndarray:
        if points_cloud.ndim != 2 or points_cloud.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape: {points_cloud.shape}")

        distances = np.linalg.norm(points_cloud, axis=1)  # (N,)

        mask = distances <= distance_threshold # (N,)

        return points_cloud[mask]

    def draw_lidar_on_image(self, image: np.ndarray, lidar_points: np.ndarray,
                            color: Tuple[int, int, int] = (0, 255, 0),
                            distance_threshold: float = 40.0,
                            point_size: int = 2) -> np.ndarray:

        processed_lidar_points = self.extract_valid_points(lidar_points, distance_threshold)

        img_points = self.project_lidar_to_image(processed_lidar_points)

        if img_points is None:
            return image

        for point in img_points:
            x, y = int(point[0]), int(point[1])

            cv2.circle(image, (x, y), point_size, color, -1)

        return image