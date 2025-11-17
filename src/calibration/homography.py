# pixel↔BEV 변환 행렬 계산/적용
from src.core.config import load_yaml
import numpy as np
import cv2

class Homography:

    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray):
        cfg = load_yaml("configs/system.yaml")
        self.bev_resolution = float(cfg["bev"]["resolution"])
        self.bev_front = float(cfg["bev"]["front"])
        self.bev_back = float(cfg["bev"]["back"])
        self.bev_left = float(cfg["bev"]["left"])
        self.bev_right = float(cfg["bev"]["right"])
        self.bev_ground_z = float(cfg["bev"]["ground_z"])

        assert intrinsics.shape == (3, 3)
        assert extrinsics.shape == (3, 4)

        self.K = intrinsics.astype(np.float32)
        self.extrinsic = extrinsics.astype(np.float32)

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.R_cw = self.extrinsic[:, :3]           # (3,3)
        self.t_cw = self.extrinsic[:, 3:4]          # (3,1)

        self.R_wc = self.R_cw.T
        self.t_wc = -self.R_wc @ self.t_cw

        self._map_x = None
        self._map_y = None
        self._bev_w = None
        self._bev_h = None

    def _create_bev_grid(self):

        x_min = -self.bev_back
        x_max = self.bev_front
        y_min = -self.bev_right
        y_max = self.bev_left

        res = self.bev_resolution

        bev_h = int((x_max - x_min) / res)  # x
        bev_w = int((y_max - y_min) / res)  # y

        xs = np.linspace(x_min, x_max, bev_h, endpoint=False) + res / 2.0
        ys = np.linspace(y_min, y_max, bev_w, endpoint=False) + res / 2.0

        Xw, Yw = np.meshgrid(xs, ys, indexing="ij")
        Zw = np.full_like(Xw, self.bev_ground_z, dtype=np.float32)

        return Xw.astype(np.float32), Yw.astype(np.float32), Zw, bev_w, bev_h

    def _build_bev_remap(self):

        Xw, Yw, Zw, bev_w, bev_h = self._create_bev_grid()

        world_points = np.stack([-Yw, -Xw, Zw], axis=-1)  # (H, W, 3)
        world_points = world_points.reshape(-1, 3).T    # (3, N)

        cam_points = self.R_wc @ world_points + self.t_wc  # (3, N)
        print(cam_points)
        Xc = cam_points[0, :]
        Yc = cam_points[1, :]
        Zc = cam_points[2, :]

        valid = Zc > 0

        eps = 1e-6
        u = self.fx * (Xc / (Zc + eps)) + self.cx
        v = self.fy * (Yc / (Zc + eps)) + self.cy

        map_x = np.full((bev_h, bev_w), -1, dtype=np.float32)
        map_y = np.full((bev_h, bev_w), -1, dtype=np.float32)

        bev_indices = np.arange(world_points.shape[1])[valid]
        bev_y_idx = bev_indices % bev_w
        bev_x_idx = bev_indices // bev_w

        bev_x_idx = (bev_h - 1) - bev_x_idx
        
        u_valid = u[valid]
        v_valid = v[valid]

        map_x[bev_x_idx, bev_y_idx] = u_valid
        map_y[bev_x_idx, bev_y_idx] = v_valid

        self._map_x = map_x
        self._map_y = map_y
        self._bev_w = bev_w
        self._bev_h = bev_h

        print("Zc min/max:", float(Zc.min()), float(Zc.max()))
        print("num valid:", int((Zc > 0).sum()), "/", Zc.size)

    def warp(self, image_bgr: np.ndarray,
             border_value=(0, 0, 0)) -> np.ndarray:

        if self._map_x is None or self._map_y is None:
            self._build_bev_remap()

        bev_bgr = cv2.remap(
            image_bgr,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return bev_bgr