# pixel↔BEV 변환 행렬 계산/적용
from src.core.config import load_yaml
import numpy as np
import cv2

class Homography:

    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray, calib_image_size: tuple = None):
        # calib_image_size: (width, height) of the image used when the intrinsics were calculated.
        # If provided and the input images use a different resolution, intrinsics will be scaled
        # automatically when `warp` is called with an image of a different size.
        cfg = load_yaml("configs/system.yaml")
        self.bev_resolution = float(cfg["bev"]["resolution"])
        self.bev_front = float(cfg["bev"]["front"])
        self.bev_back = float(cfg["bev"]["back"])
        self.bev_left = float(cfg["bev"]["left"])
        self.bev_right = float(cfg["bev"]["right"])
        self.bev_ground_z = float(cfg["bev"]["ground_z"])

        self.bev_x_min = -self.bev_back
        self.bev_x_max =  self.bev_front
from src.core.config import load_yaml
import numpy as np
import cv2


class Homography:
    """Homography / BEV utility: provides BEV warp and image<->world projections.

    This is a single, clean definition with no duplicated or nested classes.
    """

    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray, calib_image_size: tuple = None):
        cfg = load_yaml("configs/system.yaml")
        self.bev_resolution = float(cfg["bev"]["resolution"])
        self.bev_front = float(cfg["bev"]["front"])
        self.bev_back = float(cfg["bev"]["back"])
        self.bev_left = float(cfg["bev"]["left"])
        self.bev_right = float(cfg["bev"]["right"])
        self.bev_ground_z = float(cfg["bev"]["ground_z"])

        self.bev_x_min = -self.bev_back
        self.bev_x_max = self.bev_front
        self.bev_y_min = -self.bev_right
        self.bev_y_max = self.bev_left

        assert intrinsics.shape == (3, 3)
        assert extrinsics.shape == (3, 4)

        self.K = intrinsics.astype(np.float32)
        self.extrinsic = extrinsics.astype(np.float32)

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.R_cw = self.extrinsic[:, :3]
        self.t_cw = self.extrinsic[:, 3:4]

        self.R_wc = self.R_cw.T
        self.t_wc = -self.R_wc @ self.t_cw

        # remap and bev size (built lazily)
        self._map_x = None
        self._map_y = None
        self._bev_w = None
        self._bev_h = None

        self.calib_image_size = calib_image_size
        self._intrinsics_scaled_for = None

    def _create_bev_grid(self):
        x_min = -self.bev_back
        x_max = self.bev_front
        y_min = -self.bev_right
        y_max = self.bev_left

        res = self.bev_resolution
        bev_h = int((x_max - x_min) / res)
        bev_w = int((y_max - y_min) / res)

        xs = np.linspace(x_min, x_max, bev_h, endpoint=False) + res / 2.0
        ys = np.linspace(y_min, y_max, bev_w, endpoint=False) + res / 2.0

        Xw, Yw = np.meshgrid(xs, ys, indexing="ij")
        Zw = np.full_like(Xw, self.bev_ground_z, dtype=np.float32)

        return Xw.astype(np.float32), Yw.astype(np.float32), Zw, bev_w, bev_h

    def _build_bev_remap(self):
        Xw, Yw, Zw, bev_w, bev_h = self._create_bev_grid()
        world_points = np.stack([-Yw, -Xw, Zw], axis=-1).reshape(-1, 3).T

        cam_points = self.R_wc @ world_points + self.t_wc
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

        map_x[bev_x_idx, bev_y_idx] = u[valid]
        map_y[bev_x_idx, bev_y_idx] = v[valid]

        self._map_x = map_x
        self._map_y = map_y
        self._bev_w = bev_w
        self._bev_h = bev_h

    def warp(self, image_bgr: np.ndarray, border_value=(0, 0, 0)) -> np.ndarray:

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

    def pixel_to_bev_warp(self, u, v, radius=2):
        if self._map_x is None or self._map_y is None:
            self._build_bev_remap()

        diff_x = np.abs(self._map_x - u)
        diff_y = np.abs(self._map_y - v)

        score = diff_x + diff_y
        bev_x, bev_y = np.unravel_index(np.argmin(score), score.shape)

        min_score = score[bev_x, bev_y]
        if min_score > 5:
            return -1, -1

        return int(bev_x), int(bev_y)

    def bev_to_pixel(self, bev_x, bev_y):
        if not (0 <= bev_x < self._bev_h and 0 <= bev_y < self._bev_w):
            return -1, -1

        bev_x = (self._bev_h - 1) - bev_x

        Xw = - (bev_y * self.bev_resolution + self.bev_y_min)
        Yw = - (bev_x * self.bev_resolution + self.bev_x_min)
        Zw = self.bev_ground_z

        Pw = np.array([[Xw], [Yw], [Zw]], dtype=np.float32)
        Pc = self.R_wc @ Pw + self.t_wc

        Xc, Yc, Zc = Pc[0, 0], Pc[1, 0], Pc[2, 0]
        if Zc <= 0:
            return -1, -1

        u = self.fx * (Xc / Zc) + self.cx
        v = self.fy * (Yc / Zc) + self.cy

        return int(u), int(v)

    def world_to_bev_index(self, Xw: float, Yw: float) -> tuple:
        if self._bev_h is None or self._bev_w is None:
            self._build_bev_remap()

        res = self.bev_resolution
        bev_y = int(np.round(-(Xw + self.bev_y_min) / res))
        bev_x = int(np.round((self._bev_h - 1) + (Yw + self.bev_x_min) / res))

        bev_x = max(0, min(self._bev_h - 1, bev_x))
        bev_y = max(0, min(self._bev_w - 1, bev_y))

        return bev_x, bev_y

    def pixel_to_world(self, u: float, v: float, flipped: bool = True) -> tuple:
        if self._bev_h is None or self._bev_w is None:
            self._build_bev_remap()

        H = int(self._bev_h)
        W = int(self._bev_w)

        xmin = self.bev_x_min
        xmax = self.bev_x_max
        ymin = self.bev_y_min
        ymax = self.bev_y_max

        scale_x = (xmax - xmin) / float(H)
        scale_y = (ymax - ymin) / float(W)

        v_idx = float(v)
        if flipped:
            v_idx = (H - 1) - v_idx

        x_m = xmin + v_idx * scale_x
        y_m = ymin + float(u) * scale_y

        return float(x_m), float(y_m)

    def image_pixel_to_world(self, u: float, v: float) -> tuple:
        try:
            x_c_dir = (float(u) - float(self.cx)) / float(self.fx)
            y_c_dir = (float(v) - float(self.cy)) / float(self.fy)
        except Exception:
            return float(0.0), float(0.0)

        dir_cam = np.array([x_c_dir, y_c_dir, 1.0], dtype=np.float32).reshape(3, 1)
        A = self.R_wc @ dir_cam
        B = self.t_wc

        A2 = float(A[2, 0])
        B2 = float(B[2, 0])
        eps = 1e-8
        if abs(A2) < eps:
            return float(0.0), float(0.0)

        s = (float(self.bev_ground_z) - B2) / A2
        Xw = float(s * float(A[0, 0]) + float(B[0, 0]))
        Yw = float(s * float(A[1, 0]) + float(B[1, 0]))

        return Xw, Yw

    def world_to_bev_img_pixel(self, x_m: float, y_m: float, flipped: bool = True) -> tuple:
        if self._bev_h is None or self._bev_w is None:
            self._build_bev_remap()

        H = float(self._bev_h)
        W = float(self._bev_w)

        xmin = self.bev_x_min
        xmax = self.bev_x_max
        ymin = self.bev_y_min
        ymax = self.bev_y_max

        scale_x = (xmax - xmin) / H
        scale_y = (ymax - ymin) / W

        u = (y_m - ymin) / scale_y
        v_unflipped = (x_m - xmin) / scale_x

        if flipped:
            v = (self._bev_h - 1) - v_unflipped
        else:
            v = v_unflipped

        u_int = int(round(u))
        v_int = int(round(v))

        u_int = max(0, min(int(self._bev_w - 1), u_int))
        v_int = max(0, min(int(self._bev_h - 1), v_int))

        return u_int, v_int