"""BEV risk-zone mask generator and PedestrianStateManager.

Implements:
- generate_risk_zones(h, w, front_m, width_m, bev_resolution, vehicle_bev)
  returning (caution_poly, risk_poly, caution_mask, risk_mask)
- PedestrianStateManager: per-track hysteresis state manager using those polygons.

This file is self-contained and uses the same geometry as Visualizer.draw_risk_zone_bev.
"""
from collections import deque
from typing import Tuple, Optional, Dict, Any, Iterable, Union, List
import numpy as np
import cv2

from src.core.config import load_yaml
from src.calibration.homography import Homography


def _sector_pts(row_c: int, col_c: int, a0: float, a1: float, r: int, steps: int = 24):
    angles = np.linspace(a0, a1, steps)
    pts = []
    for a in angles:
        x = np.cos(a) * r
        y = np.sin(a) * r
        row = int(row_c - y)
        col = int(col_c + x)
        pts.append((col, row))
    # return array of shape (n,2)
    return np.array([(col_c, row_c)] + pts, dtype=np.int32)


def generate_risk_zones(h: int, w: int, front_m: float, width_m: float, bev_resolution: float,
                        vehicle_bev: Optional[Tuple[int, int]] = None,
                        half_ang_deg: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create caution and risk polygons and masks matching Visualizer.draw_risk_zone_bev.

    Returns:
        (caution_poly_left, caution_poly_right, caution_mask, risk_mask)
    Polygons are numpy int arrays of (N,2) in (col,row) order suitable for cv2.pointPolygonTest
    Masks are uint8 arrays with 1 where polygon covers.
    """
    # vehicle BEV location (row, col)
    if vehicle_bev is not None:
        try:
            vx, vy = int(vehicle_bev[0]), int(vehicle_bev[1])
        except Exception:
            vx, vy = h - 1, w // 2
    else:
        vx, vy = h - 1, w // 2

    vx = max(0, min(h - 1, vx))
    vy = max(0, min(w - 1, vy))

    front_px = int(round(front_m / bev_resolution))
    half_w_px = int(round(width_m / bev_resolution))

    top_x = max(0, vx - front_px)
    bottom_x = vx
    left_y = max(0, vy - half_w_px)
    right_y = min(w - 1, vy + half_w_px)

    # risk rectangle polygon (col,row) order
    risk_poly = np.array([
        (left_y, top_x),
        (right_y, top_x),
        (right_y, bottom_x),
        (left_y, bottom_x),
    ], dtype=np.int32)

    # caution fans (left and right)
    fan_r = front_px
    half_ang = np.deg2rad(half_ang_deg)

    left_sector = _sector_pts(bottom_x, left_y, np.pi / 2, np.pi / 2 + half_ang, fan_r)
    right_sector = _sector_pts(bottom_x, right_y, np.pi / 2 - half_ang, np.pi / 2, fan_r)

    # build masks
    caution_mask = np.zeros((h, w), dtype=np.uint8)
    risk_mask = np.zeros((h, w), dtype=np.uint8)

    # fill risk rectangle with value 1 in risk_mask
    cv2.fillPoly(risk_mask, [risk_poly.reshape(-1, 2)], 1)

    # fill caution sectors into caution_mask
    if left_sector.shape[0] >= 3:
        cv2.fillPoly(caution_mask, [left_sector.reshape(-1, 2)], 1)
    if right_sector.shape[0] >= 3:
        cv2.fillPoly(caution_mask, [right_sector.reshape(-1, 2)], 1)

    return left_sector, right_sector, caution_mask, risk_mask





class PedestrianStateManager:
    """Manage per-track pedestrian safety state using BEV risk/caution polygons with hysteresis.

    API:
      update_state(object_id: int, predicted_world_pos: Tuple[float,float]) -> state_str
      get_state(object_id) -> state_str
      get_history(object_id) -> deque
    """

    def __init__(self):
        # load config defaults
        cfg = load_yaml("configs/system.yaml")
        rzm_cfg = cfg.get("risk_zone_manager", {})

        enter_threshold = int(rzm_cfg.get("enter_threshold", 2))
        exit_threshold = int(rzm_cfg.get("exit_threshold", 3))
        history_len = int(rzm_cfg.get("history_len", 20))

        # thresholds
        self.enter_threshold = int(enter_threshold)
        self.exit_threshold = int(exit_threshold)
        self.history_len = int(history_len)

        # per-object records
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}

        # load bev params for world->bev conversions
        cfg = load_yaml("configs/system.yaml")
        self.bev_resolution = float(cfg["bev"]["resolution"])
        self.bev_front = float(cfg["bev"]["front"])
        self.bev_back = float(cfg["bev"]["back"])
        self.bev_left = float(cfg["bev"]["left"])
        self.bev_right = float(cfg["bev"]["right"])

        # compute bev image size used for mapping
        x_min = -self.bev_back
        x_max = self.bev_front
        y_min = -self.bev_right
        y_max = self.bev_left
        res = self.bev_resolution
        self._bev_h = int((x_max - x_min) / res)
        self._bev_w = int((y_max - y_min) / res)
        self._bev_x_min = x_min
        self._bev_y_min = y_min

        # placeholders for the current zone polygons/masks
        self.caution_left_poly = None
        self.caution_right_poly = None
        self.caution_mask = None
        self.risk_mask = None
        # defaults for zone generation (can be overridden per-call)
        self.default_width_m = float(rzm_cfg.get("width_m", 1.25))
        self.default_min_front = float(rzm_cfg.get("min_front", 5.0))
        self.default_max_front = float(rzm_cfg.get("max_front", 10.0))
        self.prediction_samples = int(rzm_cfg.get("prediction_samples", 5))
        # how many frames to hold before demoting RISK -> CAUTION
        self.risk_to_caution_hold = int(rzm_cfg.get("risk_to_caution_hold", 2))
        
    def world_to_bev(self, world_pos: Tuple[float, float], flipped: bool = True) -> Tuple[int, int]:
        """Convert world (x_m, y_m) to BEV pixel (row, col).

        Note: BEV uses flipped vertical axis in this codebase; set flipped=True to apply same convention.
        """
        x_m, y_m = float(world_pos[0]), float(world_pos[1])

        H = float(self._bev_h)
        W = float(self._bev_w)
        xmin = self._bev_x_min
        xmax = self.bev_front
        ymin = self._bev_y_min
        ymax = self.bev_left

        scale_x = (xmax - xmin) / H
        scale_y = (ymax - ymin) / W

        # u: col, v_unflipped: row
        u = (y_m - ymin) / scale_y
        v_unflipped = (x_m - xmin) / scale_x

        if flipped:
            v = (self._bev_h - 1) - v_unflipped
        else:
            v = v_unflipped

        u_int = int(round(u))
        v_int = int(round(v))

        u_int = max(0, min(self._bev_w - 1, u_int))
        v_int = max(0, min(self._bev_h - 1, v_int))

        return int(v_int), int(u_int)

    def set_zones_from_params(self, front_m: float, width_m: float, vehicle_bev: Optional[Tuple[int,int]] = None):
        """Generate and store polygons/masks for given zone parameters."""
        left_sector, right_sector, caution_mask, risk_mask = generate_risk_zones(
            self._bev_h, self._bev_w, front_m, width_m, self.bev_resolution, vehicle_bev
        )
        self.caution_left_poly = left_sector
        self.caution_right_poly = right_sector
        self.caution_mask = caution_mask
        self.risk_mask = risk_mask

    def set_zones_from_gps(self, gps_data: Any, vehicle_bev: Optional[Tuple[int,int]] = None):
        """Convenience: compute front_m from gps_data.vel_x/vel_y and set zones.

        gps_data is expected to have attributes vel_x and vel_y (meters/sec).
        This mirrors behavior in test_full_pipeline.py where front_m = clip(speed, 2.0, 10.0).
        """
        width_m = float(self.default_width_m)
        min_front = float(self.default_min_front)
        max_front = float(self.default_max_front)

        try:
            vx = float(getattr(gps_data, 'vel_x', 0.0))
            vy = float(getattr(gps_data, 'vel_y', 0.0))
            speed = float((vx * vx + vy * vy) ** 0.5)
        except Exception:
            speed = float(min_front)

        front_m = float(np.clip(speed, float(min_front), float(max_front)))
        self.set_zones_from_params(front_m=front_m, width_m=width_m, vehicle_bev=vehicle_bev)

    def _ensure_object(self, object_id: int):
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = {
                "state": "SAFE",
                "enter_count": 0,
                "exit_count": 0,
                "history": deque(maxlen=self.history_len),
                "risk_hold": 0,
            }

    def update_state(self, object_id: int, predicted_world_pos: Union[Tuple[float, float], Iterable[Tuple[float, float]]]) -> str:
        """Update state for object_id given either a single predicted world position (x_m,y_m)
        or an iterable/list of predicted world positions. If a sequence is provided, the method
        samples up to 4 evenly-spaced points across the sequence and uses the worst-severity
        point (RISK > CAUTION > SAFE) to drive the hysteresis state update.

        Returns the updated state string.
        """
        self._ensure_object(object_id)
        rec = self.tracked_objects[object_id]

        # require zones to be set
        if self.caution_mask is None or self.risk_mask is None:
            raise RuntimeError("Risk/caution zones not set; call set_zones_from_params first.")

        # normalize input: accept single tuple or iterable of tuples
        sample_points: List[Tuple[float, float]] = []
        if predicted_world_pos is None:
            raise ValueError("predicted_world_pos must be a tuple or an iterable of tuples")

        # If it's a single tuple (x,y), treat as a 1-element list
        if isinstance(predicted_world_pos, tuple) and len(predicted_world_pos) == 2 and not isinstance(predicted_world_pos[0], (list, tuple)):
            sample_points = [ (float(predicted_world_pos[0]), float(predicted_world_pos[1])) ]
        else:
            # try to iterate and coerce to list of (float,float)
            try:
                sample_points = [ (float(xy[0]), float(xy[1])) for xy in predicted_world_pos ]
            except Exception:
                # fallback: try to treat as single point
                sample_points = [ (float(predicted_world_pos[0]), float(predicted_world_pos[1])) ]

        # If multiple samples are provided, pick up to 4 evenly spaced samples (including endpoints)
        # and include the very first sample as well, producing up to 5 unique samples.
        if len(sample_points) > 1:
            n = len(sample_points)
            even_indices = [ int(round(i * (n - 1) / 3.0)) for i in range(4) ]
            # include index 0 (first sample) and combine with even_indices, preserving order and dedup
            all_indices = [0] + even_indices
            seen = set()
            unique_indices = []
            for idx in all_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            samples = [ sample_points[idx] for idx in unique_indices ]
        else:
            samples = sample_points

        # evaluate samples to find worst-case severity; priority: RISK > CAUTION > SAFE
        chosen_sample = samples[-1]
        worst = "SAFE"
        # bounds
        h_mask, w_mask = self.risk_mask.shape
        for sx, sy in samples:
            bev_row, bev_col = self.world_to_bev((sx, sy), flipped=True)
            if 0 <= bev_row < h_mask and 0 <= bev_col < w_mask:
                if self.risk_mask[bev_row, bev_col] == 1:
                    worst = "RISK"
                    chosen_sample = (sx, sy)
                    break
                if self.caution_mask[bev_row, bev_col] == 1 and worst != "RISK":
                    worst = "CAUTION"
                    chosen_sample = (sx, sy)

        prev_state = rec["state"]

        if worst == "RISK":
            # immediate transition to RISK
            rec["state"] = "RISK"
            rec["enter_count"] = 0
            rec["exit_count"] = 0
            # reset hold counter when in RISK
            rec["risk_hold"] = 0

        elif worst == "CAUTION":
            if prev_state == "SAFE":
                rec["enter_count"] += 1
                rec["exit_count"] = 0
                if rec["enter_count"] >= self.enter_threshold:
                    rec["state"] = "CAUTION"
                    rec["enter_count"] = 0
            elif prev_state == "CAUTION":
                # remain in caution and reset exit counter
                rec["exit_count"] = 0
                rec["enter_count"] = 0
                rec["state"] = "CAUTION"
            elif prev_state == "RISK":
                # leaving RISK into caution: require a short hold before committing CAUTION
                cnt = rec.get("risk_hold", 0) + 1
                rec["risk_hold"] = cnt
                if cnt >= self.risk_to_caution_hold:
                    rec["state"] = "CAUTION"
                    rec["enter_count"] = 0
                    rec["exit_count"] = 0
                    rec["risk_hold"] = 0
                else:
                    # remain in RISK until hold expires
                    rec["state"] = "RISK"

        else:
            # outside both polygons
            if prev_state == "CAUTION":
                rec["exit_count"] += 1
                if rec["exit_count"] >= self.exit_threshold:
                    rec["state"] = "SAFE"
                    rec["enter_count"] = 0
                    rec["exit_count"] = 0
            elif prev_state == "RISK":
                # require sustained exit to leave RISK
                rec["exit_count"] += 1
                if rec["exit_count"] >= self.exit_threshold:
                    rec["state"] = "SAFE"
                    rec["enter_count"] = 0
                    rec["exit_count"] = 0
            else:
                rec["state"] = "SAFE"
                rec["enter_count"] = 0
                rec["exit_count"] = 0

        # record history entry using the chosen_sample we evaluated above
        try:
            # recompute bev indices from chosen_sample to be safe
            try:
                bev_r, bev_c = self.world_to_bev(chosen_sample, flipped=True)
            except Exception:
                bev_r, bev_c = int(bev_row), int(bev_col)
            rec["history"].append({
                "world": (float(chosen_sample[0]), float(chosen_sample[1])),
                "bev": (int(bev_r), int(bev_c)),
                "state": rec["state"]
            })
        except Exception:
            # ignore history append errors
            pass

        return rec["state"]

    def get_state(self, object_id: int) -> str:
        self._ensure_object(object_id)
        return self.tracked_objects[object_id]["state"]

    def get_history(self, object_id: int):
        self._ensure_object(object_id)
        return self.tracked_objects[object_id]["history"]

    def draw_zones_on_image(self, img: np.ndarray, H: "Homography", alpha: float = 0.4) -> np.ndarray:
        """Draw current caution and risk zones onto the provided image.

        Args:
            img: original image (BGR numpy array)
            H: Homography instance with method bev_to_pixel(row,col) -> (u,v)
            alpha: overlay alpha blending factor for fills

        Returns:
            image with overlays drawn (copy)
        """
        if self.caution_mask is None or self.risk_mask is None:
            raise RuntimeError("Zones not set; call set_zones_from_params or set_zones_from_gps first")

        img_overlay = img.copy()

        # draw caution sectors (may be two separate polys)
        for poly in (self.caution_left_poly, self.caution_right_poly):
            if poly is None or poly.shape[0] < 3:
                continue
            img_pts = []
            for col, row in poly.tolist():
                u_img, v_img = H.bev_to_pixel(int(row), int(col))
                if u_img == -1 or v_img == -1:
                    continue
                img_pts.append((int(u_img), int(v_img)))
            if len(img_pts) >= 3:
                pts_arr = np.array(img_pts, dtype=np.int32)
                fill = img.copy()
                cv2.fillPoly(fill, [pts_arr], (0, 255, 255))
                img_overlay = cv2.addWeighted(fill, alpha, img_overlay, 1 - alpha, 0)
                cv2.polylines(img_overlay, [pts_arr], True, (255, 255, 255), 2)

        # draw risk rectangle as single polygon: map bbox corners of risk_mask
        ys, xs = np.where(self.risk_mask == 1)
        if ys.size > 0:
            min_row, max_row = int(ys.min()), int(ys.max())
            min_col, max_col = int(xs.min()), int(xs.max())
            bev_corners = [(min_row, min_col), (max_row, min_col), (max_row, max_col), (min_row, max_col)]
            img_pts = []
            for r, c in bev_corners:
                u_img, v_img = H.bev_to_pixel(int(r), int(c))
                if u_img == -1 or v_img == -1:
                    continue
                img_pts.append((int(u_img), int(v_img)))
            if len(img_pts) >= 3:
                pts_arr = np.array(img_pts, dtype=np.int32)
                fill = img.copy()
                cv2.fillPoly(fill, [pts_arr], (0, 0, 255))
                img_overlay = cv2.addWeighted(fill, alpha, img_overlay, 1 - alpha, 0)
                cv2.polylines(img_overlay, [pts_arr], True, (255, 255, 255), 2)

        return img_overlay

    def draw_zones_on_bev(self, bev_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Render caution and risk zones directly onto a BEV image.

        Returns a copy of bev_img with translucent fills and outlines applied.
        """
        if self.caution_mask is None or self.risk_mask is None:
            raise RuntimeError("Zones not set; call set_zones_from_params or set_zones_from_gps first")

        out = bev_img.copy()
        red = (0, 0, 255)
        yellow = (0, 255, 255)

        colored = bev_img.copy()
        colored[self.risk_mask == 1] = red
        colored[self.caution_mask == 1] = yellow

        out = cv2.addWeighted(colored, alpha, bev_img, 1 - alpha, 0)

        # draw caution outlines
        for poly in (self.caution_left_poly, self.caution_right_poly):
            if poly is None or poly.shape[0] < 3:
                continue
            cv2.polylines(out, [poly.reshape(-1, 2)], True, (255, 255, 255), 2)

        # draw risk rectangle outline using mask bbox
        ys, xs = np.where(self.risk_mask == 1)
        if ys.size > 0:
            min_row, max_row = int(ys.min()), int(ys.max())
            min_col, max_col = int(xs.min()), int(xs.max())
            risk_poly = np.array([
                (min_col, min_row),
                (max_col, min_row),
                (max_col, max_row),
                (min_col, max_row),
            ], dtype=np.int32)
            cv2.polylines(out, [risk_poly.reshape(-1, 2)], True, (255, 255, 255), 2)

        return out

    def draw_detections_on_image(self, img: np.ndarray, detections: Iterable[dict], state_getter: Optional[callable] = None) -> np.ndarray:
        """Draw detections onto an image, filling bbox interiors according to tracked state.

        Args:
            img: BGR image to draw on (will be copied)
            detections: iterable of detection dicts with keys: 'id', 'bbox' (x,y,w,h)
            state_getter: optional callable taking object_id and returning state string; if None, use internal get_state

        Returns:
            image copy with colored bbox fills and state text overlays.
        """
        out = img.copy()

        # fill colors for CAUTION and RISK; SAFE will have no fill
        state_colors = {
            "SAFE": (200, 200, 200),
            "CAUTION": (0, 215, 255),
            "RISK": (0, 0, 255),
        }

        # bbox outline color: blue
        bbox_outline = (255, 0, 0)
        # skeleton color: white
        skeleton_color = (255, 255, 255)

        if state_getter is None:
            def _sg(tid):
                try:
                    return self.get_state(int(tid))
                except Exception:
                    return "SAFE"
            state_getter = _sg

        # local skeleton definition reused from Visualizer (COCO 1-indexed pairs)
        try:
            from src.visualization.overlay_2d import Visualizer
            SKELETON = Visualizer.SKELETON
        except Exception:
            SKELETON = []

        for det in detections:
            try:
                tid = int(det.get('id', -1))
                bbox = det.get('bbox', None)
                if bbox is None or len(bbox) < 4:
                    continue
                x, y, w, h = bbox
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + w)), int(round(y + h))

                # clamp to image bounds
                h_img, w_img = out.shape[:2]
                x1 = max(0, min(w_img - 1, x1))
                x2 = max(0, min(w_img - 1, x2))
                y1 = max(0, min(h_img - 1, y1))
                y2 = max(0, min(h_img - 1, y2))

                state = state_getter(tid)

                # fill interior only for CAUTION and RISK
                if state == "CAUTION" or state == "RISK":
                    color = state_colors[state]
                    fill = out.copy()
                    cv2.rectangle(fill, (x1, y1), (x2, y2), color, -1)
                    out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)

                # always draw bbox outline in blue
                cv2.rectangle(out, (x1, y1), (x2, y2), bbox_outline, 2)

                # Draw keypoints & skeleton if available
                if 'keypoints' in det and det['keypoints']:
                    kpts = det['keypoints']
                    norm_kpts = []
                    if kpts and isinstance(kpts[0], (list, tuple)):
                        for kp in kpts:
                            try:
                                norm_kpts.append((float(kp[0]), float(kp[1])))
                            except Exception:
                                norm_kpts.append((0.0, 0.0))
                    else:
                        for i in range(0, len(kpts), 2):
                            try:
                                norm_kpts.append((float(kpts[i]), float(kpts[i+1])))
                            except Exception:
                                norm_kpts.append((0.0, 0.0))

                    # draw keypoints
                    for i, (px, py) in enumerate(norm_kpts):
                        if px > 0 and py > 0:
                            cv2.circle(out, (int(px), int(py)), 3, skeleton_color, -1)

                    # draw skeleton lines (use white)
                    for p1, p2 in SKELETON:
                        a = norm_kpts[p1 - 1] if p1 - 1 < len(norm_kpts) else (0, 0)
                        b = norm_kpts[p2 - 1] if p2 - 1 < len(norm_kpts) else (0, 0)
                        if a[0] > 0 and a[1] > 0 and b[0] > 0 and b[1] > 0:
                            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), skeleton_color, 1)

                # draw id+state label with color per state
                label = f"ID:{tid} {state}"
                lbl_col = state_colors.get(state, (255, 255, 255))
                cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lbl_col, 2)
            except Exception:
                continue

        return out
