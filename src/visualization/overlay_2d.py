import random as pyrandom
import cv2
import numpy as np


class Visualizer:
    # COCO skeleton (1-indexed)
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
        [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]

    def draw_on_img(self, img, detections):
        vis_img = img.copy()

        for det in detections:
            x, y, w, h = det['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            tid, cls_name, conf = det['id'], det['class'], det['score']

            color = (255, 0, 0) if cls_name == "person" else (0, 255, 0)
            label = f"ID:{tid} {cls_name} {conf:.2f}"

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            fu, fv = map(int, det["foot_uv"])
            cv2.circle(vis_img, (fu, fv), 5, self.id_to_color(tid), -1)

        return vis_img

    def draw_on_img_with_keypoints(self, img, detections, seq_name=None,
                                   show_legend=True, max_width=1280):
        img_h, img_w = img.shape[:2]
        vis_img = img.copy()

        for det in detections:
            tid, cls_name, score = det['id'], det['class'], det['score']
            x, y, w, h = det['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            box_color = (255, 0, 0) if cls_name == 'person' else (0, 255, 0)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(vis_img, f"ID:{tid} {cls_name} {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        box_color, 2)

            # Bottom center of bbox
            bc_u = int((x1 + x2) / 2)
            bc_v = int(y2)
            cv2.circle(vis_img, (bc_u, bc_v), 5, (0, 0, 255), -1)

            foot_uv = det.get('foot_uv', [])
            foot_type = det.get('foot_uv_type', 'detected')
            oof_info = det.get('out_of_fov_info')

            # Draw detected / OOF foot position
            if len(foot_uv) == 2:
                fu, fv = int(foot_uv[0]), int(foot_uv[1])

                if foot_type == "out_of_fov":
                    # clamp OOF positions to boundary for visualization
                    u = max(0, min(fu, img_w - 1))
                    v = max(0, min(fv, img_h - 1))
                    cv2.circle(vis_img, (u, v), 8, (0, 165, 255), -1)

                    # lowest visible keypoint indicator
                    if oof_info:
                        lkpt = oof_info.get('lowest_visible_keypoint', {})
                        luv = lkpt.get('uv', [])
                        lname = lkpt.get('name', '')

                        if len(luv) == 2:
                            lu, lv = int(luv[0]), int(luv[1])
                            cv2.drawMarker(vis_img, (lu, lv), (255, 255, 0),
                                           cv2.MARKER_DIAMOND, 12, 2)
                            self._draw_dashed_line(vis_img, (lu, lv),
                                                   (u, v), (0, 165, 255), 1)
                            cv2.putText(vis_img, f"OOF({lname})",
                                        (u + 5, v - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (0, 165, 255), 1)
                else:
                    if 0 <= fu < img_w and 0 <= fv < img_h:
                        cv2.circle(vis_img, (fu, fv), 8, (0, 255, 0), -1)

            # Keypoints & skeleton
            if 'keypoints' in det and det['keypoints']:
                kpts = det['keypoints']

                # Normalize keypoints into list of (x,y) pairs. Input may be
                # a flat list [x1,y1,x2,y2,...] or a list of lists/tuples where
                # elements may contain additional values (e.g., score).
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

                for i, (px, py) in enumerate(norm_kpts):
                    if i in (15, 16):  # ankles handled separately
                        continue
                    if px > 0 and py > 0:
                        cv2.circle(vis_img, (int(px), int(py)), 3,
                                   (0, 255, 0), -1)

                for p1, p2 in self.SKELETON:
                    # skeleton indices are 1-indexed in SKELETON
                    a = norm_kpts[p1 - 1] if p1 - 1 < len(norm_kpts) else (0, 0)
                    b = norm_kpts[p2 - 1] if p2 - 1 < len(norm_kpts) else (0, 0)
                    if a[0] > 0 and a[1] > 0 and b[0] > 0 and b[1] > 0:
                        cv2.line(vis_img, (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 (255, 255, 0), 1)

                # Ankles (highlight) indices 15 and 16 (1-indexed in SKELETON)
                la = norm_kpts[15] if 15 < len(norm_kpts) else (0, 0)
                ra = norm_kpts[16] if 16 < len(norm_kpts) else (0, 0)
                if la[0] > 0 and la[1] > 0:
                    cv2.circle(vis_img, (int(la[0]), int(la[1])),
                               5, (0, 255, 255), -1)
                if ra[0] > 0 and ra[1] > 0:
                    cv2.circle(vis_img, (int(ra[0]), int(ra[1])),
                               5, (0, 255, 255), -1)

        if seq_name:
            cv2.putText(vis_img, f"seq: {seq_name}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)

        if show_legend:
            vis_img = self._draw_legend(vis_img)

        if img_w > max_width:
            scale = max_width / img_w
            new_h = int(img_h * scale)
            vis_img = cv2.resize(vis_img, (max_width, new_h),
                                 interpolation=cv2.INTER_AREA)

        return vis_img

    def _draw_legend(self, img, start_y=70):
        legend_spacing = 40
        font_scale = 0.8
        thickness = 2
        r = 10

        cv2.circle(img, (25, start_y), r, (0, 0, 255), -1)
        cv2.putText(img, "bbox bottom",
                    (45, start_y + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness)

        cv2.circle(img, (25, start_y + legend_spacing), r, (0, 255, 0), -1)
        cv2.putText(img, "foot_uv (detected)",
                    (45, start_y + legend_spacing + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)

        cv2.circle(img, (25, start_y + legend_spacing * 2), r,
                   (0, 165, 255), -1)
        cv2.putText(img, "foot_uv (out_of_fov)",
                    (45, start_y + legend_spacing * 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)

        cv2.circle(img, (25, start_y + legend_spacing * 3), r,
                   (0, 255, 255), -1)
        cv2.putText(img, "ankle keypoint",
                    (45, start_y + legend_spacing * 3 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)

        return img

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10):
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return

        dashes = int(dist / dash_length)
        for i in range(0, dashes, 2):
            s = i / dashes
            e = min((i + 1) / dashes, 1.0)
            start = (int(pt1[0] + (pt2[0] - pt1[0]) * s),
                     int(pt1[1] + (pt2[1] - pt1[1]) * s))
            end = (int(pt1[0] + (pt2[0] - pt1[0]) * e),
                   int(pt1[1] + (pt2[1] - pt1[1]) * e))
            cv2.line(img, start, end, color, thickness)

    def draw_on_BEV(self, bev_img, id, foot_bevs):
        img = bev_img.copy()
        if not foot_bevs:
            return img

        for idx, (bx, by) in enumerate(foot_bevs):
            cv2.circle(img, (int(by), int(bx)), 5, self.id_to_color(id), -1)
            cv2.putText(img, str(idx), (int(by), int(bx)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return img

    def draw_points(self, bev_img, id, points, radius=4):
        img = bev_img.copy()
        for px, py in points:
            cv2.circle(img, (int(py), int(px)), radius,
                       self.id_to_color(id), -1)
        return img

    def draw_polyline(self, bev_img, id, points, color=(255, 0, 0),
                      thickness=2):
        if len(points) < 2:
            return bev_img

        img = bev_img.copy()
        pts = np.array([(int(y), int(x)) for x, y in points],
                       dtype=np.int32)

        for p in pts:
            cv2.circle(img, (p[0], p[1]), 3, self.id_to_color(id), -1)

        cv2.polylines(img, [pts], isClosed=False,
                      color=color, thickness=thickness)
        return img

    def id_to_color(self, track_id: int):
        rnd = pyrandom.Random(track_id)
        return (rnd.randint(50, 200),
                rnd.randint(150, 255),
                rnd.randint(0, 150))

    def draw_risk_zone_bev(
        self, bev_img, front_m, width_m, bev_resolution,
        color=(0,0,255), alpha=0.25,
        caution_color=(0,255,255), caution_alpha=0.25,
        caution_border_color=(255,255,255),
        caution_border_thickness=2,
        vehicle_bev=None
    ):
        """
        Draw risk rectangle + caution sectors around vehicle BEV location.
        """
        img = bev_img.copy()
        h, w = img.shape[:2]

        # Vehicle BEV location (row, col)
        if vehicle_bev is not None:
            try:
                vx, vy = int(vehicle_bev[0]), int(vehicle_bev[1])
            except Exception:
                vx, vy = h - 1, w // 2
        else:
            vx, vy = h - 1, w // 2

        vx = max(0, min(h - 1, vx))
        vy = max(0, min(w - 1, vy))

        front_px = int(front_m / bev_resolution)
        half_w_px = int(width_m / bev_resolution)

        top_x = max(0, vx - front_px)
        bottom_x = vx
        left_y = max(0, vy - half_w_px)
        right_y = min(w - 1, vy + half_w_px)

        # Main rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, (left_y, top_x), (right_y, bottom_x),
                      color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.rectangle(img, (left_y, top_x), (right_y, bottom_x),
                      (255, 255, 255), 2)
        cv2.putText(img, f"Risk: {front_m:.2f}m x {width_m*2}m",
                    (left_y + 5, top_x + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)

        # --- Caution fans ---
        fan_r = front_px
        half_ang = np.deg2rad(15)

        def sector_pts(row_c, col_c, a0, a1, r, steps=24):
            angles = np.linspace(a0, a1, steps)
            pts = []
            for a in angles:
                x = np.cos(a) * r
                y = np.sin(a) * r
                row = int(row_c - y)
                col = int(col_c + x)
                pts.append((col, row))
            return np.array([(col_c, row_c)] + pts, dtype=np.int32)

        left_sector = sector_pts(bottom_x, left_y,
                                 np.pi/2, np.pi/2 + half_ang, fan_r)
        right_sector = sector_pts(bottom_x, right_y,
                                  np.pi/2 - half_ang, np.pi/2, fan_r)

        overlay2 = img.copy()
        if left_sector.shape[0] >= 3:
            cv2.fillPoly(overlay2, [left_sector], caution_color)
        if right_sector.shape[0] >= 3:
            cv2.fillPoly(overlay2, [right_sector], caution_color)
        cv2.addWeighted(overlay2, caution_alpha, img, 1 - caution_alpha, 0, img)

        if left_sector.shape[0] >= 3:
            cv2.polylines(img, [left_sector], True,
                          caution_border_color, caution_border_thickness)
        if right_sector.shape[0] >= 3:
            cv2.polylines(img, [right_sector], True,
                          caution_border_color, caution_border_thickness)

        # CAUTION text near outermost sector points
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        try:
            if left_sector.shape[0] >= 2:
                lx, ly = left_sector[-1]
                cv2.putText(img, "CAUTION",
                            (lx - 40, ly - 40), font, fs,
                            (255,255,255), th, cv2.LINE_AA)
            if right_sector.shape[0] >= 2:
                rx, ry = right_sector[-1]
                cv2.putText(img, "CAUTION",
                            (rx + 10, ry - 40), font, fs,
                            (255,255,255), th, cv2.LINE_AA)
        except Exception:
            pass

        return img
