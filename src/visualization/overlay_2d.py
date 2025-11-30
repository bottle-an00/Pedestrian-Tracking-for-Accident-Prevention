# bbox, ID, 속도 overlay
import random as pyrandom
import cv2
import numpy as np


class Visualizer:
    # 키포인트 스켈레톤 연결 (COCO format, 1-indexed)
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

            track_id, cls_name, conf = det['id'], det['class'], det['score']

            color = (255, 0, 0) if cls_name == "person" else (0, 255, 0)
            label = f"ID:{track_id} {cls_name} {conf:.2f}"

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            foot_u, foot_v = map(int, det["foot_uv"])
            cv2.circle(vis_img, (foot_u, foot_v), 5, self.id_to_color(track_id), -1)

        return vis_img

    def draw_on_img_with_keypoints(self, img, detections, seq_name=None, show_legend=True, max_width=1280):
        """
        키포인트, foot_uv 타입별 시각화, 범례를 포함한 상세 시각화
        """
        img_h, img_w = img.shape[:2]
        vis_img = img.copy()

        for det in detections:
            track_id, cls_name, score = det['id'], det['class'], det['score']
            x, y, w, h = det['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            box_color = (255, 0, 0) if cls_name == 'person' else (0, 255, 0)
            label = f"ID:{track_id} {cls_name} {score:.2f}"
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            foot_uv = det.get('foot_uv', [])
            foot_uv_type = det.get('foot_uv_type', 'detected')
            out_of_fov_info = det.get('out_of_fov_info')

            # bbox 하단 중앙 (빨간색)
            bbox_center_bottom_u = int((x1 + x2) / 2)
            bbox_center_bottom_v = int(y2)
            cv2.circle(vis_img, (bbox_center_bottom_u, bbox_center_bottom_v), 5, (0, 0, 255), -1)

            # foot_uv 위치 표시
            if len(foot_uv) == 2:
                foot_u, foot_v = int(foot_uv[0]), int(foot_uv[1])

                if foot_uv_type == "out_of_fov":
                    vis_foot_u = max(0, min(foot_u, img_w - 1))
                    vis_foot_v = max(0, min(foot_v, img_h - 1))
                    cv2.circle(vis_img, (vis_foot_u, vis_foot_v), 8, (0, 165, 255), -1)

                    if out_of_fov_info:
                        lowest_kpt = out_of_fov_info.get('lowest_visible_keypoint', {})
                        lowest_uv = lowest_kpt.get('uv', [])
                        lowest_name = lowest_kpt.get('name', '')

                        if len(lowest_uv) == 2:
                            lkpt_u, lkpt_v = int(lowest_uv[0]), int(lowest_uv[1])
                            cv2.drawMarker(vis_img, (lkpt_u, lkpt_v), (255, 255, 0), cv2.MARKER_DIAMOND, 12, 2)
                            self._draw_dashed_line(vis_img, (lkpt_u, lkpt_v), (vis_foot_u, vis_foot_v), (0, 165, 255), 1)
                            cv2.putText(vis_img, f"OOF({lowest_name})", (vis_foot_u + 5, vis_foot_v - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                else:
                    if 0 <= foot_u < img_w and 0 <= foot_v < img_h:
                        cv2.circle(vis_img, (foot_u, foot_v), 8, (0, 255, 0), -1)

            # 키포인트 시각화
            if 'keypoints' in det and det['keypoints']:
                kpts = det['keypoints']

                for i, (px, py) in enumerate(kpts):
                    if i == 15 or i == 16:
                        continue
                    if px > 0 and py > 0:
                        cv2.circle(vis_img, (int(px), int(py)), 3, (0, 255, 0), -1)

                for p1_idx, p2_idx in self.SKELETON:
                    p1 = kpts[p1_idx - 1]
                    p2 = kpts[p2_idx - 1]
                    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                        cv2.line(vis_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 1)

                # 발목 키포인트 (노란색)
                left_ankle = kpts[15]
                right_ankle = kpts[16]
                if left_ankle[0] > 0 and left_ankle[1] > 0:
                    cv2.circle(vis_img, (int(left_ankle[0]), int(left_ankle[1])), 5, (0, 255, 255), -1)
                if right_ankle[0] > 0 and right_ankle[1] > 0:
                    cv2.circle(vis_img, (int(right_ankle[0]), int(right_ankle[1])), 5, (0, 255, 255), -1)

        # 시퀀스 라벨
        if seq_name:
            cv2.putText(vis_img, f"seq: {seq_name}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # 범례
        if show_legend:
            vis_img = self._draw_legend(vis_img)

        # 이미지 리사이즈
        if img_w > max_width:
            scale = max_width / img_w
            new_h = int(img_h * scale)
            vis_img = cv2.resize(vis_img, (max_width, new_h), interpolation=cv2.INTER_AREA)

        return vis_img

    def _draw_legend(self, img, start_y=70):
        """범례 그리기"""
        legend_spacing = 40
        font_scale = 0.8
        font_thickness = 2
        circle_radius = 10

        cv2.circle(img, (25, start_y), circle_radius, (0, 0, 255), -1)
        cv2.putText(img, "bbox bottom", (45, start_y + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        cv2.circle(img, (25, start_y + legend_spacing), circle_radius, (0, 255, 0), -1)
        cv2.putText(img, "foot_uv (detected)", (45, start_y + legend_spacing + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        cv2.circle(img, (25, start_y + legend_spacing * 2), circle_radius, (0, 165, 255), -1)
        cv2.putText(img, "foot_uv (out_of_fov)", (45, start_y + legend_spacing * 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        cv2.circle(img, (25, start_y + legend_spacing * 3), circle_radius, (0, 255, 255), -1)
        cv2.putText(img, "ankle keypoint", (45, start_y + legend_spacing * 3 + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return img

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10):
        """점선 그리기"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist == 0:
            return
        dashes = int(dist / dash_length)
        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)
            start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                     int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
            end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                   int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
            cv2.line(img, start, end, color, thickness)

    def draw_on_BEV(self, bev_img, id, foot_bevs):
        vis_img = bev_img.copy()
        cnt=0

        if foot_bevs is None or len(foot_bevs) == 0:
            return vis_img

        for bx, by in foot_bevs:
            cv2.circle(vis_img, (int(by), int(bx)), 5, self.id_to_color(id), -1)
            cv2.putText(vis_img, str(cnt), (int(by), int(bx)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cnt+=1

        return vis_img

    def draw_points(self, bev_img, id, points, radius=4):
        img = bev_img.copy()

        for p in points:
            x, y = int(p[0]), int(p[1])
            cv2.circle(img, (y, x), radius, self.id_to_color(id), -1)

        return img

    def draw_polyline(self, bev_img, id, points, color=(255, 0, 0), thickness=2):
        if len(points) < 2:
            return bev_img

        img = bev_img.copy()

        pts = np.array([(int(y), int(x)) for x, y in points], dtype=np.int32)
        for pt in pts:
            cv2.circle(img, (pt[0], pt[1]), 3, self.id_to_color(id), -1)

        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

        return img

    def id_to_color(self, track_id: int):
        rnd = pyrandom.Random(track_id)
        r = rnd.randint(0, 150)
        g = rnd.randint(150, 255)
        b = rnd.randint(50, 200)
        return (b, g, r)
