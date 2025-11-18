# bbox, ID, 속도 overlay
import cv2
import numpy as np

class Visualizer:
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
            cv2.circle(vis_img, (foot_u, foot_v), 5, (0, 0, 255), -1)

        return vis_img

    def draw_on_BEV(self, bev_img, foot_bevs, color=(0, 255, 0)):
        vis_img = bev_img.copy()

        for bx, by in foot_bevs:
            cv2.circle(vis_img, (int(by), int(bx)), 5, color, -1)

        return vis_img

    def draw_points(self, bev_img, points, color=(0, 0, 255), radius=4):
        img = bev_img.copy()

        for p in points:
            x, y = int(p[0]), int(p[1])
            cv2.circle(img, (y, x), radius, color, -1)

        return img

    def draw_polyline(self, bev_img, points, color=(255, 0, 0), thickness=2):
        if len(points) < 2:
            return bev_img

        img = bev_img.copy()

        pts = np.array([(int(y), int(x)) for x, y in points], dtype=np.int32)

        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

        return img