# bbox, ID, 속도 overlay

import cv2

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

    def draw_on_BEV(self, bev_img, foot_bevs):
        vis_img = bev_img.copy()

        for bx, by in foot_bevs:
            cv2.circle(vis_img, (by, bx), 5, (0, 0, 255), -1)

        return vis_img