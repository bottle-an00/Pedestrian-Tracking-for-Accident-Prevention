# src/tracking/tracker.py
# ByteTrack 기반 트래킹 모듈

from ultralytics import YOLO

from src.detection.yolo_detector import YoloDetector


class ByteTracker:
    """
    YOLO + ByteTrack 기반 트래킹 모듈
    YoloDetector의 foot 추정 로직을 재사용합니다.
    """

    def __init__(self, model_path, conf_thres_config, target_class_names=None, imgsz=640):
        # YoloDetector 생성 (YOLO 모델 1번만 로드)
        self._detector = YoloDetector(model_path, conf_thres_config, target_class_names, imgsz)

        # detector에서 필요한 속성 참조
        self.model = self._detector.model
        self.names = self._detector.names
        self.imgsz = self._detector.imgsz
        self.conf_thres_config = self._detector.conf_thres_config
        self.target_indices = self._detector.target_indices
        self.min_conf_thres = self._detector.min_conf_thres

    def process(self, img):
        """
        img → YOLO tracking → detection 딕셔너리 리스트 반환
        [{
            "id": track_id,
            "bbox": [x1, y1, w, h],
            "score": conf,
            "class": cls_name,
            "foot_uv": [u, v],
            "keypoints": [[x, y], ...],
            "foot_uv_type": "detected" | "out_of_fov",
            "out_of_fov_info": {...} | None
        }]
        """
        img_h, img_w = img.shape[:2]
        results = self.model.track(
            img,
            persist=True,
            verbose=False,
            classes=self.target_indices,
            conf=self.min_conf_thres,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz
        )[0]

        if results.boxes is None or results.boxes.id is None:
            return []

        has_keypoints = results.keypoints is not None and results.keypoints.data.shape[1] > 0

        detections = []
        for i, box in enumerate(results.boxes.data):
            x1, y1, x2, y2, track_id, conf, cls_id = map(float, box)
            track_id, cls_id = int(track_id), int(cls_id)
            cls_name = self.names.get(cls_id, "unknown")

            # 클래스별 threshold 적용
            if isinstance(self.conf_thres_config, dict):
                threshold = self.conf_thres_config.get(
                    cls_name, self.conf_thres_config.get("default", self.min_conf_thres)
                )
                if conf < threshold:
                    continue

            # --- 발 위치 추정 로직 (YoloDetector 메서드 재사용) ---
            foot_u, foot_v = (x1 + x2) / 2.0, y2  # 기본값
            foot_uv_type = "detected"
            out_of_fov_info = None
            keypoints_xy = []

            if has_keypoints and i < len(results.keypoints.data):
                kpts = results.keypoints.data[i]
                keypoints_xy = kpts[:, :2].cpu().numpy().tolist()

                foot_u, foot_v, foot_uv_type, out_of_fov_info = self._detector._estimate_foot_position(
                    kpts, x1, y1, x2, y2, img_w, img_h
                )

            detections.append({
                "id": track_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf,
                "class": cls_name,
                "foot_uv": [float(foot_u), float(foot_v)],
                "keypoints": keypoints_xy,
                "foot_uv_type": foot_uv_type,
                "out_of_fov_info": out_of_fov_info
            })

        return detections
