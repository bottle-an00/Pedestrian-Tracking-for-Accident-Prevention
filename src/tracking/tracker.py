# src/tracking/tracker.py
# ByteTrack 기반 트래킹 모듈

import torch
from src.detection.yolo_detector import YoloDetector


class ByteTracker:
    """
    YOLO + ByteTrack 기반 트래킹 모듈
    """

    def __init__(self, model_path, conf_thres_config, target_class_names=None, imgsz=640, half=True):
        """
        Args:
            model_path: YOLO 모델 경로
            conf_thres_config: dict(클래스별) 또는 float(전체)
            target_class_names: ['person', 'car'] 등
            imgsz: YOLO 입력 이미지 사이즈
            half: FP16 사용 여부 (메모리 절약)
        """
        # YoloDetector 생성 (half precision 전달)
        self._detector = YoloDetector(model_path, conf_thres_config, target_class_names, imgsz, half=half)

        # detector에서 필요한 속성 참조
        self.model = self._detector.model
        self.names = self._detector.names
        self.imgsz = self._detector.imgsz
        self.conf_thres_config = self._detector.conf_thres_config
        self.target_indices = self._detector.target_indices
        self.min_conf_thres = self._detector.min_conf_thres
        self.half = half
        self.device = self._detector.device

    def process(self, img):
        """
        img → YOLO tracking → detection 딕셔너리 리스트 반환
        [{
            "id": track_id,
            "bbox": [x1, y1, w, h],
            "score": conf,
            "class": cls_name,
            "foot_uv": [u, v],
            "keypoints": [[x, y, conf], ...],
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
            imgsz=self.imgsz,
            half=self.half,
        )[0]

        if results.boxes is None or results.boxes.id is None:
            return []

        # Raw keypoints 추출
        raw_keypoints = self._detector._extract_raw_keypoints_matched(img, results)

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

            # --- 발 위치 추정 로직 ---
            foot_u, foot_v = (x1 + x2) / 2.0, y2  # 기본값
            foot_uv_type = "detected"
            out_of_fov_info = None
            keypoints_xy = []

            # Raw keypoints 사용 (IoU 기반 매칭)
            if raw_keypoints is not None and i in raw_keypoints:
                kpts = raw_keypoints[i]  # shape: (17, 3) - x, y, conf
                keypoints_xy = kpts.tolist()

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
