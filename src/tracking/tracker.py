# src/tracking/yolo_tracker.py

import numpy as np
from ultralytics import YOLO


class BaseTracker:
    """추적기의 공통 인터페이스"""
    def __init__(self, model_path, conf_thres_config, target_class_names=None):
        self.model_path = model_path
        self.conf_thres_config = conf_thres_config
        self.target_class_names = target_class_names

    def process(self, img):
        raise NotImplementedError
        

class UltralyticsTracker(BaseTracker):
    """
    YOLO + (BoT-SORT or ByteTrack) 기반 트래킹 모듈
    - model_path: YOLO 모델
    - conf_thres_config:
        dict → 클래스별 threshold
        float → 전체 공통 threshold
    - target_class_names: ['person', 'car'] 이런 식
    - tracker_type: 'botsort' 또는 'bytetrack'
    """
    def __init__(self, model_path, conf_thres_config, target_class_names=None, tracker_type="bytetrack"):
        super().__init__(model_path, conf_thres_config, target_class_names)

        # YOLO 모델 로드
        self.model = YOLO(model_path)
        self.names = self.model.names

        # 클래스 인덱스 필터링
        if target_class_names:
            self.target_indices = [
                k for k, v in self.names.items() if v in target_class_names
            ]
        else:
            self.target_indices = None

        # conf threshold
        if isinstance(conf_thres_config, dict):
            self.min_conf_thres = min(conf_thres_config.values())
        else:
            self.min_conf_thres = conf_thres_config

        # tracker YAML (Ultralytics가 제공하는 기본 config)
        self.tracker_config = f"{tracker_type}.yaml"

    def process(self, img):
        """
        img → YOLO tracking → detection 딕셔너리 리스트 반환
        [{
            "id": track_id,
            "bbox": [x1, y1, w, h],
            "score": conf,
            "class": cls_name,
            "foot_uv": [u, v]
        }]
        """
        results = self.model.track(
            img,
            persist=True,
            verbose=False,
            classes=self.target_indices,
            conf=self.min_conf_thres,
            tracker=self.tracker_config
        )[0]

        if results.boxes is None or results.boxes.id is None:
            return []

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2 = map(float, box[:4])
            track_id, conf, cls_id = int(box[4]), float(box[5]), int(box[6])
            cls_name = self.names.get(cls_id, "unknown")

            # 클래스별 threshold 적용
            if isinstance(self.conf_thres_config, dict):
                threshold = self.conf_thres_config.get(
                    cls_name, self.conf_thres_config.get("default", self.min_conf_thres)
                )
                if conf < threshold:
                    continue

            # foot point = bounding box 아래 중앙
            foot_u, foot_v = (x1 + x2) / 2.0, y2

            detections.append({
                "id": track_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf,
                "class": cls_name,
                "foot_uv": [foot_u, foot_v],
            })

        return detections
