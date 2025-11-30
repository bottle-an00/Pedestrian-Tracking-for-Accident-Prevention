# YOLO 추론 래퍼

from ultralytics import YOLO


class YoloDetector:
    """
    YOLO 기반 객체 탐지 모듈
    - model_path: YOLO 모델 경로
    - conf_thres_config: dict(클래스별) 또는 float(전체)
    - target_class_names: ['person', 'car'] 등
    - imgsz: YOLO 입력 이미지 사이즈
    """

    # COCO 17 keypoints 인덱스
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    # 발목 추정에 사용할 수 있는 하체 키포인트 (아래에서 위 순서)
    LOWER_BODY_KEYPOINTS = [
        (15, 'left_ankle'), (16, 'right_ankle'),
        (13, 'left_knee'), (14, 'right_knee'),
        (11, 'left_hip'), (12, 'right_hip')
    ]

    def __init__(self, model_path, conf_thres_config, target_class_names=None, imgsz=640):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.imgsz = imgsz
        self.conf_thres_config = conf_thres_config

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

    def detect(self, img):
        """
        img → YOLO detection → detection 딕셔너리 리스트 반환 (트래킹 없음)
        [{
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
        results = self.model(
            img,
            verbose=False,
            classes=self.target_indices,
            conf=self.min_conf_thres,
            imgsz=self.imgsz
        )[0]

        if results.boxes is None:
            return []

        has_keypoints = results.keypoints is not None and results.keypoints.data.shape[1] > 0

        detections = []
        for i, box in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, cls_id = map(float, box)
            cls_id = int(cls_id)
            cls_name = self.names.get(cls_id, "unknown")

            # 클래스별 threshold 적용
            if isinstance(self.conf_thres_config, dict):
                threshold = self.conf_thres_config.get(
                    cls_name, self.conf_thres_config.get("default", self.min_conf_thres)
                )
                if conf < threshold:
                    continue

            # --- 발 위치 추정 로직 ---
            foot_u, foot_v = (x1 + x2) / 2.0, y2  # 기본값: 바운딩 박스 하단 중앙
            foot_uv_type = "detected"
            out_of_fov_info = None
            keypoints_xy = []

            if has_keypoints and i < len(results.keypoints.data):
                kpts = results.keypoints.data[i]
                keypoints_xy = kpts[:, :2].cpu().numpy().tolist()

                foot_u, foot_v, foot_uv_type, out_of_fov_info = self._estimate_foot_position(
                    kpts, x1, y1, x2, y2, img_w, img_h
                )

            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf,
                "class": cls_name,
                "foot_uv": [float(foot_u), float(foot_v)],
                "keypoints": keypoints_xy,
                "foot_uv_type": foot_uv_type,
                "out_of_fov_info": out_of_fov_info
            })

        return detections

    def _estimate_foot_position(self, kpts, x1, y1, x2, y2, img_w, img_h):
        """키포인트에서 발 위치를 추정합니다."""
        foot_u, foot_v = (x1 + x2) / 2.0, y2  # 기본값
        foot_uv_type = "detected"
        out_of_fov_info = None

        left_ankle = kpts[15]
        right_ankle = kpts[16]

        left_ankle_valid = left_ankle[0] > 0 and left_ankle[1] > 0
        right_ankle_valid = right_ankle[0] > 0 and right_ankle[1] > 0

        bbox_height = y2 - y1
        y_offset = bbox_height * 0.1

        if left_ankle_valid and right_ankle_valid:
            # 양쪽 발목 모두 검출: x는 평균, y는 더 아래값
            ankle_u = (float(left_ankle[0]) + float(right_ankle[0])) / 2.0
            ankle_v = max(float(left_ankle[1]), float(right_ankle[1]))
            foot_u, foot_v = ankle_u, ankle_v + y_offset
        elif left_ankle_valid:
            foot_u, foot_v = float(left_ankle[0]), float(left_ankle[1]) + y_offset
        elif right_ankle_valid:
            foot_u, foot_v = float(right_ankle[0]), float(right_ankle[1]) + y_offset
        else:
            # FOV 이탈: lowest visible keypoint에서 발목 위치 추정
            foot_u, foot_v, foot_uv_type, out_of_fov_info = self._estimate_foot_from_visible_keypoints(
                kpts, x1, y1, x2, y2, img_w, img_h
            )

        return foot_u, foot_v, foot_uv_type, out_of_fov_info

    def _estimate_foot_from_visible_keypoints(self, kpts, x1, y1, x2, y2, img_w, img_h):
        """FOV 이탈 시 lowest visible keypoint에서 발목 위치를 추정합니다."""
        bbox_height = y2 - y1

        lowest_visible = None
        lowest_visible_idx = None
        lowest_visible_name = None

        for idx, name in self.LOWER_BODY_KEYPOINTS:
            kpt = kpts[idx]
            if kpt[0] > 0 and kpt[1] > 0:
                if lowest_visible is None or kpt[1] > lowest_visible[1]:
                    lowest_visible = kpt
                    lowest_visible_idx = idx
                    lowest_visible_name = name

        if lowest_visible is None:
            return (x1 + x2) / 2.0, y2, "detected", None

        if lowest_visible_idx in [15, 16]:
            return float(lowest_visible[0]), float(lowest_visible[1]) + bbox_height * 0.1, "detected", None

        distance_ratios = {
            11: 0.50,  # left_hip -> ankle
            12: 0.50,  # right_hip -> ankle
            13: 0.25,  # left_knee -> ankle
            14: 0.25,  # right_knee -> ankle
        }

        estimated_y_offset = bbox_height * distance_ratios.get(lowest_visible_idx, 0.25)
        estimated_ankle_v = float(lowest_visible[1]) + estimated_y_offset
        estimated_ankle_u = float(lowest_visible[0])

        foot_u = estimated_ankle_u
        foot_v = estimated_ankle_v + bbox_height * 0.1

        is_out_of_fov = foot_v > img_h or foot_u < 0 or foot_u > img_w

        if is_out_of_fov:
            out_of_fov_info = {
                "lowest_visible_keypoint": {
                    "index": lowest_visible_idx,
                    "name": lowest_visible_name,
                    "uv": [float(lowest_visible[0]), float(lowest_visible[1])]
                },
                "estimated_ankle_uv": [estimated_ankle_u, estimated_ankle_v]
            }
            return foot_u, foot_v, "out_of_fov", out_of_fov_info
        else:
            return foot_u, foot_v, "detected", None
