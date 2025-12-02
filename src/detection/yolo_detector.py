# YOLO 추론 래퍼

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


class YoloDetector:
    """
    YOLO-pose 기반 객체 탐지 모듈
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

    # Keypoint confidence threshold (Ultralytics 기본값 0.5 대신 0.3 사용)
    KEYPOINT_CONF_THRESHOLD = 0.3

    def __init__(self, model_path, conf_thres_config, target_class_names=None, imgsz=640, half=True):
        """
        Args:
            model_path: YOLO 모델 경로
            conf_thres_config: dict(클래스별) 또는 float(전체)
            target_class_names: ['person', 'car'] 등
            imgsz: YOLO 입력 이미지 사이즈
            half: FP16 사용 여부 (메모리 절약, GPU만 해당)
        """
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.imgsz = imgsz
        self.conf_thres_config = conf_thres_config

        # GPU 사용 시 half precision 설정 (predict/track 호출 시 적용)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = half and self.device == 'cuda'

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

        # LetterBox for preprocessing (raw keypoint 추출용)
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)

    def detect(self, img):
        """
        img → YOLO detection → detection 딕셔너리 리스트 반환 (트래킹 없음)
        [{
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
        results = self.model(
            img,
            verbose=False,
            classes=self.target_indices,
            conf=self.min_conf_thres,
            imgsz=self.imgsz,
            half=self.half,
        )[0]

        if results.boxes is None:
            return []

        # Raw keypoints 추출 (conf < 0.5 여도 좌표 유지)
        # bbox와 매칭된 raw keypoints 반환
        raw_keypoints_matched = self._extract_raw_keypoints_matched(img, results)

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

            # Raw keypoints 사용 (bbox와 매칭됨)
            if raw_keypoints_matched is not None and i in raw_keypoints_matched:
                kpts = raw_keypoints_matched[i]  # shape: (17, 3) - x, y, conf
                keypoints_xy = kpts.tolist()

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

    def _extract_raw_keypoints_matched(self, img, results):  # noqa: ARG002 - img kept for API compatibility
        """
        results.keypoints에서 직접 keypoints 추출 (이중 forward pass 제거)

        Ultralytics는 기본적으로 conf < 0.5인 keypoint 좌표를 (0, 0)으로 바꿈.
        낮은 conf 임계값 (0.3) 사용으로 대부분의 keypoint 좌표 사용 가능하도록 변경함.

        Returns:
            dict: {box_idx: keypoints_array} 형태
        """
        if results.keypoints is None or results.boxes is None:
            return None

        try:
            # 직접 keypoints 데이터 추출 
            kpts_data = results.keypoints.data.cpu().numpy()

            matched = {}
            for i in range(len(kpts_data)):
                matched[i] = kpts_data[i]

            return matched

        except Exception as e:
            print(f"[WARN] Failed to extract keypoints: {e}")
            return None

    def _scale_boxes_no_clip(self, img1_shape, boxes, img0_shape):
        """bbox 스케일링 (clip 없음)"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2
        )

        boxes[:, 0] -= pad[0]  # x1
        boxes[:, 2] -= pad[0]  # x2
        boxes[:, 1] -= pad[1]  # y1
        boxes[:, 3] -= pad[1]  # y2
        boxes[:, :4] /= gain

        return boxes

    def _compute_iou(self, box1, box2):
        """두 bbox의 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def _scale_coords_no_clip(self, img1_shape, coords, img0_shape):
        """
        scale_coords 함수에서 clip 제거한 버전
        -> 이미지 범위 밖 좌표도 유지
        """
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2
        )

        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
        coords[..., 0] /= gain
        coords[..., 1] /= gain

        # clip 제거! (원본은 여기서 clip_coords 호출)
        return coords

    def _estimate_foot_position(self, kpts, x1, y1, x2, y2, img_w, img_h):
        """키포인트에서 발 위치를 추정합니다.

        Args:
            kpts: numpy array of shape (17, 3) - x, y, conf
        """
        foot_u, foot_v = (x1 + x2) / 2.0, y2  # 기본값
        foot_uv_type = "detected"
        out_of_fov_info = None

        left_ankle = kpts[15]
        right_ankle = kpts[16]

        # conf >= KEYPOINT_CONF_THRESHOLD 이면 유효 (기존 0.5 대신 0.3)
        left_ankle_valid = float(left_ankle[2]) >= self.KEYPOINT_CONF_THRESHOLD
        right_ankle_valid = float(right_ankle[2]) >= self.KEYPOINT_CONF_THRESHOLD

        # 각 발목이 이미지 범위 안에 있는지 체크
        la_x, la_y = float(left_ankle[0]), float(left_ankle[1])
        ra_x, ra_y = float(right_ankle[0]), float(right_ankle[1])
        left_ankle_in_fov = 0 <= la_x <= img_w and 0 <= la_y <= img_h
        right_ankle_in_fov = 0 <= ra_x <= img_w and 0 <= ra_y <= img_h

        bbox_height = y2 - y1
        y_offset = bbox_height * 0.1

        if left_ankle_valid and right_ankle_valid:
            # 양쪽 발목 모두 valid → 항상 평균 사용
            ankle_u = (la_x + ra_x) / 2.0
            ankle_v = max(la_y, ra_y)
            foot_u, foot_v = ankle_u, ankle_v + y_offset

            # FOV 여부에 따라 type만 다르게
            if left_ankle_in_fov and right_ankle_in_fov:
                foot_uv_type = "detected"
            else:
                foot_uv_type = "out_of_fov"
        elif left_ankle_valid:
            foot_u, foot_v = la_x, la_y + y_offset
            foot_uv_type = "detected" if left_ankle_in_fov else "out_of_fov"
        elif right_ankle_valid:
            foot_u, foot_v = ra_x, ra_y + y_offset
            foot_uv_type = "detected" if right_ankle_in_fov else "out_of_fov"
        else:
            # FOV 이탈: lowest visible keypoint에서 발목 위치 추정
            foot_u, foot_v, foot_uv_type, out_of_fov_info = self._estimate_foot_from_visible_keypoints(
                kpts, x1, y1, x2, y2, img_w, img_h
            )

        # 최종 foot_uv가 이미지 범위 밖이면 out_of_fov로 표시
        if foot_u < 0 or foot_u > img_w or foot_v < 0 or foot_v > img_h:
            foot_uv_type = "out_of_fov"

        return foot_u, foot_v, foot_uv_type, out_of_fov_info

    def _estimate_foot_from_visible_keypoints(self, kpts, x1, y1, x2, y2, img_w, img_h):
        """FOV 이탈 시 lowest visible keypoint에서 발목 위치 추정

        Args:
            kpts: numpy array of shape (17, 3) - x, y, conf
        """
        bbox_height = y2 - y1

        lowest_visible = None
        lowest_visible_idx = None
        lowest_visible_name = None

        for idx, name in self.LOWER_BODY_KEYPOINTS:
            kpt = kpts[idx]  # [x, y, conf]
            # conf >= threshold 이면 유효
            if float(kpt[2]) >= self.KEYPOINT_CONF_THRESHOLD:
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
