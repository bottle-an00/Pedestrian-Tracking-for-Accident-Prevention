"""
보행자 추적 시각화 백엔드 서버
- 기존 파이프라인 재사용
- BEV 변환 및 위험구역 시각화
- WebSocket 스트리밍
"""
import sys
import os
from pathlib import Path
import asyncio
import base64
from typing import Optional, List, Dict

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.app.main import DatasetLoader
from src.tracking.tracker import ByteTracker
from src.bev.bev_transformer import BevTransformer
from src.visualization.overlay_2d import Visualizer
from src.core.config import load_yaml
from src.evaluation.gt_loader import GTLoader
from src.trajectory.trajectory_manager import TrajectoryBuffer
from src.trajectory.ekf_manager import EKFManager
from src.calibration.homography import Homography
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.trajectory.bev_zone_risk_manager import PedestrianStateManager as RiskZoneManager
from src.trajectory.pedestrian_state_manager import PedestrianStateManager as TrajectoryStateManager

app = FastAPI(title="Pedestrian Tracking API")

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RealTimeProcessor:
    """기존 파이프라인을 활용한 실시간 처리"""

    def __init__(self):
        self.dataset: Optional[DatasetLoader] = None
        self.frame_idx = 0
        self.is_playing = False
        self.fps = 10
        self.seq_path: Optional[Path] = None

        # 설정 로드
        try:
            self.cfg = load_yaml('configs/system.yaml')
        except:
            self.cfg = {'bev': {'resolution': 0.05}}

        # YOLO 설정 로드 (기존 config 사용)
        try:
            self.detector_cfg = load_yaml('configs/detector/yolo_detector.yaml')
        except:
            self.detector_cfg = {}

        model_path = self.detector_cfg.get('yolo_model_path', './models/yolo/yolo11m-pose.pt')
        conf_thres = self.detector_cfg.get('conf_threshold', {'person': 0.1})
        target_classes = self.detector_cfg.get('target_classes', ['person'])
        imgsz = self.detector_cfg.get('imgsz', 1280)
        keypoint_conf_threshold = self.detector_cfg.get('keypoint_conf_threshold', None)

        # ByteTracker 사용 (기존 파이프라인과 동일)
        print(f"YOLO 모델 로딩: {model_path}")
        self.tracker = ByteTracker(
            model_path=str(model_path),
            conf_thres_config=conf_thres,
            target_class_names=target_classes,
            imgsz=imgsz,
            half=True,
            keypoint_conf_threshold=keypoint_conf_threshold
        )
        print("YOLO 모델 로드 완료")

        # BEV 변환기 (시퀀스별로 초기화)
        self.bev_transformer: Optional[BevTransformer] = None

        self.visualizer = Visualizer()
        self.last_detections: List[Dict] = []

        # GT 로더
        self.gt_loader = GTLoader(target_classes=["pedestrian"])

        # Trajectory 버퍼 (Pred/GT) - BEV용
        self.pred_trajectory = TrajectoryBuffer(max_length=50)
        self.gt_trajectory: Dict[int, List] = {}  # {track_id: [(bev_x, bev_y), ...]}

        # Trajectory 버퍼 (Pred/GT) - 카메라 뷰용 (foot_uv 좌표)
        self.pred_trajectory_uv: Dict[int, List] = {}  # {track_id: [(u, v), ...]}
        self.gt_trajectory_uv: Dict[int, List] = {}  # {track_id: [(u, v), ...]}

        # EKF Manager (Pred trajectory 필터링)
        self.ekf_manager = EKFManager()
        self.ekf_trajectory: Dict[int, List] = {}  # EKF 필터링된 trajectory

        # Risk Zone Manager (SAFE/CAUTION/RISK 상태 관리 with hysteresis)
        self.risk_zone_manager: Optional[RiskZoneManager] = None

        # Trajectory State Manager (EKF 기반 trajectory + future prediction)
        self.trajectory_state_manager: Optional[TrajectoryStateManager] = None

        # Homography 객체 저장 (risk zone 시각화에 필요)
        self.homography: Optional[Homography] = None

        # GT 라벨 디렉토리 경로
        self.image_label_dir: Optional[Path] = None

    def load_sequence(self, seq_dir: str) -> dict:
        """시퀀스 디렉토리 로드 (DatasetLoader 사용)"""
        self.seq_path = Path(seq_dir)
        self.is_playing = False

        try:
            self.dataset = DatasetLoader(seq_dir)
            # GPS 파일 로드 결과 확인
            print(f"[DEBUG] GPS files loaded: {len(self.dataset.gps_files)}")
        except Exception as e:
            return {"success": False, "error": str(e)}

        if self.dataset.num_frames == 0:
            return {"success": False, "error": "이미지 없음"}

        # BEV용 캘리브레이션 로드
        calib_cam = self.seq_path / "calib_Camera0.txt"
        calib_ext = self.seq_path / "calib_CameraToLidar0.txt"

        if calib_cam.exists() and calib_ext.exists():
            try:
                # Homography 객체 생성 후 BevTransformer에 주입 (develop 방식)
                calib_loader = CalibrationInfoLoader()
                intrinsics = calib_loader.load_camera_calibration(calib_cam)
                extrinsics = calib_loader.load_camera_extrinsics(calib_ext)

                # 첫 이미지 사이즈 확인 (calibration 스케일링용)
                first_img = cv2.imread(str(self.dataset.image_files[0]))
                if first_img is not None:
                    img_h, img_w = first_img.shape[:2]
                    calib_image_size = (img_w, img_h)
                    print(f"이미지 사이즈: {calib_image_size}")
                else:
                    calib_image_size = None

                self.homography = Homography(
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    calib_image_size=calib_image_size
                )
                self.bev_transformer = BevTransformer(homography=self.homography)
                print(f"BEV 변환기 로드: {self.seq_path.name}")

                # Risk Zone Manager 초기화 (SAFE/CAUTION/RISK 상태 관리)
                self.risk_zone_manager = RiskZoneManager()
                # 기본 risk zone 설정 (GPS 데이터 없을 때 사용)
                self.risk_zone_manager.set_zones_from_params(front_m=10.0, width_m=1.25)
                print("Risk Zone Manager 초기화 완료")

                # Trajectory State Manager 초기화 (future prediction)
                self.trajectory_state_manager = TrajectoryStateManager(
                    obs_len=10, max_missing=5, dt=0.1
                )
                print("Trajectory State Manager 초기화 완료")

            except Exception as e:
                print(f"BEV 변환기 에러: {e}")
                import traceback
                traceback.print_exc()
                self.bev_transformer = None
                self.homography = None
                self.risk_zone_manager = None
                self.trajectory_state_manager = None
        else:
            self.bev_transformer = None
            self.homography = None
            self.risk_zone_manager = None
            self.trajectory_state_manager = None

        self.frame_idx = 0
        self.last_detections = []

        # Trajectory 초기화
        self.pred_trajectory.clear()
        self.gt_trajectory = {}
        self.pred_trajectory_uv = {}
        self.gt_trajectory_uv = {}
        self.ekf_manager = EKFManager()
        self.ekf_trajectory = {}

        # Trajectory State Manager 리셋 (시퀀스 변경 시)
        if self.trajectory_state_manager:
            self.trajectory_state_manager = TrajectoryStateManager(
                obs_len=10, max_missing=5, dt=0.1
            )

        # GT 라벨 디렉토리 탐색
        # 원천데이터 경로에서 라벨링데이터 경로로 변환
        # .../원천데이터/13_전방 보행자_val/053/ -> .../라벨링데이터/13_전방 보행자_val/053/image0/
        self.image_label_dir = None

        # 방법 1: 시퀀스 내부 label/Camera0 (기존 방식)
        label_dir = self.seq_path / "label" / "Camera0"
        if label_dir.exists():
            self.image_label_dir = label_dir
            print(f"GT 라벨 디렉토리: {label_dir}")
        else:
            # 방법 2: 원천데이터 -> 라벨링데이터 경로 변환
            seq_path_str = str(self.seq_path)
            if "원천데이터" in seq_path_str:
                label_path_str = seq_path_str.replace("원천데이터", "라벨링데이터")
                label_dir = Path(label_path_str) / "image0"
                if label_dir.exists():
                    self.image_label_dir = label_dir
                    print(f"GT 라벨 디렉토리: {label_dir}")

        return {
            "success": True,
            "total_frames": self.dataset.num_frames,
            "seq_name": self.seq_path.name,
            "has_bev": self.bev_transformer is not None,
            "has_gt": self.image_label_dir is not None,
            "model_info": self.get_model_info()
        }

    def process_frame(self, idx: int = None) -> dict:
        """프레임 처리 (ByteTracker 사용)"""
        if self.dataset is None:
            return {"camera": None, "bev": None}

        if idx is not None:
            self.frame_idx = max(0, min(idx, self.dataset.num_frames - 1))

        if self.frame_idx >= self.dataset.num_frames:
            return {"camera": None, "bev": None}

        # DatasetLoader에서 이미지 경로 가져오기
        img_path = self.dataset.image_files[self.frame_idx]
        img_full = cv2.imread(img_path)
        if img_full is None:
            return {"camera": None, "bev": None}

        # ByteTracker로 검출 + 트래킹 (기존 파이프라인과 동일)
        detections = self.tracker.process(img_full)
        self.last_detections = detections

        # keypoints 형식 변환 (x,y,conf -> x,y) for visualization
        for det in detections:
            if 'keypoints' in det and det['keypoints']:
                det['keypoints'] = [(kp[0], kp[1]) for kp in det['keypoints']]

        # Pred foot_uv trajectory 누적 (카메라 뷰용)
        img_h, img_w = img_full.shape[:2]
        for det in detections:
            if det['class'] == 'person':
                tid = det['id']
                foot_uv = det.get('foot_uv', [])
                bbox = det.get('bbox', [])

                if len(foot_uv) >= 2:
                    fu, fv = foot_uv[0], foot_uv[1]

                    # foot_uv가 bbox에서 너무 멀리 떨어져 있으면 로그 출력
                    if len(bbox) >= 4:
                        bx, by, bw, bh = bbox
                        bbox_cx = bx + bw / 2
                        bbox_bottom = by + bh
                        dist_from_bbox = ((fu - bbox_cx)**2 + (fv - bbox_bottom)**2)**0.5

                        # bbox 높이의 2배 이상 떨어져 있으면 경고
                        if dist_from_bbox > bh * 2:
                            print(f"[WARN] foot_uv 이상! ID:{tid} frame:{self.frame_idx}")
                            print(f"  foot_uv=({fu:.1f}, {fv:.1f}), bbox_bottom=({bbox_cx:.1f}, {bbox_bottom:.1f})")
                            print(f"  distance={dist_from_bbox:.1f}, bbox_h={bh:.1f}")
                            print(f"  foot_uv_type={det.get('foot_uv_type')}")

                    if tid not in self.pred_trajectory_uv:
                        self.pred_trajectory_uv[tid] = []
                    self.pred_trajectory_uv[tid].append((fu, fv))
                    # 최대 50개 유지
                    if len(self.pred_trajectory_uv[tid]) > 50:
                        self.pred_trajectory_uv[tid] = self.pred_trajectory_uv[tid][-50:]

        # GT 로드 (있는 경우)
        gt_objects = []
        if self.image_label_dir:
            try:
                frame_name = Path(img_path).stem
                gt_json = self.image_label_dir / f"{frame_name}.json"
                if gt_json.exists():
                    gt_frame = self.gt_loader.load_frame(gt_json)
                    gt_objects = gt_frame.get_pedestrians()

                    # 디버그: 첫 프레임에서 GT 정보 출력
                    if self.frame_idx == 0 and gt_objects:
                        print(f"[DEBUG] GT 로드됨: {len(gt_objects)}개")
                        for gt in gt_objects[:2]:  # 처음 2개만
                            print(f"  - track_id={gt.track_id}, foot_uv={gt.foot_uv}, bbox={gt.bbox}")

                    # GT foot_uv trajectory 누적 (카메라 뷰용)
                    for gt in gt_objects:
                        foot_uv = getattr(gt, 'foot_uv', None)
                        tid = getattr(gt, 'track_id', None) or getattr(gt, 'instance_id', None)
                        if foot_uv is not None and len(foot_uv) >= 2 and tid is not None:
                            if tid not in self.gt_trajectory_uv:
                                self.gt_trajectory_uv[tid] = []
                            self.gt_trajectory_uv[tid].append((foot_uv[0], foot_uv[1]))
                            # 최대 50개 유지
                            if len(self.gt_trajectory_uv[tid]) > 50:
                                self.gt_trajectory_uv[tid] = self.gt_trajectory_uv[tid][-50:]
                else:
                    if self.frame_idx == 0:
                        print(f"[DEBUG] GT 파일 없음: {gt_json}")
            except Exception as e:
                print(f"GT 로드 에러: {e}")
                import traceback
                traceback.print_exc()
        else:
            if self.frame_idx == 0:
                print(f"[DEBUG] image_label_dir 없음")

        # BEV 해상도
        bev_resolution = float(self.cfg['bev']['resolution'])

        # GPS 데이터 로드 및 동적 risk zone 업데이트
        gps_data = self.dataset.get_gps_for_frame(self.frame_idx)
        if self.frame_idx == 0:
            print(f"[DEBUG] GPS data for frame 0: {gps_data is not None}")
            if gps_data:
                print(f"[DEBUG] GPS vel_x={gps_data.vel_x:.2f}, vel_y={gps_data.vel_y:.2f}")
        if gps_data and self.risk_zone_manager:
            try:
                # GPS 속도 기반으로 risk zone 크기 동적 조절
                self.risk_zone_manager.set_zones_from_gps(gps_data)
            except Exception as e:
                print(f"GPS 기반 risk zone 업데이트 에러: {e}")

        # 카메라 뷰 시각화 (리사이즈 없이 원본에 수행)
        camera_vis = img_full.copy()

        # 카메라 뷰 시각화는 BEV 처리 후에 수행 (상태 정보 필요)

        # BEV 변환 및 시각화
        bev_vis = None
        pedestrians_in_risk = 0
        closest_distance = None
        bev_coords = []

        if self.bev_transformer is not None and self.homography is not None:
            try:
                bev = self.bev_transformer.warp_image(img_full)
                bev_h, bev_w = bev.shape[:2]
                img_h, img_w = img_full.shape[:2]

                # 차량 위치 계산 (이미지 하단 중앙 -> BEV 좌표)
                # 하단이 BEV 범위 밖이면 위로 올라가면서 유효한 좌표 탐색
                vehicle_bev = (-1, -1)
                for offset in range(0, 400, 50):
                    test_v = float(img_h - 1 - offset)
                    bev_result = self.homography.pixel_to_bev_warp(
                        float(img_w / 2.0), test_v
                    )
                    if bev_result[0] >= 0 and bev_result[1] >= 0:
                        vehicle_bev = bev_result
                        if self.frame_idx == 0 and offset > 0:
                            print(f"[DEBUG] vehicle_bev fallback: offset={offset}px")
                        break

                if self.frame_idx == 0:
                    print(f"[DEBUG] vehicle_bev: {vehicle_bev}, bev_shape: {bev.shape}, img_shape: {img_full.shape}")

                # GPS 기반 동적 risk zone 설정 (test_full_pipeline 방식)
                if self.risk_zone_manager:
                    if gps_data:
                        self.risk_zone_manager.set_zones_from_gps(gps_data, vehicle_bev=vehicle_bev)
                    else:
                        # GPS 없으면 기본값
                        self.risk_zone_manager.set_zones_from_params(
                            front_m=10.0, width_m=1.25, vehicle_bev=vehicle_bev
                        )

                # 위험구역 시각화 (RiskZoneManager 사용)
                if self.risk_zone_manager:
                    bev_vis = self.risk_zone_manager.draw_zones_on_bev(bev.copy(), alpha=0.25)
                else:
                    bev_vis = bev.copy()

                # 보행자 foot 좌표를 BEV로 변환
                person_detections = [
                    {'id': det['id'], 'foot_uv': det['foot_uv']}
                    for det in detections if det['class'] == 'person'
                ]
                bev_coords = self.bev_transformer.foot_uv_to_foot_bev(person_detections)

                # BEV pixel -> world 좌표 변환
                timestamp = float(self.frame_idx) * 0.1
                pedestrian_states = {}
                world_detections = []
                ekf_future = {}

                for bev_det in bev_coords:
                    tid = bev_det.id
                    bx, by = bev_det.foot_bev  # (row, col) in BEV image

                    # BEV pixel -> world 좌표 변환 (test_full_pipeline 방식)
                    x_m, y_m = self.homography.pixel_to_world(float(by), float(bx), flipped=True)
                    world_detections.append((tid, (x_m, y_m)))

                    # EKF 업데이트 (world 좌표)
                    filtered_pos = self.ekf_manager.update(tid, (x_m, y_m), timestamp)
                    ekf_future[tid] = self.ekf_manager.predict_future(tid, steps=20)

                    # EKF trajectory 저장 (BEV pixel로 변환해서 저장)
                    if filtered_pos is not None and len(filtered_pos) >= 2:
                        u_bev, v_bev = self.homography.world_to_bev_img_pixel(
                            filtered_pos[0], filtered_pos[1], flipped=True
                        )
                        if tid not in self.ekf_trajectory:
                            self.ekf_trajectory[tid] = []
                        self.ekf_trajectory[tid].append((v_bev, u_bev))

                # EKF future로 상태 업데이트 (test_full_pipeline 방식)
                if self.risk_zone_manager:
                    for tid, future in ekf_future.items():
                        if future and len(future) > 0:
                            state = self.risk_zone_manager.update_state(tid, future)
                            pedestrian_states[tid] = state

                # Trajectory State Manager 업데이트
                if self.trajectory_state_manager and world_detections:
                    self.trajectory_state_manager.update(world_detections, self.frame_idx)
                    self.trajectory_state_manager.generate_future_predictions(T=20)

                if bev_coords:
                    self.pred_trajectory.add(bev_coords, float(self.frame_idx))

                # GT → BEV 변환 및 trajectory 업데이트
                if gt_objects:
                    gt_detections = [
                        {'id': gt.track_id or gt.instance_id, 'foot_uv': gt.foot_uv}
                        for gt in gt_objects if gt.foot_uv
                    ]
                    gt_bev_coords = self.bev_transformer.foot_uv_to_foot_bev(gt_detections)
                    for gt_bev in gt_bev_coords:
                        tid = gt_bev.id
                        bx, by = gt_bev.foot_bev
                        if 0 <= bx < bev_h and 0 <= by < bev_w:
                            if tid not in self.gt_trajectory:
                                self.gt_trajectory[tid] = []
                            self.gt_trajectory[tid].append((bx, by))

                # BEV에 Pred foot_uv 표시 (ID별 색상)
                for bev_det in bev_coords:
                    bev_vis = self.visualizer.draw_on_BEV(
                        bev_vis, bev_det.id, [bev_det.foot_bev]
                    )

                # BEV에 GT foot_uv 표시 (초록색)
                if gt_objects:
                    for gt_bev in gt_bev_coords:
                        bx, by = gt_bev.foot_bev
                        if 0 <= bx < bev_h and 0 <= by < bev_w:
                            cv2.circle(bev_vis, (int(by), int(bx)), 6, (0, 255, 0), -1)  # 초록색

                # 거리 계산
                if bev_coords:
                    vehicle_row, vehicle_col = bev_h - 1, bev_w // 2
                    distances = []
                    for bev_det in bev_coords:
                        bx, by = bev_det.foot_bev
                        dist_px = ((bx - vehicle_row) ** 2 + (by - vehicle_col) ** 2) ** 0.5
                        dist_m = dist_px * bev_resolution
                        distances.append(dist_m)
                    closest_distance = min(distances) if distances else None
                else:
                    closest_distance = None

                # 상태별 위험 카운트
                pedestrians_in_risk = sum(1 for s in pedestrian_states.values() if s == "RISK")
                pedestrians_in_caution = sum(1 for s in pedestrian_states.values() if s == "CAUTION")

                # EKF future trajectory 시각화 (파란색 polyline)
                for tid, future in ekf_future.items():
                    if future is None or len(future) == 0:
                        continue
                    bev_future_pts = []
                    for p in future:
                        u_bev, v_bev = self.homography.world_to_bev_img_pixel(p[0], p[1], flipped=True)
                        bev_future_pts.append((v_bev, u_bev))
                    if len(bev_future_pts) >= 2:
                        bev_vis = self.visualizer.draw_polyline(
                            bev_vis, tid, bev_future_pts, color=(255, 0, 0), thickness=2
                        )

                # EKF Pred trajectory 시각화 (ID별 색상)
                if self.ekf_trajectory:
                    bev_vis = self.visualizer.draw_trajectory_on_bev(
                        bev_vis, self.ekf_trajectory
                    )

                # GT trajectory 시각화 (초록색)
                if self.gt_trajectory:
                    bev_vis = self.visualizer.draw_gt_trajectory_on_bev(
                        bev_vis, self.gt_trajectory, color=(0, 255, 0)
                    )

                # 카메라 뷰에 risk zone 시각화 (test_full_pipeline 방식)
                if self.risk_zone_manager and self.homography:
                    try:
                        if self.frame_idx == 0:
                            # risk_mask 범위 확인
                            ys, xs = np.where(self.risk_zone_manager.risk_mask == 1)
                            if ys.size > 0:
                                print(f"[DEBUG] risk_mask range: row={ys.min()}-{ys.max()}, col={xs.min()}-{xs.max()}")
                        camera_vis = self.risk_zone_manager.draw_zones_on_image(
                            camera_vis, self.homography, alpha=0.25
                        )
                    except Exception as e:
                        print(f"카메라 risk zone 시각화 에러: {e}")
                        import traceback
                        traceback.print_exc()

                # bbox/keypoint 시각화 (상태별 색상 적용)
                if self.risk_zone_manager and pedestrian_states:
                    # RiskZoneManager의 draw_detections_on_image 사용 (상태별 bbox 색상)
                    camera_vis = self.risk_zone_manager.draw_detections_on_image(
                        camera_vis, detections,
                        state_getter=lambda tid: pedestrian_states.get(tid, "SAFE")
                    )
                else:
                    # fallback: 기존 시각화
                    camera_vis = self.visualizer.draw_on_img_with_keypoints(
                        camera_vis, detections,
                        seq_name=self.seq_path.name if self.seq_path else None,
                        show_legend=False,
                        max_width=9999
                    )

                # foot_uv 시각화 (pred: 파란색, out_of_fov: 주황색)
                for det in detections:
                    if det['class'] == 'person':
                        foot_uv = det.get('foot_uv', [])
                        foot_type = det.get('foot_uv_type', 'detected')
                        if len(foot_uv) == 2:
                            fu, fv = int(foot_uv[0]), int(foot_uv[1])
                            img_h, img_w = camera_vis.shape[:2]
                            if foot_type == "out_of_fov":
                                u = max(0, min(fu, img_w - 1))
                                v = max(0, min(fv, img_h - 1))
                                cv2.circle(camera_vis, (u, v), 8, (0, 165, 255), -1)  # 주황색
                            else:
                                if 0 <= fu < img_w and 0 <= fv < img_h:
                                    cv2.circle(camera_vis, (fu, fv), 8, (255, 0, 0), -1)  # 파란색 (BGR)

                # GT foot_uv 시각화 (현재 프레임, 초록색)
                if gt_objects:
                    camera_vis = self.visualizer.draw_gt_foot_uv(camera_vis, gt_objects, color=(0, 255, 0))

                # 카메라 뷰에 Pred trajectory 시각화 (누적, 파란색)
                if self.pred_trajectory_uv:
                    camera_vis = self.visualizer.draw_trajectory_on_image(
                        camera_vis, self.pred_trajectory_uv, color=(255, 0, 0)
                    )

                # 카메라 뷰에 GT trajectory 시각화 (누적, 초록색)
                if self.gt_trajectory_uv:
                    camera_vis = self.visualizer.draw_gt_trajectory_on_image(
                        camera_vis, self.gt_trajectory_uv, color=(0, 255, 0)
                    )

            except Exception as e:
                print(f"BEV 에러: {e}")
                import traceback
                traceback.print_exc()

        # BEV 처리 안됐을 때 fallback 시각화
        if bev_vis is None:
            camera_vis = self.visualizer.draw_on_img_with_keypoints(
                camera_vis, detections,
                seq_name=self.seq_path.name if self.seq_path else None,
                show_legend=False
            )
            if gt_objects:
                camera_vis = self.visualizer.draw_gt_foot_uv(camera_vis, gt_objects)

            # 카메라 뷰에 Pred trajectory 시각화 (누적)
            if self.pred_trajectory_uv:
                camera_vis = self.visualizer.draw_trajectory_on_image(
                    camera_vis, self.pred_trajectory_uv
                )

            # 카메라 뷰에 GT trajectory 시각화 (누적, 마젠타색)
            if self.gt_trajectory_uv:
                camera_vis = self.visualizer.draw_gt_trajectory_on_image(
                    camera_vis, self.gt_trajectory_uv
                )

        # base64 인코딩 (웹 전송용)
        def encode_image(img_to_encode, quality=85):
            if img_to_encode is None:
                return None
            _, buffer = cv2.imencode('.jpg', img_to_encode, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return base64.b64encode(buffer).decode('utf-8')

        # pedestrians_in_caution이 정의 안됐을 수 있으므로 기본값 설정
        try:
            caution_count = pedestrians_in_caution
        except NameError:
            caution_count = 0

        # GPS 속도 정보 (km/h로 변환)
        # GPS 데이터 없으면 0으로 표시 (멈춰있거나 GPS 없음)
        vehicle_speed = 0.0
        if gps_data:
            # vel_x, vel_y로 2D 속도 계산 (m/s -> km/h)
            speed_ms = (gps_data.vel_x**2 + gps_data.vel_y**2)**0.5
            vehicle_speed = round(speed_ms * 3.6, 1)  # m/s -> km/h

        return {
            "camera": encode_image(camera_vis),
            "bev": encode_image(bev_vis),
            "frame_idx": self.frame_idx,
            "total_frames": self.dataset.num_frames,
            "frame_name": Path(img_path).name,
            "detections": {
                "count": len([d for d in detections if d['class'] == 'person']),
                "in_risk_zone": pedestrians_in_risk,
                "in_caution_zone": caution_count,
                "closest_distance": round(closest_distance, 2) if closest_distance else None
            },
            "vehicle_speed": vehicle_speed
        }

    def next_frame(self) -> dict:
        """다음 프레임으로 이동"""
        if self.dataset and self.frame_idx < self.dataset.num_frames - 1:
            self.frame_idx += 1
        return self.process_frame()

    def stop(self):
        """재생 중지"""
        self.is_playing = False

    def get_model_info(self) -> dict:
        """현재 로드된 모델 정보 반환"""
        model_path = self.detector_cfg.get('yolo_model_path', 'N/A')
        conf_thres = self.detector_cfg.get('conf_threshold', {})
        keypoint_thres = self.detector_cfg.get('keypoint_conf_threshold', 0.3)
        imgsz = self.detector_cfg.get('imgsz', 1280)

        return {
            "yolo_model": Path(model_path).name if model_path else "N/A",
            "conf_threshold": conf_thres,
            "keypoint_threshold": keypoint_thres,
            "imgsz": imgsz
        }


# 전역 프로세서
processor = RealTimeProcessor()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Pedestrian Tracking API"}


@app.get("/api/sequences")
async def list_sequences():
    """시퀀스 목록 반환"""
    # 환경변수 > config > 기본경로 순으로 체크
    base_path_str = os.environ.get("DATA_PATH")

    if not base_path_str:
        try:
            cfg = load_yaml('configs/system.yaml')
            base_path_str = cfg.get('data', {}).get('sequence_dir', '')
        except:
            pass

    if not base_path_str:
        # 기본 경로
        base_path_str = "/media/casey/Casey_Save/차량비전/178.융합센서_다중객체_추적_및_예측데이터/01.데이터/2.Validation/원천데이터/13_전방 보행자_val"

    base_path = Path(base_path_str)

    if not base_path.exists():
        return {"sequences": [], "error": f"경로 없음: {base_path}"}

    sequences = []
    for seq_dir in sorted(base_path.iterdir()):
        if seq_dir.is_dir():
            sequences.append({
                "name": seq_dir.name,
                "path": str(seq_dir)
            })

    return {"sequences": sequences}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket 스트리밍 (실시간 YOLO 검출)"""
    await websocket.accept()
    print("클라이언트 연결됨")

    try:
        while True:
            # 메시지 수신 (타임아웃으로 논블로킹)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.01
                )

                command = data.get("command")
                print(f"명령: {command}")

                if command == "load":
                    processor.stop()
                    result = processor.load_sequence(data.get("path", ""))
                    await websocket.send_json({"type": "load_result", **result})

                    if result.get("success"):
                        frame_data = processor.process_frame(0)
                        await websocket.send_json({"type": "frame", **frame_data})

                elif command == "play":
                    processor.fps = data.get("fps", 10)
                    processor.is_playing = True
                    print(f"재생 시작: {processor.fps} FPS")

                elif command == "pause":
                    processor.is_playing = False
                    print("일시정지")
                    await websocket.send_json({"type": "paused", "frame_idx": processor.frame_idx})

                elif command == "stop":
                    processor.is_playing = False
                    processor.frame_idx = 0
                    frame_data = processor.process_frame(0)
                    await websocket.send_json({"type": "frame", **frame_data})

                elif command == "seek":
                    processor.is_playing = False
                    idx = data.get("frame_idx", 0)
                    frame_data = processor.process_frame(idx)
                    await websocket.send_json({"type": "frame", **frame_data})

                elif command == "step":
                    processor.is_playing = False
                    frame_data = processor.next_frame()
                    await websocket.send_json({"type": "frame", **frame_data})

            except asyncio.TimeoutError:
                pass

            # 재생 중이면 다음 프레임 처리
            if processor.is_playing and processor.dataset:
                if processor.frame_idx < processor.dataset.num_frames - 1:
                    frame_data = processor.next_frame()
                    await websocket.send_json({"type": "frame", **frame_data})
                    await asyncio.sleep(1.0 / processor.fps)
                else:
                    processor.is_playing = False
                    await websocket.send_json({"type": "finished"})

    except WebSocketDisconnect:
        print("클라이언트 연결 해제")
        processor.stop()
    except Exception as e:
        print(f"WebSocket 에러: {e}")
        import traceback
        traceback.print_exc()
        processor.stop()


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    # 서버 시작 후 브라우저 자동 열기
    def open_browser():
        import time
        time.sleep(1)  # 서버 시작 대기
        webbrowser.open("file://" + str(PROJECT_ROOT / "ui" / "frontend" / "index.html"))

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
