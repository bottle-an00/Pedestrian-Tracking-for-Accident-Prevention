# scripts/run_pipeline.py가 호출하는 메인 파이프라인

import os
import cv2
from pathlib import Path

from src.core.config import load_yaml
from src.tracking.tracker import ByteTracker
from src.visualization.overlay_2d import Visualizer
from src.io.image_loader import ImageLoader
from src.io.pcd_loader import PcdLoader
from src.io.gps_loader import GpsImuLoader
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.bev.bev_transformer import BevTransformer, Detections_bev
from src.trajectory.trajectory_manager import TrajectoryBuffer
from src.trajectory.ekf_manager import EKFManager
from src.trajectory.pedestrian_state_manager import PedestrianStateManager
from src.trajectory.bev_zone_risk_manager import PedestrianStateManager as BevZoneManager


class DatasetLoader:
    """
    데이터셋 폴더에서 이미지, 라이다, GPS 등 관련 데이터를 로드하고
    프레임 단위로 공급하는 클래스.
    기존 src/io/ 로더들을 활용합니다.
    """
    def __init__(self, base_path):
        self.base_path = Path(base_path)

        # 기존 로더 사용
        self.image_loader = ImageLoader()
        self.pcd_loader = PcdLoader()
        self.gps_loader = GpsImuLoader()

        # 이미지, 라이다, GPS 경로 로드
        image_dir = self.base_path / "image0"
        lidar_dir = self.base_path / "lidar"
        gps_dir = self.base_path / "gps"

        self.image_files = self.image_loader.list_img_paths(image_dir) if image_dir.exists() else []
        self.lidar_files = self.pcd_loader.list_pcd_paths(lidar_dir) if lidar_dir.exists() else []
        self.gps_files = self.gps_loader.list_gps_imu_paths(gps_dir) if gps_dir.exists() else []

        self.num_frames = len(self.image_files)
        self.idx = 0

        print(f"Dataset loaded from: {base_path}")
        print(f"   - Found {len(self.image_files)} images.")
        print(f"   - Found {len(self.lidar_files)} lidar scans.")
        print(f"   - Found {len(self.gps_files)} GPS files.")

    def get_gps_for_frame(self, frame_idx):
        if frame_idx < len(self.gps_files):
            try:
                return self.gps_loader.load_data(self.gps_files[frame_idx])
            except Exception:
                return None
        return None

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.num_frames:
            raise StopIteration

        packet = {
            "frame_id": self.idx,
            "image_path": self.image_files[self.idx],
            "lidar_path": self.lidar_files[self.idx] if self.idx < len(self.lidar_files) else None,
            "gps": self.get_gps_for_frame(self.idx)
        }
        self.idx += 1
        return packet


class Pipeline:
    """YOLO 탐지 + 추적 파이프라인"""

    def __init__(self, config_path="configs/detector/yolo_detector.yaml"):
        # 설정 로드
        self.cfg = load_yaml(config_path)
        self.system_cfg = load_yaml("configs/system.yaml")

        self.yolo_model_path = self.cfg.get("yolo_model_path", "./models/yolo/yolo11n.pt")
        self.conf_threshold = self.cfg.get("conf_threshold", {"default": 0.5})
        self.target_classes = self.cfg.get("target_classes", ["person"])
        self.imgsz = self.cfg.get("imgsz", 1280)
        self.visualize = True

        self.vis = Visualizer()
        # initialize components used by full pipeline
        self.calib_loader = CalibrationInfoLoader()

        # tracker (YOLO based)
        self.tracker = ByteTracker(
            self.yolo_model_path,
            self.conf_threshold,
            self.target_classes,
            imgsz=self.imgsz
        )

        # placeholders - constructed later if calibration available
        self.homography = None
        self.bev_conv = None

        # common helpers
        self.trajectory_buffer = TrajectoryBuffer(max_length=100)
        self.ekf_manager = EKFManager()

        self.state_manager = PedestrianStateManager(obs_len=10, max_missing=5, ekf_constructor=None)

        self.bev_risk_manager = BevZoneManager()

        print("Pipeline initialized.")

    def run(self, base_dir, target_sequence=None):
        """
        base_dir: 시퀀스들이 있는 상위 폴더
        target_sequence: 특정 시퀀스만 실행 (None이면 전체)
        """
        try:
            all_sequences = sorted([f for f in os.listdir(base_dir)
                                    if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()])
        except FileNotFoundError:
            print(f"Error: Base directory not found at {base_dir}")
            return

        sequences_to_run = []
        if target_sequence:
            if str(target_sequence) in all_sequences:
                sequences_to_run.append(str(target_sequence))
                print(f"Target sequence '{target_sequence}' found.")
            else:
                print(f"Error: Target sequence '{target_sequence}' not found")
                return
        else:
            sequences_to_run = all_sequences
            print("Running all available sequences.")

        for seq in sequences_to_run:
            seq_path = os.path.join(base_dir, seq)
            self.run_sequence(seq_path)

    def run_sequence(self, seq_path):
        print(f"\n{'='*20} Running sequence: {os.path.basename(seq_path)} {'='*20}")
        dataset = DatasetLoader(seq_path)

        tracker = ByteTracker(
            self.yolo_model_path,
            self.conf_threshold,
            self.target_classes,
            imgsz=self.imgsz
        )

        # 결과 저장 폴더
        output_dir = Path(seq_path) / "outputs" / "detection"
        output_dir.mkdir(parents=True, exist_ok=True)

        for frame_data in dataset:
            img = cv2.imread(frame_data["image_path"])
            if img is None:
                continue

            detections = tracker.process(img)

            if self.visualize:
                vis_img = self.vis.draw_on_img_with_keypoints(
                    img, detections, seq_name=os.path.basename(seq_path)
                )

                # 결과 저장
                img_name = Path(frame_data["image_path"]).name
                cv2.imwrite(str(output_dir / f"det_{img_name}"), vis_img)

                # GUI 표시 시도 (실패해도 계속 진행)
                try:
                    cv2.imshow("Detection Result", vis_img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("Quit requested.")
                        cv2.destroyAllWindows()
                        return
                except cv2.error:
                    pass  # GUI 없으면 파일 저장만

            print(f"Frame {frame_data['frame_id']} - {len(detections)} detections")

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def run_full_pipeline_sequence(self, seq_path):
        """Full pipeline: detection -> BEV -> EKFManager -> risk manager -> save visual outputs

        This method embeds the tmp_full_pipeline logic inside the Pipeline class. It attempts
        to load camera/LiDAR calibration from `system.yaml` and runs per-frame processing.
        """
        print(f"\n{'='*20} Running FULL pipeline: {os.path.basename(seq_path)} {'='*20}")

        self._dataset = DatasetLoader(seq_path)

        cam_p = Path(seq_path) / 'calib_Camera0.txt'
        lidar_cam_p = Path(seq_path) / 'calib_CameraToLidar0.txt'

        if cam_p and lidar_cam_p and self.calib_loader is not None:
            intrinsics = self.calib_loader.load_camera_calibration(Path(cam_p))
            extrinsics = self.calib_loader.load_camera_extrinsics(Path(lidar_cam_p))
            self.homography = Homography(intrinsics=intrinsics, extrinsics=extrinsics)
            self.bev_conv = BevTransformer(homography=self.homography)

        self._out_root = Path(seq_path) / "outputs" / "full_pipeline"
        self._out_root.mkdir(parents=True, exist_ok=True)

        self._out_ekf_bev = self._out_root / "ekf_bev"
        self._out_ekf_img = self._out_root / "ekf_img"

        for p in [self._out_ekf_bev, self._out_ekf_img]:
            p.mkdir(parents=True, exist_ok=True)

        img_iter = self._dataset.image_loader.iter_imgs_cv2(Path(seq_path) / 'image0')
        gps_iter = self._dataset.gps_loader.iter_data(Path(seq_path) / 'gps')

        H = self.homography
        bev_conv = self.bev_conv
        tracker = self.tracker
        bev_risk_manager = self.bev_risk_manager
        ekf_manager = self.ekf_manager
        vis = self.vis
        out_ekf_bev = self._out_ekf_bev
        out_ekf_img = self._out_ekf_img

        frame_idx = 0
        # Main loop
        for (img_path, image), (gps_path, gps_data) in zip(img_iter, gps_iter):
            print(f"[Full Pipeline] Processing {img_path.name}")

            detections = tracker.process(image)

            bev_img = H.warp(image)
            foot_bevs = bev_conv.foot_uv_to_foot_bev(detections)

            foot_bevs_m = []
            for fb in foot_bevs:
                bx, by = fb.foot_bev
                x_m, y_m = H.pixel_to_world(float(by), float(bx), flipped=True)
                foot_bevs_m.append(Detections_bev(id=fb.id, foot_bev=(x_m, y_m)))

            # vehicale_bev setting
            vehicle_bev = bev_risk_manager.compute_vehicle_bev(image,H)
            bev_risk_manager.set_zones_from_gps(gps_data, vehicle_bev=vehicle_bev)

            ekf_now = {}
            ekf_future = {}

            # Use external ekf_manager for current estimates and future predictions
            for d in foot_bevs_m:
                try:
                    tid = int(d.id)
                except Exception:
                    tid = d.id
                xy = ekf_manager.update(tid, d.foot_bev, gps_data.time)
                ekf_now[tid] = xy
                ekf_future[tid] = ekf_manager.predict_future(tid, steps=20)

            ekf_bev_img = bev_img.copy()
            ekf_img = image.copy()

            for tid, xy in ekf_now.items():
                if xy is None:
                    continue
                lx, ly = float(xy[0]), float(xy[1])
                u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
                ekf_bev_img = vis.draw_points(ekf_bev_img, tid, [(v_bev, u_bev)], radius=6)
                u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
                if u_img != -1 and v_img != -1:
                    ekf_img = vis.draw_on_img(ekf_img, [{"bbox": (0,0,0,0), "id": tid, "class": "ekf", "score": 1.0, "foot_uv": (u_img, v_img)}])

            for tid, future in ekf_future.items():
                if future is None:
                    continue
                bev_future_pts = []
                img_uvs = []
                for p in future:
                    lx, ly = float(p[0]), float(p[1])
                    u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
                    bev_future_pts.append((v_bev, u_bev))
                    u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
                    if u_img != -1 and v_img != -1:
                        img_uvs.append((u_img, v_img))

                if len(bev_future_pts) == 1:
                    ekf_bev_img = vis.draw_points(ekf_bev_img, tid, bev_future_pts, radius=4)
                elif bev_future_pts:
                    ekf_bev_img = vis.draw_polyline(ekf_bev_img, tid, bev_future_pts, color=(255, 0, 0), thickness=2)

                if img_uvs:
                    ekf_img = vis.draw_on_img(ekf_img, [{"bbox": (0,0,0,0), "id": tid, "class": "ekf_future", "score": 1.0, "foot_uv": uv} for uv in img_uvs])

            # Update risk states
            obj_states = {}
            for tid, future in ekf_future.items():
                state = "SAFE"
                if future and len(future) > 0:
                    updated_state = bev_risk_manager.update_state(tid, future)
                    state = updated_state
                obj_states[tid] = state

            frame_idx += 1

            # Visualizations
            gt_vis = ekf_bev_img.copy()
            gt_vis = bev_risk_manager.draw_zones_on_bev(gt_vis, alpha=0.25)
            ekf_img = bev_risk_manager.draw_zones_on_image(ekf_img, H, alpha=0.25)
            ekf_img = bev_risk_manager.draw_detections_on_image(ekf_img, detections)

            cv2.imwrite(str(out_ekf_bev / f"ekf_bev_gt_{img_path.name}"), gt_vis)
            cv2.imwrite(str(out_ekf_img / f"ekf_img_{img_path.name}"), ekf_img)

