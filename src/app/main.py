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
        gps_dir = self.base_path / "gps_imu"

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
        self.imgsz = self.cfg.get("imgsz", 640)
        self.visualize = True

        self.vis = Visualizer()

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
