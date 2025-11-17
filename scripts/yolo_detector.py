#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO

# ROS
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
# 기존 패키지 이름에 맞게 수정
from pedestrian_tracking.msg import GpsInfo, DetectionInfo, PipelinePacket


class DatasetLoader:
    """
    데이터셋 폴더에서 이미지, 라이다, GPS 등 관련 데이터를 로드하고
    프레임 단위로 공급하는 클래스.
    """
    def __init__(self, base_path):
        self.base_path = base_path
        
        self.image_files = sorted(glob(os.path.join(base_path, "image0", "*.jpg")))
        self.lidar_files = sorted(glob(os.path.join(base_path, "lidar", "*.pcd")))
        self.gps_data = self.load_gps(os.path.join(base_path, "gps100hz.csv"))

        self.num_frames = len(self.image_files)
        self.idx = 0

        rospy.loginfo(f"Dataset loaded from: {base_path}")
        rospy.loginfo(f"   - Found {len(self.image_files)} images.")
        rospy.loginfo(f"   - Found {len(self.lidar_files)} lidar scans.")
        rospy.loginfo(f"   - Found {len(self.gps_data)} GPS points.")

    def load_gps(self, path):
        """gps100hz.csv 파일을 읽어 GPS 데이터 리스트를 반환합니다."""
        gps_list = []
        if not os.path.exists(path):
            rospy.logwarn(f"Warning: GPS file not found at {path}")
            return gps_list
            
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    gps_list.append({
                        "time": float(parts[0]),
                        "lat": float(parts[1]),
                        "lon": float(parts[2]),
                        "alt": float(parts[3])
                    })
                except (ValueError, IndexError):
                    continue
        return gps_list

    def get_gps_for_frame(self, frame_idx):
        if frame_idx < len(self.gps_data):
            return self.gps_data[frame_idx]
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


class CalibrationLoader:
    """
    시퀀스 폴더에서 카메라 내부/외부 파라미터 파일을 로드하는 클래스.
    """
    def __init__(self, seq_path):
        self.seq_path = seq_path
        self.K = None
        self.image_size = None
        self.T_cam_lidar = None
        self.T_lidar_cam = None

        try:
            intrinsic_path = os.path.join(seq_path, "calib_Camera0.txt")
            extrinsic_path = os.path.join(seq_path, "calib_CameraToLidar0.txt")
            
            self.K, self.image_size = self._load_intrinsic(intrinsic_path)
            self.T_cam_lidar = self._load_extrinsic_3x4(extrinsic_path)
            self.T_lidar_cam = np.linalg.inv(self.T_cam_lidar)
            
            rospy.loginfo(f"Calibration loaded for sequence: {os.path.basename(seq_path)}")

        except (FileNotFoundError, ValueError) as e:
            rospy.logerr(f"Error loading calibration files for {os.path.basename(seq_path)}: {e}")
            raise

    def _load_intrinsic(self, path):
        """calib_Camera0.txt 파일에서 카메라 내부 파라미터(K)를 로드합니다."""
        with open(path, 'r') as f:
            lines = f.readlines()

        p_line = None
        K_matrix_values = []

        for line in lines:
            clean_line = line.strip().lower()
            if clean_line.startswith('p') or clean_line.startswith('k'):
                p_line = line
                break

        if p_line:
            values_str = p_line.strip().split(':')[1].strip()
            K_matrix_values = list(map(float, values_str.split()))
        else:
            matrix_rows = []
            for line in lines:
                line = line.strip()
                if not line: continue
                try:
                    row = list(map(float, line.split()))
                    matrix_rows.append(row)
                except ValueError:
                    continue
            if matrix_rows:
                K_matrix_values = [val for row in matrix_rows for val in row]

        if not K_matrix_values or len(K_matrix_values) < 9:
            raise ValueError(f"Could not parse a valid 3x3 intrinsic matrix from {path}")
            
        K = np.array(K_matrix_values).reshape(3, -1)[:, :3]
        
        width, height = -1, -1
        for line in lines:
            if line.startswith('S_rect_00:'):
                dims = list(map(float, line.strip().split()[1:]))
                width, height = int(dims[0]), int(dims[1])
                break
        
        return K, (width, height)

    def _load_extrinsic_3x4(self, path):
        """3x4 변환 행렬 파일을 4x4 동차 변환 행렬로 로드합니다."""
        try:
            # 숫자만 있는 파일의 경우, loadtxt로 한 번에 읽기
            mat = np.loadtxt(path)
            if mat.shape == (3, 4):
                return np.vstack([mat, [0, 0, 0, 1]])
        except ValueError:
            # loadtxt가 실패하면 (파일에 키워드가 섞여있을 경우) 수동으로 파싱
            pass

        # 키워드가 포함된 파일을 수동으로 파싱
        with open(path, 'r') as f:
            lines = f.readlines()
        
        matrix_rows = []
        for line in lines:
            parts = line.strip().split(':')[-1].split()
            if len(parts) == 4:
                matrix_rows.append([float(p) for p in parts])

        if len(matrix_rows) < 3:
            raise ValueError(f"Could not parse a valid 3x4 extrinsic matrix from {path}")

        mat = np.array(matrix_rows[:3])
        return np.vstack([mat, [0, 0, 0, 1]])


class BaseTracker:
    """모든 추적기 클래스를 위한 기본 인터페이스"""
    def __init__(self, model_path, conf_thres_config, target_class_names=None):
        self.model_path = model_path
        self.conf_thres_config = conf_thres_config
        self.target_class_names = target_class_names

    def process(self, img):
        """이미지를 처리하고 detection 리스트를 반환"""
        raise NotImplementedError

class UltralyticsTracker(BaseTracker):
    """Ultralytics 내장 추적기(BoT-SORT, ByteTrack)를 사용하는 클래스."""
    def __init__(self, model_path, conf_thres_config, target_class_names=None, tracker_type="botsort"):
        super().__init__(model_path, conf_thres_config, target_class_names)
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.conf_thres_config = conf_thres_config
        if isinstance(conf_thres_config, dict): self.min_conf_thres = min(conf_thres_config.values())
        else: self.min_conf_thres = conf_thres_config
        self.target_indices = [k for k, v in self.names.items() if v in target_class_names] if target_class_names else None
        self.tracker_config = f"{tracker_type}.yaml"
        rospy.loginfo(f"ObjectTracker initialized with '{self.tracker_config}'")
        rospy.loginfo(f"Tracking target classes: {[self.names[i] for i in self.target_indices] if self.target_indices else 'All'}")

    def process(self, img):
        results = self.model.track(img, persist=True, verbose=False, classes=self.target_indices, conf=self.min_conf_thres, tracker=self.tracker_config)[0]
        
        if results.boxes is None or results.boxes.id is None:
            return []

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2 = map(float, box[:4])
            track_id, conf, cls_id = int(box[4]), float(box[5]), int(box[6])
            cls_name = self.names.get(cls_id, "unknown")
            if isinstance(self.conf_thres_config, dict):
                threshold = self.conf_thres_config.get(cls_name, self.conf_thres_config.get('default', self.min_conf_thres))
                if conf < threshold: continue
            foot_u, foot_v = (x1 + x2) / 2.0, y2
            detections.append({
                "id": track_id, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": conf,
                "class": cls_name, "foot_uv": [foot_u, foot_v],
            })
        return detections

class DeepSORTTracker(BaseTracker):
    """
    별도로 구현된 DeepSORT 로직을 사용하는 클래스 (자리 표시자).
    나중에 이 클래스 내부만 구현하면 됩니다.
    """
    def __init__(self, model_path, conf_thres_config, target_class_names=None):
        super().__init__(model_path, conf_thres_config, target_class_names)
        rospy.logwarn("⚠️  DeepSORTTracker is a placeholder. You need to implement the tracking logic.")

    def process(self, img):
        rospy.logwarn_throttle(10, "DeepSORT processing is not implemented yet.")
        return []


class PipelineNode:
    def __init__(self):
        rospy.init_node('pipeline_node', anonymous=True)
        
        # ROS 파라미터 로드
        self.base_dir = rospy.get_param('~dataset_base_dir')
        self.yolo_model_path = rospy.get_param('~yolo_model_path')
        self.conf_threshold = rospy.get_param('~conf_threshold')
        self.target_classes = rospy.get_param('~target_classes')
        self.tracker_type = rospy.get_param('~tracker', 'bytetrack')
        self.loop_rate = rospy.get_param('~loop_rate', 10) # Hz
        self.visualize = rospy.get_param('~visualize', True)

        # Publisher 설정
        self.packet_pub = rospy.Publisher('pipeline_packet', PipelinePacket, queue_size=10)
        self.vis_image_pub = rospy.Publisher('pipeline_vis_image', Image, queue_size=10)
        self.bridge = CvBridge()

        rospy.loginfo("✅ PipelineNode initialized.")

    def run(self):
        try:
            sequences = sorted([f for f in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, f)) and f.isdigit()])
        except FileNotFoundError:
            rospy.logerr(f"❌ Error: Base directory not found at {self.base_dir}")
            return

        for seq in sequences:
            if rospy.is_shutdown(): break
            seq_path = os.path.join(self.base_dir, seq)
            self.run_sequence(seq_path)

    def run_sequence(self, seq_path):
        rospy.loginfo(f"\n{'='*20} Running sequence: {os.path.basename(seq_path)} {'='*20}")
        dataset = DatasetLoader(seq_path)
        try:
            calib = CalibrationLoader(seq_path)
        except (FileNotFoundError, ValueError):
            rospy.logwarn(f"⚠️  Skipping sequence {os.path.basename(seq_path)} due to calibration error.")
            return

        # --- 추적기 팩토리(Factory) 로직 ---
        if self.tracker_type == 'botsort':
            tracker = UltralyticsTracker(self.yolo_model_path, self.conf_threshold, self.target_classes, tracker_type="botsort")
        elif self.tracker_type == 'bytetrack':
            tracker = UltralyticsTracker(self.yolo_model_path, self.conf_threshold, self.target_classes, tracker_type="bytetrack")
        elif self.tracker_type == 'deepsort':
            tracker = DeepSORTTracker(self.yolo_model_path, self.conf_threshold, self.target_classes)
        else:
            rospy.logerr(f"❌ Unknown tracker type: {self.tracker_type}. Shutting down.")
            rospy.signal_shutdown("Unknown tracker type")
            return

        for frame_data in dataset:
            if rospy.is_shutdown(): break
            
            img = cv2.imread(frame_data["image_path"])
            if img is None: continue

            detections = tracker.process(img)

            # --- ROS 메시지 생성 및 발행 ---
            header = Header(stamp=rospy.Time.now(), frame_id=os.path.basename(seq_path))
            
            packet_msg = PipelinePacket(header=header, frame_id=frame_data['frame_id'])
            if frame_data['gps']:
                packet_msg.gps = GpsInfo(**frame_data['gps'])
            
            packet_msg.detections = [DetectionInfo(id=det['id'], bbox=det['bbox'], score=det['score'],
                                                    class_name=det['class'], foot_uv=det['foot_uv'])
                                      for det in detections]
            
            self.packet_pub.publish(packet_msg)

            if self.visualize:
                vis_img = self.create_visualization(img, detections, os.path.basename(seq_path))
                vis_img_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
                vis_img_msg.header = header
                self.vis_image_pub.publish(vis_img_msg)

                # [수정] OpenCV 창을 직접 띄워서 시각화합니다.
                cv2.imshow("Detection Result", vis_img)
                cv2.waitKey(1) # GUI 이벤트 처리를 위해 이 줄이 필수적입니다.
            
            rospy.loginfo(f"Published packet for frame {frame_data['frame_id']} in sequence {os.path.basename(seq_path)}")
            if self.loop_rate > 0:
                rospy.sleep(1.0 / self.loop_rate)

    def create_visualization(self, img, detections, seq_name):
        vis_img = img.copy()
        for det in detections:
            track_id, cls_name, score = det['id'], det['class'], det['score']
            x, y, w, h = det['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            
            box_color = (255, 0, 0) if cls_name == 'person' else (0, 255, 0)
            label = f"ID:{track_id} {cls_name} {score:.2f}"
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            foot_u, foot_v = map(int, det['foot_uv'])
            cv2.circle(vis_img, (foot_u, foot_v), 5, (0, 0, 255), -1)
        
        seq_label = f"seq: {seq_name}"
        cv2.putText(vis_img, seq_label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        return vis_img

    def shutdown_hook(self):
        """노드 종료 시 호출되는 함수"""
        rospy.loginfo("Shutting down. Closing OpenCV windows.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = PipelineNode()
        node.run()
    except rospy.ROSInterruptException:
        pass