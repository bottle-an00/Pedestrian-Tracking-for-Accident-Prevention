"""
보행자 추적 시각화 백엔드 서버
- YOLO 실시간 검출
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
from glob import glob

from src.detection.yolo_detector import YoloDetector
from src.bev.bev_transformer import BevTransformer
from src.visualization.overlay_2d import Visualizer
from src.core.config import load_yaml

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
    """실시간 YOLO 검출 + BEV 변환 처리"""

    def __init__(self):
        self.raw_frames = []
        self.frame_idx = 0
        self.is_playing = False
        self.fps = 10
        self.seq_path: Optional[Path] = None

        # 설정 로드
        try:
            self.cfg = load_yaml('configs/system.yaml')
        except:
            self.cfg = {'bev': {'resolution': 0.05}}

        # YOLO 검출기 초기화
        model_path = PROJECT_ROOT / "models" / "yolo" / "yolo11m-pose.pt"
        print(f"YOLO 모델 로딩: {model_path}")
        self.detector = YoloDetector(
            model_path=str(model_path),
            conf_thres_config={"person": 0.1},
            target_class_names=["person"],
            imgsz=1280,
            half=True
        )
        print("YOLO 모델 로드 완료")

        # BEV 변환기 (시퀀스별로 초기화)
        self.bev_transformer: Optional[BevTransformer] = None

        self.visualizer = Visualizer()
        self.last_detections: List[Dict] = []

    def load_sequence(self, seq_dir: str) -> dict:
        """시퀀스 디렉토리 로드"""
        self.seq_path = Path(seq_dir)
        self.is_playing = False

        # 이미지 찾기
        image_dir = self.seq_path / "image0"
        if not image_dir.exists():
            image_dir = self.seq_path

        self.raw_frames = sorted(glob(str(image_dir / "*.jpg")))
        if not self.raw_frames:
            self.raw_frames = sorted(glob(str(image_dir / "*.png")))

        if not self.raw_frames:
            return {"success": False, "error": "이미지 없음"}

        # BEV용 캘리브레이션 로드
        calib_cam = self.seq_path / "calib_Camera0.txt"
        calib_ext = self.seq_path / "calib_CameraToLidar0.txt"

        if calib_cam.exists() and calib_ext.exists():
            try:
                self.bev_transformer = BevTransformer(
                    intrinsics_path=str(calib_cam),
                    extrinsics_path=str(calib_ext)
                )
                print(f"BEV 변환기 로드: {self.seq_path.name}")
            except Exception as e:
                print(f"BEV 변환기 에러: {e}")
                self.bev_transformer = None
        else:
            self.bev_transformer = None

        self.frame_idx = 0
        self.last_detections = []

        return {
            "success": True,
            "total_frames": len(self.raw_frames),
            "seq_name": self.seq_path.name,
            "has_bev": self.bev_transformer is not None
        }

    def process_frame(self, idx: int = None) -> dict:
        """프레임 처리 (YOLO 검출 + BEV 변환)"""
        if idx is not None:
            self.frame_idx = max(0, min(idx, len(self.raw_frames) - 1))

        if not self.raw_frames or self.frame_idx >= len(self.raw_frames):
            return {"camera": None, "bev": None}

        # 원본 이미지 로드 (리사이즈 X, YOLO 내부에서 처리)
        img_full = cv2.imread(self.raw_frames[self.frame_idx])
        if img_full is None:
            return {"camera": None, "bev": None}

        # YOLO 검출
        detections = self.detector.detect(img_full)
        self.last_detections = detections

        # 임시 track ID 부여 + keypoints 형식 변환 (x,y,conf -> x,y)
        for i, det in enumerate(detections):
            det['id'] = i + 1
            if 'keypoints' in det and det['keypoints']:
                det['keypoints'] = [(kp[0], kp[1]) for kp in det['keypoints']]

        # 카메라 뷰에 검출 결과 시각화
        camera_vis = self.visualizer.draw_on_img_with_keypoints(
            img_full, detections,
            seq_name=self.seq_path.name if self.seq_path else None,
            show_legend=True,
            max_width=1280
        )

        # BEV 변환 및 시각화
        bev_vis = None
        pedestrians_in_risk = 0
        closest_distance = None

        if self.bev_transformer is not None:
            try:
                bev = self.bev_transformer.warp_image(img_full)

                # 위험구역 시각화
                bev_vis = self.visualizer.draw_risk_zone_bev(
                    bev,
                    front_m=10.0,
                    width_m=2.0,
                    bev_resolution=float(self.cfg['bev']['resolution'])
                )

                # 보행자 foot 좌표를 BEV로 변환해서 표시
                for det in detections:
                    if det['class'] == 'person':
                        foot_uv = det['foot_uv']
                        bev_coords = self.bev_transformer.foot_uv_to_foot_bev([{
                            'id': det['id'],
                            'foot_uv': foot_uv
                        }])

                        if bev_coords:
                            for bev_det in bev_coords:
                                bx, by = bev_det.foot_bev

                                # BEV에 보행자 표시
                                cv2.circle(bev_vis, (int(by), int(bx)), 8,
                                          self.visualizer.id_to_color(det['id']), -1)
                                cv2.putText(bev_vis, f"ID:{det['id']}",
                                           (int(by) + 10, int(bx)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                           (255, 255, 255), 1)

                                # 차량으로부터 거리 계산 (BEV 하단 중앙이 차량 위치)
                                h, w = bev_vis.shape[:2]
                                vehicle_pos = (h - 1, w // 2)
                                dist_px = np.sqrt((bx - vehicle_pos[0])**2 + (by - vehicle_pos[1])**2)
                                dist_m = dist_px * float(self.cfg['bev']['resolution'])

                                if closest_distance is None or dist_m < closest_distance:
                                    closest_distance = dist_m

                                # 위험구역 내 보행자 체크 (전방 10m, 폭 4m)
                                if dist_m < 10 and abs(by - vehicle_pos[1]) * float(self.cfg['bev']['resolution']) < 2:
                                    pedestrians_in_risk += 1

            except Exception as e:
                print(f"BEV 에러: {e}")

        # base64 인코딩 (웹 전송용)
        def encode_image(img_to_encode, quality=85):
            if img_to_encode is None:
                return None
            _, buffer = cv2.imencode('.jpg', img_to_encode, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return base64.b64encode(buffer).decode('utf-8')

        return {
            "camera": encode_image(camera_vis),
            "bev": encode_image(bev_vis),
            "frame_idx": self.frame_idx,
            "total_frames": len(self.raw_frames),
            "frame_name": Path(self.raw_frames[self.frame_idx]).name,
            "detections": {
                "count": len([d for d in detections if d['class'] == 'person']),
                "in_risk_zone": pedestrians_in_risk,
                "closest_distance": round(closest_distance, 2) if closest_distance else None
            }
        }

    def next_frame(self) -> dict:
        """다음 프레임으로 이동"""
        if self.frame_idx < len(self.raw_frames) - 1:
            self.frame_idx += 1
        return self.process_frame()

    def stop(self):
        """재생 중지"""
        self.is_playing = False


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
            if processor.is_playing:
                if processor.frame_idx < len(processor.raw_frames) - 1:
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
