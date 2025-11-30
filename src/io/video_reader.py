# 데이터 입출력, 동기화
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

class VideoReader:
    def __init__(self, video_path: str | Path,
                 color: bool = True,
                 step: int = 1):

        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"[VideoReader] Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"[VideoReader] Cannot open video file: {self.video_path}")

        self.color = color
        self.step = step

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self) -> int:
        return self.frame_count // self.step

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_idx = 0
        current_frame = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if current_frame % self.step == 0:
                yield frame_idx, frame
                frame_idx += 1

            current_frame += 1

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def seek(self,frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()