import numpy as np
from typing import Dict
from .ekf_tracker import EKFTracker

class EKFManager:
    def __init__(self):
        self.trackers: Dict[int, EKFTracker] = {}
        self.prev_time: Dict[int, float] = {}

    def update(self, track_id: int, foot_bev: np.ndarray, timestamp: float):
        pos = np.asarray(foot_bev[:2], dtype=np.float32)

        # 첫 등장 → tracker 생성 후 위치만 초기화
        if track_id not in self.trackers:
            self.trackers[track_id] = EKFTracker(pos, dt=0.01)
            self.prev_time[track_id] = timestamp
            return pos

        # dt 계산
        dt = timestamp - self.prev_time[track_id]
        if dt <= 0:
            dt = 0.01
        self.prev_time[track_id] = timestamp

        # dt 갱신 + F, Q 갱신
        tracker = self.trackers[track_id]
        tracker.dt = dt
        tracker._update_F_Q(dt)

        print(f"[EKFManager] Updated dt for track_id {track_id}: {dt}")

        # EKF 업데이트
        tracker.predict()
        return tracker.update(pos)

    def predict_future(self, track_id: int, steps=10):
        if track_id not in self.trackers:
            return None
        return self.trackers[track_id].predict_future(steps)
