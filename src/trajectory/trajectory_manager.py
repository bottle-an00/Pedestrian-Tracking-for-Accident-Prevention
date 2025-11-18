from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List
from src.bev.bev_transformer import Detections_bev

@dataclass
class TrajectoryPoint:
    track_id: int
    u: float
    v: float
    t: float

    @property
    def foot_bev(self):
        return (self.u, self.v)

class TrajectoryBuffer:
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.buffer = defaultdict(lambda: deque(maxlen=self.max_length))


    def add(self, compensated_detections: List[Detections_bev], timestamp: float):
        for detection in compensated_detections:
            track_id = detection.id
            u_bev, v_bev = detection.foot_bev

            traj_point = TrajectoryPoint(
                track_id=track_id,
                u=float(u_bev),
                v=float(v_bev),
                t=float(timestamp)
            )

            self.buffer[track_id].append(traj_point)


    def get(self, track_id: int) -> List[TrajectoryPoint]:
        return list(self.buffer.get(track_id, []))

    def get_all(self) -> dict:
        return {track_id: list(points) for track_id, points in self.buffer.items()}

    def latest(self, track_id: int):
        if track_id not in self.buffer:
            return None

        if len(self.buffer[track_id]) == 0:
            return None

        return self.buffer[track_id][-1]


    def exist(self, track_id: int) -> bool:
        return track_id in self.buffer


    def remove(self, track_id: int):
        if track_id in self.buffer:
            del self.buffer[track_id]


    def clear(self):
        self.buffer.clear()
