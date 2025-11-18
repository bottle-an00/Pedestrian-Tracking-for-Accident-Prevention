# GNSS 기반 ego-motion 추정/보정
import numpy as np
from typing import List

from src.io.gps_loader import GpsImuSample
from src.bev.bev_transformer import Detections_bev
from src.trajectory.trajectory_manager import TrajectoryPoint

class EgoMotionCompensator:
    def __init__(self):
        pass

    def compensate(self, detections_bev, ego_info: GpsImuSample):
        compensated_detections = []

        yaw = np.deg2rad(ego_info.heading)
        c, s = np.cos(yaw), np.sin(yaw)

        R = np.array([[c, -s],
                      [s,  c]])
        T = np.array([ego_info.x, ego_info.y])

        for det in detections_bev:
            p_local = np.array(det.foot_bev, dtype=float)
            p_global = R @ p_local + T

            compensated_detections.append(
                Detections_bev(
                    id=det.id,
                    foot_bev=(p_global[0], p_global[1])
                )
            )

        return compensated_detections
    
    def inv_compensate(self, traj_point:TrajectoryPoint, ego_info: GpsImuSample):
        
        yaw = np.deg2rad(ego_info.heading)
        c, s = np.cos(yaw), np.sin(yaw)

        R_inv = np.array([[ c,  s],
                          [-s,  c]])   # R(-yaw)
        T = np.array([ego_info.x, ego_info.y])

        p_global = np.array([traj_point.u, traj_point.v])
        p_local = R_inv @ (p_global - T)

        return TrajectoryPoint(
            track_id=traj_point.track_id,
            u=float(p_local[0]),
            v=float(p_local[1]),
            t=traj_point.t
        )

    def inv_compensate_all(self, trajectory_points, ego_info):
        return [self.inv_compensate(p, ego_info) for p in trajectory_points]
