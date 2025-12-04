"""
EKF Manager for Multi-Pedestrian Tracking in BEV

This module manages multiple EKF trackers (one per pedestrian) and handles:
- Smart initialization with velocity estimation from recent trajectory history
- Dynamic dt calculation based on timestamps
- Track lifecycle management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

from .ekf_tracker_cv import EKFTrackerCV
from ..core.config import load_yaml


class EKFManager:
    """
    Manager for multiple EKF trackers (one per pedestrian track_id).
    
    Handles:
    - Creation of new trackers with smart velocity initialization
    - Per-track timestamp management for dt calculation
    - Configuration loading from YAML
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize EKF Manager.
        
        Args:
            config_path: Path to EKF config YAML file. 
                        If None, uses default: "configs/prediction/ekf.yaml"
        """
        # Load configuration
        if config_path is None:
            config_path = "configs/prediction/ekf.yaml"
        
        try:
            self.config = load_yaml(config_path)
        except Exception as e:
            print(f"[EKFManager] Warning: Could not load config from {config_path}: {e}")
            print("[EKFManager] Using default parameters")
            self.config = {}
        
        # Extract config parameters with defaults
        self.sigma_a = self.config.get("sigma_a", 3.0)
        self.sigma_meas = self.config.get("sigma_meas", 0.1)
        self.sigma_pos_init = self.config.get("sigma_pos_init", 0.2)
        self.sigma_vel_init = self.config.get("sigma_vel_init", 1.0)
        self.default_dt = self.config.get("default_dt", 0.1)
        self.use_distance_dependent_R = self.config.get("use_distance_dependent_R", False)
        self.distance_R_alpha = self.config.get("distance_R_alpha", 0.05)
        self.min_history_for_vel_init = self.config.get("min_history_for_vel_init", 1)
        self.max_history_for_vel_init = self.config.get("max_history_for_vel_init", 5)
        self.debug = self.config.get("enable_debug_logging", False)
        
        # Storage
        self.trackers: Dict[int, EKFTrackerCV] = {}
        self.prev_time: Dict[int, float] = {}
        
        # Position history for velocity initialization
        # track_id -> deque of (timestamp, position)
        self.position_history: Dict[int, deque] = {}
        
        if self.debug:
            print(f"[EKFManager] Initialized with config:")
            print(f"  sigma_a={self.sigma_a}, sigma_meas={self.sigma_meas}")
            print(f"  sigma_pos_init={self.sigma_pos_init}, sigma_vel_init={self.sigma_vel_init}")
            print(f"  default_dt={self.default_dt}")
    
    def _estimate_initial_velocity(
        self, 
        track_id: int, 
        current_pos: np.ndarray, 
        current_time: float
    ) -> Tuple[float, float]:
        """
        Estimate initial velocity from recent position history.
        
        Strategy:
        1. Use the last N positions (where N = min_history_for_vel_init ~ max_history_for_vel_init)
        2. Compute velocity as average over those intervals
        3. If insufficient history, return (0, 0)
        
        Args:
            track_id: Track ID
            current_pos: Current position [X, Y] in BEV local [m]
            current_time: Current timestamp [s]
        
        Returns:
            (vx, vy) in BEV local [m/s]
        """
        if track_id not in self.position_history:
            return (0.0, 0.0)
        
        history = list(self.position_history[track_id])
        
        if len(history) < self.min_history_for_vel_init:
            return (0.0, 0.0)
        
        # Use up to max_history positions
        history = history[-self.max_history_for_vel_init:]
        
        # Compute velocities for each consecutive pair
        velocities = []
        for i in range(len(history)):
            t_prev, pos_prev = history[i]
            
            if i == len(history) - 1:
                # Use current position as the end point
                t_next, pos_next = current_time, current_pos
            else:
                t_next, pos_next = history[i + 1]
            
            dt = t_next - t_prev
            if dt > 1e-6:  # Avoid division by zero
                vx = (pos_next[0] - pos_prev[0]) / dt
                vy = (pos_next[1] - pos_prev[1]) / dt
                velocities.append((vx, vy))
        
        if not velocities:
            return (0.0, 0.0)
        
        # Average velocities
        vx_avg = np.mean([v[0] for v in velocities])
        vy_avg = np.mean([v[1] for v in velocities])
        
        if self.debug:
            print(f"[EKFManager] Track {track_id}: estimated vel from {len(velocities)} intervals")
            print(f"  vx={vx_avg:.3f}, vy={vy_avg:.3f} m/s")
        
        return (float(vx_avg), float(vy_avg))
    
    def update(
        self, 
        track_id: int, 
        foot_bev: np.ndarray, 
        timestamp: float
    ) -> np.ndarray:
        """
        Update EKF for a specific track with a new measurement.
        
        Args:
            track_id: Unique track ID
            foot_bev: Position measurement [X, Y] in BEV local [m]
            timestamp: Measurement timestamp [s]
        
        Returns:
            Updated position [X, Y] in BEV local [m]
        """
        pos = np.asarray(foot_bev[:2], dtype=np.float64)
        
        # First appearance of this track → create new tracker
        if track_id not in self.trackers:
            # Estimate initial velocity from history (if available)
            initial_vel = self._estimate_initial_velocity(track_id, pos, timestamp)
            
            # Create new EKF tracker with estimated velocity
            self.trackers[track_id] = EKFTrackerCV(
                initial_pos=(pos[0], pos[1]),
                initial_vel=initial_vel,
                dt=self.default_dt,
                sigma_a=self.sigma_a,
                sigma_meas=self.sigma_meas,
                sigma_pos_init=self.sigma_pos_init,
                sigma_vel_init=self.sigma_vel_init,
                use_distance_dependent_R=self.use_distance_dependent_R,
                distance_R_alpha=self.distance_R_alpha,
                debug=self.debug
            )
            
            self.prev_time[track_id] = timestamp
            
            # Initialize position history
            if track_id not in self.position_history:
                self.position_history[track_id] = deque(
                    maxlen=self.max_history_for_vel_init
                )
            
            # Add current position to history
            self.position_history[track_id].append((timestamp, pos.copy()))
            
            if self.debug:
                print(f"[EKFManager] Created new tracker for track_id={track_id}")
                print(f"  initial_pos={pos}, initial_vel={initial_vel}")
            
            return pos
        
        # Existing track → compute dt and update
        dt = timestamp - self.prev_time[track_id]
        if dt <= 0:
            dt = self.default_dt
            if self.debug:
                print(f"[EKFManager] Warning: non-positive dt for track {track_id}, using default")
        
        self.prev_time[track_id] = timestamp
        
        # Add to position history (before EKF update)
        self.position_history[track_id].append((timestamp, pos.copy()))
        
        # EKF update with measurement
        tracker = self.trackers[track_id]
        updated_pos = tracker.update(pos, dt=dt)
        
        if self.debug:
            print(f"[EKFManager] Updated track_id={track_id}, dt={dt:.4f}")
            print(f"  measurement={pos}, updated_pos={updated_pos}")
            print(f"  velocity={tracker.get_velocity()}")
        
        return updated_pos
    
    def predict_future(
        self, 
        track_id: int, 
        steps: int = 10,
        dt: Optional[float] = None
    ) -> Optional[List[np.ndarray]]:
        """
        Predict future trajectory for a specific track.
        
        Args:
            track_id: Track ID
            steps: Number of future time steps
            dt: Time step [s]. If None, uses default_dt.
        
        Returns:
            List of predicted positions [[X1,Y1], [X2,Y2], ...] or None if track doesn't exist
        """
        if track_id not in self.trackers:
            return None
        
        if dt is None:
            dt = self.default_dt
        
        return self.trackers[track_id].predict_future(steps=steps, dt=dt)
    
    def get_state(self, track_id: int) -> Optional[np.ndarray]:
        """Get current state [X, Y, Vx, Vy] for a track."""
        if track_id not in self.trackers:
            return None
        return self.trackers[track_id].get_state()
    
    def get_velocity(self, track_id: int) -> Optional[np.ndarray]:
        """Get current velocity [Vx, Vy] for a track."""
        if track_id not in self.trackers:
            return None
        return self.trackers[track_id].get_velocity()
    
    def remove_track(self, track_id: int):
        """Remove a track from the manager."""
        if track_id in self.trackers:
            del self.trackers[track_id]
        if track_id in self.prev_time:
            del self.prev_time[track_id]
        if track_id in self.position_history:
            del self.position_history[track_id]
        
        if self.debug:
            print(f"[EKFManager] Removed track_id={track_id}")
    
    def get_all_track_ids(self) -> List[int]:
        """Get list of all active track IDs."""
        return list(self.trackers.keys())

