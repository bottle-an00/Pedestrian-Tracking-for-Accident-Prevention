from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path
from collections import deque
from .bev_zone_risk_manager import PedestrianStateManager as BevZoneManager

@dataclass
class TrackState:
    id: int
    position: Tuple[float, float]
    ekf: object
    history: List[Tuple[float, float]] = field(default_factory=list)
    obs_count: int = 0
    missing_count: int = 0
    last_detected_frame: int = -1
    future_traj: Optional[List[Tuple[float, float]]] = None
    # timestamp-aware fields
    prev_time: Optional[float] = None
    # deque of (timestamp, (x,y)) for velocity initialization
    position_history: deque = field(default_factory=lambda: deque(maxlen=5))

    def update_with_detection(self, pos: Tuple[float, float], frame_idx: int):
        # pos: (x, y) in meters (vehicle-local or global depending on caller)
        self.position = (float(pos[0]), float(pos[1]))
        self.history.append(self.position)
        self.obs_count += 1
        self.missing_count = 0
        self.last_detected_frame = int(frame_idx)

        # update EKF with the new measurement if EKF supports update()
        try:
            # Prefer update(pos, dt) signature if available
            if hasattr(self.ekf, 'update'):
                try:
                    # attempt to pass frame-aligned dt if caller set it on TrackState
                    if hasattr(self, 'dt'):
                        # some EKF implementations accept (z, dt)
                        self.ekf.update(self.position, getattr(self, 'dt'))
                    else:
                        self.ekf.update(self.position)
                except TypeError:
                    # fallback: update(pos) only
                    self.ekf.update(self.position)
        except Exception:
            # EKF may expect a numpy array or different API; try converting
            try:
                if hasattr(self.ekf, 'update'):
                    try:
                        if hasattr(self, 'dt'):
                            self.ekf.update(np.asarray(self.position, dtype=np.float32), getattr(self, 'dt'))
                        else:
                            self.ekf.update(np.asarray(self.position, dtype=np.float32))
                    except TypeError:
                        self.ekf.update(np.asarray(self.position, dtype=np.float32))
            except Exception:
                # If EKF has different API, ignore here; user-supplied ekf_constructor
                # should provide an object compatible with .update() and .predict_future().
                pass

    def update_missing(self):
        self.missing_count += 1
        # keep history continuity by re-adding last known position
        if self.history:
            self.history.append(self.history[-1])

    def add_position_history(self, timestamp: Optional[float], pos: Tuple[float, float]):
        try:
            if timestamp is not None:
                self.position_history.append((float(timestamp), (float(pos[0]), float(pos[1]))))
        except Exception:
            # ignore malformed timestamps
            pass

    def estimate_initial_velocity(self, current_time: Optional[float], current_pos: Tuple[float, float], min_samples: int = 1, max_samples: int = 5) -> Tuple[float, float]:
        # Similar heuristic to EKFManager: average velocities from recent timestamped history
        hist = list(self.position_history)
        if not hist or len(hist) < min_samples:
            return (0.0, 0.0)

        # consider up to max_samples
        hist = hist[-max_samples:]
        velocities = []
        for i in range(len(hist)):
            t_prev, pos_prev = hist[i]
            if i == len(hist) - 1:
                if current_time is None:
                    continue
                t_next, pos_next = float(current_time), (float(current_pos[0]), float(current_pos[1]))
            else:
                t_next, pos_next = hist[i + 1]
            dt = float(t_next) - float(t_prev)
            if dt > 1e-6:
                vx = (pos_next[0] - pos_prev[0]) / dt
                vy = (pos_next[1] - pos_prev[1]) / dt
                velocities.append((vx, vy))

        if not velocities:
            return (0.0, 0.0)

        vx_avg = float(np.mean([v[0] for v in velocities]))
        vy_avg = float(np.mean([v[1] for v in velocities]))
        return (vx_avg, vy_avg)

    def ready_for_future_prediction(self, obs_len: int) -> bool:
        return self.obs_count >= obs_len


class PedestrianStateManager:
    def __init__(self, obs_len: int = 10, max_missing: int = 5, ekf_constructor: Callable[[], object] = None, dt: float = 1.0):
        """
        PedestrianStateManager expects incoming detection positions to already be local BEV
        coordinates (meters). Do not pass pixel coordinates to `update()`.
        """
        # configuration
        self.obs_len = int(obs_len)
        self.max_missing = int(max_missing)
        self.ekf_constructor = ekf_constructor
        self.dt = float(dt)

        # runtime state
        self.active_tracks: Dict[int, TrackState] = {}

        # provide a default EKF constructor if none supplied (use CA with tuned defaults)
        if self.ekf_constructor is None:
            try:
                from .ekf_tracker import EKFTracker
                default_q = 0.05
                default_R = 0.5
                default_dt = 0.1
                # keep default constructor but allow caller to override
                self.ekf_constructor = lambda q=default_q, R=default_R, dt=default_dt: EKFTracker(initial_pos=(0.0, 0.0), dt=dt, q=q, R_scale=R)
            except Exception:
                # if EKFTracker not available, leave ekf_constructor as None
                self.ekf_constructor = None

    def update(self, detections: List[Tuple[int, Tuple[float, float]]], frame_idx: int, timestamp: Optional[float] = None):
        """
        detections: list of (id, (x,y)) where (x,y) are positions in same frame coordinates
        frame_idx: current frame index
        """
        seen_ids = set()

        # Update or create tracks from detections
        for det in detections:
            tid, pos = det
            seen_ids.add(tid)
            if tid not in self.active_tracks:
                # construct EKF instance if constructor provided
                ekf = self.ekf_constructor() if self.ekf_constructor is not None else None
                # incoming pos MUST be local BEV meters (x_m, y_m)
                lx, ly = float(pos[0]), float(pos[1])
                ts = TrackState(id=int(tid), position=(float(lx), float(ly)), ekf=ekf)
                # set per-track prev_time and position_history if timestamp provided
                if timestamp is not None:
                    ts.prev_time = float(timestamp)
                    ts.add_position_history(timestamp, (lx, ly))
                self.active_tracks[int(tid)] = ts
            else:
                ts = self.active_tracks[int(tid)]

            # ts.update_with_detection expects local BEV meters
            ts.update_with_detection((float(pos[0]), float(pos[1])), frame_idx)
            # record timestamped history for velocity initialization
            ts.add_position_history(timestamp, (float(pos[0]), float(pos[1])))

            # If we have at least two observations, and EKF supports set_velocity, initialize velocity
            # If we have at least two timestamped observations, and EKF supports set_velocity, initialize velocity
            if ts.ekf is not None:
                try:
                    # attempt to estimate velocity from timestamped history first
                    init_vx, init_vy = ts.estimate_initial_velocity(current_time=timestamp, current_pos=(float(pos[0]), float(pos[1])), min_samples=1, max_samples=5)
                    if hasattr(ts.ekf, 'set_velocity'):
                        ts.ekf.set_velocity(init_vx, init_vy)
                except Exception:
                    # fallback to simple difference over manager dt if no timestamps
                    try:
                        if len(ts.history) >= 2:
                            x1, y1 = ts.history[-2]
                            x2, y2 = ts.history[-1]
                            vx = (x2 - x1) / float(self.dt)
                            vy = (y2 - y1) / float(self.dt)
                            if hasattr(ts.ekf, 'set_velocity'):
                                ts.ekf.set_velocity(vx, vy)
                    except Exception:
                        pass

        # For tracks not seen in this frame, increment missing_count
        to_remove = []
        for tid, ts in list(self.active_tracks.items()):
            if tid not in seen_ids:
                # advance EKF state for missing detection by calling predict(dt) if available
                try:
                    if ts.ekf is not None and hasattr(ts.ekf, 'predict'):
                        # compute dt from prev_time if available
                        try:
                            if ts.prev_time is not None and timestamp is not None:
                                dt_missing = float(timestamp) - float(ts.prev_time)
                                if dt_missing <= 0:
                                    dt_missing = self.dt
                            else:
                                dt_missing = self.dt
                            try:
                                ts.ekf.predict(dt=dt_missing)
                            except TypeError:
                                try:
                                    ts.ekf.predict(dt_missing)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

                ts.update_missing()
                if ts.missing_count > self.max_missing:
                    to_remove.append(tid)

        # Remove expired tracks
        for tid in to_remove:
            del self.active_tracks[tid]

    def get_future_ready_tracks(self) -> List[TrackState]:
        return [ts for ts in self.active_tracks.values() if ts.ready_for_future_prediction(self.obs_len)]

    def generate_future_predictions(self, T: int):
        for ts in self.get_future_ready_tracks():
            if ts.ekf is None:
                ts.future_traj = None
                continue

            # prefer forecast API: forecast_future or predict_future
            preds = None
            try:
                preds = ts.ekf.forecast_future(T)
            except Exception:
                try:
                    # if per-track prev_time exists, some EKF implementations accept dt; try manager dt fallback
                    preds = None
                    try:
                        if ts.prev_time is not None and hasattr(ts.ekf, 'predict_future'):
                            preds = ts.ekf.predict_future(T)
                        else:
                            preds = ts.ekf.predict_future(T)
                    except Exception:
                        preds = ts.ekf.predict_future(T)
                except Exception:
                    preds = None

            if preds is None:
                ts.future_traj = None
                continue

            # ensure list of tuples
            ts.future_traj = [(float(p[0]), float(p[1])) for p in preds]

        # return mapping for convenience
        return {ts.id: ts.future_traj for ts in self.get_future_ready_tracks()}

    def save_predictions_json(self, frame_idx: int, out_dir: str, eval_mode: bool = True, bev_transformer=None):
        """
        Dump per-frame prediction JSON in evaluation mode.

        Format:
        {
          "frame": frame_idx,
          "objects": [ {"id": id, "pos": {"x":..,"y":..}, "obs_count": n, "future": [{"x":..,"y":..}, ...] }, ... ]
        }

        Writes to: <out_dir>/pred_json/frame_000123.json
        """
        if not eval_mode:
            return None

        out_path = Path(out_dir) / "pred_json"
        out_path.mkdir(parents=True, exist_ok=True)

        entry = {"frame": int(frame_idx), "objects": [], "meta": {"coord": "world_m", "units": "meters"}}

        for ts in self.active_tracks.values():
            # ts.position is expected to be local BEV meters (x_m, y_m)
            lx, ly = float(ts.position[0]), float(ts.position[1])

            obj = {
                "id": int(ts.id),
                "pos": {"x": lx, "y": ly},
                "obs_count": int(ts.obs_count),
                "future": []
            }

            # if a bev_transformer (Homography or BevTransformer) is provided, also include BEV pixel indices
            try:
                if bev_transformer is not None:
                    # accept either object with .world_to_bev_img_pixel or .homography
                    bev_h = bev_transformer.homography if hasattr(bev_transformer, 'homography') else bev_transformer
                    u_bev, v_bev = bev_h.world_to_bev_img_pixel(lx, ly, flipped=True)
                    obj['pos_bev'] = {'u': int(u_bev), 'v': int(v_bev)}
            except Exception:
                pass

            if ts.ready_for_future_prediction(self.obs_len) and ts.future_traj:
                # future_traj entries are expected to be in local BEV meters
                obj["future"] = [{"x": float(p[0]), "y": float(p[1])} for p in ts.future_traj]
                # also add BEV pixel futures when possible
                try:
                    if bev_transformer is not None:
                        bev_h = bev_transformer.homography if hasattr(bev_transformer, 'homography') else bev_transformer
                        future_bev = []
                        for p in ts.future_traj:
                            u_b, v_b = bev_h.world_to_bev_img_pixel(p[0], p[1], flipped=True)
                            future_bev.append({'u': int(u_b), 'v': int(v_b)})
                        obj['future_bev'] = future_bev
                except Exception:
                    pass

            entry["objects"].append(obj)

        file_path = out_path / f"frame_{int(frame_idx):06d}.json"
        with open(file_path, "w") as jf:
            json.dump(entry, jf, indent=2)

        return str(file_path)
