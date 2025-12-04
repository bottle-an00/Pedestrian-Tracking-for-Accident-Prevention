import numpy as np
from collections import deque
from typing import Tuple, Optional

# Import the existing EKF implementation. Adjust path if package layout differs.
try:
    from src.trajectory.ekf_tracker_cv import EKFTrackerCV
except Exception:
    try:
        # fallback to relative import if executed as package
        from .ekf_tracker_cv import EKFTrackerCV
    except Exception:
        # final fallback: bare module name (user may adjust)
        from ekf_tracker_cv import EKFTrackerCV


def _cv_F_Q(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create CV state transition F and process noise Q for given dt and sigma_a."""
    dt = float(dt)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt

    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    q11 = dt4 / 4.0
    q12 = dt3 / 2.0
    q22 = dt2
    sigma_a_sq = float(sigma_a) ** 2
    Q = sigma_a_sq * np.array([
        [q11, 0.0, q12, 0.0],
        [0.0, q11, 0.0, q12],
        [q12, 0.0, q22, 0.0],
        [0.0, q12, 0.0, q22],
    ], dtype=np.float64)

    return F, Q


def rollout_future_from_state(x0: np.ndarray, P0: np.ndarray, sigma_a: float, steps: int, dt: float) -> np.ndarray:
    """
    Pure CV rollout from (x0, P0) returning positions (steps,2) without touching EKF.
    """
    x = x0.copy().astype(np.float64)
    P = P0.copy().astype(np.float64)

    positions = np.zeros((int(steps), 2), dtype=np.float64)
    for i in range(int(steps)):
        F, Q = _cv_F_Q(dt, sigma_a)
        x = F @ x
        P = F @ P @ F.T + Q
        positions[i, :] = x[:2].copy()

    return positions


class RealTimeEKFWithLagSmoothing:
    """Wrapper for EKFTrackerCV providing fixed-lag smoothing for future rollouts.

    - update(...) calls underlying EKF.update and stores filtered states
    - predict_future_with_smoothing runs RTS smoothing over history and rolls out from smoothed state
    """

    def __init__(self, ekf: EKFTrackerCV, lag_size: int = 5, future_horizon: int = 20, default_dt: float = 0.1, sigma_a: Optional[float] = None):
        self.ekf = ekf
        self.lag_size = int(lag_size)
        self.future_horizon = int(future_horizon)
        self.default_dt = float(default_dt)
        # sigma_a fallback: prefer provided, else read from ekf if available
        if sigma_a is None:
            self.sigma_a = float(getattr(self.ekf, 'sigma_a', 1.0))
        else:
            self.sigma_a = float(sigma_a)

        maxlen = self.lag_size + 1
        # history deques (most recent appended to right)
        self._x_hist = deque(maxlen=maxlen)  # each item shape (4,)
        self._P_hist = deque(maxlen=maxlen)  # each item shape (4,4)
        self._F_hist = deque(maxlen=maxlen)  # each item shape (4,4)
        self._Q_hist = deque(maxlen=maxlen)  # each item shape (4,4)

    def update(self, z: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Real-time EKF update: calls underlying EKF.update(z, dt) and appends filtered state to history.
        Returns current filtered position [X, Y].
        """
        if dt is None:
            dt = self.default_dt

        # forward-only update on underlying EKF
        # EKFTrackerCV.update accepts numpy array-like
        self.ekf.update(np.asarray(z, dtype=np.float64), dt=dt)

        # read filtered state and copies of covariance/transition
        x = np.asarray(self.ekf.get_state(), dtype=np.float64).copy()
        # read P, F, Q directly (public attributes)
        P = np.asarray(self.ekf.P, dtype=np.float64).copy()
        F = np.asarray(self.ekf.F, dtype=np.float64).copy()
        Q = np.asarray(self.ekf.Q, dtype=np.float64).copy()

        # append to history
        self._x_hist.append(x)
        self._P_hist.append(P)
        self._F_hist.append(F)
        self._Q_hist.append(Q)

        return x[:2].copy()

    def _run_fixed_lag_smoothing(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run RTS smoothing over the history window and return smoothed states and covariances.
        Returns xs_s (T,4) and Ps_s (T,4,4) where T = len(history)
        """
        T = len(self._x_hist)
        if T == 0:
            # no history: return current ekf state
            xcur = np.asarray(self.ekf.get_state(), dtype=np.float64).reshape(1, 4).copy()
            Pcur = np.asarray(getattr(self.ekf, 'P', np.eye(4)), dtype=np.float64).reshape(1, 4, 4).copy()
            return xcur, Pcur

        # Build arrays
        xs_f = np.vstack([np.asarray(x, dtype=np.float64).reshape(1, 4) for x in self._x_hist]).reshape(T, 4)
        Ps_f = np.stack([np.asarray(P, dtype=np.float64) for P in self._P_hist], axis=0)
        Fs = np.stack([np.asarray(F, dtype=np.float64) for F in self._F_hist], axis=0)
        Qs = np.stack([np.asarray(Q, dtype=np.float64) for Q in self._Q_hist], axis=0)

        xs_s = np.zeros_like(xs_f, dtype=np.float64)
        Ps_s = np.zeros_like(Ps_f, dtype=np.float64)

        # Initialize last element
        xs_s[-1] = xs_f[-1].copy()
        Ps_s[-1] = Ps_f[-1].copy()

        # Backward RTS pass
        for k in range(T - 2, -1, -1):
            Fk = Fs[k]
            Pk = Ps_f[k]
            # predicted covariance
            P_pred = Fk @ Pk @ Fk.T + Qs[k]
            # compute smoothing gain Ck; use pseudo-inverse for stability
            try:
                Ck = Pk @ Fk.T @ np.linalg.inv(P_pred)
            except Exception:
                Ck = Pk @ Fk.T @ np.linalg.pinv(P_pred)

            # state prediction from filtered state
            x_pred = Fk @ xs_f[k]
            xs_s[k] = xs_f[k] + Ck @ (xs_s[k + 1] - x_pred)
            Ps_s[k] = Pk + Ck @ (Ps_s[k + 1] - P_pred) @ Ck.T

        return xs_s, Ps_s

    def predict_future_with_smoothing(self, horizon_steps: Optional[int] = None, dt: Optional[float] = None, sigma_a: Optional[float] = None) -> np.ndarray:
        """
        Run fixed-lag smoothing over recent history and rollout CV future from smoothed current state.
        Returns positions array shape (horizon_steps, 2).
        """
        if horizon_steps is None:
            horizon_steps = int(self.future_horizon)
        if dt is None:
            dt = float(self.default_dt)
        if sigma_a is None:
            sigma_a = float(getattr(self.ekf, 'sigma_a', self.sigma_a))

        # run smoothing on copies
        xs_s, Ps_s = self._run_fixed_lag_smoothing()

        # take last smoothed state
        x0_smooth = xs_s[-1].copy()
        P0_smooth = Ps_s[-1].copy()

        # rollout CV from smoothed state
        future_pos = rollout_future_from_state(x0_smooth, P0_smooth, sigma_a, int(horizon_steps), float(dt))
        return future_pos

    # convenience getters
    def get_current_filtered_state(self) -> np.ndarray:
        return np.asarray(self.ekf.get_state(), dtype=np.float64).copy()

    def get_current_filtered_position(self) -> np.ndarray:
        return np.asarray(self.ekf.get_position(), dtype=np.float64).copy()
