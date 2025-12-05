import numpy as np
from typing import Optional, List, Tuple


class EKFTrackerCA:
    """
    2D Constant Acceleration Extended Kalman Filter (linearized here since H is linear).

    State vector (6): [X, Y, Vx, Vy, Ax, Ay]^T

    Measurement: z = [X_meas, Y_meas]^T

    This class mirrors the interface of the existing CV EKF so it can be
    substituted in the pipeline where a predict/update API is expected.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float] = (0.0, 0.0),
        initial_vel: Optional[Tuple[float, float]] = None,
        initial_acc: Optional[Tuple[float, float]] = None,
        dt: float = 0.1,
        sigma_a_rw: float = 1.0,
        sigma_meas: float = 0.3,
        sigma_pos_init: float = 0.2,
        sigma_vel_init: float = 1.0,
        sigma_acc_init: float = 2.0,
        debug: bool = False,
    ):
        self.default_dt = float(dt)
        self.sigma_a_rw = float(sigma_a_rw)  # random-walk std of acceleration
        self.sigma_meas = float(sigma_meas)
        self.debug = bool(debug)

        vx, vy = (initial_vel if initial_vel is not None else (0.0, 0.0))
        ax, ay = (initial_acc if initial_acc is not None else (0.0, 0.0))

        # state: [X, Y, Vx, Vy, Ax, Ay]
        self.x = np.array([
            float(initial_pos[0]),
            float(initial_pos[1]),
            float(vx),
            float(vy),
            float(ax),
            float(ay),
        ], dtype=np.float64)

        # initial covariance
        self.P = np.diag([
            sigma_pos_init ** 2,
            sigma_pos_init ** 2,
            sigma_vel_init ** 2,
            sigma_vel_init ** 2,
            sigma_acc_init ** 2,
            sigma_acc_init ** 2,
        ]).astype(np.float64)

        # measurement matrix H (2x6) - we observe position only
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)

        # measurement noise
        self.R = np.eye(2, dtype=np.float64) * (self.sigma_meas ** 2)

        # placeholders for F and Q (updated on predict)
        self.F = np.eye(6, dtype=np.float64)
        self.Q = np.zeros((6, 6), dtype=np.float64)

        # initialize F/Q with default dt
        self._update_F_Q(self.default_dt)

        if self.debug:
            print(f"[EKFTrackerCA] init pos={initial_pos} vel={(vx,vy)} acc={(ax,ay)}")

    def _F_matrix(self, dt: float) -> np.ndarray:
        dt = float(dt)
        dt2 = 0.5 * (dt ** 2)
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0, dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return F

    def _Q_matrix(self, dt: float) -> np.ndarray:
        dt = float(dt)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # Q for 1D state [pos, vel, acc]
        q11 = dt4 / 4.0
        q12 = dt3 / 2.0
        q13 = dt2 / 2.0
        q22 = dt2
        q23 = dt
        q33 = 1.0

        Q1D = np.array([
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33],
        ], dtype=np.float64)

        # scale by sigma_a_rw^2
        Q1D = (self.sigma_a_rw ** 2) * Q1D

        # block diag for 2D: [X-block, Y-block]
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = Q1D
        Q[3:6, 3:6] = Q1D
        # Note: ordering here matches state [X, Y, Vx, Vy, Ax, Ay] if we map
        # positions/velocities/acc appropriately. We constructed Q1D for [pos,vel,acc]
        # but need to place terms to align with our state indexing. Above mapping
        # uses contiguous blocks; for our state ordering we must permute.

        # The chosen state ordering is [X, Y, Vx, Vy, Ax, Ay]. To build Q correctly
        # we will assemble Q by placing Q1D into the (pos,vel,acc) slots for X and Y.
        Q_correct = np.zeros((6, 6), dtype=np.float64)
        # X-block -> indices (0:X,2:Vx,4:Ax) mapping [0,2,4]
        idx_x = [0, 2, 4]
        idx_y = [1, 3, 5]
        for i in range(3):
            for j in range(3):
                Q_correct[idx_x[i], idx_x[j]] = Q1D[i, j]
                Q_correct[idx_y[i], idx_y[j]] = Q1D[i, j]

        return Q_correct

    def _update_F_Q(self, dt: float):
        self.F = self._F_matrix(dt)
        self.Q = self._Q_matrix(dt)

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        if dt is None:
            dt = self.default_dt
        # update F and Q for dt
        self._update_F_Q(dt)

        # Predict state and covariance
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if self.debug:
            print(f"[CAEKF2D] Predict dt={dt:.4f} x_pred={self.x}")

        # return predicted position
        return self.x[0:2].copy()

    def update(self, z: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)

        # Optionally call predict with dt first if provided
        if dt is not None:
            self.predict(dt)

        # Update measurement noise R (kept constant here)
        self.R = np.eye(2, dtype=np.float64) * (self.sigma_meas ** 2)

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.P.shape[0], dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        if self.debug:
            print(f"[CAEKF2D] Update z={z}, x_upd={self.x}")

        return self.x[0:2].copy()

    def predict_future(self, steps: int = 10, dt: Optional[float] = None) -> List[np.ndarray]:
        if dt is None:
            dt = self.default_dt

        # backup
        x_b = self.x.copy()
        P_b = self.P.copy()

        # ensure F/Q for dt
        self._update_F_Q(dt)

        preds: List[np.ndarray] = []
        for _ in range(steps):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            preds.append(self.x[0:2].copy())

        # restore
        self.x = x_b
        self.P = P_b

        return preds

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        return self.x[0:2].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def set_velocity(self, vx: float, vy: float):
        self.x[2] = float(vx)
        self.x[3] = float(vy)

    def set_acceleration(self, ax: float, ay: float):
        self.x[4] = float(ax)
        self.x[5] = float(ay)


def make_ca_ekf(*args, **kwargs) -> "EKFTrackerCA":
    """Helper factory for convenience."""
    return EKFTrackerCA(*args, **kwargs)
