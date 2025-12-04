"""
2D Constant Velocity (CV) EKF Tracker for Pedestrian Tracking in BEV

This module implements an Extended Kalman Filter with a Constant Velocity motion model
for tracking pedestrians in Bird's Eye View (BEV) local coordinates.

State Vector:
    x = [X, Y, Vx, Vy]^T
    - X, Y: position in BEV local frame [m]
    - Vx, Vy: velocity in BEV local frame [m/s]

Motion Model:
    - Constant Velocity with white acceleration noise
    - State transition: F(dt) as defined in the CV model

Measurement Model:
    - Measures position only: z = [X_meas, Y_meas]^T in BEV local [m]
    - Measurement matrix: H = [[1,0,0,0], [0,1,0,0]]

Assumptions:
    - Input measurements are already converted from BEV pixels to BEV local [m]
    - Time step dt is variable and based on actual timestamps
"""

import numpy as np
from typing import Optional, Tuple


class EKFTrackerCV:
    """
    2D Constant Velocity EKF Tracker in BEV local coordinates [m].

    Args:
        initial_pos: Initial position (X, Y) in BEV local [m]
        initial_vel: Initial velocity (Vx, Vy) in BEV local [m/s].
                     If None, defaults to (0, 0)
        dt: Default time step [s]. Used when actual dt is not provided.
        sigma_a: Process noise - std dev of acceleration [m/s^2]
        sigma_meas: Measurement noise - std dev of position [m]
        sigma_pos_init: Initial position uncertainty [m]
        sigma_vel_init: Initial velocity uncertainty [m/s]
        use_distance_dependent_R: If True, scale R with distance
        distance_R_alpha: Scaling factor for distance-dependent R
        debug: Enable debug logging
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float],
        initial_vel: Optional[Tuple[float, float]] = None,
        dt: float = 0.1,
        sigma_a: float = 3.0,
        sigma_meas: float = 0.3,
        sigma_pos_init: float = 0.2,
        sigma_vel_init: float = 1.0,
        use_distance_dependent_R: bool = False,
        distance_R_alpha: float = 0.05,
        debug: bool = False,
    ):
        # Configuration parameters
        self.default_dt = float(dt)
        self.sigma_a = float(sigma_a)
        # measurement sigma (configurable)
        self.sigma_meas = float(sigma_meas)
        self.base_sigma_meas = float(self.sigma_meas)
        self.use_distance_dependent_R = use_distance_dependent_R
        self.distance_R_alpha = float(distance_R_alpha)
        self.debug = bool(debug)

        # State vector: [X, Y, Vx, Vy]^T in BEV local [m], [m/s]
        vx, vy = initial_vel if initial_vel is not None else (0.0, 0.0)
        self.x = np.array([
            float(initial_pos[0]),  # X [m]
            float(initial_pos[1]),  # Y [m]
            float(vx),              # Vx [m/s]
            float(vy),              # Vy [m/s]
        ], dtype=np.float64)

        # Initial covariance matrix P0
        # Larger uncertainty on velocity than position
        self.P = np.diag([
            sigma_pos_init ** 2,  # var(X)
            sigma_pos_init ** 2,  # var(Y)
            sigma_vel_init ** 2,  # var(Vx)
            sigma_vel_init ** 2,  # var(Vy)
        ]).astype(np.float64)

        # Measurement matrix H: we observe position only
        # z = H * x = [X, Y]^T
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R (will be updated if distance-dependent)
        self.R_base = np.eye(2, dtype=np.float64) * (self.base_sigma_meas ** 2)
        self.R = self.R_base.copy()

        # State transition matrix F and process noise Q
        # Will be computed dynamically based on dt
        self.F = np.eye(4, dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64)

        # Update F and Q with default dt
        self._update_F_Q(self.default_dt)

        if self.debug:
            print(f"[EKFTrackerCV] Initialized at pos={initial_pos}, vel={initial_vel}")
            print(f"  sigma_a={self.sigma_a}, sigma_meas={self.sigma_meas}")
            print(f"  base_sigma_meas={self.base_sigma_meas}")

    def _update_F_Q(self, dt: float):
        """
        Update state transition matrix F and process noise Q based on time step dt.

        F(dt) for Constant Velocity model:
            [[1, 0, dt, 0 ],
             [0, 1, 0,  dt],
             [0, 0, 1,  0 ],
             [0, 0, 0,  1 ]]

        Q(dt) for white acceleration noise (sigma_a):
            Q = [[ Q_1D,  0    ],
                 [ 0,     Q_1D ]]

            where Q_1D = sigma_a^2 * [[ dt^4/4,  dt^3/2 ],
                                       [ dt^3/2,  dt^2   ]]
        """
        dt = float(dt)

        # State transition matrix F(dt)
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Process noise Q(dt) - white acceleration model
        # Build Q_1D for one dimension
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q11 = dt4 / 4.0
        q12 = dt3 / 2.0
        q22 = dt2

        sigma_a_sq = self.sigma_a ** 2

        # Q matrix for 2D (x and y are independent)
        self.Q = sigma_a_sq * np.array([
            [q11, 0.0, q12, 0.0],  # X  row
            [0.0, q11, 0.0, q12],  # Y  row
            [q12, 0.0, q22, 0.0],  # Vx row
            [0.0, q12, 0.0, q22]   # Vy row
        ], dtype=np.float64)

        if self.debug:
            print(f"[EKFTrackerCV] Updated F and Q for dt={dt:.4f}")

    def _update_R(self):
        """
        Update measurement noise R based on current state (distance-dependent if enabled).

        If use_distance_dependent_R is True:
            sigma_eff = sigma_meas * (1 + alpha * range)
            where range = sqrt(X^2 + Y^2)
        """
        if not self.use_distance_dependent_R:
            self.R = self.R_base.copy()
            return

        # Compute distance from origin
        X, Y = self.x[0], self.x[1]
        range_m = np.sqrt(X**2 + Y**2)

        # Scale sigma_meas with distance
        scale_factor = 1.0 + self.distance_R_alpha * range_m
        sigma_eff = self.sigma_meas * scale_factor

        self.R = np.eye(2, dtype=np.float64) * (sigma_eff ** 2)

        if self.debug:
            print(f"[EKFTrackerCV] Updated R: range={range_m:.2f}m, sigma_eff={sigma_eff:.4f}")

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Prediction step of the Kalman filter.

        Args:
            dt: Time step [s]. If None, uses default_dt.

        Returns:
            Predicted position [X, Y] in BEV local [m]
        """
        if dt is None:
            dt = self.default_dt

        # Update F and Q if dt changed
        if abs(dt - self.default_dt) > 1e-6:
            self._update_F_Q(dt)

        # Predict state: x_pred = F * x
        self.x = self.F @ self.x

        # Predict covariance: P_pred = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update R for distance-dependent noise
        self._update_R()

        if self.debug:
            print(f"[EKFTrackerCV] Predict: dt={dt:.4f}, x_pred={self.x}")

        return self.x[:2].copy()

    def update(self, z: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Update step of the Kalman filter with a new measurement.

        Args:
            z: Measurement [X_meas, Y_meas] in BEV local [m]
            dt: Time step since last update [s]. If provided, calls predict(dt) first.

        Returns:
            Updated position [X, Y] in BEV local [m]
        """
        z = np.asarray(z, dtype=np.float64)

        # If dt is provided, do predict step first
        if dt is not None:
            self.predict(dt)

        # Ensure R is up-to-date (distance-dependent R or base R)
        self._update_R()

        # Innovation: y = z - H * x
        y = z - (self.H @ self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state: x = x + K * y
        self.x = self.x + K @ y

        # Update covariance: P = (I - K * H) * P
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        if self.debug:
            print(f"[EKFTrackerCV] Update: z={z}, x_upd={self.x}, innovation={y}")

        return self.x[:2].copy()

    def predict_future(self, steps: int = 10, dt: Optional[float] = None) -> list:
        """
        Predict future trajectory without updating the filter state.

        Args:
            steps: Number of future time steps to predict
            dt: Time step [s]. If None, uses default_dt.

        Returns:
            List of predicted positions [[X1, Y1], [X2, Y2], ...]
        """
        if dt is None:
            dt = self.default_dt

        # Backup current state
        x_backup = self.x.copy()
        P_backup = self.P.copy()

        # Update F and Q for prediction dt
        self._update_F_Q(dt)

        predictions = []
        for _ in range(steps):
            # Predict next state
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            predictions.append(self.x[:2].copy())

        # Restore original state
        self.x = x_backup
        self.P = P_backup

        return predictions

    def get_state(self) -> np.ndarray:
        """Get current state [X, Y, Vx, Vy]."""
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        """Get current position [X, Y] in BEV local [m]."""
        return self.x[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity [Vx, Vy] in BEV local [m/s]."""
        return self.x[2:].copy()

    def set_velocity(self, vx: float, vy: float):
        """
        Manually set velocity (useful for initialization or debugging).

        Args:
            vx: Velocity in X direction [m/s]
            vy: Velocity in Y direction [m/s]
        """
        self.x[2] = float(vx)
        self.x[3] = float(vy)

        if self.debug:
            print(f"[EKFTrackerCV] Velocity manually set to ({vx:.3f}, {vy:.3f}) m/s")

