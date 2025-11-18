import numpy as np

class EKFTracker:
    def __init__(self, initial_pos, dt=0.05):
        self.dt = dt

        self.x = np.array([initial_pos[0], initial_pos[1],
                           0.0, 0.0,
                           0.0, 0.0], dtype=np.float32)

        self.P = np.eye(6, dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.R = np.eye(2, dtype=np.float32) * 0.2

        self._update_F_Q(dt)

    def _update_F_Q(self, dt):
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1,  0, dt, 0],
            [0, 0, 0,  1, 0, dt],
            [0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 1]
        ], dtype=np.float32)

        q = 1.5
        self.Q = q * np.array([
            [dt**4/4,    0,         dt**3/2,   0,        dt**2/2, 0],
            [0,       dt**4/4,     0,         dt**3/2,   0,        dt**2/2],
            [dt**3/2,   0,         dt**2,      0,        dt,       0],
            [0,       dt**3/2,     0,         dt**2,      0,       dt],
            [dt**2/2,   0,         dt,         0,        1,        0],
            [0,       dt**2/2,     0,         dt,        0,        1],
        ], dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z):
        z = np.asarray(z, dtype=np.float32)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2]

    def predict_future(self, steps=10):
        preds = []
        x_backup = self.x.copy()
        P_backup = self.P.copy()

        for _ in range(steps):
            self.predict()
            preds.append(self.x[:2].copy())

        self.x = x_backup
        self.P = P_backup

        return preds
