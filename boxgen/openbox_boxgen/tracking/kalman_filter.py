"""Per-track constant-velocity Kalman filter on ``filterpy.kalman.KalmanFilter``.

State (10): ``x, y, z, yaw, l, w, h, dx, dy, dz`` -- constant velocity on the centre, yaw
and size constant; measurement (7): ``x, y, z, yaw, l, w, h``.  One step is one frame.
filterpy's own ``predict`` / ``update`` (Joseph-form covariance update, ``numpy.linalg.inv``,
float64) are part of the released numerics, so the filter is used as-is.
"""
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanTrack:
    """One tracked object: the Kalman filter plus AB3DMOT's hit / age bookkeeping.

    ``initial_state`` is the (7,) ``[x, y, z, yaw, l, w, h]`` of the spawning detection and
    ``cfg`` the ``tracking.kalman`` section (covariance and process-noise scales).
    """

    def __init__(self, initial_state, track_id, cfg):
        """initial_state: float (7,) [x, y, z, yaw, l, w, h]; track_id: int; cfg: ``tracking.kalman`` section."""
        self.id = track_id
        self.hits = 1
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # x' = x + dx, y' = y + dy, z' = z + dz; every other state is carried over.
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # The measurement is the first seven states.
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # Velocity block first, then the whole matrix (R stays filterpy's identity).
        self.kf.P[7:, 7:] *= cfg.velocity_initial_covariance_scale
        self.kf.P *= cfg.initial_covariance_scale
        self.kf.Q[7:, 7:] *= cfg.velocity_process_noise_scale

        self.kf.x[:7] = initial_state.reshape((7, 1))
