"""3-D box container and corner geometry for the AB3DMOT tracker.

``corners_camcoord`` is the original KITTI camera-frame formulation (rotation about y,
height along -y from the box origin, footprint in the x-z plane) applied unchanged to
lidar-frame boxes.  The affinities -- and so the released track ids -- depend on it, so
it must not be "fixed".
"""
import numpy as np


class Box3D:
    """One box: centre ``(x, y, z)``, size ``(l, w, h)``, heading ``ry``; the corners are cached."""

    def __init__(self, x, y, z, l, w, h, ry):
        """x/y/z: float centre; l/w/h: float size; ry: float heading (rad)."""
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.ry = ry
        self.corners_3d_cam = None

    def corners_camcoord(self):
        """Eight corners (8, 3), KITTI camera convention: 0-3 on the y = 0 face, 4-7 on y = -h.

                    1 -------- 0
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7
        """
        if self.corners_3d_cam is not None:
            return self.corners_3d_cam

        c = np.cos(self.ry)
        s = np.sin(self.ry)
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
        l, w, h = self.l, self.w, self.h

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + self.x
        corners_3d[1, :] = corners_3d[1, :] + self.y
        corners_3d[2, :] = corners_3d[2, :] + self.z
        corners_3d = np.transpose(corners_3d)
        self.corners_3d_cam = corners_3d

        return corners_3d
