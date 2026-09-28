"""AB3DMOT (Xinshuo Weng): a baseline 3-D multi-object tracker, Waymo path.

Per frame ``track`` runs, in this fixed order: count the frame; predict every live track
one step and wrap its yaw into ``[-pi, pi)``; associate (``matching.data_association``);
Kalman-update the matched tracks after making predicted and observed yaw acute; spawn one
track per unmatched detection, ids handed out in association order; emit and prune,
iterating the track list in REVERSED order.  The step order, the reversed emission and
the id order are all part of the released numerics.
"""
import numpy as np

from .box import Box3D
from .kalman_filter import KalmanTrack
from .matching import data_association


def wrap_angle(theta):
    """One ``2*pi`` step towards ``[-pi, pi)``; values beyond ``3*pi`` stay out of range.

    ``theta`` may be a 1-element view into a filter state, in which case the in-place
    operators also write the view (harmless: callers assign the result back to that slot).
    """
    if theta >= np.pi:
        theta -= np.pi * 2
    if theta < -np.pi:
        theta += np.pi * 2

    return theta


def correct_orientation(theta_pre, theta_obs):
    """Wrap both yaws, then flip / unwrap the predicted one so the two differ by less than 90 degrees."""
    theta_pre = wrap_angle(theta_pre)
    theta_obs = wrap_angle(theta_obs)

    if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
        theta_pre += np.pi
        theta_pre = wrap_angle(theta_pre)

    if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
        if theta_obs > 0:
            theta_pre += np.pi * 2
        else:
            theta_pre -= np.pi * 2

    return theta_pre, theta_obs


class AB3DMOT:
    """Single-category tracker.

    ``cfg`` is the category's ``tracking.categories[<cat>]`` section (``algorithm``,
    ``metric``, ``threshold``, ``min_hits``, ``max_age``), ``kalman_cfg`` the
    ``tracking.kalman`` section and ``first_track_id`` the first id handed out;
    ``next_track_id`` holds the advanced counter afterwards.
    """

    def __init__(self, cfg, kalman_cfg, first_track_id):
        """cfg / kalman_cfg: config sections, first_track_id: int (see the class docstring)."""
        self.algm = cfg.algorithm
        self.metric = cfg.metric
        self.thres = cfg.threshold
        if cfg.metric == 'dist_3d':
            self.thres *= -1  # a distance threshold; the affinity is the negated distance
        self.max_age = cfg.max_age
        self.min_hits = cfg.min_hits
        self.kalman_cfg = kalman_cfg

        self.trackers = []
        self.frame_count = 0
        self.next_track_id = first_track_id

    def track(self, dets):
        """Process one frame; call once per frame in sequence order, even without detections.

        ``dets`` is an (N, 7) ``[x, y, z, l, w, h, yaw]`` array in one fixed frame.  Returns
        the (M, 8) float64 ``[x, y, z, l, w, h, yaw, track_id]`` rows emitted this frame, or
        ``np.zeros((0, 7))`` when nothing is emitted.
        """
        self.frame_count += 1

        dets = [Box3D(*det[:7]) for det in dets]

        trks = []
        for trk in self.trackers:
            trk.kf.predict()
            trk.kf.x[3] = wrap_angle(trk.kf.x[3])

            trk.time_since_update += 1
            state = trk.kf.x.reshape((-1))[:7]  # x, y, z, yaw, l, w, h
            trks.append(Box3D(state[0], state[1], state[2], state[4], state[5], state[6], state[3]))

        matched, unmatched_dets, unmatched_trks = \
            data_association(dets, trks, self.metric, self.thres, self.algm)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                assert len(d) == 1, 'track %d matched %d detections' % (trk.id, len(d))

                trk.time_since_update = 0
                trk.hits += 1

                det = dets[d[0]]
                bbox3d = np.array([det.x, det.y, det.z, det.ry, det.l, det.w, det.h])
                trk.kf.x[3], bbox3d[3] = correct_orientation(trk.kf.x[3], bbox3d[3])

                trk.kf.update(bbox3d)

                trk.kf.x[3] = wrap_angle(trk.kf.x[3])

        for i in unmatched_dets:
            det = dets[i]
            self.trackers.append(KalmanTrack(np.array([det.x, det.y, det.z, det.ry, det.l, det.w, det.h]),
                                             self.next_track_id, self.kalman_cfg))
            self.next_track_id += 1

        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                state = trk.kf.x[:7].reshape((7, ))  # x, y, z, yaw, l, w, h
                d = np.array([state[0], state[1], state[2], state[4], state[5], state[6], state[3]])
                results.append(np.concatenate((d, [float(trk.id)])).reshape(-1))
            num_trks -= 1

            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(num_trks)

        if len(results) > 0:
            return np.array(results)
        return np.zeros((0, 7))
