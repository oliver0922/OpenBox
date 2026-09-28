"""Detection-to-track assignment for AB3DMOT.

The affinity matrix (rows = detections, columns = tracks) is float32 ``giou_3d`` or
negative ``dist3d``; the assignment is Hungarian (``linear_sum_assignment`` on the negated
matrix) or greedy (ascending ``np.argsort`` of the negated matrix, first come first served),
and assigned pairs whose affinity is strictly below the threshold are rejected.  The
float32 rounding before the threshold test and the order of the unmatched indices
(unassigned first, rejected pairs after, in assignment order) decide the track ids.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .dist_metrics import dist3d, giou_3d


def greedy_matching(cost_matrix):
    """Pair rows and columns in ascending cost order, each at most once.

    cost_matrix: float (D, T) -> int (K, 2) ``[det, trk]`` pairs.
    """
    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)


def data_association(dets, trks, metric, threshold, algm):
    """Assign ``Box3D`` detections to ``Box3D`` tracks.

    ``metric`` is ``'giou_3d'`` or ``'dist_3d'``, ``algm`` is ``'hungar'`` or ``'greedy'``.
    Returns ``(matches, unmatched_dets, unmatched_trks)``: a (K, 2) int array of
    ``[det_index, trk_index]`` and two index arrays.
    """
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.array([], dtype=int)
    if len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.array([], dtype=int), np.arange(len(trks))

    aff_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            if metric == 'giou_3d':
                aff_matrix[d, t] = giou_3d(det, trk)
            elif metric == 'dist_3d':
                aff_matrix[d, t] = -dist3d(det, trk)
            else:
                raise ValueError('unsupported affinity metric: %s' % metric)

    if algm == 'hungar':
        row_ind, col_ind = linear_sum_assignment(-aff_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
    elif algm == 'greedy':
        matched_indices = greedy_matching(-aff_matrix)
    else:
        raise ValueError('unsupported association algorithm: %s' % algm)

    unmatched_dets = []
    for d, det in enumerate(dets):
        if (d not in matched_indices[:, 0]):
            unmatched_dets.append(d)
    unmatched_trks = []
    for t, trk in enumerate(trks):
        if (t not in matched_indices[:, 1]):
            unmatched_trks.append(t)

    matches = []
    for m in matched_indices:
        if (aff_matrix[m[0], m[1]] < threshold):
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
