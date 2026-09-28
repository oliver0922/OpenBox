"""HDBSCAN clustering of the aggregated static SAM points and cluster-to-instance matching.

Stage-1 "cluster and match" step: :class:`HDBSCANClusterer` clusters the SDF-filtered,
multi-frame aggregated static points (xyz only); the caller drops the noise
points with ``inlier_mask`` and :class:`ClusterMatcher` relabels every cluster to
the majority SAM instance id of its points, so the boxes generated downstream
are per SAM instance (clusters sharing a majority id are merged).

``hdbscan.HDBSCAN`` is deterministic for a fixed library build but not
bit-reproducible across hdbscan / numpy versions, and the majority vote
(``Counter.most_common``) breaks ties by first occurrence, so the point order of
the inputs is part of the numerics.
"""
from collections import Counter

import hdbscan
import numpy as np


class HDBSCANClusterer:
    """``hdbscan.HDBSCAN`` with the fixed release keywords; :meth:`fit` sets ``labels`` and ``inlier_mask``."""

    def __init__(self, cfg):
        """cfg: the ``hdbscan`` config section (min_cluster_size / min_samples / leaf_size: int)."""
        min_cluster_size, min_samples, leaf_size = cfg.min_cluster_size, cfg.min_samples, cfg.leaf_size
        self.clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                         gen_min_span_tree=True, leaf_size=leaf_size,
                                         metric='euclidean', min_cluster_size=min_cluster_size,
                                         min_samples=min_samples, cluster_selection_method='eom')
        self.labels = None
        self.inlier_mask = None

    def fit(self, points):
        """Cluster the (N, 3) ``points``: ``labels`` is (N,) int64 (-1 = noise), ``inlier_mask`` = ``labels >= 0``."""
        self.labels = self.clusterer.fit(points).labels_.copy()
        self.inlier_mask = self.labels >= 0
        return self.labels


class ClusterMatcher:
    """Relabel every HDBSCAN cluster to the majority SAM instance id of its points.

    ``instance_id_list`` (M,) and ``hdbscan_idxs`` (M,) are the SAM instance id
    and the HDBSCAN label of every inlier point, aligned and in their original
    order (the majority vote breaks ties by first occurrence).
    """

    def __init__(self, instance_id_list, hdbscan_idxs):
        """instance_id_list: (M,) SAM instance ids; hdbscan_idxs: (M,) int cluster labels (aligned)."""
        self.instance_id_list = instance_id_list
        self.hdbscan_idxs = hdbscan_idxs

    def match(self):
        """(M,) majority SAM instance id of the cluster each point belongs to (dtype of ``instance_id_list``)."""
        cluster_to_instance_id = {}
        for cluster_label in np.unique(self.hdbscan_idxs):
            member_instance_ids = self.instance_id_list[self.hdbscan_idxs == cluster_label]
            cluster_to_instance_id[cluster_label], _ = Counter(member_instance_ids).most_common(1)[0]
        return np.array(list(map(lambda label: cluster_to_instance_id[label], self.hdbscan_idxs)))
