"""Nearest-neighbour masks for the static/dynamic split of SAM points and for
stripping SAM-covered points out of the full LiDAR cloud.

Every ``sq_threshold`` is compared against a SQUARED L2 distance: the verified
release run used faiss ``IndexFlatL2`` on the GPU, which returns squared float32
distances, so 0.15 means an effective radius of sqrt(0.15) = 0.387 m.

The GPU faiss path is the reference implementation and is required for
byte-identical reproduction (install ``faiss-gpu-cu12``; import torch first).
Without a CUDA-capable faiss the code falls back to a ``cKDTree`` whose float64
distance is squared before the comparison -- numerically equivalent except for
points exactly on the threshold boundary, which can flip the odd box
(observed: 1 box of 10k on one of 20 scenes).  Do not "fix" either path to a
plain Euclidean comparison: that changes the classification everywhere.
"""
import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401  -- must precede faiss: both use the nvidia-* wheels torch installed
    import faiss
    _FAISS_GPU = hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
except Exception as error:  # noqa: BLE001
    faiss = None
    _FAISS_GPU = False
    logger.warning('faiss unavailable (%r): cKDTree fallback, borderline points can differ from the release run',
                   error)
if faiss is not None and not _FAISS_GPU:
    logger.warning('faiss has no GPU: cKDTree fallback, borderline points can differ from the release run')

_gpu_index = None


def _faiss_sq_nn(pool_pc, query_pc):
    """faiss-GPU k=1 search: float32 SQUARED distances and indices, exactly as the release run."""
    global _gpu_index
    if _gpu_index is None:
        _gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    _gpu_index.reset()
    _gpu_index.add(np.ascontiguousarray(pool_pc, dtype=np.float32))
    dist, index = _gpu_index.search(np.ascontiguousarray(query_pc, dtype=np.float32), 1)
    return dist.flatten(), index.flatten()


def _kdtree_sq_nn(pool_pc, query_pc):
    """cKDTree fallback k=1 search: pool_pc (P, 3), query_pc (Q, 3) ->
    ((Q,) float64 SQUARED distances, (Q,) int indices into pool_pc).
    """
    tree = cKDTree(np.ascontiguousarray(pool_pc, dtype=np.float32))
    dist, index = tree.query(np.ascontiguousarray(query_pc, dtype=np.float32), k=1)
    return (dist.astype(np.float64) ** 2).flatten(), index.flatten()


def _sq_nn(pool_pc, query_pc):
    """Nearest neighbour of each query point: pool_pc (P, 3), query_pc (Q, 3) ->
    ((Q,) squared distances, (Q,) indices); faiss-GPU when available, else cKDTree.
    """
    if _FAISS_GPU:
        return _faiss_sq_nn(pool_pc, query_pc)
    return _kdtree_sq_nn(pool_pc, query_pc)


def is_pool(pool_pc, query_pc, sq_threshold):
    """(Q,) mask over ``query_pc``: True where the nearest ``pool_pc`` point is within ``sq_threshold``."""
    sq_dist, _ = _sq_nn(pool_pc, query_pc)
    return sq_dist < sq_threshold


def is_pool_for_heavy_query(pool_pc, query_pc, sq_threshold):
    """(Q,) mask over ``query_pc``: True for the query points that are the nearest
    neighbour of at least one ``pool_pc`` point within ``sq_threshold``.

    The index is built on ``query_pc`` and each pool point looks up its single
    nearest query point, so only that one query point per pool point can be
    flagged -- not every query point inside the radius.
    """
    sq_dist, nearest_query_index = _sq_nn(query_pc, pool_pc)
    hit_query_index = nearest_query_index[sq_dist < sq_threshold]
    return np.isin(np.arange(len(query_pc)), hit_query_index)


def get_target_removed_heavy_pc(source_pc, target_pc, sq_threshold):
    """``source_pc`` minus the single nearest source point (within ``sq_threshold``) of each target point."""
    return source_pc[~is_pool_for_heavy_query(target_pc, source_pc, sq_threshold)]
