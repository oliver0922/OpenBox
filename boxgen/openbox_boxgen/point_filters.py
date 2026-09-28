"""Point-level filters of the pseudo-box pipeline.

``dynamic_removal_with_ppscore`` splits one frame's refined SAM points into a
static and a dynamic set (ego frame): a no-aug SAM point is static when a
static LiDAR point (ppscore >= threshold) has it as nearest neighbour AND it
lies on the static SDF surface; an instance follows the majority of its
points.  Deformable classes are forced static for the deformable box path.

``SDFPointFilter`` runs once per scene on the aggregated static points
(world frame, deformable instances removed): every point is tied to its
nearest mesh vertex, vertices dominated by non-SAM points are discarded, and
per instance only the points on the largest connected triangle cluster of the
instance's sub-mesh survive.  Instances that lose all their points are
diverted to the single-frame box path together with their frame indices.

The ``*_sq_threshold`` values are compared with SQUARED L2 distances (the
faiss IndexFlatL2 semantics of the release, see ``neighbors.py``);
``point_to_sdf`` and ``max_cluster_filtering`` use plain Euclidean cKDTree
distances.  Instances are always visited in ``np.unique`` (sorted) order and
their points concatenated in that order: downstream HDBSCAN is order-sensitive.
"""
import logging

import numpy as np
from scipy.spatial import cKDTree

from .neighbors import is_pool, is_pool_for_heavy_query

logger = logging.getLogger(__name__)


# Sentinel vertex index: "no mesh vertex within the surface radius".
NO_VERTEX = -1


def point_to_sdf(vert, pcd, radius):
    """(N,) int64 index into ``vert`` of the nearest vertex of each ``pcd`` point,
    ``NO_VERTEX`` where it is farther than ``radius`` (plain Euclidean, not squared).

    Every point is an independent exact nearest-neighbour query, so the parallel
    search and the ``distance_upper_bound`` pruning (points beyond ``radius`` come
    back as inf and map to ``NO_VERTEX`` anyway) give the same result as a plain
    one-by-one search.
    """
    if len(pcd) == 0:
        return np.array([], dtype=np.int64)
    dist, nearest_vertex = cKDTree(vert).query(pcd, k=1, distance_upper_bound=radius, workers=-1)
    return np.where(dist < radius, nearest_vertex, NO_VERTEX)


def largest_mesh_patch(mesh, vertex_ids):
    """Sub-mesh of ``mesh`` spanned by ``vertex_ids`` and the vertices of its largest connected triangle patch.

    Returns ``(sub_mesh, patch_vertices)`` or ``None`` when the sub-mesh has no
    complete triangle.  ``sub_mesh`` is the FULL sub-mesh (``select_by_index`` of
    the unique ids) with vertex and triangle normals computed -- the open3d call
    sequence of the release; ``patch_vertices`` are the float64 (V, 3) vertices
    of the patch with the most triangles (first patch on a tie) in
    ``np.unique(triangle indices)`` order.
    """
    sub_mesh = mesh.select_by_index(np.unique(vertex_ids))
    sub_mesh.compute_vertex_normals()
    sub_mesh.compute_triangle_normals()

    triangle_cluster_ids = np.asarray(sub_mesh.cluster_connected_triangles()[0])  # does not modify the mesh
    if len(triangle_cluster_ids) == 0:
        return None

    largest_cluster = np.argmax(np.bincount(triangle_cluster_ids))
    patch_triangles = np.asarray(sub_mesh.triangles)[triangle_cluster_ids == largest_cluster]
    patch_vertices = np.array(sub_mesh.vertices)[np.unique(patch_triangles)]
    return sub_mesh, patch_vertices


def points_rigid_transform(cloud, pose):
    """Apply the 4x4 homogeneous ``pose`` to the xyz columns of ``cloud``; returns (N, 3) float32.

    The points are truncated to float32 and multiplied as ``pose @ points.T`` through
    ``np.asmatrix`` (float64 accumulation) before the cast back to float32 -- rewriting this
    as ``cloud @ R.T + t`` changes the low-order bits.  An empty input is returned as-is.
    """
    cloud = np.array(cloud)
    if cloud.shape[0] == 0:
        return cloud
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    pose_mat = np.asmatrix(pose)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.asmatrix(mat)
    transformed_mat = pose_mat * mat.T
    transformed = np.array(transformed_mat.T, dtype=np.float32)
    return transformed[:, 0:3]


def dynamic_removal_with_ppscore(cfg, sam_pc, sam_color, sam_id, no_aug_sam_pc, H, lidar_pc, pose,
                                 vert, instance_cls_dict, deformable_classes,
                                 ppscore_threshold, sq_threshold, get_dynamics):
    """Split one frame's refined SAM points into static and dynamic sets.

    The vote is taken on the no-aug SAM points (``no_aug_sam_pc``, (K, 4) xyz +
    instance id) while the emitted points are the refined SAM points
    (``sam_pc``, ``sam_color``, ``sam_id``) of the same instance id; ids
    present only in ``sam_pc`` are dropped.  ``H`` is the ppscore of every
    ``lidar_pc`` point, ``pose`` the ego-to-world matrix and ``vert`` the mesh
    vertices in the world frame.

    ``cfg`` is the ``sdf_filter`` section (``sam_to_surface_sq_threshold``,
    ``static_instance_min_fraction``); ``ppscore_threshold`` and ``sq_threshold``
    (SAM-to-static-LiDAR, squared) are the ``static_dynamic_split`` values.

    Returns ``(static_pc, static_color, static_id)`` and, when ``get_dynamics``,
    the same triple of the dynamic set appended.  Empty sets are float64
    zero-size arrays of shape (0, 3) / (0, 3) / (0, 1).
    """
    posed_vert = points_rigid_transform(vert, np.linalg.inv(pose))
    static_lidar = lidar_pc[H >= ppscore_threshold]
    no_aug_xyz = no_aug_sam_pc[:, :3]

    sam_near_static_lidar = is_pool_for_heavy_query(static_lidar, no_aug_xyz, sq_threshold)
    sam_near_surface = is_pool(posed_vert, no_aug_xyz, cfg.sam_to_surface_sq_threshold)
    sam_is_static = np.logical_and(sam_near_static_lidar, sam_near_surface)

    static_points, static_colors, static_ids = [], [], []
    dynamic_points, dynamic_colors, dynamic_ids = [], [], []

    for instance_id in np.unique(no_aug_sam_pc[:, 3]):
        selected = sam_id.flatten() == instance_id
        xyz = sam_pc[selected][:, :3]
        color = sam_color[selected]
        ids = instance_id * np.ones((np.sum(selected), 1))

        if instance_id in instance_cls_dict and instance_cls_dict[instance_id] in deformable_classes:
            static_points.append(xyz)
            static_colors.append(color)
            static_ids.append(ids)
            continue

        in_instance = no_aug_sam_pc[:, 3] == instance_id
        static_sam_size = np.sum(np.logical_and(in_instance, sam_is_static))
        dynamic_sam_size = np.sum(np.logical_and(in_instance, ~sam_is_static))

        if static_sam_size / (dynamic_sam_size + static_sam_size + 1e-6) > cfg.static_instance_min_fraction:
            static_points.append(xyz)
            static_colors.append(color)
            static_ids.append(ids)
        else:
            dynamic_points.append(xyz)
            dynamic_colors.append(color)
            dynamic_ids.append(ids)

    if len(static_points) == 0:
        static_set = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 1)))
    else:
        static_set = (np.concatenate(static_points, axis=0),
                      np.concatenate(static_colors, axis=0),
                      np.concatenate(static_ids, axis=0))
    if not get_dynamics:
        return static_set
    if len(dynamic_points) == 0:
        return static_set + (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 1)))
    return static_set + (np.concatenate(dynamic_points, axis=0),
                         np.concatenate(dynamic_colors, axis=0),
                         np.concatenate(dynamic_ids, axis=0))


class SDFPointFilter:
    """Scene-level outlier removal of the aggregated SAM points against the static mesh.

    Reads the ``sdf_filter`` config section, the loader's static surface in the
    world frame (``vert`` (V, 3) and the open3d ``mesh`` built from it) and the
    deformable stage's concatenated SAM points ((N, 3) points / colours, (N,) ids
    and frame indices) plus the remaining non-SAM LiDAR points (M, 3).
    """

    def __init__(self, cfg, loader, split):
        """cfg: the full config (uses ``sdf_filter``); loader: the SceneLoader (mesh
        ``vert`` (V, 3) / ``mesh``, ``scene_idx``); split: the
        :class:`~openbox_boxgen.deformable.DeformableSplit` whose ``static_sam_pc`` (N, 3),
        ``static_colors`` (N, 3), ``static_ids`` (N,), ``recorded_frame_idx`` (N,) and
        ``non_sam_pc`` (M, 3) are filtered.
        """
        self.cfg = cfg.sdf_filter
        self.vert = loader.vert
        self.mesh = loader.mesh
        self.scene_idx = loader.scene_idx
        self.static_sam_pc = split.static_sam_pc
        self.static_colors = split.static_colors
        self.static_ids = split.static_ids
        self.non_sam_pc = split.non_sam_pc
        self.recorded_frame_idx = split.recorded_frame_idx

    def sdf_vertex_voting(self, positive_ind_lst, negative_ind_lst):
        """Per-vertex vote between SAM and non-SAM points.

        ``positive_ind_lst`` / ``negative_ind_lst`` are the nearest-vertex index
        of every SAM / non-SAM point (``NO_VERTEX`` when out of range).  Returns
        ``positive_ind_lst`` with the vertices set to ``NO_VERTEX`` where the SAM
        count is not above ``vertex_vote_min_sam_fraction`` of the total count.
        """
        sam_bincount = np.bincount(positive_ind_lst[positive_ind_lst != NO_VERTEX])
        non_sam_bincount = np.bincount(negative_ind_lst[negative_ind_lst != NO_VERTEX])
        # pad the shorter histogram with float zeros: the comparison below is then float64
        if len(sam_bincount) < len(non_sam_bincount):
            sam_bincount = np.concatenate([sam_bincount, np.zeros(len(non_sam_bincount) - len(sam_bincount))])
        elif len(sam_bincount) > len(non_sam_bincount):
            non_sam_bincount = np.concatenate([non_sam_bincount, np.zeros(len(sam_bincount) - len(non_sam_bincount))])
        surviving_vertices = np.where(
            sam_bincount > (sam_bincount + non_sam_bincount) * self.cfg.vertex_vote_min_sam_fraction,
            np.arange(len(sam_bincount)), NO_VERTEX)
        return np.where(np.isin(positive_ind_lst, surviving_vertices), positive_ind_lst, NO_VERTEX)

    def max_cluster_filtering(self, pc, vert_ind, mesh):
        """Mask of the instance points ``pc`` lying within ``max_cluster_vertex_dist`` of the
        largest connected triangle patch of the sub-mesh spanned by ``vert_ind``
        (:func:`largest_mesh_patch`); None when that sub-mesh has no complete triangle."""
        patch = largest_mesh_patch(mesh, vert_ind)
        if patch is None:
            return None
        _, patch_vertices = patch
        dist_to_cluster, _ = cKDTree(patch_vertices).query(pc, k=1)
        return dist_to_cluster < self.cfg.max_cluster_vertex_dist

    def filter(self):
        """Run the scene-level filter.

        Returns the 7-tuple ``(multi_frame_pc, multi_frame_id, multi_frame_color,
        single_frame_pcd, single_frame_color, single_frame_id, recorded_frame)``.
        The first three are None when no aggregated point survived, the last
        four when no instance was diverted to the single-frame path.
        """
        sam_vertex_idx = point_to_sdf(self.vert, self.static_sam_pc, self.cfg.sdf_vertex_radius)
        non_sam_vertex_idx = point_to_sdf(self.vert, self.non_sam_pc, self.cfg.sdf_vertex_radius)
        sam_vertex_idx = self.sdf_vertex_voting(sam_vertex_idx, non_sam_vertex_idx)

        has_vertex = sam_vertex_idx != NO_VERTEX
        sam_vertex_idx = sam_vertex_idx[has_vertex]
        pc_list = self.static_sam_pc[has_vertex]
        color_list = self.static_colors[has_vertex]
        id_list = self.static_ids[has_vertex]

        # instances that lost every point go to the single-frame path with their frame indices
        removed_pc = self.static_sam_pc[~has_vertex]
        removed_color = self.static_colors[~has_vertex]
        removed_id = self.static_ids[~has_vertex]
        removed_frame_idx = self.recorded_frame_idx[~has_vertex]
        surviving_ids = np.unique(id_list)

        single_frame_pcd, single_frame_color, single_frame_id, recorded_frame = [], [], [], []
        for inst_id in np.unique(removed_id):
            if inst_id in surviving_ids:
                continue
            instance_mask = removed_id == inst_id
            single_frame_pcd.append(removed_pc[instance_mask])
            single_frame_color.append(removed_color[instance_mask])
            single_frame_id.append(removed_id[instance_mask])
            recorded_frame.append(removed_frame_idx[instance_mask])
        logger.debug("scene-%d: %d instances diverted to the single-frame path",
                     self.scene_idx, len(single_frame_pcd))

        # per instance keep the points on the largest connected cluster of its sub-mesh
        new_pc_list, new_id_list, new_color_list = [], [], []
        for inst_id in np.unique(id_list):
            in_instance = id_list == inst_id
            pc = pc_list[in_instance]
            on_cluster = self.max_cluster_filtering(pc, sam_vertex_idx[in_instance], self.mesh)
            if on_cluster is None:
                logger.debug("scene-%s: instance %s dropped, sub-mesh has no triangle", self.scene_idx, inst_id)
                continue
            new_pc_list.append(pc[on_cluster])
            new_id_list.append(id_list[in_instance][on_cluster])
            new_color_list.append(color_list[in_instance][on_cluster])

        if len(new_pc_list) == 0:
            multi_frame = (None, None, None)
        else:
            multi_frame = (np.concatenate(new_pc_list, axis=0),
                           np.concatenate(new_id_list, axis=0),
                           np.concatenate(new_color_list, axis=0))

        if len(single_frame_pcd) == 0:
            single_frame = (None, None, None, None)
        else:
            single_frame = (np.concatenate(single_frame_pcd, axis=0),
                            np.concatenate(single_frame_color, axis=0),
                            np.concatenate(single_frame_id, axis=0),
                            np.concatenate(recorded_frame, axis=0))

        return multi_frame + single_frame
