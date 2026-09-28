"""Multi-frame (aggregated, static) pseudo-box generation for stage 1.

Per SAM instance id -- in ascending ``np.unique`` order, which fixes the order of
the box list fed to NMS -- the generator:

1. snaps the instance's static SAM points to the static SDF mesh
   (:func:`openbox_boxgen.point_filters.point_to_sdf`) and drops points that hit no vertex;
2. selects the instance sub-mesh from the matched vertices, keeps the largest
   connected-triangle patch and appends that patch's vertices to the points;
3. removes statistical outliers (open3d);
4. fits an initial yaw-aligned box (``BoundingBox.make_box('closeness_to_edge')``);
5. refines it on the mesh surface with :func:`openbox_boxgen.boxes.refine_bbox`
   (surface extension, 2-D IoU candidate vote, height refinement against the
   full non-SAM + SAM point cloud).

Output per instance: a float32 ``[cx, cy, cz, l, w, h, yaw]`` row (z = box
centre) in the scene (frame-0) frame, the class name, and the number of instance
points before mesh-vertex augmentation and outlier removal (the NMS score).
Deformable classes are boxed per frame by ``deformable.py`` and skipped here.
"""
import logging
from typing import NamedTuple

import numpy as np
import open3d as o3d

from .boxes import BoundingBox, refine_bbox
from .point_filters import NO_VERTEX, largest_mesh_patch, point_to_sdf

logger = logging.getLogger(__name__)


class MultiFrameBoxes(NamedTuple):
    """Result of :meth:`MultiFrameBoxGen.run`; the three lists are index-aligned."""
    boxes: list  # float32 (7,) [cx, cy, cz, l, w, h, yaw] per instance, z = box centre
    num_points: list  # instance point count before augmentation (the NMS score)
    class_names: list  # fine class name per instance (key of ``statistical_box_size``)


class MultiFrameBoxGen:
    """One refined box per static (non-deformable) SAM instance.

    Reads the ``multi_frame`` config section (outlier_nb_neighbors, outlier_std_ratio,
    thres_parallel, thres_num_normal_vectors, z_refine_threshold),
    ``sdf_filter.sdf_vertex_radius`` and the loader's mesh, camera calibration, 2D
    boxes and instance classes.  ``non_sam_points`` (P, 3) and ``sam_points`` (S, 3)
    only feed the ground estimate of the height refinement; ``cluster_points``
    (N, 3) are the HDBSCAN inliers of ``sam_points`` and ``point_instance_ids`` (N,)
    their majority SAM instance ids (:class:`openbox_boxgen.clustering.ClusterMatcher`).
    """

    def __init__(self, cfg, loader, non_sam_points, sam_points, cluster_points, point_instance_ids):
        """cfg: the full config (uses ``multi_frame``, ``box_fit``, ``classes`` and
        ``sdf_filter.sdf_vertex_radius``); loader: the SceneLoader (mesh, camera
        calibration, 2D boxes, instance classes).  The point arrays are described in
        the class docstring.  Eagerly builds ``full_pc_list`` and keeps only the
        cluster points on the mesh surface.
        """
        self.cfg = cfg  # full config: multi_frame, box_fit, classes, sdf_filter sections are read
        self.statistical_box_size = cfg.classes.statistical_box_size
        self.deformable_classes = cfg.classes.deformable
        self.intrinsic_dict_list = loader.intrinsic_list
        self.extrinsic_dict_list = loader.projection_mat_list
        self.bbox_2d_info = loader.bbox_2d_info
        self.cameras = loader.cameras
        self.mesh = loader.mesh
        self.instance_class_dict = loader.instance_cls_dict
        self.scene_idx = loader.scene_idx

        # non-SAM (float32) + SAM points (float32, or float64 when the deformable split
        # upcast an empty frame) for the ground estimate; refine_z only masks and sorts
        # this cloud, so the concatenation order is irrelevant.
        self.full_pc_list = np.concatenate([non_sam_points, sam_points], axis=0)

        # Snap every cluster point to its nearest mesh vertex and drop the points
        # farther than the SDF surface radius from the mesh.
        vertex_ids = point_to_sdf(loader.vert, cluster_points, cfg.sdf_filter.sdf_vertex_radius)
        on_mesh = vertex_ids != NO_VERTEX
        self.vertex_ids = vertex_ids[on_mesh]
        self.cluster_points = cluster_points[on_mesh]
        self.point_instance_ids = point_instance_ids[on_mesh]

    def run(self):
        """Box every instance id in ascending ``np.unique`` order -> :class:`MultiFrameBoxes`.

        Deformable instances, instances whose sub-mesh has no triangle and
        instances with no point left after outlier removal are skipped entirely.
        """
        boxes = []
        num_points = []
        class_names = []
        for instance_id in np.unique(self.point_instance_ids):
            class_name = self.instance_class_dict[instance_id]
            if class_name in self.deformable_classes:
                continue

            instance_mask = self.point_instance_ids == instance_id
            instance_points = self.cluster_points[instance_mask]

            patch = self._largest_mesh_patch(instance_points, instance_mask)
            if patch is None:
                logger.warning('scene %s instance %s: sub-mesh has no triangles, instance skipped',
                               self.scene_idx, instance_id)
                continue
            augmented_points, sub_mesh = patch

            box = self._fit_and_refine_box(augmented_points, instance_id, sub_mesh, class_name)
            if box is None:
                logger.warning('scene %s instance %s: no points left after outlier removal, instance skipped',
                               self.scene_idx, instance_id)
                continue
            boxes.append(box)
            class_names.append(class_name)
            # Raw instance points, before augmentation and outlier removal.
            num_points.append(len(instance_points))
        return MultiFrameBoxes(boxes, num_points, class_names)

    def _largest_mesh_patch(self, instance_points, instance_mask):
        """Instance sub-mesh plus the points augmented with its largest connected patch.

        Returns ``(augmented_points, sub_mesh)`` or ``None`` when the sub-mesh has
        no triangle (:func:`~openbox_boxgen.point_filters.largest_mesh_patch`).
        The patch vertices follow ``instance_points`` in ``np.unique(triangles)``
        order (the rectangle fit consumes this order).  ``sub_mesh`` is the FULL
        instance sub-mesh with vertex and triangle normals, not the largest patch
        alone; that is what ``refine_bbox`` expects.
        """
        patch = largest_mesh_patch(self.mesh, self.vertex_ids[instance_mask])
        if patch is None:
            return None
        sub_mesh, patch_vertices = patch
        return np.concatenate([instance_points, patch_vertices], axis=0), sub_mesh

    def _fit_and_refine_box(self, instance_points, instance_id, sub_mesh, class_name):
        """Outlier removal -> initial rectangle fit -> mesh-surface refinement.

        Returns the (7,) float32 ``[cx, cy, cz, l, w, h, yaw]`` with z = box
        centre, or ``None`` when no point survives the outlier removal.
        """
        instance_pc_src = o3d.geometry.PointCloud()
        instance_pc_src.points = o3d.utility.Vector3dVector(instance_points)
        _, inlier_indices = instance_pc_src.remove_statistical_outlier(
            nb_neighbors=self.cfg.multi_frame.outlier_nb_neighbors, std_ratio=self.cfg.multi_frame.outlier_std_ratio)
        instance_pc_src = instance_pc_src.select_by_index(inlier_indices)
        if len(instance_pc_src.points) == 0:
            return None

        initial_box = BoundingBox().make_box(instance_pc_src, 'closeness_to_edge', self.cfg.box_fit)
        initial_obb = initial_box.get_o3d_instance()

        refined_obb = refine_bbox(self.cfg, initial_obb, self.statistical_box_size[class_name],
                                  sub_mesh, instance_id, self.intrinsic_dict_list,
                                  self.extrinsic_dict_list, self.bbox_2d_info, self.cameras,
                                  self.scene_idx, self.full_pc_list)

        # Centre -> bottom centre here, bottom centre -> centre again in
        # get_np_instance(); the round trip is not exact in floating point (part of the numerics).
        center = refined_obb.center
        dimension = refined_obb.extent
        new_center = np.array([center[0], center[1], center[2] - dimension[2]/2])
        rotation = refined_obb.R
        angle = np.arctan2(rotation[1, 0], rotation[0, 0])
        return BoundingBox(new_center, dimension, angle).get_np_instance().astype(np.float32)
