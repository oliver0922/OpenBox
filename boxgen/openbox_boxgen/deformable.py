"""Per-frame boxes for the deformable classes (person, bicycle).

Deformable instances do not aggregate well across frames, so they bypass the
SDF-filter / HDBSCAN aggregation path.  :class:`DeformableBoxGen` runs first
in stage 1 and does two things:

* it strips the deformable-class points out of the per-frame SAM point lists
  (mutating the caller's lists IN PLACE) and regroups the remaining points by
  ascending instance id -- that regrouped, frame-major order is what the SDF
  filter and HDBSCAN see afterwards;
* for every deformable instance of every frame it fits one box from the largest
  HDBSCAN cluster of the instance points (after statistical outlier removal),
  extends it to the statistical class size and snaps its bottom to the local
  ground (``refine_z_deform``).
"""
from typing import NamedTuple

import hdbscan
import numpy as np
import open3d as o3d

from .boxes import BoundingBox, refine_z_deform


class DeformableSplit(NamedTuple):
    """Result of :meth:`DeformableBoxGen.run` -- the scene split into two parts:
    the deformable instances, boxed per frame (``deformable_boxes`` as ``{frame_idx:
    [(class_name, float32 (7,)), ...]}`` world-frame boxes ``[cx, cy, cz, l, w, h, yaw]``,
    cz = box centre, a key for every frame), and everything that REMAINS for the
    static path: the scene-wide, frame-major concatenation of the non-deformable SAM
    points (``static_sam_pc`` (N, 3), ``static_ids`` (N,), ``static_colors`` (N, 3);
    ``recorded_frame_idx`` (N,) is the frame of every row) and the concatenated
    non-SAM points ``non_sam_pc`` (M, 3).
    """
    recorded_frame_idx: np.ndarray
    static_sam_pc: np.ndarray
    static_ids: np.ndarray
    static_colors: np.ndarray
    non_sam_pc: np.ndarray
    deformable_boxes: dict


class DeformableBoxGen:
    """Split the deformable instances out of the SAM point lists and box them per frame.

    Reads the ``deformable`` config section and the loader's per-frame static SAM
    lists (points / colours / ids).  Those loader lists are MUTATED IN PLACE by
    :meth:`run`: afterwards they hold only the non-deformable points of each frame,
    regrouped by ascending instance id, with the ids flattened to 1-D.  The SDF
    filter and HDBSCAN consume exactly that reordered sequence.
    """

    def __init__(self, cfg, loader):
        """cfg: the full config (uses ``deformable``, ``box_fit`` and ``classes``); loader: a loaded
        :class:`~openbox_boxgen.io.SceneLoader` whose per-frame static SAM lists
        (world frame, (N_f, 3) points / (N_f, 3) colours / (N_f,) or (N_f, 1) ids)
        and background points (M_f, 3) are read -- and MUTATED, see class docstring.
        """
        self.cfg = cfg.deformable
        self.fit_cfg = cfg.box_fit
        self.statistical_box_size = cfg.classes.statistical_box_size
        self.deformable_classes = cfg.classes.deformable
        self.instance_cls_dict = loader.instance_cls_dict
        self.static_sam_pc_list = loader.static_sam_pc_world_list
        self.static_id_list = loader.static_sam_id_list
        self.static_color_list = loader.static_sam_color_list
        self.non_sam_pc_list = loader.background_pc_world_list
        self.scene_idx = loader.scene_idx
        self.deformable_boxes_all = dict()

    def gen_bbox(self, instance_pts, sam_pc_frame, non_sam_pc_frame, cls):
        """Fit, extend to the class prior and ground-snap one box (world frame, float32
        (7,)), or ``None`` when there are too few points or the fit is too big for the
        class.  ``sam_pc_frame`` is the frame's ORIGINAL SAM points (before the split).

        instance_pts: (P, 3) xyz; sam_pc_frame / non_sam_pc_frame: (N, 3) /
        (M, 3) xyz; cls: str class name.
        """
        if len(instance_pts) < self.cfg.min_points_for_box_fit:
            return None
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(instance_pts))
        box = BoundingBox().make_box(cloud, 'closeness_to_edge', self.fit_cfg)
        stat_size = np.array(self.statistical_box_size[cls])
        if (box.s > stat_size * self.cfg.too_big_ratio).any():
            return None
        box.s = np.maximum(box.s, stat_size)
        box = refine_z_deform(box, np.concatenate([sam_pc_frame, non_sam_pc_frame], axis=0), stat_size,
                              self.fit_cfg.ground_quantile)
        return box.get_np_instance().astype(np.float32)

    def process_instance(self, id_mask, sam_pc_frame, non_sam_pc_frame, cls):
        """Largest HDBSCAN cluster -> statistical outlier removal -> :meth:`gen_bbox`.

        id_mask: (N,) bool over the frame's points; sam_pc_frame /
        non_sam_pc_frame: (N, 3) / (M, 3) xyz; cls: str -> float32 (7,) box or None.
        """
        instance_pts = sam_pc_frame[id_mask]
        if len(instance_pts) >= self.cfg.min_points_for_clustering:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.cfg.hdbscan.min_cluster_size,
                min_samples=self.cfg.hdbscan.min_samples,
                cluster_selection_epsilon=self.cfg.hdbscan.cluster_selection_epsilon)
            clusterer.fit(instance_pts[:, :3])
            non_noise_labels = clusterer.labels_[clusterer.labels_ != -1]
            if len(non_noise_labels) == 0:
                return None
            # size ties resolve to the lowest label (bincount + argmax)
            instance_pts = instance_pts[clusterer.labels_ == np.argmax(np.bincount(non_noise_labels))]

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(instance_pts))
        if len(cloud.points) > 1:
            _, inlier_idx = cloud.remove_statistical_outlier(
                nb_neighbors=len(cloud.points) // self.cfg.outlier_neighbour_fraction,
                std_ratio=self.cfg.outlier_std_ratio)
            cloud = cloud.select_by_index(inlier_idx)
        return self.gen_bbox(np.array(cloud.points), sam_pc_frame, non_sam_pc_frame, cls)

    def process_frame(self, frame_idx):
        """Box the deformable instances of one frame and rewrite the frame's point lists.

        Instance ids are visited in ``np.unique`` (ascending) order; non-deformable
        instances are re-appended to the frame's point / colour / id lists in that
        order (which regroups the frame by instance id and flattens the ids),
        deformable instances produce boxes and their points are dropped.

        frame_idx: int -> list of (str class, float32 (7,)) world-frame boxes.
        """
        deformable_boxes = []
        new_pc, new_color, new_id = [], [], []
        id_frame = self.static_id_list[frame_idx]
        sam_pc_frame = self.static_sam_pc_list[frame_idx]
        color_frame = self.static_color_list[frame_idx]
        non_sam_pc_frame = self.non_sam_pc_list[frame_idx]

        for instance_id in np.unique(id_frame):
            try:
                cls = self.instance_cls_dict[instance_id]
            except KeyError:
                raise ValueError('instance id {} not in instance_cls_dict in scene {}'.format(
                    instance_id, self.scene_idx))
            id_mask = id_frame.flatten() == instance_id
            if cls not in self.deformable_classes:
                new_pc.append(sam_pc_frame[id_mask])
                new_color.append(color_frame[id_mask])
                new_id.append(np.ones_like(id_frame[id_mask]) * instance_id)
                continue
            deformable_box = self.process_instance(id_mask, sam_pc_frame, non_sam_pc_frame, cls)
            if deformable_box is not None:
                deformable_boxes.append((cls, deformable_box))

        if new_pc:
            self.static_sam_pc_list[frame_idx] = np.concatenate(new_pc, axis=0)
            self.static_color_list[frame_idx] = np.concatenate(new_color, axis=0)
            self.static_id_list[frame_idx] = np.concatenate(new_id, axis=0).flatten()
        else:
            # float64 on purpose: an empty float64 frame upcasts the scene-wide concatenation
            self.static_sam_pc_list[frame_idx] = np.zeros((0, 3))
            self.static_color_list[frame_idx] = np.zeros((0, 3))
            self.static_id_list[frame_idx] = np.zeros((0))
        return deformable_boxes

    def run(self):
        """Process every frame, then flatten the non-deformable points scene-wide -> DeformableSplit."""
        for frame_idx in range(len(self.static_sam_pc_list)):
            self.deformable_boxes_all[frame_idx] = self.process_frame(frame_idx)

        recorded_frame_idx = []
        for frame_idx in range(len(self.static_sam_pc_list)):
            recorded_frame_idx.extend([frame_idx] * len(self.static_sam_pc_list[frame_idx]))
        return DeformableSplit(
            recorded_frame_idx=np.array(recorded_frame_idx),
            static_sam_pc=np.concatenate(self.static_sam_pc_list, axis=0),
            static_ids=np.concatenate(self.static_id_list, axis=0).flatten(),
            static_colors=np.concatenate(self.static_color_list, axis=0),
            non_sam_pc=np.concatenate(self.non_sam_pc_list, axis=0),
            deformable_boxes=self.deformable_boxes_all,
        )
