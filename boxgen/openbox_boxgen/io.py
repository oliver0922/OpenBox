"""Per-scene input loading for stage 1 (box generation) on Waymo.

``SceneLoader`` reads everything ``generate_boxes.py`` needs for one scene
``scene-{idx}`` and splits the SAM instance points of every frame into a static
and a dynamic part.  Per scene directory ``S = {scene_root}/scene-{idx}`` it reads

* ``S/static_vert{vl}_{tr}.bin`` (float64 (N, 3)) and ``S/static_tri{vl}_{tr}.bin``
  (int32 (M, 3)): the static vdbfusion mesh,
* ``S/instance_classname_dict.pkl`` (instance id -> class name) and
  ``S/agg_mask.json`` (2-D box / mask info per instance and camera),
* ``S/pose/{f:06d}.bin`` (float64 4x4 ego poses; frame 000000 is mandatory),
* ``S/pointcloud/{f:06d}.bin`` (float32 (N, 3) raw LiDAR, sensor frame),
* ``S/refined_sam_pc/{adaptive_name}/{f:06d}.bin`` (float64 (N, 4) xyz + id),
* ``S/refined_sam_color/{adaptive_name}/{f:06d}.bin`` (float32 (N, 3), optional),
* ``S/merged_sam_pc/no_aug/{f:06d}.bin`` (float32 (N, 4) xyz + id),
* ``S/{CAM}/intrinsic/{f:06d}.bin`` (float64 3x3) and
  ``S/{CAM}/projection_mat/{f:06d}.bin`` (float64 4x4) for the five cameras,

and per processed Waymo segment ``P = {processed_root}/{segment}`` (line ``idx``
of ``split_file``) ``P/{f:04d}.npy`` (LiDAR, only xyz used) and
``P/ppscore/{f:04d}.npy`` (per-point static probability).

The world frame of a scene is the ego frame of ``pose/000000.bin``; every
per-frame point list except ``raw_pc_sensor_list`` is expressed in it.  The
``*_sq_threshold`` values of ``cfg_split`` are compared against SQUARED L2
distances (``neighbors.py``): 0.15 is an effective radius of 0.387 m.
"""
import json
import logging
import os
import pickle
from typing import NamedTuple

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .neighbors import get_target_removed_heavy_pc
from .point_filters import dynamic_removal_with_ppscore
from .point_filters import points_rigid_transform

logger = logging.getLogger(__name__)

# Instance colours when refined_sam_color/ is absent (it is for the release data).
# Drawn at import time: the seed reset of the process-wide numpy RNG is part of
# the numerics.
np.random.seed(0)
COLOR_LOOKUP = np.random.rand(10000, 3)


def read_segment_name(split_file, scene_idx):
    """Waymo segment name on line ``scene_idx`` of ``split_file`` (``.tfrecord`` stripped)."""
    with open(split_file, "r") as handle:
        seq_list = handle.readlines()
    if scene_idx >= len(seq_list):
        raise ValueError(f"scene {scene_idx}: {split_file} has only {len(seq_list)} lines")
    return seq_list[scene_idx].strip().split(".")[0]


class CameraConventions(NamedTuple):
    """Dataset camera conventions of the 2D-mask vote, from the ``data`` config section."""
    to_opencv: np.ndarray   # int (4, 4): camera frame -> OpenCV frame, applied after projection_mat
    image_bounds: dict      # camera name -> [min_u, min_v, max_u, max_v] pixel clip bounds


class SceneLoader:
    """Load and pre-split all per-scene inputs of the box-generation stage.

    ``cfg_data``, ``cfg_split`` and ``cfg_sdf_filter`` are the ``data``,
    ``static_dynamic_split`` and ``sdf_filter`` sections of the configuration
    (the last two parametrise ``dynamic_removal_with_ppscore``); instances of
    the ``deformable_classes`` are forced static by the split so that the
    deformable generator can consume them.

    Attributes (one entry per frame whose raw point cloud exists, in frame
    order; ``B`` = background, ``S`` = static SAM, ``D`` = dynamic SAM):
        vert: float64 (V, 3) static mesh vertices (world frame).
        mesh: ``o3d.geometry.TriangleMesh`` of the static surface.
        instance_cls_dict: instance id -> class name.
        bbox_2d_info: parsed ``agg_mask.json``.
        pose_list: float64 4x4 ``inv(pose_0) @ pose_f`` (one per existing pose file).
        raw_pc_sensor_list: float32 (N, 3) FOV-cropped raw LiDAR, SENSOR frame.
        background_pc_world_list: float32 (B, 3) raw LiDAR minus SAM points.
        static_sam_pc_world_list, static_sam_color_list, static_sam_id_list:
            float32 (S, 3) points, (S, 3) colours, float64 (S, 1) instance ids.
        dynamic_sam_pc_world_list, dynamic_sam_color_list, dynamic_sam_id_list:
            float32 (D, 3) points, (D, 3) colours, float64 (D,) instance ids.
        intrinsic_list, projection_mat_list: one dict camera name -> float64
            3x3 / 4x4 per frame of the configured range (indexed by frame index).
        cameras: :class:`CameraConventions` (``data.camera_to_opencv`` / ``data.image_bounds``).
    """

    def __init__(self, cfg, scene_idx, scene_root, processed_root, split_file):
        """cfg: the full loaded config (uses cfg.data / .static_dynamic_split /
        .sdf_filter / .classes.deformable); scene_idx: int; scene_root /
        processed_root / split_file: str paths.  Loads everything eagerly
        (mesh, poses, frames, cameras).
        """
        self.cfg_data = cfg.data
        self.cfg_split = cfg.static_dynamic_split
        self.cfg_sdf_filter = cfg.sdf_filter
        self.deformable_classes = cfg.classes.deformable
        self.cam_locs = cfg.data.cameras  # {camera id: name}; iteration order matters
        self.cameras = CameraConventions(np.array(cfg.data.camera_to_opencv), cfg.data.image_bounds)
        self.scene_idx = scene_idx
        self.scene_path = os.path.join(scene_root, f"scene-{scene_idx}")
        self.sequence_path = os.path.join(processed_root, read_segment_name(split_file, scene_idx))
        self.frames = range(cfg.data.frame_start, cfg.data.frame_end + 1)  # inclusive range
        sam_dir = os.path.join(self.scene_path, "refined_sam_pc", cfg.data.adaptive_name)
        if not os.path.isdir(sam_dir):  # fail once here, not with one warning per frame
            raise FileNotFoundError(f"{sam_dir}: refined SAM points for data.adaptive_name not found")

        self._load_static_mesh()
        with open(os.path.join(self.scene_path, "instance_classname_dict.pkl"), "rb") as handle:
            self.instance_cls_dict = pickle.load(handle)
        with open(os.path.join(self.scene_path, "agg_mask.json"), "r") as handle:
            self.bbox_2d_info = json.load(handle)
        self._load_pose_list()
        self._load_frames()
        self._load_camera_info()

    def _load_static_mesh(self):
        """Read the static vdbfusion mesh bins -> sets ``self.vert`` (float64 (V, 3))
        and ``self.mesh`` (o3d TriangleMesh with vertex/triangle normals and adjacency).
        """
        mesh_tag = f"{self.cfg_data.mesh_voxel_length}_{self.cfg_data.mesh_truncation}"
        vert_path = os.path.join(self.scene_path, f"static_vert{mesh_tag}.bin")
        tri_path = os.path.join(self.scene_path, f"static_tri{mesh_tag}.bin")
        vert = np.fromfile(vert_path, dtype=np.float64).reshape(-1, 3)
        tri = np.fromfile(tri_path, dtype=np.int32).reshape(-1, 3)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vert), o3d.utility.Vector3iVector(tri)
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.compute_adjacency_list()
        self.vert = vert
        self.mesh = mesh

    def _load_pose_list(self):
        """``pose_list[i] = inv(pose_000000) @ pose`` for every frame in range with a pose file.

        Frames without a pose file are skipped WITHOUT a placeholder, so the list
        is aligned with frame indices only when ``frame_start`` is 0 and no file
        is missing; ``pose_frames`` records the frame of each entry and
        ``_load_frames`` (which indexes by frame index) checks the alignment.
        """
        pose_dir = os.path.join(self.scene_path, "pose")
        center_pose = np.fromfile(os.path.join(pose_dir, "000000.bin"), dtype=np.float64).reshape(4, 4)
        pose_list, pose_frames = [], []
        for frame_idx in self.frames:
            try:
                pose = np.fromfile(os.path.join(pose_dir, f"{frame_idx:06d}.bin"), dtype=np.float64).reshape(4, 4)
            except (OSError, ValueError):
                logger.debug("scene-%d: no pose for frame %d", self.scene_idx, frame_idx)
                continue
            pose = np.linalg.inv(center_pose) @ pose
            pose_list.append(pose)
            pose_frames.append(frame_idx)
        self.pose_list = pose_list
        self.pose_frames = pose_frames

    def _load_sam_frame(self, frame_idx):
        """Refined SAM xyz (float64 (N, 3)), colours (N, 3), ids (float64 (N, 1)) and the
        no-aug SAM cloud (float32 (N, 4)) of one frame, sensor frame.

        Any failure (missing or truncated file, id outside the colour table) yields
        the empty float64 arrays ``(0, 3), (0, 3), (0, 1), (0, 4)``.
        """
        frame_file = f"{frame_idx:06d}.bin"
        adaptive_name = self.cfg_data.adaptive_name
        try:
            sam_pc = np.fromfile(os.path.join(self.scene_path, "refined_sam_pc", adaptive_name, frame_file),
                                 dtype=np.float64).reshape(-1, 4)
            color_path = os.path.join(self.scene_path, "refined_sam_color", adaptive_name, frame_file)
            if os.path.exists(color_path):
                sam_color = np.fromfile(color_path, dtype=np.float32).reshape(-1, 3)
            else:
                sam_color = COLOR_LOOKUP[sam_pc[:, 3].astype(int)]
            no_aug_sam = np.fromfile(os.path.join(self.scene_path, "merged_sam_pc", "no_aug", frame_file),
                                     dtype=np.float32).reshape(-1, 4)
            sam_id = sam_pc[:, 3:]
            sam_xyz = sam_pc[:, :3]
        except Exception as err:  # any error -> empty SAM frame
            logger.warning("scene-%d: SAM frame %d unreadable (%r); using an empty frame",
                           self.scene_idx, frame_idx, err)
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 1)), np.zeros((0, 4))
        return sam_xyz, sam_color, sam_id, no_aug_sam

    def _load_frames(self):
        """Per frame: FOV crop the raw cloud, strip the SAM points off it (sensor
        frame), split the SAM points by ppscore, then move everything but the raw
        cloud to the world frame -- in this order.
        """
        raw_pc_dir = os.path.join(self.scene_path, "pointcloud")
        ppscore_dir = os.path.join(self.sequence_path, "ppscore")

        raw_pc_sensor_list = []
        background_pc_world_list = []
        static_sam_pc_world_list, static_sam_color_list, static_sam_id_list = [], [], []
        dynamic_sam_pc_world_list, dynamic_sam_color_list, dynamic_sam_id_list = [], [], []

        for frame_idx in tqdm(self.frames, desc=f"scene-{self.scene_idx} load"):
            try:
                raw_pc = np.fromfile(os.path.join(raw_pc_dir, f"{frame_idx:06d}.bin"), dtype=np.float32).reshape(-1, 3)
            except (OSError, ValueError):
                logger.debug("scene-%d: no raw point cloud for frame %d", self.scene_idx, frame_idx)
                continue

            sam_xyz, sam_color, sam_id, no_aug_sam = self._load_sam_frame(frame_idx)

            fov = np.arctan2(raw_pc[:, 1], raw_pc[:, 0])
            raw_pc = raw_pc[np.abs(fov) < self.cfg_data.fov_half_angle_rad]
            raw_pc_sensor_list.append(raw_pc)
            if frame_idx >= len(self.pose_frames) or self.pose_frames[frame_idx] != frame_idx:
                raise RuntimeError(f"scene-{self.scene_idx}: pose_list is not aligned with frame {frame_idx} "
                                   "(data.frame_start must be 0 and no pose file may be missing)")
            pose = self.pose_list[frame_idx]

            background_pc = get_target_removed_heavy_pc(raw_pc, sam_xyz, self.cfg_split.sam_removal_sq_threshold)
            background_pc = points_rigid_transform(background_pc, pose)
            background_pc_world_list.append(background_pc)

            lidar_pc = np.load(os.path.join(self.sequence_path, f"{frame_idx:04d}.npy"))[:, :3]
            ppscore = np.load(os.path.join(ppscore_dir, f"{frame_idx:04d}.npy"))
            (static_xyz, static_color, static_id,
             dynamic_xyz, dynamic_color, dynamic_id) = dynamic_removal_with_ppscore(
                cfg=self.cfg_sdf_filter,
                sam_pc=sam_xyz, sam_color=sam_color, sam_id=sam_id, no_aug_sam_pc=no_aug_sam,
                H=ppscore, lidar_pc=lidar_pc, pose=pose, vert=self.vert,
                instance_cls_dict=self.instance_cls_dict, deformable_classes=self.deformable_classes,
                ppscore_threshold=self.cfg_split.ppscore_static_threshold,
                sq_threshold=self.cfg_split.sam_to_lidar_sq_threshold, get_dynamics=True)

            static_xyz = points_rigid_transform(static_xyz, pose)
            static_sam_pc_world_list.append(static_xyz)
            static_sam_color_list.append(static_color)
            static_sam_id_list.append(static_id)

            dynamic_xyz = points_rigid_transform(dynamic_xyz, pose)
            dynamic_sam_pc_world_list.append(dynamic_xyz)
            dynamic_sam_color_list.append(dynamic_color)
            dynamic_sam_id_list.append(dynamic_id.flatten())

        self.raw_pc_sensor_list = raw_pc_sensor_list
        self.background_pc_world_list = background_pc_world_list
        self.static_sam_pc_world_list = static_sam_pc_world_list
        self.static_sam_color_list = static_sam_color_list
        self.static_sam_id_list = static_sam_id_list
        self.dynamic_sam_pc_world_list = dynamic_sam_pc_world_list
        self.dynamic_sam_color_list = dynamic_sam_color_list
        self.dynamic_sam_id_list = dynamic_sam_id_list

    def _load_camera_info(self):
        """Per frame in range, per camera: intrinsics (3x3) and projection matrix (4x4).

        One dict per frame is appended even when no camera file exists for it, so
        these lists have one entry per frame of the range (unlike the point lists).
        """
        intrinsic_list = []
        projection_mat_list = []
        for frame_idx in self.frames:
            intrinsic_dict = {}
            projection_mat_dict = {}
            for cam_name in self.cam_locs.values():
                cam_dir = os.path.join(self.scene_path, cam_name)
                intrinsic_path = os.path.join(cam_dir, "intrinsic", f"{frame_idx:06d}.bin")
                projection_path = os.path.join(cam_dir, "projection_mat", f"{frame_idx:06d}.bin")
                try:
                    intrinsic_dict[cam_name] = np.fromfile(intrinsic_path, dtype=np.float64).reshape(3, 3)
                    projection_mat_dict[cam_name] = np.fromfile(projection_path, dtype=np.float64).reshape(4, 4)
                except (OSError, ValueError):
                    logger.debug("scene-%d: no %s camera files for frame %d", self.scene_idx, cam_name, frame_idx)
                    continue
            intrinsic_list.append(intrinsic_dict)
            projection_mat_list.append(projection_mat_dict)
        self.intrinsic_list = intrinsic_list
        self.projection_mat_list = projection_mat_list

    def get_dynamic_frame_pcds(self, single_frame_pcd, single_frame_id, recorded_frame_idx):
        """Merge the SDF-filter single-frame leftovers into the per-frame dynamic clouds.

        ``single_frame_pcd`` (M, 3) / ``single_frame_id`` (M,) are the world-frame
        points the SDF filter diverted to the single-frame path and
        ``recorded_frame_idx`` (M,) their load position (index into the point
        lists); all three may be None.  Returns ``(pcd_per_frame, id_per_frame)``:
        per loaded frame the concatenation ``[leftovers of that frame, dynamic SAM
        points of that frame]``, or the dynamic lists themselves when the
        leftovers are None.
        """
        dynamic_pc_list = self.dynamic_sam_pc_world_list
        dynamic_id_list = self.dynamic_sam_id_list
        pcd_per_frame = [[] for _ in range(len(dynamic_pc_list))]
        id_per_frame = [[] for _ in range(len(dynamic_pc_list))]

        for frame_pos in range(len(dynamic_pc_list)):
            if single_frame_pcd is None or recorded_frame_idx is None:
                pcd_per_frame[frame_pos] = dynamic_pc_list[frame_pos]
                id_per_frame[frame_pos] = dynamic_id_list[frame_pos]
            else:
                frame_mask = recorded_frame_idx == frame_pos
                pcd_per_frame[frame_pos] = np.concatenate([single_frame_pcd[frame_mask], dynamic_pc_list[frame_pos]])
                id_per_frame[frame_pos] = np.concatenate([single_frame_id[frame_mask], dynamic_id_list[frame_pos]])

        return pcd_per_frame, id_per_frame
