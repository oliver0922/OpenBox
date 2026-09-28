"""Generate OpenPCDet-style processed lidar frames and pose infos from Waymo TFRecords.

Writes, per training segment:
    $OUT/processed/<segment>/NNNN.npy          float32 (N, 6): x, y, z, intensity,
                                               elongation, NLZ flag (vehicle frame);
                                               read by steps 4-6
    $OUT/processed/<segment>/<segment>.pkl     per-frame dicts with the 4x4 ego
                                               `pose` (float32); read by
                                               compute_ppscore.py (step 5)
    $OUT/processed/<segment>/masked_points/NNNN.npy
                                               the rows of NNNN.npy whose lidar
                                               return projects into a camera image
                                               (same 6 columns); the box-generation
                                               stages count points in boxes on it
    $OUT/processed/<segment>/<segment>_fov.pkl per-frame OpenPCDet infos (frame
                                               meta, float32 pose, camera image
                                               shapes, GT boxes in front of the
                                               vehicle, per-lidar counts of the
                                               masked points); the box-generation
                                               stages copy their frame records
                                               from it and replace `annos`

Example:
    python waymo_processed_gen.py \
        --waymo-root $WAYMO --out-root $OUT/processed \
        --scene-start 0 --scene-end 797 --workers 16
"""

import argparse
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils


# ---------------------------------------------------------------------------
# Adapted from OpenPCDet (pcdet/datasets/waymo/waymo_utils.py).
# ---------------------------------------------------------------------------
def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation




WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def generate_labels(frame, pose):
    """Collect one frame's laser labels in the OpenPCDet ``annos`` layout.

    frame: Waymo Frame proto; pose: float32 (4, 4) vehicle-to-global -> dict with
    name (str), difficulty, dimensions (l, w, h), location, heading_angles,
    obj_ids, tracking_difficulty, num_points_in_gt, speed_global, accel_global
    (all np.ndarray over the K labels, 'unknown' class dropped) and
    gt_boxes_lidar float64 (K, 9): x, y, z, l, w, h, heading, speed_x, speed_y
    with the speed rotated into the vehicle frame.
    """
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    for label in frame.laser_labels:
        box = label.box
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[label.type])
        difficulty.append(label.detection_difficulty_level)
        tracking_difficulty.append(label.tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])
        locations.append([box.center_x, box.center_y, box.center_z])
        obj_ids.append(label.id)
        num_points_in_gt.append(label.num_lidar_points_in_box)
        speeds.append([label.metadata.speed_x, label.metadata.speed_y])
        accelerations.append([label.metadata.accel_x, label.metadata.accel_y])

    annotations = {
        'name': np.array(obj_name),
        'difficulty': np.array(difficulty),
        'dimensions': np.array(dimensions),
        'location': np.array(locations),
        'heading_angles': np.array(heading_angles),
        'obj_ids': np.array(obj_ids),
        'tracking_difficulty': np.array(tracking_difficulty),
        'num_points_in_gt': np.array(num_points_in_gt),
        'speed_global': np.array(speeds),
        'accel_global': np.array(accelerations),
    }
    keep = [i for i, name in enumerate(annotations['name']) if name != 'unknown']
    annotations = {key: value[keep] for key, value in annotations.items()}
    if len(annotations['name']) > 0:
        global_speed = np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)
        speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))[:, :2]
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], speed],
            axis=1)
    else:
        gt_boxes_lidar = np.zeros((0, 9))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def filter_fov_annos(annotations):
    """Keep only the labels in front of the vehicle (box centre x > 0); annotations: generate_labels dict -> same dict."""
    valid = annotations['gt_boxes_lidar'][:, 0] > 0
    return {key: value[valid] for key, value in annotations.items()}


def save_lidar_points(frame, cur_save_path, masked_save_path, use_two_returns=True):
    """Save one frame's lidar sweep as an OpenPCDet-style .npy, plus its camera-visible subset.

    frame: Waymo Frame proto; cur_save_path / masked_save_path: Path/str of the
    two output .npy files; use_two_returns: bool -> (list of 5 int per-lidar
    point counts, the same counts for the masked subset). Both files are
    float32 (N, 6): x, y, z, intensity, elongation, NLZ flag (vehicle frame);
    the masked file keeps the rows whose first camera-projection slot is set,
    i.e. returns that fall inside one of the five camera images.  The
    projection of the FIRST return is used for both returns (as in the
    original OpenPCDet-derived run).
    """
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    def stack(point_list, nlz_list, intensity_list, elongation_list):
        """Per-lidar lists -> float32 (N, 6) rows in lidar order, plus the per-lidar counts."""
        rows = np.concatenate([
            np.concatenate(point_list, axis=0),
            np.concatenate(intensity_list, axis=0).reshape(-1, 1),
            np.concatenate(elongation_list, axis=0).reshape(-1, 1),
            np.concatenate(nlz_list, axis=0).reshape(-1, 1),
        ], axis=-1).astype(np.float32)
        return rows, [point.shape[0] for point in point_list]

    # 3d points in vehicle frame.
    save_points, num_points_of_each_lidar = stack(points, points_in_NLZ_flag, points_intensity, points_elongation)
    np.save(cur_save_path, save_points)

    # camera-visible subset: rows whose first camera-projection slot names a camera
    masks = [np.where(cp[..., 0] != 0)[0] for cp in cp_points]
    masked_points, num_masked_points_of_each_lidar = stack(
        [p[m] for p, m in zip(points, masks)], [p[m] for p, m in zip(points_in_NLZ_flag, masks)],
        [p[m] for p, m in zip(points_intensity, masks)], [p[m] for p, m in zip(points_elongation, masks)])
    np.save(masked_save_path, masked_points)
    return num_points_of_each_lidar, num_masked_points_of_each_lidar


def process_segment(record_path: Path, out_root: Path) -> int:
    """Convert one TFRecord segment into NNNN.npy frames, masked_points/, a pose pkl and the fov info pkl.

    record_path: Path to the .tfrecord; out_root: Path output root -> int
    number of frames written. ``<segment>.pkl`` is a list of {frame_id: str,
    pose: float32 (4, 4)} dicts; ``<segment>_fov.pkl`` is the list of full
    OpenPCDet frame infos described in the module docstring.
    """
    tf.config.set_visible_devices([], "GPU")
    segment_name = record_path.name.replace(".tfrecord", "")
    segment_dir = out_root / segment_name
    pkl_path = segment_dir / (segment_name + ".pkl")
    if pkl_path.exists():
        # A pkl with more than {frame_id, pose} belongs to another tool
        # (OpenPCDet's training info pkl lives at this exact path) — never
        # clobber it. A pose-only pkl from a previous run of this script is
        # fine to rewrite.
        try:
            with open(pkl_path, "rb") as stream:
                existing = pickle.load(stream)
            foreign = bool(existing) and set(existing[0]) != {"frame_id", "pose"}
        except Exception:
            foreign = True
        if foreign:
            raise RuntimeError(
                f"refusing to overwrite {pkl_path}: it is not a pose-only pkl "
                "written by this script (it looks like an OpenPCDet info pkl, "
                "or is unreadable). Point --out-root at a separate directory."
            )
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "masked_points").mkdir(exist_ok=True)

    dataset = tf.data.TFRecordDataset(str(record_path), compression_type="")
    frame = dataset_pb2.Frame()
    infos, fov_infos = [], []
    for frame_idx, data in enumerate(dataset):
        frame.ParseFromString(bytes(data.numpy()))
        _, num_masked_points = save_lidar_points(
            frame, segment_dir / ("%04d.npy" % frame_idx), segment_dir / "masked_points" / ("%04d.npy" % frame_idx))
        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        frame_id = segment_name + ("_%03d" % frame_idx)
        infos.append({"frame_id": frame_id, "pose": pose})
        # key order and dtypes follow OpenPCDet's info records
        fov_infos.append({
            "point_cloud": {"num_features": 5, "lidar_sequence": segment_name, "sample_idx": frame_idx},
            "frame_id": frame_id,
            "metadata": {"context_name": frame.context.name, "timestamp_micros": frame.timestamp_micros},
            "image": {"image_shape_%d" % j: (calibration.height, calibration.width)
                      for j, calibration in enumerate(frame.context.camera_calibrations[:5])},
            "pose": pose,
            "annos": filter_fov_annos(generate_labels(frame, pose)),
            "num_points_of_each_lidar": num_masked_points,
        })

    with open(segment_dir / (segment_name + ".pkl"), "wb") as stream:
        pickle.dump(infos, stream)
    with open(segment_dir / (segment_name + "_fov.pkl"), "wb") as stream:
        pickle.dump(fov_infos, stream)
    return len(infos)


def main() -> None:
    """CLI entry point: map segments over a spawn Pool of --workers processes."""
    parser = argparse.ArgumentParser(
        description="Extract OpenPCDet-style npy frames + pose pkls from Waymo TFRecords.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--waymo-root", type=Path, required=True,
                        help="Waymo root with train/*.tfrecord and ImageSets/train.txt")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="Output root, e.g. $OUT/processed (use a NEW directory, "
                             "not an OpenPCDet root).")
    parser.add_argument("--scene-start", type=int, default=0,
                        help="first scene index (line of ImageSets/train.txt)")
    parser.add_argument("--scene-end", type=int, default=797,
                        help="last scene index, INCLUSIVE (797 = all 798 scenes)")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of worker processes (one segment each)")
    args = parser.parse_args()

    split_file = args.waymo_root / "ImageSets" / "train.txt"
    with split_file.open("r") as stream:
        record_names = [line.strip() for line in stream if line.strip()]
    scene_end = args.scene_end if args.scene_end is not None else len(record_names) - 1
    selected = record_names[args.scene_start : scene_end + 1]

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        jobs = [
            pool.apply_async(process_segment, (args.waymo_root / "train" / name, args.out_root))
            for name in selected
        ]
        total = sum(job.get() for job in tqdm(jobs, desc="segments"))
    print(f"done: {len(selected)} segments, {total} frames")


if __name__ == "__main__":
    main()
