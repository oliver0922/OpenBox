"""Generate point-cloud, annotation, ego-pose, camera-mask (mask_1/mask_2), and image files from Waymo TFRecords.

Example:
    python waymo_file_gen.py \
        --waymo-root $WAYMO --output-root $OUT/scenes \
        --scene-start 0 --scene-end 797 --workers 16
"""

import argparse
import io
import multiprocessing
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

FIRST_PROJECTION_CAMERA_COLUMN = 0
SECOND_PROJECTION_CAMERA_COLUMN = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Waymo TFRecords into the per-scene directory layout ($OUT/scenes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--waymo-root",
        type=Path,
        required=True,
        help="Waymo root: train/*.tfrecord + ImageSets/train.txt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory in which scene-N output directories will be created (e.g. $OUT/scenes).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Conversion device. cpu (default) is deterministic; "
        "gpu is faster but shifts point coordinates by up to ~5 mm.",
    )
    parser.add_argument(
        "--scene-start",
        type=int,
        default=0,
        help="First scene index to process.",
    )
    parser.add_argument(
        "--scene-end",
        type=int,
        default=797,
        help="Last scene index to process (inclusive; 797 = all scenes).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of scenes to process in parallel "
        "(with --device gpu, capped at the number of GPUs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print paths without reading TFRecords or writing files.",
    )
    return parser


def configure_gpu_worker(gpu_index: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    gpu = gpus[gpu_index]
    tf.config.set_visible_devices(gpu, "GPU")
    tf.config.experimental.set_memory_growth(gpu, True)


def configure_cpu_worker(_worker_index: int) -> None:
    tf.config.set_visible_devices([], "GPU")


def resolve_paths(args: argparse.Namespace):
    waymo_root = args.waymo_root.expanduser().resolve()
    split_file = waymo_root / "ImageSets" / "train.txt"
    records_dir = waymo_root / "train"
    output_root = args.output_root.expanduser().resolve(strict=False)
    return waymo_root, split_file, records_dir, output_root


def load_record_names(split_file: Path):
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    with split_file.open("r", encoding="utf-8") as stream:
        record_names = [line.strip() for line in stream if line.strip()]

    if not record_names:
        raise ValueError(f"Split file is empty: {split_file}")
    return record_names


def extract_point_cloud(frame: dataset_pb2.Frame):
    parsed = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(parsed) == 4:
        # waymo-open-dataset >= 1.5 also returns segmentation labels
        range_images, camera_projections, _seg_labels, top_pose = parsed
    else:
        range_images, camera_projections, top_pose = parsed
    points_first, projections_first = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, top_pose, 0
    )
    points_second, projections_second = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, top_pose, 1
    )

    points = np.concatenate(
        (np.vstack(points_first), np.vstack(points_second)), axis=0
    )
    projections = np.concatenate(
        (np.vstack(projections_first), np.vstack(projections_second)), axis=0
    )
    return points, projections


def extract_annotations(frame: dataset_pb2.Frame) -> np.ndarray:
    annotations = []
    for label in frame.laser_labels:
        box = label.box
        annotations.append(
            [
                box.center_x,
                box.center_y,
                box.center_z,
                box.length,
                box.width,
                box.height,
                box.heading,
            ]
        )
    return np.asarray(annotations)


def build_output_paths(
    output_root: Path, scene_idx: int, frame_idx: int
):
    scene_dir = output_root / f"scene-{scene_idx}"
    frame_name = f"{frame_idx:06d}"
    core_paths = {
        "pointcloud": scene_dir / "pointcloud" / f"{frame_name}.bin",
        "projection": scene_dir
        / "pointcloud_projection"
        / f"{frame_name}.bin",
        "annotations": scene_dir / "annotations" / f"{frame_name}.bin",
        "pose": scene_dir / "pose" / f"{frame_name}.bin",
    }
    camera_paths = {
        camera_id: {
            "mask_1": scene_dir / camera_name / "mask_1" / f"{frame_name}.bin",
            "mask_2": scene_dir / camera_name / "mask_2" / f"{frame_name}.bin",
            "image": scene_dir / camera_name / "image" / f"{frame_name}.jpeg",
        }
        for camera_id, camera_name in CAMERA_NAMES.items()
    }
    return core_paths, camera_paths


def save_frame(
    frame: dataset_pb2.Frame,
    points: np.ndarray,
    projections: np.ndarray,
    annotations: np.ndarray,
    core_paths,
    camera_paths,
) -> None:
    images_by_camera = {image.name: image.image for image in frame.images}
    missing_cameras = sorted(set(CAMERA_NAMES) - set(images_by_camera))
    if missing_cameras:
        raise ValueError(f"Frame is missing camera images: {missing_cameras}")

    output_paths = list(core_paths.values())
    for paths_by_type in camera_paths.values():
        output_paths.extend(paths_by_type.values())
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    points.tofile(str(core_paths["pointcloud"]))
    projections.tofile(str(core_paths["projection"]))
    annotations.tofile(str(core_paths["annotations"]))
    pose = np.asarray(frame.pose.transform, dtype=np.float64)
    pose.tofile(str(core_paths["pose"]))

    for camera_id in CAMERA_NAMES:
        # projections columns: (cam_id, x, y) for the first camera a point
        # projects into, then (cam_id, x, y) for the second one.
        mask_1 = np.where(projections[..., FIRST_PROJECTION_CAMERA_COLUMN] == camera_id)[0]
        mask_2 = np.where(projections[..., SECOND_PROJECTION_CAMERA_COLUMN] == camera_id)[0]
        mask_1.tofile(str(camera_paths[camera_id]["mask_1"]))
        mask_2.tofile(str(camera_paths[camera_id]["mask_2"]))
        with Image.open(io.BytesIO(images_by_camera[camera_id])) as image:
            image.save(str(camera_paths[camera_id]["image"]), format="JPEG")


def process_scene(
    scene_idx: int,
    record_path: Path,
    output_root: Path,
) -> int:
    dataset = tf.data.TFRecordDataset(str(record_path))
    frame = dataset_pb2.Frame()
    processed_frames = 0

    for frame_idx, data in enumerate(dataset):
        core_paths, camera_paths = build_output_paths(
            output_root, scene_idx, frame_idx
        )

        frame.ParseFromString(bytes(data.numpy()))
        points, projections = extract_point_cloud(frame)
        annotations = extract_annotations(frame)
        save_frame(frame, points, projections, annotations, core_paths, camera_paths)
        processed_frames += 1

    return processed_frames


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    waymo_root, split_file, records_dir, output_root = resolve_paths(args)
    record_names = load_record_names(split_file)
    scene_end = args.scene_end if args.scene_end is not None else len(record_names) - 1
    if not 0 <= args.scene_start <= scene_end < len(record_names):
        parser.error(
            f"scene range must satisfy 0 <= scene-start <= scene-end <= {len(record_names) - 1}"
        )
    selected_records = [
        (scene_idx, record_names[scene_idx])
        for scene_idx in range(args.scene_start, scene_end + 1)
    ]

    print(f"Waymo root:   {waymo_root}")
    print(f"Split file:   {split_file}")
    print(f"Records dir:  {records_dir}")
    print(f"Output root:  {output_root}")
    print(f"Scenes:       {args.scene_start}..{scene_end} ({len(selected_records)} total)")
    print(f"Device:       {args.device}")

    if args.dry_run:
        print("Dry run complete; no TFRecords were read and no files were written.")
        return

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            parser.error("--device gpu requested but no GPU is visible")
        if args.workers > len(gpus):
            parser.error(
                f"--workers cannot exceed the number of GPUs ({len(gpus)}) with --device gpu"
            )
        worker_initializer = configure_gpu_worker
    else:
        worker_initializer = configure_cpu_worker

    print(f"Workers:      {args.workers}")
    context = multiprocessing.get_context("spawn")
    pools = []
    try:
        for worker_index in range(args.workers):
            pools.append(
                context.Pool(
                    processes=1,
                    initializer=worker_initializer,
                    initargs=(worker_index,),
                )
            )

        frame_counts = []
        for scene_idx, record_name in selected_records:
            frame_counts.append(
                pools[scene_idx % args.workers].apply_async(
                    process_scene,
                    (scene_idx, records_dir / record_name, output_root),
                )
            )

        total_frames = sum(
            result.get() for result in tqdm(frame_counts, desc="scenes")
        )
    except BaseException:
        for pool in pools:
            pool.terminate()
        raise
    else:
        for pool in pools:
            pool.close()
    finally:
        for pool in pools:
            pool.join()

    print(
        f"Finished: {len(selected_records)} scenes, {total_frames} frames written "
        f"to {output_root}"
    )


if __name__ == "__main__":
    main()
