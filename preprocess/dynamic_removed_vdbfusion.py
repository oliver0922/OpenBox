"""Build a static-background TSDF mesh per Waymo scene with vdbfusion.

For every frame of a scene:
  1. load the raw lidar points ($OUT/processed/<segment>/NNNN.npy, step 0)
     and the per-point persistence score ($OUT/ppscore/<segment>/ppscore/NNNN.npy,
     step 5, compute_ppscore.py),
  2. keep only static points (ppscore > --ppscore-threshold),
  3. transform them into the scene's frame-0 coordinates via the ego poses
     ($OUT/scenes/scene-N/pose/NNNNNN.bin, step 1, waymo_file_gen.py),
  4. integrate them into a TSDF volume (voxel --voxel-length m, truncation
     --sdf-trunc m).
After all frames, extract a triangle mesh and write per scene:
    <out-root>/scene-N/static_vert{vl}_{tr}.bin   float64 flat (V, 3)
    <out-root>/scene-N/static_tri{vl}_{tr}.bin    int32   flat (T, 3)

Example:
    python dynamic_removed_vdbfusion.py \
        --scene-data-root $OUT/scenes --processed-data-root $OUT/processed \
        --ppscore-root $OUT/ppscore \
        --split-file $WAYMO/ImageSets/train.txt \
        --scene-start 0 --scene-end 797 --out-root $OUT/mesh
"""

import argparse
import os
from pathlib import Path

import numpy as np
import vdbfusion
from tqdm import tqdm

SENSOR_HEIGHT_M = 1.7  # approximate lidar height above the ego origin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse ppscore-static lidar points into a per-scene TSDF mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-data-root",
        type=Path,
        required=True,
        help="Root holding scene-N/pose/ dirs (read only), e.g. $OUT/scenes.",
    )
    parser.add_argument(
        "--processed-data-root",
        type=Path,
        required=True,
        help="OpenPCDet-style processed Waymo root with <segment>/NNNN.npy "
        "(read only), e.g. $OUT/processed.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Split file whose line N is the segment name of scene-N "
        "($WAYMO/ImageSets/train.txt).",
    )
    parser.add_argument(
        "--ppscore-root",
        type=Path,
        default=None,
        help="Root holding <segment>/ppscore/NNNN.npy, i.e. compute_ppscore.py's "
        "--out-root ($OUT/ppscore). Defaults to --processed-data-root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root; scene-N/static_{vert,tri}*.bin are written here. "
        "Point this at a NEW directory - existing files are overwritten.",
    )
    parser.add_argument("--scene-start", type=int, default=0, help="first scene index")
    parser.add_argument(
        "--scene-end", type=int, default=797, help="last scene index (inclusive)"
    )
    parser.add_argument("--num-frames", type=int, default=200, help="frames per scene")
    parser.add_argument("--voxel-length", type=float, default=0.15)
    parser.add_argument("--sdf-trunc", type=float, default=0.4)
    parser.add_argument("--ppscore-threshold", type=float, default=0.7)
    parser.add_argument(
        "--min-weight",
        type=float,
        default=5.0,
        help="min TSDF weight for mesh extraction",
    )
    return parser


def load_split(split_file: Path):
    with split_file.open("r") as stream:
        return [line.strip().split(".")[0] for line in stream if line.strip()]


def process_scene(
    scene_idx: int, segment_name: str, args: argparse.Namespace
) -> "tuple[int, int, int] | None":
    scene_dir = args.scene_data_root / f"scene-{scene_idx}"
    if not scene_dir.is_dir():
        return None
    segment_dir = args.processed_data_root / segment_name
    ppscore_root = args.ppscore_root or args.processed_data_root
    ppscore_dir = ppscore_root / segment_name / "ppscore"

    center_pose = np.fromfile(
        scene_dir / "pose" / "000000.bin", dtype=np.float64
    ).reshape(4, 4)
    center_pose_inv = np.linalg.inv(center_pose)

    volume = vdbfusion.VDBVolume(args.voxel_length, args.sdf_trunc, False)

    integrated = 0
    for frame_idx in range(args.num_frames):
        ppscore_path = ppscore_dir / f"{frame_idx:04d}.npy"
        if not ppscore_path.is_file():
            continue  # Waymo segments have ~198 frames; indices past the end have no file
        ppscore = np.load(ppscore_path)
        lidar_points = np.load(segment_dir / f"{frame_idx:04d}.npy")[:, :3]
        statics = lidar_points[ppscore > args.ppscore_threshold]

        pose = np.fromfile(
            scene_dir / "pose" / f"{frame_idx:06d}.bin", dtype=np.float64
        ).reshape(4, 4)
        pose = center_pose_inv @ pose

        statics = np.concatenate([statics, np.ones((statics.shape[0], 1))], axis=1)
        statics = (statics @ pose.T)[:, :3]

        sensor_origin = pose[:3, 3] + [0, 0, SENSOR_HEIGHT_M]
        volume.integrate(statics.astype(np.float64), sensor_origin)
        integrated += 1

    vertices, triangles = volume.extract_triangle_mesh(True, args.min_weight)
    vertices = np.array(vertices)
    triangles = np.array(triangles, dtype=np.int32)

    out_dir = args.out_root / f"scene-{scene_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.voxel_length}_{args.sdf_trunc}"
    vertices.tofile(out_dir / f"static_vert{suffix}.bin")
    triangles.tofile(out_dir / f"static_tri{suffix}.bin")
    return integrated, len(vertices), len(triangles)


def main() -> None:
    args = build_parser().parse_args()
    segment_names = load_split(args.split_file)

    for scene_idx in tqdm(
        range(args.scene_start, args.scene_end + 1), desc="scenes"
    ):
        result = process_scene(scene_idx, segment_names[scene_idx], args)
        if result is None:
            continue
        frames, n_vert, n_tri = result
        tqdm.write(
            f"scene-{scene_idx}: frames={frames} vert={n_vert} tri={n_tri}"
        )


if __name__ == "__main__":
    main()
