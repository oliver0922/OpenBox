# Adaptive Bounding Box Generation

Turns the per-scene instance point clouds from [`preprocess/`](../preprocess/README.md)
into per-frame 3-D pseudo labels (the stage that produced the OpenBox
labels). Three scripts run in order; every tunable lives in one
YAML (`configs/waymo.yaml`), the scripts take only paths and a `--cfg`.

| # | Script | Produces |
|---|--------|----------|
| 1 | `generate_boxes.py` | one set of boxes per **scene**, in the scene's world frame |
| 2 | `build_pseudo_infos.py` | per-**frame** OpenPCDet infos (`<segment>_openbox_untracked.pkl`) |
| 3 | `track_boxes.py` | tracked boxes — the final labels (`<segment>_openbox.pkl`) |

**Dataset scope.** The code is written for and verified on the Waymo Open
Dataset. Everything dataset-specific is in `configs/waymo.yaml`: camera ids and
names, the camera-frame-to-OpenCV rotation and per-camera image bounds used by
the 2D-mask vote, and the class size priors. Porting to another dataset (e.g.
nuScenes) needs a matching config plus its own preprocessing
(`preprocess/` reads Waymo TFRecords) and intrinsic layout (`proj_points`
unpacks Waymo's 9-vector intrinsics).

## Setup

Runs in the same `openbox` conda environment as the preprocessing
([env recipe](../preprocess/README.md#environment)) plus four additions:

```bash
conda activate openbox
# faiss with CUDA-12 GPU support. --no-deps is REQUIRED: it must share the
# nvidia-* libraries torch already installed; letting pip pull faiss's own
# newer nvidia wheels breaks torch's cuBLAS at import.
pip uninstall -y faiss-cpu && pip install --no-deps faiss-gpu-cu12==1.14.1.post1
pip install --no-deps filterpy==1.4.5 pyquaternion==0.9.9 llvmlite==0.44.0 numba==0.61.0
# The two CUDA ops (rotated-BEV NMS, points-in-boxes) ship with this folder as a
# minimal vendored subset of OpenPCDet (boxgen/pcdet/, Apache-2.0) — no OpenPCDet
# clone needed.  Build them once against this env's torch:
cd boxgen && TORCH_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace   # 8.6 = your GPU's compute capability; omit the variable to autodetect
export PYTHONPATH=/path/to/OpenBox/boxgen:$PYTHONPATH   # serves openbox_boxgen AND pcdet
```

A CUDA GPU is required: the neighbour searches run on faiss-GPU (see
*Neighbour thresholds*) and NMS / point counting on the pcdet CUDA ops.

## Inputs

Assembled from the preprocessing outputs (`$OUT` layout of
`preprocess/README.md`; the step that produces each entry is marked):

```
--scene-root/scene-N/
├── pointcloud/NNNNNN.bin                          # step 1
├── pose/NNNNNN.bin                                # step 1
├── <CAM>/intrinsic/NNNNNN.bin                     # step 1
├── <CAM>/projection_mat/NNNNNN.bin                # step 1
├── merged_sam_pc/no_aug/NNNNNN.bin                # step 3
├── instance_classname_dict.pkl                    # step 2
├── agg_mask.json                                  # step 3 (adaptive run)
├── refined_sam_pc/adaptive1200_30_50_10_2/NNNNNN.bin   # step 4
├── refined_sam_color/adaptive1200_30_50_10_2/NNNNNN.bin # optional; a fixed colour table is used when absent
├── static_vert0.15_0.4.bin                        # step 6
└── static_tri0.15_0.4.bin                         # step 6

--processed-root/<segment>/
├── NNNN.npy                                       # step 0
├── ppscore/NNNN.npy                               # step 5
├── <segment>_fov.pkl                              # step 0
└── masked_points/NNNN.npy                         # step 0
```

`masked_points/` holds the lidar returns that project into one of the five
camera images; the point counts stored with the labels are taken on it.
`<segment>_fov.pkl` holds the per-frame OpenPCDet info records (frame meta,
pose, camera image shapes) that stage 2 copies before replacing their
`annos` with the pseudo labels.

## Running

```bash
python generate_boxes.py --scenes 0,1,2 \
    --scene-root $DATA/waymo_sam2 --processed-root $DATA/waymo_processed_data_v0_5_0 \
    --split-file $DATA/ImageSets/train.txt --out-root $DATA/waymo_sam2

python build_pseudo_infos.py --scenes 0,1,2 \
    --scene-root $DATA/waymo_sam2 --processed-root $DATA/waymo_processed_data_v0_5_0 \
    --split-file $DATA/ImageSets/train.txt \
    --out-root $DATA/waymo_processed_data_v0_5_0_static --workers 2

python track_boxes.py --scenes 0,1,2 \
    --infos-root $DATA/waymo_processed_data_v0_5_0_static \
    --processed-root $DATA/waymo_processed_data_v0_5_0 \
    --split-file $DATA/ImageSets/train.txt \
    --out-root $DATA/waymo_processed_data_v0_5_0_static --workers 2
```

All three scripts take `--gpu N` for the CUDA NMS / points-in-boxes ops; the
faiss neighbour search of stage 1 always uses GPU 0, so set
`CUDA_VISIBLE_DEVICES` to move a whole run to another GPU.

Stage 1 writes one file per scene, `scene-N/openbox_boxes.pkl` — a dict with
`static_boxes` (float32 (K, 8) `[x, y, z, l, w, h, yaw, num_points]`, world
frame), `static_classes`, `deformable_boxes` and `single_frame_boxes` (the last
two per frame). A scene whose static points all fall away stores
`static_boxes = None`; stage 2 then also drops its deformable boxes, exactly like
the original run did. Stage 2 writes
`<segment>_openbox_untracked.pkl`, stage 3 reads it and writes the final
`<segment>_openbox.pkl` (tags: `--version` / `--input-version` /
`--output-version`). In the stage-3 output the `tracking_id` column carries
the heading angle: the final NMS drops the id column (kept as in the original
labels).

All hyperparameters — HDBSCAN, split/SDF thresholds, per-stage NMS
(`stage1: 1e-9/300`, `tracking: 0/300`), box-fit ratios and the rectangle-fit /
ground-snapping constants (`box_fit`), AB3DMOT per-class parameters — are in
[`configs/waymo.yaml`](configs/waymo.yaml) with comments.

## Acknowledgements and licences

- `pcdet/` is an unmodified subset of
  [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache-2.0; see
  `pcdet/LICENSE` and `pcdet/NOTICE`): the rotated-BEV NMS and points-in-boxes
  CUDA ops.
- `openbox_boxgen/tracking/` is derived from
  [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) (Xinshuo Weng, CMU), adapted
  to the Waymo frame with per-class settings and a forward + backward pass.
  AB3DMOT is distributed under a non-commercial research licence, reproduced in
  `openbox_boxgen/tracking/LICENSE_AB3DMOT`; that code is subject to its terms.
- The closeness-to-edge rectangle fit follows Zhang, Wang and Wang, "Efficient
  L-shape fitting for vehicle detection using laser scanners" (IV 2017).
- Everything else builds on [Open3D](https://www.open3d.org/),
  [hdbscan](https://github.com/scikit-learn-contrib/hdbscan),
  [faiss](https://github.com/facebookresearch/faiss),
  [filterpy](https://github.com/rlabbe/filterpy) and the
  [Waymo Open Dataset](https://waymo.com/open/) tools.

## Neighbour thresholds (do not "fix" these)

The neighbour searches compare thresholds against **squared** L2 distances
(faiss `IndexFlatL2` semantics): `0.15` means a radius of `sqrt(0.15)=0.387 m`.
`openbox_boxgen/neighbors.py` runs faiss on the GPU (a cKDTree fallback with
the same squared-distance semantics exists; it can differ on borderline points).
