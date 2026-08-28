# Grounded-SAM-2 (Waymo multicam generation)

Generates instance masks and every derived artifact for Waymo multi-camera
sequences using Grounding DINO detection + SAM2 mask tracking.
A single run of `gen_multicam_samv2_data_unified.py` produces, per scene
(`<out_dir>` = `$OUT/sam` in the pipeline layout):

```
<out_dir>/scene-N/<CAM>/erosion/mask_data/mask_XXXXXX.npy   raw instance masks (uint16, 0 = background)
<out_dir>/scene-N/<CAM>/erosion/json_data/mask_XXXXXX.json  instance id / class / bbox
<out_dir>/scene-N/<CAM>/erosion/result/*.jpeg               annotated (fixed erosion, 3x3 cross, 15 iters)
<out_dir>/scene-N/<CAM>/result/*.jpeg                       annotated (no erosion)
<out_dir>/scene-N/<CAM>/adaptive_erosion/<P>/result/*.jpeg  annotated (width-adaptive erosion)
<out_dir>/scene-N/<CAM>/visualization/uppc_*_sam/*.bin      3D unprojection (from no-erosion masks)
<out_dir>/scene-N/<CAM>/adaptive<P>/visualization/...       3D unprojection (from adaptive masks)
<out_dir>/scene-N/instance_classname_dict.pkl               {instance_id: class_name}
```
`<P>` = `width_thres_1`\_`width_thres_2`\_`iteration_0`\_`iteration_1`\_`iteration_2` (default `1200_30_50_10_2`)

## Repository layout

```
Grounded-SAM-2/
├── gen_multicam_samv2_data_unified.py   # entry script (multi-GPU)
├── utils/
│   ├── mask_dictionary_model.py         # tracking mask dictionary (IoU matching)
│   └── multicam_common_utils_unified.py # annotated images + 3D unprojection
├── sam2/                                # SAM2 package (pure Python — no CUDA extension)
├── sam2_configs/                        # SAM2 hydra configs
├── checkpoints/sam2_hiera_large.pt      # SAM2 checkpoint (not in git, see below)
├── parent_files/                        # files to copy into the parent mmdetection fork
├── requirements_groundedsam.txt         # pinned pip packages
└── setup.py                             # exposes sam2/sam2_configs top-level (no extension build)
```

## Requirements (hardware / driver / framework)

The results depend on the exact PyTorch / mmcv / numpy versions below (and on
the GPU generation, see the notes), so keep them pinned:

| | requirement |
|---|---|
| GPU | CUDA GPU with ~32 GB VRAM (peak usage ≈31 GB); NVIDIA Ampere or newer for the TF32 path |
| NVIDIA driver | ≥ 525.60.13 (runtime requirement of cu121 wheels); the driver version can affect rare mask-boundary pixels |
| CUDA toolkit | 12.1 (nvcc), used once to build mmcv 2.1.0 from source — no torch2.3-compatible mmcv wheel exists |
| Python | 3.10.15 |
| PyTorch | 2.3.1+cu121 (mmcv 2.1.0 binary ops are ABI-coupled to it — do not mix versions; other releases such as 2.4.1 change the results) |
| mmcv / mmengine / mmdet | 2.1.0 / 0.10.5 / 3.3.0 |
| numpy / opencv / supervision | 1.26.3 / 4.10.0.84 / 0.22.0 |

Notes:
- The script enables TF32 on Ampere+ GPUs. Pre-Ampere GPUs still run, but their
  numerics differ from the TF32 path.
- The SAM2 CUDA extension (`sam2._C`) is intentionally absent — do not build it.
  The CUDA toolkit is needed only for the one-time mmcv build; nothing else in
  the stack compiles.

## Parent repository setup (everything bundled in `parent_files/`)

This repo is meant to be used as a **git submodule of an mmdetection fork**
(v3.3.x line). All custom mmdet files and configs (the SAMV2 inferencer, the
modified detector/head/dataset files, and the grounding-DINO finetune configs)
are bundled under `parent_files/` with parent-repo-relative paths — copying it
over the fork is the whole setup:

```bash
cd <parent mmdetection repo>
cp -r <this repo>/parent_files/* .
```

### Detector weight (finetuned Grounding DINO swin-b)

The 2.7 GB finetuned checkpoint is hosted on Hugging Face:
[`oliver0922/GroundingDINOfintune`](https://huggingface.co/oliver0922/GroundingDINOfintune)
(`groundingdinofintune.pth`). `gen_multicam_samv2_data_unified.py` downloads it
automatically on first run when `--weights` is not given. The repo is gated
(auto-approved): log in once with `huggingface-cli login` and accept the access
request on the model page — `hf_hub_download` then picks up your token by itself. To fetch it manually
(e.g. for an offline machine):

```bash
python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('oliver0922/GroundingDINOfintune', 'groundingdinofintune.pth'))"
```

Also required, not part of any repo:

1. **Input data** — the step-1 scene tree (`$OUT/scenes/scene-N/`, produced by
   `preprocess/waymo_file_gen.py`):
   - `<CAM>/image/*.jpeg` (FRONT / FRONT_LEFT / FRONT_RIGHT: 1920x1280, SIDE_LEFT / SIDE_RIGHT: 1920x886)
   - `pointcloud/`, `pointcloud_projection/` (inputs for the 3D unprojection)
2. **BERT**: `bert-base-uncased` is fetched from HuggingFace on first run
   (offline machines need it in the HF cache).
3. **NLTK data**: `glip.py` auto-downloads it if missing (offline machines need `~/nltk_data`).

### Adding as a submodule

The import paths are `groundedsamv2.*`, so the **submodule path must be named
`groundedsamv2`**:

```bash
cd <parent repo>
git submodule add <this repo URL> groundedsamv2
```

Two sys.path entries are needed at runtime: the parent repo root (resolves
`mmdet` and the `groundedsamv2.*` prefix) and the submodule dir itself
(hydra imports `sam2_configs` / `sam2` top-level).

```bash
# recommended
export PYTHONPATH=<parent repo>:<parent repo>/groundedsamv2
```

Alternative: `pip install -e <parent repo>` plus `pip install -e ./groundedsamv2`.
Caveat: with modern pip, an editable install maps only the declared packages
(`mmdet`, `sam2`, `sam2_configs`) — the `groundedsamv2.*` prefix is then still
unresolved, so the parent-root PYTHONPATH entry is required anyway. When in
doubt, just use the PYTHONPATH line above.

## Conda environment (groundedsam)

Python 3.10.15 / PyTorch 2.3.1+cu121. Keep the versions pinned — the results
depend on them.

```bash
conda create -n groundedsam python=3.10.15 -y
conda activate groundedsam

# PyTorch cu121 builds come from the pytorch index, not PyPI
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# mmcv 2.1.0 must be built from source against torch 2.3.1 — no compatible
# wheel exists (PyPI: sdist only; the openmmlab torch2.3 index starts at
# 2.2.0; the torch2.1 wheel is ABI-incompatible with torch 2.3). The build
# needs a CUDA 12.1 nvcc (CUDA_HOME) and setuptools<81 (pkg_resources).
pip install "setuptools==60.2.0" wheel ninja
CUDA_HOME=<cuda-12.1-toolkit> FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS=32 \
    pip install mmcv==2.1.0 --no-build-isolation --no-binary mmcv

# everything else, pinned. --no-deps: the file is a complete freeze, and a
# plain resolver pass backtracks into re-building the mmcv sdist.
grep -v '^mmcv==' requirements_groundedsam.txt > /tmp/req.txt
pip install --no-deps -r /tmp/req.txt

# mmdet (parent repo, editable)
pip install -e <parent repo path>

# this repo (exposes sam2/sam2_configs top-level)
pip install -e <this repo path>
```

### SAM2 checkpoint

`checkpoints/sam2_hiera_large.pt` (~858 MB) is not tracked by git. The step-2
script downloads it from the official SAM2 release into `checkpoints/` next to
itself on first run (or pass an explicit path with `--sam2_checkpoint`). To
fetch it by hand:

```bash
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## Running

```bash
# $OUT = the pipeline output root (see preprocess/README.md "Output layout");
# $OUT/scenes is step 1's output, $OUT/sam is where this step writes.
python groundedsamv2/gen_multicam_samv2_data_unified.py \
    configs/grounding_dino/grounding_dino_swin-b_finetune_8xb4_1x_nus.py \
    --data_dir $OUT/scenes --out_dir $OUT/sam \
    --scene-start 0 --scene-end 797 \
    --gpus 0,1,2,3,4,5,6,7
```

- `--gpus`: one worker process per GPU; scenes are distributed across workers.
  Models are loaded once per worker, and each worker's CPU threads are capped at
  `cores / num_workers` to keep the load average sane.
- The camera order (FRONT → FRONT_LEFT → FRONT_RIGHT → SIDE_LEFT → SIDE_RIGHT)
  and the instance-id continuity across cameras within a scene are part of the
  algorithm — do not change them.