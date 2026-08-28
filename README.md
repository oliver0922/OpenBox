<p align="center">
  <h1 align="center">OpenBox: Annotate Any Bounding Boxes in 3D</h1>
  <p align="center">
    <a href="https://oliver0922.github.io/">In-Jae Lee</a>
    ·
    <a href="https://scholar.google.com/citations?user=0EZ-YmIAAAAJ&hl=ko">Moongyeom Kim</a>
    ·
    <a href="https://kwonyoung9120.github.io/">Kwonyoung Ryu</a>
    ·
    <a href="https://pmusacchio.github.io/">Pierre Musacchio</a>
    ·
    <a href="https://jaesik.info/">Jaesik Park</a>
  </p>
  <p align="center">Seoul National University &nbsp;·&nbsp; POSTECH</p>
  <h3 align="center">NeurIPS 2025 (Spotlight)</h3>
  <p align="center">
    <a href="https://www.arxiv.org/abs/2512.01352">Paper</a>
    |
    <a href="https://oliver0922.github.io/OpenBox/">Project Page</a>
    |
    <a href="https://youtu.be/Si0VvsvM2O4">Video</a>
  </p>
</p>

OpenBox is a **two-stage automatic annotation pipeline** that produces
high-quality 3D bounding boxes for LiDAR scenes **without any 3D labels or
self-training iterations**, by leveraging 2D vision foundation models:

1. **Cross-modal instance alignment** — 2D instances (Grounding DINO +
   SAM2) are lifted to 3D, merged across cameras, and cleaned by
   **context-aware refinement** (Patchwork++ ground removal, HDBSCAN
   clustering, majority voting) to handle noisy LiDAR-image projections.
2. **Adaptive bounding box generation** — instances are categorized as
   rigid-static / rigid-dynamic / deformable; static objects use
   surface-aware filtering with SDFs from a static background mesh, dynamic
   ones use visibility-based box extension via 2D tracking.

OpenBox outperforms prior unsupervised auto-labeling baselines on Waymo,
Lyft Level 5, and nuScenes, and supports open-vocabulary annotation of novel
classes (strollers, fire hydrants, dogs, ...).

## Setup

All data-generation steps run in a single conda environment (`openbox`,
python 3.10 / PyTorch 2.3.1 cu121). The environment recipe, required
checkpoints, and the mmdetection-fork setup for the Grounding DINO + SAM2
stage are documented step-by-step in
[preprocess/README.md](preprocess/README.md) and
[Grounded-SAM-2/README.md](Grounded-SAM-2/README.md).

## Dataset Preparation + Context-aware Refinement

**[→ preprocess/README.md](preprocess/README.md)**

Turns raw Waymo TFRecords into everything the box-generation stage consumes —
camera-aligned point clouds, cross-camera instance point clouds with
context-aware refinement, per-point persistence scores, and a
dynamic-object-free background mesh. Each of the seven steps (0–6) is a
single script with explicit input/output roots.

## Adaptive Bounding Box Generation

Stage 2 (instance categorization, SDF-based surface-aware filtering, and
visibility-based box extension) is not part of this repository yet; it will be
released together with the full codebase.

## BibTeX

```bibtex
@inproceedings{Lee_OpenBox_NeurIPS_2025,
  author    = {In-Jae Lee and Mungyeom Kim and Kwonyoung Ryu and Pierre Musacchio and Jaesik Park},
  title     = {OpenBox: Annotate Any Bounding Boxes in 3D},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

## Acknowledgements

OpenBox builds on
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) /
[MMDetection](https://github.com/open-mmlab/mmdetection),
[SAM2](https://github.com/facebookresearch/sam2),
[Patchwork++](https://github.com/url-kaist/patchwork-plusplus),
[HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan),
[VDBFusion](https://github.com/PRBonn/vdbfusion),
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet), and
[CPD](https://github.com/hailanyi/CPD). We thank the authors of these
projects.
