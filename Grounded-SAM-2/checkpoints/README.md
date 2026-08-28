# Checkpoints (not in git)

Step 2 needs two checkpoints. Neither has to be fetched by hand:

| file | size | source |
|---|---|---|
| `sam2_hiera_large.pt` | 857 MB | official [SAM2 release](https://github.com/facebookresearch/sam2) — auto-downloaded into this directory on the first run of `gen_multicam_samv2_data_unified.py` (or pass `--sam2_checkpoint`) |
| `groundingdinofintune.pth` | 2.7 GB | Grounding DINO swin-b fine-tuned for this project — auto-downloaded from [HF `oliver0922/GroundingDINOfintune`](https://huggingface.co/oliver0922/GroundingDINOfintune) on the first run of `gen_multicam_samv2_data_unified.py` (or pass an explicit path via `--weights`) |

Nothing has to be placed by hand: the SAM2 weight lands in this directory, the
detector weight in the Hugging Face cache.
