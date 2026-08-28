_base_ = [
    './grounding_dino_swin-t_finetune_8xb2_1x_nus.py',
]

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'  # noqa
model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
)

train_dataloader = dict(
    batch_size=4,
     num_workers=4,)
val_dataloader = dict(
    batch_size=1,
     num_workers=2,)
test_dataloader = dict(
    batch_size=1,
     num_workers=2,)

auto_scale_lr = dict(enable=False, base_batch_size=32)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=20)
