model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        partial_bn=True),
    cls_head=dict(
        type='TRNHead',
        num_classes=174,
        in_channels=2048,
        num_segments=8,
        spatial_type='avg',
        relation_type='TRNMultiScale',
        hidden_dim=256,
        dropout_ratio=0.8,
        init_std=0.001),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[30, 45])
total_epochs = 50
checkpoint_config = dict(interval=1)
log_config = dict(interval=10,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='WandbLoggerHook',
                           init_kwargs={
                               'entity': "cares",
                               'project': "wlasl-model-ablation",
                               'group': "trn",
                           },
                           log_artifact=True)
                  ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'RawframeDataset'
data_root = 'data/wlasl/rawframes'
data_root_val = 'data/wlasl/rawframes'
ann_file_train = 'data/wlasl/train_annotations.txt'
ann_file_val = 'data/wlasl/test_annotations.txt'
ann_file_test = 'data/wlasl/test_annotations.txt'
gpu_ids = range(0, 1)
sthv2_flip_label_map = dict({
    86: 87,
    87: 86,
    93: 94,
    94: 93,
    166: 167,
    167: 166
})
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        flip_label_map=dict({
            86: 87,
            87: 86,
            93: 94,
            94: 93,
            166: 167,
            167: 166
        })),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        twice_sample=True,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/train_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames', clip_len=1, frame_interval=1,
                num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(
                type='Flip',
                flip_ratio=0.5,
                flip_label_map=dict({
                    86: 87,
                    87: 86,
                    93: 94,
                    94: 93,
                    166: 167,
                    167: 166
                })),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/test_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/test_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                twice_sample=True,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='ThreeCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
find_unused_parameters = True
work_dir = './work_dirs/trn_r50_1x1x8_50e_sthv2_rgb/'
omnisource = False
module_hooks = []
