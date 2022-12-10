model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
dataset_type = 'RawframeDataset'
data_root = 'data/wlasl/rawframes'
data_root_val = 'data/wlasl/rawframes'
split = 1
clip_length = 16
ann_file_train = 'data/wlasl/train_annotations.txt'
ann_file_val = 'data/wlasl/val_annotations.txt'
ann_file_test = 'data/wlasl/test_annotations.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Normalize', mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Normalize', mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/train_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='RandomCrop', size=112),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[104, 117, 128],
                std=[1, 1, 1],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/val_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='CenterCrop', crop_size=112),
            dict(
                type='Normalize',
                mean=[104, 117, 128],
                std=[1, 1, 1],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/test_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=10,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='CenterCrop', crop_size=112),
            dict(
                type='Normalize',
                mean=[104, 117, 128],
                std=[1, 1, 1],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 45
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(entity='cares', project='wlasl-100-c3d'),
            log_artifact=True)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/c3d_wlasl_1_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
