default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet'),
    logger=dict(type='LoggerHook', interval=5, _scope_='mmdet'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=2,
        _scope_='mmdet',
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmdet'),
    visualization=dict(type='DetVisualizationHook', _scope_='mmdet'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmdet')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer',
    _scope_='mmdet')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, _scope_='mmdet')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
resume = False
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1000,
    val_interval=10,
    dynamic_intervals=[(280, 1)],
    _scope_='mmdet')
val_cfg = dict(type='ValLoop', _scope_='mmdet')
test_cfg = dict(type='TestLoop', _scope_='mmdet')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=30),
    dict(
        type='CosineAnnealingLR',
        eta_min=5e-05,
        begin=990,
        end=1000,
        T_max=990,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
auto_scale_lr = dict(enable=False, base_batch_size=16)
dataset_type = 'CocoDataset'
data_root = 'balloon/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
        _scope_='mmdet'),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
        _scope_='mmdet'),
    dict(type='RandomCrop', crop_size=(640, 640), _scope_='mmdet'),
    dict(type='YOLOXHSVRandomAug', _scope_='mmdet'),
    dict(type='RandomFlip', prob=0.5, _scope_='mmdet'),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmdet'),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
        _scope_='mmdet'),
    dict(type='PackDetInputs', _scope_='mmdet')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True, _scope_='mmdet'),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmdet'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
        _scope_='mmdet')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmdet'),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='balloon/',
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='CachedMosaic',
                img_scale=(640, 640),
                pad_val=114.0,
                max_cached_images=20,
                random_pop=False),
            dict(
                type='RandomResize',
                scale=(1280, 1280),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='CachedMixUp',
                img_scale=(640, 640),
                ratio_range=(1.0, 1.0),
                max_cached_images=10,
                random_pop=False,
                pad_val=(114, 114, 114),
                prob=0.5),
            dict(type='PackDetInputs')
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(classes=('balloon', ), palette=[(220, 20, 60)])),
    pin_memory=True)
val_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='balloon/',
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(classes=('balloon', ), palette=[(220, 20, 60)])))
test_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='balloon/',
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(classes=('balloon', ), palette=[(220, 20, 60)])))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='balloon/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10),
    _scope_='mmdet')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='balloon/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10),
    _scope_='mmdet')
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100),
    _scope_='mmdet')
img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (960, 960),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'Pad',
                        'size': (960, 960),
                        'pad_val': {
                            'img': (114, 114, 114)
                        }
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]],
        _scope_='mmdet')
]
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
    _scope_='mmdet')
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
max_epochs = 1000
stage2_num_epochs = 1
base_lr = 0.001
interval = 10
custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=999,
        switch_pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomResize',
                scale=(640, 640),
                ratio_range=(0.1, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad', size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ])
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
train_batch_size_per_gpu = 8
train_num_workers = 2
metainfo = dict(classes=('balloon', ), palette=[(220, 20, 60)])
launcher = 'none'
work_dir = './work_dirs\\rtmdet_tiny_1xb4-20e_balloon'
