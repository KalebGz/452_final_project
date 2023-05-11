checkpoint_config = dict(interval=20) # save n  epochs
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])log
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/media/scazlab/UNTITLED/20_balls/mmdetection/checkpoints/yolact_r101_1x8_coco_20200908-4cbe9101' #  coco weights
resume_from = None
workflow = [('train', 1), ('val', 1)]
img_size = 550
model = dict(
    type='YOLACT',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        zero_init_residual=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=1, # change num classes here
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[
                7.971014492753623, 15.714285714285714, 30.555555555555557,
                61.111111111111114, 110.0
            ],
            centers=[(3.9855072463768115, 3.9855072463768115),
                     (7.857142857142857, 7.857142857142857),
                     (15.277777777777779, 15.277777777777779),
                     (30.555555555555557, 30.555555555555557), (55.0, 55.0)]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True),
    mask_head=dict(
        type='YOLACTProtonet',
        in_channels=256,
        num_protos=32,
        num_classes=1, #  num classes 
        max_masks_to_train=100,
        loss_mask_weight=6.125),
    segm_head=dict(
        type='YOLACTSegmHead',
        num_classes=1, #  num classes 
        in_channels=256,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.4, 
        top_k=200,
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = '/media/scazlab/UNTITLED/20_balls'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomCrop', crop_size=(480, 640)), # height x width
    dict(type='Resize', img_scale=(480, 640), keep_ratio=True), 
    dict(type='RandomFlip', flip_ratio=0.5), # !!!stack overflow mmdetection enforces flipping augmentation
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'), 
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2, # batch size
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/media/scazlab/UNTITLED/20_balls/ouput.json', #json
        img_prefix=
        '/media/scazlab/UNTITLED/20_balls/images', # imagewes
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomCrop', crop_size=(480, 640)), 
            dict(type='Resize', img_scale=(480, 640), keep_ratio=True), 
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        classes=('1'), # all classes in the dataset
    val=dict(
        type='CocoDataset',
        ann_file=
        '/media/scazlab/UNTITLED/20_balls/ouput.json', # json
        img_prefix=
        '/media/scazlab/UNTITLED/20_balls/images', # images 
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('1')), 
    t   est=dict(
        type='CocoDataset',
        aann_file=
        '/media/scazlab/UNTITLED/20_balls/ouput.json', #json
        img_prefix=
        '/media/scazlab/UNTITLED/20_balls/images', #  images 
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(480, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('1'))) #  classes 
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005) 
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])
runner = dict(type='EpochBasedRunner', max_epochs=100) # number epochs
cudnn_benchmark = True
evaluation = dict(metric=['bbox', 'segm'])
classes = ('1') # all classes in the dataset
work_dir = '/media/scazlab/UNTITLED/20_balls/trained_weights' # where to store weight
gpu_ids = range(0, 1)
