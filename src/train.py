import os
import torch
print('Pytorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print('MMDet:', mmdet.__version__)

# Check mmcv installation
import mmcv
from mmcv import Config
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector

cfg = Config.fromfile('/mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py')


# Modify dataset type and path
classes = ['building']
cfg.dataset_type = 'COCODataset'

cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 8


cfg.data.test.img_prefix = 'train/images'
cfg.data.test.data_root = '/data/'
cfg.data.test.ann_file = 'val.json'
cfg.data.test.classes = classes

cfg.data.train.img_prefix = 'train/images'
cfg.data.train.data_root = '/data/'
cfg.data.train.ann_file = 'train.json'
cfg.data.train.classes = classes
cfg.data.train.seg_prefix= '/data/train/masks'

cfg.data.val.img_prefix = '/data/train/images'
cfg.data.val.data_root = '/data/'
cfg.data.val.ann_file = 'val.json'
cfg.data.val.classes = classes

for i in cfg.model.roi_head.bbox_head:
    i['num_classes'] = 1
for i in cfg.model.roi_head.mask_head:
    i['num_classes'] = 1
cfg.load_from = 'detectors_htc_r50_1x_coco-329b1453.pth'

cfg.work_dir = '/model'

cfg.optimizer.lr = 0.0002
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# # Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = ['bbox', 'segm']
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = [0]
cfg.device = 'cuda'

cfg.runner.max_epochs = 1

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
cfg.dump('/model/model.py')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
