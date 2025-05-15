import os
import pprint
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

import logging

options = edict()

options.name = 'p2m'
options.version = 20250513
options.num_workers = 1
options.num_gpus = 1
options.pin_memory = True

options.log_dir = "logs"
options.log_level = "info"
options.summary_dir = "summary"
options.checkpoint_dir = "checkpoints"
# options.checkpoint = 'C:/Users/Administrator/Desktop/Pixel2Mesh/datasets/pretrained/resnet.pth.tar'
options.checkpoint = 'C:/Users/Administrator/Desktop/Pixel2Mesh/checkpoints/3DBubbles_001000.pt'

options.dataset = edict()
options.dataset.name = "shapenet"
options.dataset.original_data_path = 'C:/DataSet_DOOR/dataset_Reconstruction_normalization'
options.dataset.temporal_data_path = 'C:/DataSet_DOOR/dataset_Reconstruction_normalization/self_supervision'
options.dataset.sequence_length = 32
options.dataset.view_count = 6
options.dataset.spatial_data_path = 'C:/DataSet_DOOR/dataset_Reconstruction_normalization/projection'
options.dataset.subset_train = None
options.dataset.subset_eval = "eval"
options.dataset.camera_f = [248.0, 248.0]
options.dataset.camera_c = [111.5, 111.5]
options.dataset.mesh_pos = [0., 0., -0.8]
options.dataset.normalization = True
# options.dataset.normalization = False
options.dataset.num_classes = 1

options.dataset.shapenet = edict()
options.dataset.shapenet.num_points = 2000
options.dataset.shapenet.resize_with_constant_border = False

options.dataset.predict = edict()
options.dataset.predict.folder = "C:/Users/Administrator/Desktop/Pixel2Mesh/predict/images"

options.model = edict()
options.model.name = "pixel2mesh"
options.model.hidden_dim = 192
options.model.last_hidden_dim = 192
options.model.coord_dim = 3
options.model.backbone = "resnet50"
options.model.gconv_activation = True
# provide a boundary for z, so that z will never be equal to 0, on denominator
# if z is greater than 0, it will never be less than z;
# if z is less than 0, it will never be greater than z.
options.model.z_threshold = 0
# align with original tensorflow model
# please follow experiments/tensorflow.yml
options.model.align_with_tensorflow = False

options.loss = edict()
options.loss.weights = edict()
options.loss.weights.normal = 1.6e-4
options.loss.weights.edge = 0.3
options.loss.weights.laplace = 0.5
options.loss.weights.move = 0.1
options.loss.weights.constant = 1.
options.loss.weights.chamfer = [1., 1., 1.]
options.loss.weights.chamfer_opposite = 1.
options.loss.weights.reconst = 0.

options.loss.original_weight = 1
options.loss.temporal_weight = 0.5
options.loss.spatial_weight = 0.5
options.loss.spatial_consistency_weight = 0.5
options.loss.spatial_gt_weight = 0.5

options.train = edict()
options.train.num_epochs = 500
options.train.batch_size = 32
options.train.spatial_batch_size = 1
options.train.temporal_batch_size = 1
options.train.summary_steps = 50
options.train.checkpoint_steps = 1000
options.train.test_epochs = 10
options.train.use_augmentation = True
options.train.shuffle = True

options.test = edict()
options.test.dataset = []
options.test.summary_steps = 50
options.test.batch_size = 50
options.test.shuffle = False
options.test.weighted_mean = False

options.optim = edict()
options.optim.name = "adam"

options.optim.adam_beta1 = 0.9
options.optim.sgd_momentum = 0.9
options.optim.lr = 5.0E-5
options.optim.wd = 1.0E-6
options.optim.lr_step = [30, 45]
options.optim.lr_factor = 0.1

def create_logger(cfg, phase='train'):
    log_file = '{}_{}.log'.format(cfg.version, phase)
    final_log_file = os.path.join(cfg.log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    if cfg.log_level == "info":
        logger.setLevel(logging.INFO)
    elif cfg.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError("Log level has to be one of info and debug")
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def _update_dict(full_key, val, d):
    for vk, vv in val.items():
        if vk not in d:
            raise ValueError("{}.{} does not exist in options".format(full_key, vk))
        if isinstance(vv, list):
            d[vk] = np.array(vv)
        elif isinstance(vv, dict):
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:
            d[vk] = vv


def _update_options(options_file):
    # do scan twice
    # in the first round, MODEL.NAME is located so that we can initialize MODEL.EXTRA
    # in the second round, we update everything

    with open(options_file) as f:
        options_dict = yaml.safe_load(f)
        # do a dfs on `BASED_ON` options files
        if "based_on" in options_dict:
            for base_options in options_dict["based_on"]:
                _update_options(os.path.join(os.path.dirname(options_file), base_options))
            options_dict.pop("based_on")
        _update_dict("", options_dict, options)


def update_options(options_file):
    _update_options(options_file)


def gen_options(options_file):
    def to_dict(ed):
        ret = dict(ed)
        for k, v in ret.items():
            if isinstance(v, edict):
                ret[k] = to_dict(v)
            elif isinstance(v, np.ndarray):
                ret[k] = v.tolist()
        return ret

    cfg = to_dict(options)

    with open(options_file, 'w') as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False)


def slugify(filename):
    filename = os.path.relpath(filename, ".")
    if filename.startswith("experiments/"):
        filename = filename[len("experiments/"):]
    return os.path.splitext(filename)[0].lower().replace("/", "_").replace(".", "_")


def reset_options(options, args, phase='train'):
    if hasattr(args, "batch_size") and args.batch_size:
        options.train.batch_size = options.test.batch_size = args.batch_size
    if hasattr(args, "version") and args.version:
        options.version = args.version
    if hasattr(args, "num_epochs") and args.num_epochs:
        options.train.num_epochs = args.num_epochs
    if hasattr(args, "checkpoint") and args.checkpoint:
        options.checkpoint = args.checkpoint
    if hasattr(args, "folder") and args.folder:
        options.dataset.predict.folder = args.folder
    if hasattr(args, "gpus") and args.gpus:
        options.num_gpus = args.gpus
    if hasattr(args, "shuffle") and args.shuffle:
        options.train.shuffle = options.test.shuffle = True

    options.name = args.name

    if options.version is None:
        prefix = ""
        # if args.options:
        #     prefix = slugify(args.options) + "_"
        options.version = prefix + datetime.now().strftime('%m%d%H%M%S')  # ignore %Y
    options.log_dir = os.path.join(options.log_dir, options.name)
    print('=> creating {}'.format(options.log_dir))
    os.makedirs(options.log_dir, exist_ok=True)

    # options.checkpoint_dir = os.path.join(options.checkpoint_dir, options.name, options.version)
    # print('=> creating {}'.format(options.checkpoint_dir))
    # os.makedirs(options.checkpoint_dir, exist_ok=True)

    # options.summary_dir = os.path.join(options.summary_dir, options.name, options.version)
    # print('=> creating {}'.format(options.summary_dir))
    # os.makedirs(options.summary_dir, exist_ok=True)

    logger = create_logger(options, phase=phase)
    options_text = pprint.pformat(vars(options))
    logger.info(options_text)

    print('=> creating summary writer')
    writer = SummaryWriter(options.summary_dir)

    return logger, writer


if __name__ == "__main__":
    parser = ArgumentParser("Read options and freeze")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    update_options(args.input)
    gen_options(args.output)
