# borrowed from https://gist.github.com/fmassa/c0fbb9fe7bf53b533b5cc241f5c8234c
# this is the main entrypoint
import torch
import time
import torch.nn as nn
import numpy as np
import tqdm
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg, set_global_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import build_model
# from detectron2.utils.analysis import _flatten_to_tuple
from detectron2.utils.flop_count import flop_count
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n


def _flatten_to_tuple(outputs):
    result = []
    if isinstance(outputs, torch.Tensor):
        result.append(outputs)
    elif isinstance(outputs, (list, tuple)):
        for v in outputs:
            result.extend(_flatten_to_tuple(v))
    elif isinstance(outputs, dict):
        for _, v in outputs.items():
            result.extend(_flatten_to_tuple(v))
    elif isinstance(outputs, Instances):
        result.extend(_flatten_to_tuple(outputs.get_fields()))
    elif isinstance(outputs, (Boxes, BitMasks, ImageList)):
        result.append(outputs.tensor)
    else:
        log_first_n(
            logging.WARN,
            f"Output of type {type(outputs)} not included in flops/activations count.",
            n=10,
        )
    return tuple(result)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()


# get the first 100 images of COCO val2017
# PATH_TO_COCO = "/path/to/coco/"
args = default_argument_parser().parse_args()
cfg = setup(args)
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)


class WrapModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        if isinstance(
                model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
        ):
            self.model = model.module
        else:
            self.model = model

    def forward(self, image):
        # jit requires the input/output to be Tensors
        inputs = [{"image": image}]
        outputs = self.model.forward(inputs)
        # Only the subgraph that computes the returned tuple of tensor will be
        # counted. So we flatten everything we found to tuple of tensors.
        return _flatten_to_tuple(outputs)


for dataset_name in cfg.DATASETS.TEST:
    dataset = build_detection_test_loader(cfg, dataset_name)
    images = []
    dataloader = iter(dataset)
    for _ in range(100):
        images.append(next(dataloader))

    results = {}
    # for model_name in ['detr_resnet50']:
    #     model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
    #     model.to(device)

    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(images):
            # inputs = img[0]['image']
            inputs = torch.rand(3, 1200, 800)
            res = flop_count(WrapModel(model).train(False), (inputs,))
            # t = measure_time(model, inputs)
            tmp.append(sum(res.values()))
            # tmp2.append(t)

    results = {'flops': fmt_res(np.array(tmp))}

    print('=============================')
    print('')
    for k, v in results.items():
        print(' ', k, ':', v)


# , 'time': fmt_res(np.array(tmp2))