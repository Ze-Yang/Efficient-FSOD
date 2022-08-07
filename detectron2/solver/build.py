# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List
import torch
import logging
from detectron2.config import CfgNode

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
logger = logging.getLogger(__name__)


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        # elif 'reweight' in key:
        #     lr = cfg.SOLVER.BASE_LR * 10
        #     logger.info(
        #         'The lr for {} is multiply by {}.'.format(key, lr/cfg.SOLVER.BASE_LR)
        #     )
        # elif ('box_head.' in key or 'cls_score.' in key) and cfg.PHASE == 2 and cfg.METHOD == 'ours':
        #     lr = cfg.SOLVER.BASE_LR * 0.1
        #     logger.info(
        #         'The lr for {} is multiply by 0.1.'.format(key)
        #     )
        elif key.endswith(".bias"):
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if ('roi_heads.box_head' in key or 'roi_heads.res5' in key) and cfg.PHASE == 2 \
                and cfg.MODEL.ROI_BOX_HEAD.LR_FACTOR != 1.0:
            lr = lr * cfg.MODEL.ROI_BOX_HEAD.LR_FACTOR
            logger.info('The lr for {} is multiply by {}.'.format(key, cfg.MODEL.ROI_BOX_HEAD.LR_FACTOR))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
