# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from visualize_reweight import tensor_display

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    build_dataloader
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    inference_context
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

logger = logging.getLogger("detectron2")


def init_reweight(cfg, model, data_loader):
    """
    Initialize reweight parameters.
    """
    logger.info("Initializing reweight parameters...")
    cls_dict_feat = {i: [] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
    cls_dict_act = {i: [] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
    with inference_context(model), torch.no_grad():
        for data, _ in zip(data_loader, range(cfg.INIT_ITERS)):
            feat, act = model(data, init=True)
            cls_dict_act = {key: cls_dict_act[key] + act[key] for key in cls_dict_act.keys()}
            cls_dict_feat = {key: cls_dict_feat[key] + feat[key] for key in cls_dict_feat.keys()}  # per image accumulation

    cls_dict_feat = {key: torch.stack(value, dim=0) if value else torch.empty(0, device='cuda')
                for key, value in cls_dict_feat.items()}  # concat to tensor
    dict_list = comm.all_gather(cls_dict_feat)  # gather from all GPUs
    cls_dict_feat = {key: torch.cat([dict_list[i][key] for i in range(len(dict_list))], dim=0)
                for key in cls_dict_feat.keys()}  # concat the results
    # if comm.is_main_process():
    #     file = 'results/support_10shot_in.pkl'
    #     import pickle
    #     with open(file, 'wb') as f:
    #         pickle.dump(cls_dict_feat, f, pickle.HIGHEST_PROTOCOL)
    # comm.synchronize()
    # exit(0)
    cls_feat = torch.sigmoid(torch.stack([x.mean(0) for x in cls_dict_feat.values()], dim=0))
    cls_feat = cls_feat / cls_feat.mean(1)[:, None]
    if cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads_Incre':
        cls_feat = torch.cat([torch.ones((1, cls_feat.size(1)), device=cls_feat.device), cls_feat[15:]], dim=0)

    cls_dict_act = {key: torch.stack(value, dim=0) if value else torch.empty(0, device='cuda')
                     for key, value in cls_dict_act.items()}  # concat to tensor
    dict_list = comm.all_gather(cls_dict_act)  # gather from all GPUs
    cls_dict_act = {key: torch.cat([dict_list[i][key] for i in range(len(dict_list))], dim=0)
                     for key in cls_dict_act.keys()}  # concat the results
    cls_act = torch.stack([x.mean(0) for x in cls_dict_act.values()], dim=0)
    cls_act = cls_act / cls_act.norm(dim=1)[:, None]

    if comm.get_world_size() > 1:
        # model.module.roi_heads.reweight.weight.data = cls_tensor if cfg.MODEL.MASK_ON else cls_tensor[15:]
        model.module.roi_heads.reweight.weight.data = cls_feat
        if cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads_Incre':
            model.module.roi_heads.box_predictor.cls_score_noval.weight.data = cls_act[15:]
        # else:
        #     cls_act = torch.cat([cls_act, model.module.roi_heads.box_predictor.cls_score.weight[-1][None, :]], dim=0)
        #     model.module.roi_heads.box_predictor.cls_score.weight.data = cls_act
    else:
        # model.roi_heads.reweight.weight.data = cls_tensor if cfg.MODEL.MASK_ON else cls_tensor[15:]
        model.roi_heads.reweight.weight.data = cls_feat
        if cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads_Incre':
            model.roi_heads.box_predictor.cls_score_noval.weight.data = cls_act[15:]
        # else:
        #     cls_act = torch.cat([cls_act, model.roi_heads.box_predictor.cls_score.weight[-1][None, :]], dim=0)
        #     model.roi_heads.box_predictor.cls_score.weight.data = cls_act
            # logger.info(cls_act.size(), model.roi_heads.box_predictor.cls_score.weight.data.size())

    num_cls = [value.size(0) for value in cls_dict_feat.values()]
    logger.info('{}'.format(num_cls))
    del cls_dict_feat
    del dict_list
    del num_cls
    del cls_feat
    del cls_dict_act
    del cls_act
    return None


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name, output_folder)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, args=None):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        if args is not None and args.retest is True:
            evaluator.load()
            results_i = evaluator.evaluate()
        else:
            results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, cfg, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.LOAD_FILE, resume=resume).get("iteration", -1) + 1 if cfg.PHASE == 2 \
        else checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # file = 'results/learned_fullshot_5999.pkl'
    # a = model.roi_heads.reweight.weight.data
    # import pickle
    # with open(file, 'wb') as f:
    #     pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    dataset, dataset_dicts = build_detection_train_loader(cfg, get_dataset=True)

    if cfg.PHASE == 2 and cfg.METHOD == 'ours':
        # initialize the reweight parameters
        init_reweight(cfg, model, build_dataloader(cfg, dataset, dataset_dicts))
        if comm.get_world_size() > 1:
            logger.info('Reweight after init: {}'.format(model.module.roi_heads.reweight.weight))
            if cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads_Incre':
                logger.info('Cls_N after init: {}'.format(model.module.roi_heads.box_predictor.cls_score_noval.weight))
        else:
            logger.info('Reweight after init: {}'.format(model.roi_heads.reweight.weight))
            if cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads_Incre':
                logger.info('Cls_N after init: {}'.format(model.roi_heads.box_predictor.cls_score_noval.weight))
    # elif cfg.PHASE == 2 and cfg.METHOD == 'ft':
    #     # For incremental baseline finetuning
    #     checkpoint = checkpointer._load_file(cfg.LOAD_FILE)
    #     checkpoint_state_dict = checkpoint.pop("model")
    #     cls_score = checkpoint_state_dict['roi_heads.box_predictor.cls_score.weight']
    #     if comm.get_world_size() > 1:
    #         tmp = model.module.roi_heads.box_predictor.cls_score.weight.data
    #         tmp[:15] = cls_score[:15]
    #         tmp[-1] = cls_score[-1]
    #         model.module.roi_heads.box_predictor.cls_score.weight.data = tmp
    #         logger.info('Init Base and Bg classifier')
    #     else:
    #         tmp = model.roi_heads.box_predictor.cls_score.weight.data
    #         tmp[:15] = cls_score[:15]
    #         tmp[-1] = cls_score[-1]
    #         model.roi_heads.box_predictor.cls_score.weight.data = tmp
    #         logger.info('Init Base and Bg classifier')

    # mean = torch.mean(model.roi_heads.reweight.weight.data, dim=1)
    # var = torch.var(model.roi_heads.reweight.weight.data, dim=1)
    # tensor_display(model.roi_heads.reweight.weight.data)
    # exit(0)

    assert model.training, 'Model.train() must be True during training.'
    logger.info("Starting training from iteration {}".format(start_iter))
    data_loader = build_dataloader(cfg, dataset, dataset_dicts)
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.LOAD_FILE, resume=args.resume
        )
        return do_test(cfg, model, args)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
