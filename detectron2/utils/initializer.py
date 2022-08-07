import torch
import torch.nn.functional as F
import detectron2.utils.comm as comm
from detectron2.evaluation import inference_context
import logging

logger = logging.getLogger(__name__)


def weight_cal(cfg, model, data_loader, cls_norm):
    """
    Calculate the weight parameters for initialization.
    """
    logger.info("Calculating the weight parameters...")
    cls_dict_act = {i: [] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
    with inference_context(model), torch.no_grad():
        for data, _ in zip(data_loader, range(cfg.INIT_ITERS)):
            act = model(data, init=True)
            cls_dict_act = {key: cls_dict_act[key] + act[key] for key in cls_dict_act.keys()}

    cls_dict_act = {key: torch.stack(value, dim=0) if value else torch.empty(0, device='cuda')
                     for key, value in cls_dict_act.items()}  # concat to tensor
    dict_list = comm.all_gather(cls_dict_act)  # gather from all GPUs
    cls_dict_act = {key: torch.cat([dict_list[i][key] for i in range(len(dict_list))], dim=0)
                     for key in cls_dict_act.keys()}  # concat the results

    if cfg.MODEL.ROI_BOX_HEAD.PREDICTOR == 'FastRCNNOutputLayers':
        if cfg.MODEL.ROI_BOX_HEAD.ALR:
            # KI w/ ALR
            cls_avg = cls_norm.mean()
            norm_base = torch.stack([x.mean(0).norm() for x in cls_dict_act.values()])[:cls_norm.size(0)]
            base_avg = norm_base.mean()
            ratio = base_avg / cls_avg
            cls_act = torch.stack([x.mean(0) / ratio for x in cls_dict_act.values()], dim=0)
        else:
            # KI w/o ALR
            cls_act = F.normalize(torch.stack([x.mean(0) for x in cls_dict_act.values()], dim=0))
    elif cfg.MODEL.ROI_BOX_HEAD.PREDICTOR == 'CosineSimOutputLayers':
        cls_act = F.normalize(torch.stack([F.normalize(x).mean(0) for x in cls_dict_act.values()], dim=0))
    else:
        raise NotImplementedError("KI initializer does not support {}."
                                  .format(cfg.MODEL.ROI_BOX_HEAD.PREDICTOR))
    return cls_act


def init_weight(cfg, model, dataloader, checkpointer):
    checkpoint = checkpointer._load_file(cfg.MODEL.WEIGHTS)
    checkpoint_state_dict = checkpoint.pop("model")
    cls_score = checkpoint_state_dict['roi_heads.box_predictor.cls_score.weight']
    cls_norm = cls_score[:-1].norm(dim=1).cuda()
    num_base = 0 if cfg.SETTING == 'Transfer' else cls_score.size(0) - 1

    if cfg.METHOD in ['ours']:
        cls_all = weight_cal(cfg, model, dataloader, cls_norm)
        cls_novel = cls_all[num_base:]
        logger.info('Novel class weights:\n{}'.format(cls_novel))

    if cfg.MODEL.ROI_BOX_HEAD.PREDICTOR == 'CosineSimOutputLayers':
        cls_score_norm = F.normalize(cls_score)
        cls_base, cls_bg = cls_score_norm.split([cls_score_norm.size(0) - 1, 1], dim=0)
    else:
        cls_base, cls_bg = cls_score.split([cls_score.size(0) - 1, 1], dim=0)
        cls_bias = checkpoint_state_dict.get('roi_heads.box_predictor.cls_score.bias')
        if cls_bias is not None:
            bias_base, bias_bg = cls_bias.split([cls_bias.size(0) - 1, 1])

    # initialize box classifier
    if comm.get_world_size() > 1:
        cls_score = model.module.roi_heads.box_predictor.cls_score.weight.data
        cls_score[:num_base] = cls_base[:num_base]
        if cfg.METHOD in ['ours']:
            cls_score[num_base:-1] = cls_novel
        cls_score[-1] = cls_bg
        model.module.roi_heads.box_predictor.cls_score.weight.data = cls_score
        if cls_bias is not None:
            if model.module.roi_heads.box_predictor.cls_score.bias is not None:
                bias = model.module.roi_heads.box_predictor.cls_score.bias.data
                bias[:num_base] = bias_base[:num_base]
                bias[-1] = bias_bg
                model.module.roi_heads.box_predictor.cls_score.bias.data = bias
                logger.info("'roi_heads.box_predictor.cls_score.bias' initialized.")
        else:
            if model.module.roi_heads.box_predictor.cls_score.bias is not None:
                logger.info("'roi_heads.box_predictor.cls_score.bias' not initialized, "
                            "since it does not exist in the checkpoint.")
    else:
        cls_score = model.roi_heads.box_predictor.cls_score.weight.data
        cls_score[:num_base] = cls_base[:num_base]
        if cfg.METHOD in ['ours']:
            cls_score[num_base:-1] = cls_novel
        cls_score[-1] = cls_bg
        model.roi_heads.box_predictor.cls_score.weight.data = cls_score
        if cls_bias is not None:
            if model.roi_heads.box_predictor.cls_score.bias is not None:
                bias = model.roi_heads.box_predictor.cls_score.bias.data
                bias[:num_base] = bias_base[:num_base]
                bias[-1] = bias_bg
                model.roi_heads.box_predictor.cls_score.bias.data = bias
                logger.info("'roi_heads.box_predictor.cls_score.bias' initialized.")
        else:
            if model.roi_heads.box_predictor.cls_score.bias is not None:
                logger.info("'roi_heads.box_predictor.cls_score.bias' not initialized, "
                            "since it does not exist in the checkpoint.")

    # initialize box regressor
    if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG is False:
        bbox_pred_w_base = checkpoint_state_dict['roi_heads.box_predictor.bbox_pred.weight']
        bbox_pred_b_base = checkpoint_state_dict['roi_heads.box_predictor.bbox_pred.bias']
        if comm.get_world_size() > 1:
            bbox_pred_w = model.module.roi_heads.box_predictor.bbox_pred.weight.data
            bbox_pred_w[:4*num_base] = bbox_pred_w_base[:4*num_base]
            model.module.roi_heads.box_predictor.bbox_pred.weight.data = bbox_pred_w
            bbox_pred_b = model.module.roi_heads.box_predictor.bbox_pred.bias.data
            bbox_pred_b[:4*num_base] = bbox_pred_b_base[:4*num_base]
            model.module.roi_heads.box_predictor.bbox_pred.bias.data = bbox_pred_b
        else:
            bbox_pred_w = model.roi_heads.box_predictor.bbox_pred.weight.data
            bbox_pred_w[:4*num_base] = bbox_pred_w_base[:4*num_base]
            model.roi_heads.box_predictor.bbox_pred.weight.data = bbox_pred_w
            bbox_pred_b = model.roi_heads.box_predictor.bbox_pred.bias.data
            bbox_pred_b[:4*num_base] = bbox_pred_b_base[:4*num_base]
            model.roi_heads.box_predictor.bbox_pred.bias.data = bbox_pred_b

    if cfg.METHOD in ['ours']:
        checkpointer.save("model_init", iteration=0)
