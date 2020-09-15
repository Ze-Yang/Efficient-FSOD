import torch
import torch.nn.functional as F
import detectron2.utils.comm as comm
from detectron2.evaluation import inference_context
import logging

logger = logging.getLogger(__name__)


def weight_cal(cfg, model, data_loader):
    """
    Calculate the weight parameters for initialization.
    """
    logger.info("Calculating the weight parameters...")
    if cfg.METHOD == 'ours':
        cls_dict_feat = {i: [] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
    cls_dict_act = {i: [] for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
    with inference_context(model), torch.no_grad():
        for data, _ in zip(data_loader, range(cfg.INIT_ITERS)):
            if cfg.METHOD == 'ours':
                feat, act = model(data, init=True)
                cls_dict_feat = {key: cls_dict_feat[key] + feat[key] for key in cls_dict_feat.keys()}
            else:
                act = model(data, init=True)
            cls_dict_act = {key: cls_dict_act[key] + act[key] for key in cls_dict_act.keys()}

    if cfg.METHOD == 'ours':
        cls_dict_feat = {key: torch.stack(value, dim=0) if value else torch.empty(0, device='cuda')
                         for key, value in cls_dict_feat.items()}  # concat to tensor
        dict_list = comm.all_gather(cls_dict_feat)  # gather from all GPUs
        cls_dict_feat = {key: torch.cat([dict_list[i][key] for i in range(len(dict_list))], dim=0)
                         for key in cls_dict_feat.keys()}  # concat the results
        cls_feat = torch.sigmoid(torch.stack([x.mean(0) for x in cls_dict_feat.values()], dim=0))
        cls_feat = cls_feat / cls_feat.mean(1)[:, None]
        if cfg.SETTING == 'Incremental':
            cls_feat = torch.cat([torch.ones((1, cls_feat.size(1)), device=cls_feat.device), cls_feat[15:]], dim=0)

    cls_dict_act = {key: torch.stack(value, dim=0) if value else torch.empty(0, device='cuda')
                     for key, value in cls_dict_act.items()}  # concat to tensor
    dict_list = comm.all_gather(cls_dict_act)  # gather from all GPUs
    cls_dict_act = {key: torch.cat([dict_list[i][key] for i in range(len(dict_list))], dim=0)
                     for key in cls_dict_act.keys()}  # concat the results
    cls_act = torch.stack([F.normalize(x).mean(0) for x in cls_dict_act.values()], dim=0)
    cls_act = F.normalize(cls_act)

    # num_cls = [value.size(0) for value in cls_dict_feat.values()]
    # logger.info('{}'.format(num_cls))
    del cls_dict_act
    del dict_list
    del data_loader
    if cfg.METHOD == 'ours':
        del cls_dict_feat
        return cls_feat, cls_act
    else:
        return cls_act


def init_weight(cfg, model, dataloader, checkpointer):
    checkpoint = checkpointer._load_file(cfg.MODEL.WEIGHTS)
    checkpoint_state_dict = checkpoint.pop("model")
    cls_score = checkpoint_state_dict['roi_heads.box_predictor.cls_score.weight']
    # if cfg.METHOD == 'ours':
    #     # initialize the reweight parameters
    #     cls_feat, cls_novel = weight_cal(cfg, model, dataloader)
    #     if comm.get_world_size() > 1:
    #         model.module.roi_heads.reweight.weight.data = cls_feat
    #         logger.info('Reweight after init: {}'.format(model.module.roi_heads.reweight.weight))
    #         if cfg.SETTING == 'Incremental':
    #             model.module.roi_heads.box_predictor.cls_score_novel.weight.data = cls_novel[15:]
    #             logger.info('Cls_N after init: {}'.format(model.module.roi_heads.box_predictor.cls_score_novel.weight))
    #     else:
    #         model.roi_heads.reweight.weight.data = cls_feat
    #         logger.info('Reweight after init: {}'.format(model.roi_heads.reweight.weight))
    #         if cfg.SETTING == 'Incremental':
    #             model.roi_heads.box_predictor.cls_score_novel.weight.data = cls_novel[15:]
    #             logger.info('Cls_N after init: {}'.format(model.roi_heads.box_predictor.cls_score_novel.weight))
    # elif cfg.METHOD in ['meta', 'imprinted', 'ft']:
    # only support voc
    if cfg.METHOD in ['meta', 'imprinted']:
        cls_all = weight_cal(cfg, model, dataloader)
        num_base = 0 if cfg.SETTING == 'Transfer' else cls_score.size(0) - 1
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
        if comm.get_world_size() > 1:
            cls_score = model.module.roi_heads.box_predictor.cls_score.weight.data
            cls_score[:num_base] = cls_base[:num_base]
            if cfg.METHOD in ['meta', 'imprinted']:
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
            # logger.info('ram_norm:{}'.format(cls_score.norm(dim=-1).mean()))
            # logger.info('ram_var:{}'.format(cls_score.var(dim=-1).mean()))
            cls_score[:num_base] = cls_base[:num_base]
            if cfg.METHOD in ['meta', 'imprinted']:
                cls_score[num_base:-1] = cls_novel
            cls_score[-1] = cls_bg
            # logger.info('base_norm:{}'.format(cls_base.norm(dim=-1).mean()))
            # logger.info('base_var:{}'.format(cls_base.var(dim=-1).mean()))
            # logger.info('novel_norm:{}'.format(cls_novel.norm(dim=-1).mean()))
            # logger.info('novel_var:{}'.format(cls_novel.var(dim=-1).mean()))
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
                    logger.info("'roi_heads.box_predictor.cls_score.bias' not initialized"
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
        if cfg.METHOD == 'meta':
            if hasattr(model.module.roi_heads, 'meta_head'):
                del model.module.roi_heads.meta_head
            if hasattr(model.module.roi_heads, 'meta_pooler'):
                del model.module.roi_heads.meta_pooler
            model.module.roi_heads.meta_on = False
        checkpointer.save("model_init", iteration=0)
