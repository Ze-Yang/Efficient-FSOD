# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer

logger = logging.getLogger(__name__)

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        if cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
            if cfg.MODEL.RPN.ENABLE_DECOUPLE:
                self.affine_rpn = AffineLayer(num_channels=self.backbone.output_shape()['res4'].channels, bias=True)
            if cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                self.affine_rcnn = AffineLayer(num_channels=self.backbone.output_shape()['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info('Freeze backbone parameters')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            logger.info('Freeze proposal generator parameters')

        if cfg.MODEL.ROI_BOX_HEAD.FREEZE:
            if cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
                for p in self.roi_heads.box_head.parameters():
                    p.requires_grad = False
                logger.info('Freeze box_head parameters')
            elif cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
                for p in self.roi_heads.res5.parameters():
                    p.requires_grad = False
                logger.info('Freeze res5 parameters')
            else:
                raise NotImplementedError("Freeze roi_box_head does not support {}."
                                          .format(cfg.MODEL.ROI_HEADS.NAME))

        if cfg.MODEL.ROI_META_HEAD.FREEZE:
            for p in self.roi_heads.meta_head.parameters():
                p.requires_grad = False
            logger.info('Freeze roi_meta_head parameters')

        if cfg.MODEL.ROI_MASK_HEAD.FREEZE:
            for p in self.roi_heads.mask_head.parameters():
                p.requires_grad = False
            for p in self.roi_heads.mask_head.predictor.parameters():
                p.requires_grad = True
            logger.info('Freeze roi_mask_head parameters')

    def unfreeze_box_head(self, optimizer, scheduler):
        if self.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = True
        elif self.cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = True
        else:
            raise NotImplementedError("Unfreeze_box_head does not support {}."
                                      .format(self.cfg.MODEL.ROI_HEADS.NAME))
        base_lrs = []
        optimizer.param_groups = []
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.cfg.SOLVER.BASE_LR
            weight_decay = self.cfg.SOLVER.WEIGHT_DECAY
            if key.endswith("norm.weight") or key.endswith("norm.bias"):
                weight_decay = self.cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key.endswith(".bias"):
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = self.cfg.SOLVER.BASE_LR * self.cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = self.cfg.SOLVER.WEIGHT_DECAY_BIAS
            if ('res5' in key or 'box_head' in key) and self.cfg.PHASE == 2 and \
                    self.cfg.MODEL.ROI_BOX_HEAD.LR_FACTOR != 1.0:
                lr = lr * self.cfg.MODEL.ROI_BOX_HEAD.LR_FACTOR
                logger.info('The lr for {} is multiply by 0.1.'.format(key))
            optimizer.add_param_group({"params": [value], "lr": lr, "weight_decay": weight_decay, "initial_lr": lr})
            base_lrs.append(lr)
        scheduler.base_lrs = base_lrs
        logger.info('Unfreeze roi_box_head parameters')

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs, init=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
            init: bool, indicating whether it is initialization process.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, init=init)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            if self.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
                features_de_rpn = {k: decouple_layer(features[k], scale) for k in features}
            elif self.cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            else:
                raise NotImplementedError("Gradient decoupling does not support {}."
                                          .format(self.cfg.MODEL.ROI_HEADS.NAME))

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            if self.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
                features_de_rcnn = {k: decouple_layer(features[k], scale) for k in features}
            elif self.cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            else:
                raise NotImplementedError("Gradient decoupling does not support {}."
                                          .format(self.cfg.MODEL.ROI_HEADS.NAME))

        _, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, init=False):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
            init (bool): indicating whether it is initialization process.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if init:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                elif "targets" in batched_inputs[0]:
                    log_first_n(
                        logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
                    )
                    gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None
                features_de_rcnn = features
                if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                    scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                    if self.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
                        features_de_rcnn = {k: decouple_layer(features[k], scale) for k in features}
                    elif self.cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
                        features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
                    else:
                        raise NotImplementedError("Gradient decoupling does not support {}."
                                                  .format(self.cfg.MODEL.ROI_HEADS.NAME))
                return self.roi_heads(images, features_de_rcnn, None, gt_instances)
            else:
                features_de_rpn = features
                if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                    scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                    if self.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
                        features_de_rpn = {k: decouple_layer(features[k], scale) for k in features}
                    elif self.cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
                        features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
                    else:
                        raise NotImplementedError("Gradient decoupling does not support {}."
                                                  .format(self.cfg.MODEL.ROI_HEADS.NAME))
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features_de_rpn, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                features_de_rcnn = features
                if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                    scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                    if self.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
                        features_de_rcnn = {k: decouple_layer(features[k], scale) for k in features}
                    elif self.cfg.MODEL.ROI_HEADS.NAME == "Res5ROIHeads":
                        features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
                    else:
                        raise NotImplementedError("Gradient decoupling does not support {}."
                                                  .format(self.cfg.MODEL.ROI_HEADS.NAME))
                results, _ = self.roi_heads(images, features_de_rcnn, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info('Freeze backbone parameters')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            logger.info('Freeze proposal generator parameters')

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
