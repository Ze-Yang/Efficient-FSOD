# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
import logging
import detectron2.utils.comm as comm
from detectron2.config import global_cfg as cfg
import os
from .c2_model_loading import align_and_update_state_dicts
from typing import List, Optional


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # if cfg.SETTING == 'Incremental' and \
        #         cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads':
        #     logger = logging.getLogger(__name__)
        #     logger.info("Initializing box_head for novel classes.")
        #     dict = {}
        #     for key, value in checkpoint["model"].items():
        #         if 'box_head' in key:
        #             key = key.replace('box_head', 'box_head_novel')
        #             dict[key] = value
        #     checkpoint["model"].update(dict)
        # for non-caffe2 models, use standard ways to load it
        return super()._load_model(checkpoint)

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        # do not load 'iteration' unless args.resume is True
        # checkpointables is not None implies that args.resume is False
        if cfg.PHASE == 2 and checkpointables is not None:
            try:
                checkpoint.pop('iteration')
            except KeyError:
                logger = logging.getLogger(__name__)
                logger.warning("No key 'iteration' found in the loaded model.")
        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore

        # return any further checkpoint data
        return checkpoint
