# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import numpy as np
from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock
from .. import MetadataCatalog, DatasetCatalog
from .fsod_categories import FSOD_CATEGORIES_TRAIN, FSOD_CATEGORIES_TEST
# for debug
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.data.datasets.fsod_categories import FSOD_CATEGORIES_TRAIN, FSOD_CATEGORIES_TEST

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_fsod_json", "register_fsod_instances", "get_fsod_instances_meta"]


def register_fsod_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "fsod_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda cfg=None: load_fsod_json(cfg, json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_fsod_json(cfg, json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "fsod" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
        # id_map = meta.thing_dataset_id_to_contiguous_id
        # id_of_int = list(id_map.keys())[:60] if 'nonvoc' in dataset_name else list(id_map.keys())[:20]  # id of interest

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    # cls_num = {i: 0 for i in id_of_int}
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        # find = False
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            # cls_id = obj["category_id"]
            # if not find:
            #     if cls_num[cls_id] < cfg.DATASETS.SHOT:
            #         find = True
            #         cls_num[cls_id] += 1
            #     else:
            #         cls_id = -1
            # else:
            #     cls_id = -1

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
                # if 'train' in dataset_name and cfg.PHASE == 2:
                #     obj["category_id"] = id_map[cls_id] if cls_id != -1 else -1
                # else:
                #     obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        # if 'train' in dataset_name and cfg.PHASE == 2:
        #     if find:
        #         record["annotations"] = objs
        #         dataset_dicts.append(record)
        #     if min(cls_num.values()) == cfg.DATASETS.SHOT:
        #         break
        # else:
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def get_fsod_instances_meta(dataset_name, split):
    """
    Load FSOD metadata.

    Args:
        dataset_name (str): FSOD dataset name without the split name.
        split: split name of FSOD dataset.

    Returns:
        dict: FSOD metadata with keys: thing_classes
    """
    assert dataset_name == "fsod", "No built-in metadata for dataset {}".format(dataset_name)
    assert len(FSOD_CATEGORIES_TRAIN) == 800
    assert len(FSOD_CATEGORIES_TEST) == 200
    if split == 'fsod_train':
        FSOD_CATEGORIES = FSOD_CATEGORIES_TRAIN
    elif split == 'fsod_test':
        FSOD_CATEGORIES = FSOD_CATEGORIES_TEST
    else:
        raise ValueError("No {} split for FSOD dataset".format(split))
    cat_ids = [k["id"] for k in FSOD_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    fsod_categories = sorted(FSOD_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in fsod_categories]
    meta = {"thing_classes": thing_classes}
    return meta


if __name__ == "__main__":
    """
    Test the FSOD json dataset loader.

    Usage:
        python -m detectron2.data.datasets.fsod \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "fsod_train", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])
    dicts = load_fsod_json(None, sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    # import matplotlib.pyplot as plt
    # import cv2
    # def is_jpg(filename):
    #     try:
    #         i = Image.open(filename)
    #         return i.format
    #     except IOError:
    #         return False

    # for d in dicts:
    #     from PIL import Image, ImageOps
    #     img = Image.open('./datasets/fsod/images/part_2/train_part_d/d9/d9af31bc74d18334.jpg')
    #     image = ImageOps.exif_transpose(img)
    #     print(d["file_name"])
    # exit(0)
    #     if d["file_name"] in ['./datasets/fsod/images/part_1/n03476684/n03476684_8799.jpg',
    #                           './datasets/fsod/images/part_1/n03805280/n03805280_2499.jpg']:
    #         Image.open(d["file_name"]).save(d["file_name"])
    #     format = is_jpg(d["file_name"])
    #     if format == 'JPEG':
    #         pass
    #     else:
    #         try:
    #             Image.open(d["file_name"]).save(d["file_name"])
    #         except IOError:
    #             print("cannot convert", d["file_name"])
    #         if is_jpg(d["file_name"]) == 'JPEG':
    #             print(d["file_name"], 'convert successfully.')
    #     img = np.asarray(Image.open(d["file_name"]))
    #     if img.ndim == 2 or img.shape[-1] != 3:
    #         img = cv2.imread(d["file_name"])
    #         im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         plt.imshow(im_rgb)
    #         plt.show()
    #         cv2.imwrite(d["file_name"], img)
    #         print(d["file_name"])

    dirname = "fsod-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
