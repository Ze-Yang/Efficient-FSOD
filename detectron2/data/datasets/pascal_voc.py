# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.config import global_cfg as cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_pascal_voc"]


# fmt: off
CLASS_NAMES_origin = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CLASS_NAMES_split1 = [
    'aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
    'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
    'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa'
]

CLASS_NAMES_split2 = [
    'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
    'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
    'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa'
]

CLASS_NAMES_split3 = [
    'aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
    'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa'
]

CLASS_NAMES_split4 = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "chair",
    "diningtable",  "motorbike", "person", "pottedplant",  "sofa", "train",
    "tvmonitor", "cat", "cow", "dog", "horse", "sheep",
]

CLASS_NAMES = (CLASS_NAMES_origin, CLASS_NAMES_split1,
               CLASS_NAMES_split2, CLASS_NAMES_split3,
               CLASS_NAMES_split4)
# fmt: on


def load_voc_instances(name: str, dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        name: name of the dataset
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    thing_classes = MetadataCatalog.get(name).thing_classes

    # legacy data shots, which are constructed by sampling only one instance of interest per image.
    # if 'train' in split and cfg.PHASE == 2:
    #     fileids = []
    #     for cls_name in CLASS_NAMES[cls_split]:
    #         with PathManager.open(os.path.join(dirname, "ImageSets", "Main", "1_box", cls_name + ".txt")) as f:
    #             fileids.extend(np.loadtxt(f, dtype=np.str)[:cfg.DATASETS.SHOT])
    # else:
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            difficult = int(obj.find("difficult").text)
            if cfg.PHASE == 2 and difficult == 1 and 'train' in split:
                continue
            if cfg.PHASE == 1 and cls not in thing_classes:
                continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": thing_classes.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(name, dirname, split))
    if 'split' in name:
        index = name.find('split')
        cls_split = int(name[index + len('split')])
    else:
        cls_split = 0
    if 'base' in name:
        class_name = CLASS_NAMES[cls_split][:15]
    else:
        class_name = CLASS_NAMES[cls_split]
    MetadataCatalog.get(name).set(
        thing_classes=class_name, dirname=dirname, year=year, split=split
    )
