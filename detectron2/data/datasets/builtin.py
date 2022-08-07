# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from .fsod import register_fsod_instances, get_fsod_instances_meta
from .lvis import register_lvis_instances, get_lvis_instances_meta
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .pascal_voc import register_pascal_voc
from .builtin_meta import _get_builtin_metadata


# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2014_valminusminival_nonvoc_split": (
        "coco/val2014",
        "coco/annotations/split_nonvoc_instances_valminusminival2014.json",
    ),

    "coco_2014_valminusminival_voc_split": (
        "coco/val2014",
        "coco/annotations/split_voc_instances_valminusminival2014.json",
    ),
    "coco_2014_train_10shot_FSRW_voc_split": ("coco/trainval2014", "coco/annotations/novel_10shot.json"),
    "coco_2014_train_10shot_MPSR_all": ("coco/train2014", "coco/annotations/instances_train2014_10shot_novel_standard.json"),
    "coco_2014_val_10shot_MPSR_all": ("coco/val2014", "coco/annotations/instances_val2014_10shot_novel_standard.json"),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_train_voc_split": ("coco/train2017", "coco/annotations/split_voc_instances_train2017.json"),
    "coco_2017_train_nonvoc_split": ("coco/train2017", "coco/annotations/split_nonvoc_instances_train2017.json"),
    "coco_2017_train_all": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_val_voc_split": ("coco/val2017", "coco/annotations/split_voc_instances_val2017.json"),
    "coco_2017_val_nonvoc_split": ("coco/val2017", "coco/annotations/split_nonvoc_instances_val2017.json"),
    "coco_2017_val_all": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name, key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


# ==== Predefined datasets and splits for FSOD ==========


_PREDEFINED_SPLITS_FSOD = {
    "fsod": {
        "fsod_train": ("fsod/images", "fsod/annotations/fsod_train.json"),
        "fsod_test": ("fsod/images", "fsod/annotations/fsod_test.json"),
    }
}


def register_all_fsod(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_FSOD.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_fsod_instances(
                key,
                get_fsod_instances_meta(dataset_name, key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_train_base": ("coco/", "lvis/lvis_v1_train_base.json"),
        "lvis_v1_train_novel": ("coco/", "lvis/lvis_v1_train_novel.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_val_base": ("coco/", "lvis/lvis_v1_val_base.json"),
        "lvis_v1_val_novel": ("coco/", "lvis/lvis_v1_val_novel.json"),
        "lvis_v1_val_all": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_train_base": ("coco/", "lvis/lvis_v0.5_train_base.json"),
        "lvis_v0.5_train_novel": ("coco/", "lvis/lvis_v0.5_train_novel.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_base": ("coco/", "lvis/lvis_v0.5_val_base.json"),
        "lvis_v0.5_val_novel": ("coco/", "lvis/lvis_v0.5_val_novel.json"),
        "lvis_v0.5_val_all": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
        # few-shot sets
        "lvis_v0.5_train_10shot": ("coco/", "lvis/lvis_v0.5_train.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}


def register_all_cityscapes(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
        # To do the first stage training with images that contains no novel class
        # objects, uncomment the following 6 lines and comment its counterparts.
        # ("voc_2007_trainval_split1_base", "VOC2007", "trainval_split1_base"),
        # ("voc_2007_trainval_split2_base", "VOC2007", "trainval_split2_base"),
        # ("voc_2007_trainval_split3_base", "VOC2007", "trainval_split3_base"),
        # ("voc_2012_trainval_split1_base", "VOC2012", "trainval_split1_base"),
        # ("voc_2012_trainval_split2_base", "VOC2012", "trainval_split2_base"),
        # ("voc_2012_trainval_split3_base", "VOC2012", "trainval_split3_base"),
        ("voc_2007_trainval_split1_base", "VOC2007", "trainval"),
        ("voc_2007_trainval_split2_base", "VOC2007", "trainval"),
        ("voc_2007_trainval_split3_base", "VOC2007", "trainval"),
        ("voc_2007_trainval_split4_base", "VOC2007", "trainval"),
        ("voc_2012_trainval_split1_base", "VOC2012", "trainval"),
        ("voc_2012_trainval_split2_base", "VOC2012", "trainval"),
        ("voc_2012_trainval_split3_base", "VOC2012", "trainval"),
        ("voc_2012_trainval_split4_base", "VOC2012", "trainval"),
        ("voc_2007_test_split1_base", "VOC2007", "test"),
        ("voc_2007_test_split2_base", "VOC2007", "test"),
        ("voc_2007_test_split3_base", "VOC2007", "test"),
        ("voc_2007_trainval_split1_1shot", "VOC2007", "trainval_1shot_FSRW"),
        ("voc_2007_trainval_split2_1shot", "VOC2007", "trainval_1shot_FSRW"),
        ("voc_2007_trainval_split3_1shot", "VOC2007", "trainval_1shot_FSRW"),
        ("voc_2012_trainval_split1_1shot", "VOC2012", "trainval_1shot_FSRW"),
        ("voc_2012_trainval_split2_1shot", "VOC2012", "trainval_1shot_FSRW"),
        ("voc_2012_trainval_split3_1shot", "VOC2012", "trainval_1shot_FSRW"),
        ("voc_2007_trainval_split1_2shot", "VOC2007", "trainval_2shot_FSRW"),
        ("voc_2007_trainval_split2_2shot", "VOC2007", "trainval_2shot_FSRW"),
        ("voc_2007_trainval_split3_2shot", "VOC2007", "trainval_2shot_FSRW"),
        ("voc_2012_trainval_split1_2shot", "VOC2012", "trainval_2shot_FSRW"),
        ("voc_2012_trainval_split2_2shot", "VOC2012", "trainval_2shot_FSRW"),
        ("voc_2012_trainval_split3_2shot", "VOC2012", "trainval_2shot_FSRW"),
        ("voc_2007_trainval_split1_3shot", "VOC2007", "trainval_3shot_FSRW"),
        ("voc_2007_trainval_split2_3shot", "VOC2007", "trainval_3shot_FSRW"),
        ("voc_2007_trainval_split3_3shot", "VOC2007", "trainval_3shot_FSRW"),
        ("voc_2007_trainval_split4_3shot", "VOC2007", "trainval_3shot_FSRW"),
        ("voc_2012_trainval_split1_3shot", "VOC2012", "trainval_3shot_FSRW"),
        ("voc_2012_trainval_split2_3shot", "VOC2012", "trainval_3shot_FSRW"),
        ("voc_2012_trainval_split3_3shot", "VOC2012", "trainval_3shot_FSRW"),
        ("voc_2012_trainval_split4_3shot", "VOC2012", "trainval_3shot_FSRW"),
        ("voc_2007_trainval_split1_5shot", "VOC2007", "trainval_5shot_FSRW"),
        ("voc_2007_trainval_split2_5shot", "VOC2007", "trainval_5shot_FSRW"),
        ("voc_2007_trainval_split3_5shot", "VOC2007", "trainval_5shot_FSRW"),
        ("voc_2012_trainval_split1_5shot", "VOC2012", "trainval_5shot_FSRW"),
        ("voc_2012_trainval_split2_5shot", "VOC2012", "trainval_5shot_FSRW"),
        ("voc_2012_trainval_split3_5shot", "VOC2012", "trainval_5shot_FSRW"),
        ("voc_2007_trainval_split1_10shot", "VOC2007", "trainval_10shot_FSRW"),
        ("voc_2007_trainval_split2_10shot", "VOC2007", "trainval_10shot_FSRW"),
        ("voc_2007_trainval_split3_10shot", "VOC2007", "trainval_10shot_FSRW"),
        ("voc_2007_trainval_split4_10shot", "VOC2007", "trainval_10shot_FSRW"),
        ("voc_2012_trainval_split1_10shot", "VOC2012", "trainval_10shot_FSRW"),
        ("voc_2012_trainval_split2_10shot", "VOC2012", "trainval_10shot_FSRW"),
        ("voc_2012_trainval_split3_10shot", "VOC2012", "trainval_10shot_FSRW"),
        ("voc_2012_trainval_split4_10shot", "VOC2012", "trainval_10shot_FSRW"),
        ("voc_2007_test_split1_all", "VOC2007", "test"),
        ("voc_2007_test_split2_all", "VOC2007", "test"),
        ("voc_2007_test_split3_all", "VOC2007", "test"),
        ("voc_2007_test_split4_all", "VOC2007", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
register_all_fsod()
register_all_lvis()
register_all_cityscapes()
register_all_pascal_voc()
