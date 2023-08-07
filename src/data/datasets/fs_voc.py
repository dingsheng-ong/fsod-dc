import os
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

from .builtin_meta import PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES


def _get_pascal_voc_meta(phase: str, ns: int):
    base_classes = PASCAL_VOC_BASE_CATEGORIES.get(ns, [])
    novel_classes = PASCAL_VOC_NOVEL_CATEGORIES.get(ns, [])
    all_classes = base_classes + novel_classes

    if phase == "all":
        thing_classes = all_classes
    elif phase == "base":
        thing_classes = base_classes
    elif phase == "novel":
        thing_classes = novel_classes
    else:
        raise NotImplementedError(f"Unknown phase: {phase}")

    return {
        "thing_classes": thing_classes,
        "base_classes": base_classes,
        "novel_classes": novel_classes,
    }


def _load_pascal_voc_instances(
    metadata: dict, root: str, split: str, shot: int = None, seed: int = None
):
    dataset_dicts = []
    if not (shot is None or seed is None):
        fdir = os.path.join(root, "vocsplit", f"seed{seed}")
        fpattr = "box_{shot}shot_{cls}_train.txt"
        for cls in metadata["thing_classes"]:
            cls_datasets_dicts_ = []
            with PathManager.open(
                os.path.join(fdir, fpattr.format(shot=shot, cls=cls))
            ) as f:
                img_ids = np.loadtxt(f, dtype=str).tolist()
                if isinstance(img_ids, str):
                    img_ids = [img_ids]
                img_ids = [i.split(os.sep)[-1].split(".jpg")[0] for i in img_ids]
                for img_id in img_ids:
                    year = 2012 if "_" in img_id else 2007
                    if not year in metadata["years"]:
                        continue
                    dirname = metadata["dirname"].format(year=year)
                    annofile = os.path.join(dirname, "Annotations", f"{img_id}.xml")
                    jpegfile = os.path.join(dirname, "JPEGImages", f"{img_id}.jpg")
                    with PathManager.open(annofile) as f:
                        tree = ET.parse(f)
                    height = int(tree.findall("./size/height")[0].text)
                    width = int(tree.findall("./size/width")[0].text)
                    for obj in tree.findall("object"):
                        record = {
                            "file_name": jpegfile,
                            "image_id": img_id,
                            "height": height,
                            "width": width,
                        }
                        if obj.find("name").text != cls:
                            continue
                        # difficult = int(obj.find("difficult").text)
                        # if difficult == 1:
                        #     continue
                        bbox = obj.find("bndbox")
                        bbox = [
                            float(bbox.find(k).text)
                            for k in ["xmin", "ymin", "xmax", "ymax"]
                        ]
                        bbox[0] -= 1.0
                        bbox[1] -= 1.0
                        record.update(
                            {
                                "annotations": [
                                    {
                                        "category_id": metadata["thing_classes"].index(
                                            cls
                                        ),
                                        "bbox": bbox,
                                        "bbox_mode": BoxMode.XYXY_ABS,
                                    }
                                ]
                            }
                        )
                        cls_datasets_dicts_.append(record)
            if len(cls_datasets_dicts_) > shot:
                cls_datasets_dicts_ = np.random.choice(
                    cls_datasets_dicts_, shot, replace=False
                )
            dataset_dicts.extend(cls_datasets_dicts_)
    else:
        for year in metadata["years"]:
            dirname = metadata["dirname"].format(year=year)
            with PathManager.open(
                os.path.join(dirname, "ImageSets", "Main", f"{split}.txt")
            ) as f:
                for img_id in np.loadtxt(f, dtype=str):

                    dirname = metadata["dirname"].format(year=year)
                    annodir = PathManager.get_local_path(
                        os.path.join(dirname, "Annotations")
                    )
                    jpegdir = PathManager.get_local_path(
                        os.path.join(dirname, "JPEGImages")
                    )

                    annofile = os.path.join(annodir, f"{img_id}.xml")
                    jpegfile = os.path.join(jpegdir, f"{img_id}.jpg")
                    with PathManager.open(annofile) as f:
                        tree = ET.parse(f)

                    record = {
                        "file_name": jpegfile,
                        "image_id": img_id,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                        "annotations": [],
                    }
                    for obj in tree.findall("object"):
                        cls = obj.find("name").text
                        if not cls in metadata["thing_classes"]:
                            continue
                        # difficult = int(obj.find("difficult").text)
                        # if difficult == 1:
                        #     continue
                        bbox = obj.find("bndbox")
                        bbox = [
                            float(bbox.find(k).text)
                            for k in ["xmin", "ymin", "xmax", "ymax"]
                        ]
                        bbox[0] -= 1.0
                        bbox[1] -= 1.0
                        record["annotations"].append(
                            {
                                "category_id": metadata["thing_classes"].index(cls),
                                "bbox": bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                            }
                        )
                    dataset_dicts.append(record)
    return dataset_dicts


def register_few_shot_pascal_voc(root: str, name: str):
    _, year, split, phase, *args = name.split("_")
    ns = None
    if phase[-1].isdigit():
        phase, ns = phase[:-1], int(phase[-1])
    if len(args) > 0:
        assert len(args) == 2, f"Unsupported dataset: {name}"
        shot, seed = args
        shot = int(shot.replace("shot", ""))
        seed = int(seed.replace("seed", ""))
    else:
        shot, seed = None, None

    assert phase in ["base", "novel", "all"], f"Unknown phase: {phase}"
    if phase in ["base", "novel"]:
        assert ns in [1, 2, 3], f"Unknown novel split: {ns}"
    assert split in ["trainval", "train", "val", "test"], f"Unknown split: {split}"

    metadata = _get_pascal_voc_meta(phase, ns)
    metadata.update(
        {
            "evaluator_type": "pascal_voc",
            "dirname": os.path.join(root, "VOC{year}"),
            "years": tuple(map(int, year.split("+"))),
            "split": split,
        }
    )
    if len(metadata["years"]) == 1:
        year = int(metadata["years"][0])
        metadata.update(
            {
                "dirname": os.path.join(root, f"VOC{year}"),
                "year": year,
            }
        )

    DatasetCatalog.register(
        name,
        lambda: _load_pascal_voc_instances(metadata, root, split, shot, seed),
    )
    MetadataCatalog.get(name).set(**metadata)
