import contextlib
import io
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from .builtin_meta import COCO_CATEGORIES, COCO_NOVEL_CATEGORIES


def _get_coco_meta(phase: str):
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]

    novel_ids = [k["id"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [k["name"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    base_categories = [
        k
        for k in COCO_CATEGORIES
        if k["isthing"] == 1 and k["name"] not in novel_classes
    ]
    base_ids = [k["id"] for k in base_categories]
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
    base_classes = [k["name"] for k in base_categories]

    metadata = {
        "novel_dataset_id_to_contiguous_id": novel_dataset_id_to_contiguous_id,
        "novel_classes": novel_classes,
        "base_dataset_id_to_contiguous_id": base_dataset_id_to_contiguous_id,
        "base_classes": base_classes,
    }

    if phase == "all":
        pass
    elif phase == "base":
        thing_classes = base_classes
        thing_dataset_id_to_contiguous_id = base_dataset_id_to_contiguous_id
    elif phase == "novel":
        thing_classes = novel_classes
        thing_dataset_id_to_contiguous_id = novel_dataset_id_to_contiguous_id
    else:
        raise NotImplementedError(f"Unknown phase: {phase}")

    metadata.update(
        {
            "thing_classes": thing_classes,
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_colors": thing_colors,
        }
    )
    return metadata


def _load_coco_json(
    metadata: dict, root: str, split: str, shot: int = None, seed: int = None
):
    if not (shot is None or seed is None):
        imgid2ann = {}
        annofile_pattrn = "full_box_{shot}shot_{cls}_trainval.json"
        annodir = os.path.join(root, "cocosplit", f"seed{seed}")
        for cls in metadata["thing_classes"]:
            annofile = os.path.join(annodir, annofile_pattrn.format(shot=shot, cls=cls))
            annofile = PathManager.get_local_path(annofile)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(annofile)
            img_ids = sorted(list(coco_api.imgs.keys()))
            for img_id in img_ids:
                if img_id not in imgid2ann:
                    imgid2ann[img_id] = [
                        coco_api.loadImgs([img_id])[0],
                        coco_api.imgToAnns[img_id],
                    ]
                else:
                    for item in coco_api.imgToAnns[img_id]:
                        imgid2ann[img_id][1].append(item)
        imgs, anno = [], []
        for img_id in imgid2ann:
            imgs.append(imgid2ann[img_id][0])
            anno.append(imgid2ann[img_id][1])
    else:
        annofile = "{}5k.json".format("trainvalno" if "train" in split else "")
        annofile = os.path.join(root, "cocosplit", "datasplit", annofile)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(annofile)
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anno = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anno = list(zip(imgs, anno))
    id_mapping = metadata["thing_dataset_id_to_contiguous_id"]
    dataset_dicts = []

    splits = (
        ["train", "val"]
        if split == "trainval"
        else [
            split,
        ]
    )
    for (img_dict, anno_dict_list) in imgs_anno:
        _, splitdir, *_ = img_dict["file_name"].split("_")
        record = {
            "file_name": os.path.join(root, "coco", splitdir, img_dict["file_name"]),
            "height": img_dict["height"],
            "width": img_dict["width"],
            "annotations": [],
        }
        if not any([splitdir.startswith(k) for k in splits]):
            continue
        image_id = record["image_id"] = img_dict["id"]
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0
            obj = {
                key: anno[key]
                for key in ["id", "iscrowd", "bbox", "category_id"]
                if key in anno
            }
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if obj["category_id"] in id_mapping:
                obj["category_id"] = id_mapping[obj["category_id"]]
                record["annotations"].append(obj)
        dataset_dicts.append(record)

    return dataset_dicts


def register_few_shot_coco(root: str, name: str):
    _, _, split, phase, *args = name.split("_")
    if len(args) > 0:
        assert len(args) == 2, f"Unsupported dataset: {name}"
        shot, seed = args
        shot = int(shot.replace("shot", ""))
        seed = int(seed.replace("seed", ""))
    else:
        shot, seed = None, None

    assert split in ["trainval", "train", "val", "test"], f"Unknown split: {split}"
    assert phase in ["base", "novel", "all"], f"Unknown phase: {phase}"

    metadata = _get_coco_meta(phase)
    metadata.update({"evaluator_type": "coco"})

    json_file = "{}5k.json".format("trainvalno" if "train" in split else "")
    json_file = os.path.join(root, "cocosplit", "datasplit", json_file)
    metadata.update({"json_file": json_file})

    DatasetCatalog.register(
        name, lambda: _load_coco_json(metadata, root, split, shot, seed)
    )
    MetadataCatalog.get(name).set(**metadata)
