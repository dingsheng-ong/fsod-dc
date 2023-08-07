import os
import re

from .fs_coco import register_few_shot_coco
from .fs_voc import register_few_shot_pascal_voc


def register_all_fewshot_coco(root="./datasets"):
    _datasets = [
        "coco_2014_trainval_base",
        "coco_2014_val_base",
        "coco_2014_trainval_novel_{shot}shot_seed{seed}",
        "coco_2014_trainval_all_{shot}shot_seed{seed}",
        "coco_2014_val_novel",
        "coco_2014_val_all",
    ]
    for dataset in _datasets:
        if re.match(r"(.*)\{(.*?)\}(.*)", dataset):
            subset = set()
            for shot in [1, 2, 3, 5, 10, 30]:
                for seed in range(10):
                    subset.add(dataset.format(shot=shot, seed=seed))
            for dataset in subset:
                register_few_shot_coco(root, dataset)
                FEW_SHOT_DATASETS.append(dataset)
        else:
            register_few_shot_coco(root, dataset)
            FEW_SHOT_DATASETS.append(dataset)


def register_all_fewshot_pascal_voc(root="./datasets"):
    _datasets = [
        "voc_2007_test_all{ns}",
        "voc_2007_test_base{ns}",
        "voc_2007_test_novel{ns}",
        "voc_2007+2012_trainval_base{ns}",
        "voc_2007+2012_trainval_all{ns}_{shot}shot_seed{seed}",
        "voc_2007+2012_trainval_base{ns}_{shot}shot_seed{seed}",
        "voc_2007+2012_trainval_novel{ns}_{shot}shot_seed{seed}",
    ]
    for dataset in _datasets:
        if re.match(r"(.*)\{(.*?)\}(.*)", dataset):
            subset = set()
            for ns in [1, 2, 3]:
                for shot in [1, 2, 3, 5, 10]:
                    for seed in range(30):
                        subset.add(dataset.format(ns=ns, shot=shot, seed=seed))
            for dataset in subset:
                register_few_shot_pascal_voc(root, dataset)
                FEW_SHOT_DATASETS.append(dataset)
        else:
            register_few_shot_pascal_voc(root, dataset)
            FEW_SHOT_DATASETS.append(dataset)


if __name__.endswith(".builtin"):
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "dataset"))
    FEW_SHOT_DATASETS = []
    register_all_fewshot_coco(_root)
    register_all_fewshot_pascal_voc(_root)
    FEW_SHOT_DATASETS.sort()
