#!/usr/bin/env python3
import argparse
import os
from collections import OrderedDict

import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.env import seed_all_rng

__doc__ = f"""This script processes the weights established in the base training
phase before utilizing them to initialize the model during the novel
fine-tuning phase. The script will save the processed file using the
same name as input weight file with the setting appended to the end.

Example:
    $ {os.path.basename(__file__)} ./weight.pth -d voc -m remove

    This command will save the new weight file to `./weight-fsod.pth`.
"""


def initialize_parameters(sd: OrderedDict, dataset: str):
    if dataset == "voc":
        class_indices = list(range(15))
        N = 20
    elif dataset == "coco":
        class_indices = (
            [7, 9, 10, 11, 12, 13, 20, 21, 22, 23]
            + [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
            + [34, 35, 36, 37, 38, 40, 41, 42, 43, 44]
            + [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
            + [55, 59, 61, 63, 64, 65, 66, 67, 68, 69]
            + [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
        )
        N = 80
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    for key in [
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
    ]:
        n = N * 4 if "bbox_pred" in key else N + 1
        if key.endswith("bias"):
            params = torch.zeros(n)
        else:
            params = torch.randn(n, *sd[key].shape[1:]).mul(0.01)
        params = params.to(sd[key])
        indices = class_indices[:]
        base_params = sd[key]
        if "bbox_pred" in key:
            indices = torch.tensor(indices).unsqueeze(1)
            indices = indices * 4 + torch.arange(4)
            indices = indices.view(-1).tolist()
        else:
            base_params = base_params[:-1]
        params[indices] = base_params
        sd[key] = params


def remove_parameters(sd: OrderedDict):
    roi_params_keys = [
        "roi_heads.box_predictor.cls_score",
        "roi_heads.box_predictor.bbox_pred",
    ]
    for name in roi_params_keys:
        keys = tuple(filter(lambda p: p.startswith(name), sd.keys()))
        for key in keys:
            del sd[key]


def main(args):
    if args.seed > -1:
        seed_all_rng(args.seed)

    inp_path = os.path.abspath(args.path[0])
    bname, ext = os.path.splitext(os.path.basename(inp_path))
    fname = f"{bname}-fsod{ext}"
    out_path = os.path.join(os.path.dirname(inp_path), fname)

    sd = torch.load(inp_path, map_location="cpu")
    for key in ["scheduler", "optimizer", "iteration"]:
        if key in sd:
            del sd[key]
    if "model" in sd:
        sd = sd["model"]

    if args.method == "init":
        initialize_parameters(sd, args.dataset)
    elif args.method == "remove":
        remove_parameters(sd)
    else:
        raise NotImplementedError(f"Unknown method: `{args.method}`")

    torch.save(sd, out_path)
    print(f"save to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("path", default=None, nargs=1)
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["coco", "voc"],
        help="choose dataset (choices: `coco`, `voc)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=True,
        choices=["init", "remove"],
        help="choose method (choices: `init`, `remove`)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="RNG seed, (default: -1)",
    )
    args = parser.parse_args()
    main(args)
