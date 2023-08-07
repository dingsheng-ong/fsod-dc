#!/usr/bin/env python3
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import sem

__doc__ = f"""This script processes the results and formats them into a table.
By providing the path to the checkpoint directory, the script will extract all
the results found in the directory and store the table in csv format in the
same directory.

Example:
    $ {os.path.basename(__file__)} checkpoints/voc/baseline

    This command will produce the table and save to:
    checkpoints/voc/baseline/results.txt
"""


def write(mssg: str, file=None):
    sys.stdout.write(mssg)
    sys.stdout.flush()
    if file is not None:
        file.write(mssg)
        file.flush()


def tabulate(data: dict, split: int, shot: int, file=None):
    title = f"{shot:2d}-shot"
    if split is not None:
        title = f"VOC {title}, NS: {split}"
    else:
        title = f"COCO {title}"

    data = {
        k: data[k] for k in filter(lambda k: not np.isnan(np.nanmean(data[k])), data)
    }
    ncol = len(data)
    nc_c = ncol * 7 + 5

    write_fn = lambda mssg: write(mssg, file=file)
    write_fn(f'┌{(nc_c - 1) * "─"}┐\n')
    write_fn(f"│{title[:nc_c]:^{nc_c - 1}s}│\n")
    write_fn(f'├────{"┬──────" * ncol}┤\n')
    write_fn(f'│  \ │{"│".join([f"{k[:7]:^6s}" for k in data])}│\n')
    write_fn(f'├────{"┼──────" * ncol}┤\n')

    nrow = len(data[next(iter(data))])
    for i in range(nrow):
        values = [data[k][i] for k in data]
        write_fn(f'│ {i:02d} │{"│".join([f"{v:^6.1f}" for v in values])}│\n')

    if nrow > 0:
        write_fn(f'├────{"┼──────" * ncol}┤\n')
        mean = [np.nanmean(data[k]) for k in data]
        serr = [float(sem(data[k], nan_policy="omit")) for k in data]
        write_fn(f'│  μ │{"│".join([f"{v:^6.1f}" for v in mean])}│\n')
        write_fn(f'│  ε │{"│".join([f"{v:^6.1f}" for v in serr])}│\n')
    write_fn(f'└────{"┴──────" * ncol}┘\n')


def summarize(data: dict, file=None):
    write_fn = lambda mssg: write(mssg, file=file)

    if len(data) == 0:
        return

    _, split, shot, _ = list(data.keys())[0]
    is_coco = split == None

    if is_coco:
        write_fn("┌─────┬────────────────────────────────────┐\n")
        write_fn(
            "│  k  │" + "".join([f"{k:^6d}" for k in [1, 2, 3, 5, 10, 30]]) + "│\n"
        )
        write_fn("├─────┼────────────────────────────────────┤\n")
        for metric in ["AP", "bAP", "nAP"]:
            write_fn(f"│ {metric:>3s} │")
            for shot in [1, 2, 3, 5, 10, 30]:
                if data.get(("coco", None, shot, "fsod")):
                    score = f"{np.mean(data[('coco', None, shot, 'fsod')][f'{metric}']):.1f}"
                    if len(score) < 4:
                        score = (" " * (5 - len(score))) + score
                else:
                    score = "-"
                write_fn(f"{score:^6s}")
            write_fn(f"│\n")
        write_fn("└─────┴────────────────────────────────────┘\n")
    else:
        write_fn("┌───────┬" + "┬".join(("─" * 30,) * 3) + "┐\n")
        write_fn("│ Split │" + "│".join([f"{k + 1:^30d}" for k in range(3)]) + "│\n")
        write_fn("├───────┼" + "┼".join(("─" * 30,) * 3) + "┤\n")
        write_fn("│   k   │")
        for _ in range(3):
            write_fn("".join([f"{k:^6d}" for k in [1, 2, 3, 5, 10]]))
            write_fn("│")
        write_fn("\n")
        write_fn("├───────┼" + "┼".join(("─" * 30,) * 3) + "┤\n")
        for metric in ["AP50", "bAP50", "nAP50"]:
            write_fn(f"│ {metric:>5s} │")
            for split in [1, 2, 3]:
                for shot in [1, 2, 3, 5, 10]:
                    if data.get(("voc", split, shot, "fsod")):
                        score = f"{np.mean(data[('voc', split, shot, 'fsod')][f'{metric}']):.1f}"
                        if len(score) < 4:
                            score = (" " * (5 - len(score))) + score
                    else:
                        score = "-"
                    write_fn(f"{score:^6s}")
                write_fn(f"│")
            write_fn(f"\n")
        write_fn("└───────┴" + "┴".join(("─" * 30,) * 3) + "┘\n")


def main(args):
    root = args.path[0]

    shots = {"coco": [1, 2, 3, 5, 10, 30], "voc": [1, 2, 3, 5, 10]}

    table = open(os.path.join(root, "results.txt"), "w")
    results = {}
    for setting in ["fsod"]:
        # deduce dataset by searching for split directories
        splits = []
        if os.path.exists(os.path.join(root, setting)):
            splits.append(None)
        else:
            for ns in [1, 2, 3]:
                if os.path.exists(os.path.join(root, f"{setting}{ns}")):
                    splits.append(ns)
        if len(splits) == 0:
            continue
        dataset = "voc" if splits[0] else "coco"
        for split in splits:
            setting_dir = os.path.join(root, f"{setting}{split if split else ''}")
            for shot in shots[dataset]:
                metrics = defaultdict(list)
                shot_dir = os.path.join(setting_dir, f"{shot}shot")
                if not os.path.exists(shot_dir):
                    continue
                for seed in range(30):
                    ckpt_dir = os.path.join(shot_dir, f"seed{seed}")
                    log_file = os.path.join(ckpt_dir, "log.txt")
                    if not os.path.exists(log_file):
                        continue
                    raw = open(log_file, "r").read().strip().split("\n")
                    if not ("copypaste" in raw[-1] or "copypaste" in raw[-2]):
                        continue
                    keys = raw[-2].split("copypaste: ")[-1].split(",")
                    vals = raw[-1].split("copypaste: ")[-1].split(",")
                    for i in range(len(keys)):
                        k = keys[i]
                        try:
                            v = float(vals[i])
                        except:
                            v = float("nan")
                        metrics[k].append(v)

                if len(metrics.keys()) > 0 and args.verbose:
                    tabulate(metrics, split, shot, file=table)

                key = (dataset, split, shot, setting)
                if len(metrics):
                    results[key] = metrics

    summarize(results, file=table)
    keys = [
        (k, f"{k[0]}_{k[3]}{k[1] if k[1] else ''}_{k[2]}shot") for k in results.keys()
    ]
    results = {b: results[a] for a, b in keys}
    json.dump(results, open(os.path.join(root, "results.json"), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("path", default=None, nargs=1)

    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
