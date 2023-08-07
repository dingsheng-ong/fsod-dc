import os

import torch
from detectron2.config import CfgNode, get_cfg
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger


def setup(args):
    cfg = get_cfg()

    # additional variables
    cfg.MODEL.ROI_HEADS.DROPOUT = 0.0
    cfg.MODEL.ROI_HEADS.FREEZE_FE = False

    cfg.MODEL.REFINE = CfgNode()
    cfg.MODEL.REFINE.MOMENTUM = 0.0
    cfg.MODEL.REFINE.NUM_CLASSES = 0
    cfg.MODEL.CALIBRATE = False

    cfg.SOLVER.BACKBONE_LR_FACTOR = 1.0

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info(
        "Rank of current process: {}. World size: {}".format(
            rank, comm.get_world_size()
        )
    )
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    return cfg
