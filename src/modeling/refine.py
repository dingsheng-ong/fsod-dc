import logging

import torch
from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils import comm, events
from einops import reduce
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("detectron2")


class Refine(nn.Module):
    @configurable
    def __init__(
        self,
        input_dim: int,
        *,
        pooler: ROIPooler,
        num_classes: int,
        momentum: float = 0.1,
        warmup_iters: int = 0,
        eps=1e-12,
    ):
        super().__init__()
        logger.info(f"[Refine] n = {num_classes}, α = {momentum}")
        self.pooler = pooler
        self.num_classes = num_classes
        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.eps = eps
        self.fc = nn.Conv2d(input_dim, input_dim, 1, 1, 0, bias=True)
        self.fc.weight.data = torch.eye(input_dim).reshape(input_dim, -1, 1, 1)
        nn.init.zeros_(self.fc.bias.data)
        self.register_buffer("centroids", torch.zeros(num_classes, input_dim) + eps)

    @classmethod
    def from_config(cls, cfg: CfgNode, input_shape: ShapeSpec):
        pooler = ROIPooler(
            output_size=(1, 1),
            scales=(1 / input_shape.stride,),
            sampling_ratio=(0),
            pooler_type="ROIAlignV2",
        )
        return {
            "input_dim": input_shape.channels,
            "pooler": pooler,
            "num_classes": cfg.MODEL.REFINE.NUM_CLASSES,
            "momentum": cfg.MODEL.REFINE.MOMENTUM,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
        }

    @property
    def iterations(self):
        if not hasattr(self, "storage"):
            self.storage = events.get_event_storage()
        return self.storage.iter

    @torch.no_grad()
    def update_centroids(self, features, proposals):
        if self.momentum == 0:
            return

        if self.iterations < self.warmup_iters:
            return

        gt_boxes = [x.gt_boxes for x in proposals]
        features = torch.flatten(self.pooler([features], gt_boxes), start_dim=1)

        simm = F.normalize(F.dropout(features, p=0.5), dim=1).matmul(
            F.normalize(self.centroids, dim=1).T
        )
        mask = torch.zeros_like(simm).scatter(1, simm.argmax(dim=1, keepdim=True), 1.0)

        sum_x = mask.T.matmul(features)
        count = mask.sum(dim=0).unsqueeze(1)

        world_size = comm.get_world_size()
        if world_size > 1:
            sum_x_gt = [torch.empty_like(sum_x) for _ in range(world_size)]
            count_gt = [torch.empty_like(count) for _ in range(world_size)]
            dist.all_gather(sum_x_gt, sum_x)
            dist.all_gather(count_gt, count)
            sum_x_gt = torch.stack(sum_x_gt, dim=0).sum(dim=0)
            count_gt = torch.stack(count_gt, dim=0).sum(dim=0)
        else:
            sum_x_gt = sum_x
            count_gt = count

        centroids = sum_x_gt / count.clamp_min(1)
        alpha = (count_gt > 0).float() * self.momentum
        self.centroids.set_((1 - alpha) * self.centroids + alpha * centroids)

    def forward(self, features: torch.Tensor):
        if self.training:
            if self.iterations < self.warmup_iters:
                return F.relu(self.fc(features))

        sim = torch.einsum(
            "bchw,nc->bnhw",
            F.normalize(features, dim=1),
            F.normalize(self.centroids, dim=1),
        )
        mask = torch.zeros_like(sim).scatter(1, sim.argmax(dim=1, keepdim=True), 1.0)

        sum_x = torch.einsum("bnhw,bchw->bnc", mask, features)
        count = reduce(mask, "b n h w -> b n ()", "sum")
        centroids = sum_x / count.clamp_min(1)
        delta = torch.einsum("bnhw,bnc->bchw", mask, centroids) - features
        alpha = torch.exp(-delta.square().mean(dim=1, keepdim=True))
        features = F.relu(self.fc(features + alpha * delta))

        return features
