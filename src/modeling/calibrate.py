import logging
import pickle as pkl
from collections import defaultdict
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, configurable
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.structures import ImageList
from einops import repeat

logger = logging.getLogger("detectron2")


@lru_cache(maxsize=None)
def extract_support_rois(cfg: bytes):
    cfg = pkl.loads(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    cfg.defrost()
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res5"]
    cfg.freeze()

    backbone = build_backbone(cfg)
    backbone.stem.fc = torch.nn.Linear(2048, 1000)
    _level = logger.getEffectiveLevel()
    # suppress logging
    logger.setLevel(logging.CRITICAL)
    DetectionCheckpointer(backbone).resume_or_load("./pretrain/R-101.pkl")
    logger.setLevel(_level)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.to(device)
    pooler = ROIPooler(
        output_size=(1, 1),
        scales=(1 / 32.0,),
        sampling_ratio=(0),
        pooler_type="ROIAlignV2",
    )

    px_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(device)
    px_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(device)

    def read_image(filename):
        image = utils.read_image(filename, format=cfg.INPUT.FORMAT).copy()
        image = torch.from_numpy(image).permute(2, 0, 1).to(device)
        image = (image - px_mean) / px_std
        image = ImageList.from_tensors([image], 0)
        return image.tensor

    support_set = []
    for dataset in cfg.DATASETS.TRAIN:
        support_set.extend(DatasetCatalog.get(dataset))

    samples = defaultdict(list)
    for data in support_set:
        image = read_image(data["file_name"])
        instances = utils.annotations_to_instances(
            data["annotations"], image.shape[-2:]
        )
        boxes = instances.gt_boxes.to(device)
        features = torch.flatten(pooler([backbone(image)["res5"]], [boxes]), 1)
        # features = F.normalize(features, dim=1)
        for i, cat_id in enumerate(instances.gt_classes):
            samples[cat_id.item()].append(features[i])

    support_features = []
    masks = []
    shot = int(cfg.DATASETS.TRAIN[0].split("_")[4].replace("shot", ""))
    for key in sorted(samples):
        if len(samples[key]) < shot:
            feature = torch.stack(samples[key])
            n, c = feature.shape
            l = shot - n
            ph = torch.zeros(l, c).to(feature)
            feature = torch.cat([feature, ph], dim=0)
            mask = torch.ones(n + l, 1).to(feature)
            mask[-l:] = 0
        else:
            feature = torch.stack(samples[key])
            mask = torch.ones(shot, 1).to(feature)
        support_features.append(feature)
        masks.append(mask)

    support_features = torch.stack(support_features)
    masks = torch.stack(masks)
    return support_features.detach().cpu(), masks.detach().cpu()


class Calibrate(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        dropout: nn.Module = nn.Identity(),
        supp_features: torch.Tensor = None,
        transform_func: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(input_shape, **kwargs)
        self.dropout = dropout
        self.register_buffer("fsup", supp_features)
        self.fc = transform_func

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)
        nn.init.zeros_(self.cls_score.bias)

    @classmethod
    def from_config(cls, cfg: CfgNode, input_shape: "ShapeSpec"):
        ret = super().from_config(cfg, input_shape)

        supp_feats, mask = extract_support_rois(pkl.dumps(cfg))
        ret["supp_features"] = torch.sum(supp_feats * mask, dim=1) / mask.sum(dim=1)

        fc = cls.learn_mapping(supp_feats, mask, cfg.MODEL.DEVICE)
        ret["transform_func"] = fc

        if cfg.MODEL.ROI_HEADS.DROPOUT > 0:
            drop_ratio = cfg.MODEL.ROI_HEADS.DROPOUT
            ret["dropout"] = nn.Dropout(p=drop_ratio)
            logger.info((f"[CLS] Use dropout: p = {ret['dropout'].p}"))

        return ret

    @classmethod
    @lru_cache(maxsize=None)
    def learn_mapping(
        cls, supp_features, mask, device_id: str = "cpu", eps: float = 1e-6
    ):
        device = torch.device(device_id)
        N, k, d = supp_features.shape
        x = supp_features.reshape(N * k, -1).to(device)
        m = mask.reshape(N * k, 1).to(device)
        m = m.matmul(m.T)
        fc = nn.Linear(d, d, bias=False)
        nn.init.normal_(fc.weight.data, 0, 0.01)
        if fc.bias is not None:
            nn.init.zeros_(fc.bias.data)
        fc.to(device)
        t = F.one_hot(repeat(torch.arange(N), "N -> (N k)", k=k), num_classes=N)
        t = t.matmul(t.T).float().to(device)
        opt = torch.optim.SGD(fc.parameters(), lr=1.0)
        loss = torch.tensor(torch.inf)
        prev_loss = 0
        while abs(loss.item() - prev_loss) > eps:
            prev_loss = loss.item()
            z = fc(x)
            loss = F.mse_loss(cls.cosine_similarity(z, z), t, reduction="none")
            loss = torch.sum(loss * m) / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        logger.info(f"Loss: {loss.item():.4f}")
        return fc.cpu()

    @staticmethod
    def cosine_similarity(x, y, eps=1e-8):
        return F.normalize(x, dim=-1, eps=eps).matmul(F.normalize(y, dim=-1, eps=eps).T)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        scores = self.cls_score(self.dropout(x))
        cos_simm = self.cosine_similarity(self.fc(x), self.fc(self.fsup))
        with torch.no_grad():
            tau_cls = scores.norm(dim=1, keepdim=True)
            tau_cos = cos_simm.norm(dim=1, keepdim=True)
        scores = scores * tau_cos
        cos_simm = cos_simm * tau_cls
        cls_scores, bg_score = scores.split(cos_simm.shape[1], dim=1)
        scores = torch.cat([cls_scores + cos_simm, bg_score + tau_cls], dim=1) / 2
        return scores, proposal_deltas
