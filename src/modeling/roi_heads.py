from typing import Dict, List, Union

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    FastRCNNOutputLayers,
    Res5ROIHeads,
    select_foreground_proposals,
)
from detectron2.structures import Instances
from torch import nn

from src.modeling.calibrate import Calibrate


class FsodFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        dropout: nn.Module = nn.Identity(),
    ):
        super().__init__(
            input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
        )
        self.dropout = dropout

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        p_drop = cfg.MODEL.ROI_HEADS.DROPOUT
        if p_drop > 0:
            dropout = nn.Dropout(p=p_drop)
        else:
            dropout = nn.Identity()
        ret["dropout"] = dropout
        return ret

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        scores = self.cls_score(self.dropout(x))
        return scores, proposal_deltas


@ROI_HEADS_REGISTRY.register()
class FsodRes5ROIHeads(Res5ROIHeads):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        _, out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = (
            Calibrate if cfg.MODEL.CALIBRATE else FsodFastRCNNOutputLayers
        )(cfg, ShapeSpec(channels=out_channels, height=1, width=1))
        return ret

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
