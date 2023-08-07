from typing import Dict, List, Optional

import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from torch import nn

from .refine import Refine

__all__ = ["FsodRCNN"]


@META_ARCH_REGISTRY.register()
class FsodRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *args, **kwargs):
        refine = kwargs.pop("refine", nn.Identity())
        super().__init__(*args, **kwargs)
        self.refine = refine

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["refine"] = Refine(cfg, ret["backbone"].output_shape()["res4"])
        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features_refined = {"res4": self.refine(features["res4"])}

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images,
                {k: v.detach() for k, v in features_refined.items()},
                gt_instances,
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        proposals = self.roi_heads.label_and_sample_proposals(proposals, gt_instances)
        self.refine.update_centroids(features["res4"], proposals)
        _, detector_losses = self.roi_heads(features_refined, proposals)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features_refined = {"res4": self.refine(features["res4"])}

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features_refined, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(features_refined, proposals)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features_refined, detected_instances
            )

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return FsodRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
