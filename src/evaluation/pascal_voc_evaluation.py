import json
import logging
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        super().__init__()
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        anno_dir = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(anno_dir, "{img_name}.xml")
        self._image_set_path = os.path.join(
            meta.dirname, "ImageSets", "Main", f"{meta.split}.txt"
        )
        self._class_names = meta.thing_classes
        self._base_classes = meta.get("base_classes", None)
        self._novel_classes = meta.get("novel_classes", None)
        assert meta.year in [2007, 2012], f"Unkown year: {meta.year}"
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2")

    def reset(self):
        self._predictions = defaultdict(list)

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]
            instances = out["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    (
                        image_id,
                        float(np.round(score, 3)),
                        float(np.round(xmin, 1)),
                        float(np.round(ymin, 1)),
                        float(np.round(xmax, 1)),
                        float(np.round(ymax, 1)),
                    )
                )

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            f"Evaluating {self._dataset_name} "
            f"using {2007 if self._is_2007 else 2012} metric. "
            f"Note that results do not use the official Matlab API."
        )

        aps = defaultdict(list)
        for cls_id, cls_name in enumerate(self._class_names):
            detections = predictions.get(cls_id, [])
            for thresh in range(50, 100, 5):
                *_, ap = self._voc_eval(
                    detections,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                )
                aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {
            "AP": np.mean(list(mAP.values())),
            "AP50": mAP[50],
            "AP75": mAP[75],
        }

        if self._base_classes is not None:
            base_aps = defaultdict(list)
            for thresh, ap in aps.items():
                base_aps[thresh] = [
                    ap[i]
                    for i, c in enumerate(self._class_names)
                    if c in self._base_classes
                ]
            bAP = {iou: np.mean(x) for iou, x in base_aps.items()}
            ret["bbox"].update(
                {"bAP": np.mean(list(bAP.values())), "bAP50": bAP[50], "bAP75": bAP[75]}
            )

        if self._novel_classes is not None:
            novel_aps = defaultdict(list)
            for thresh, ap in aps.items():
                novel_aps[thresh] = [
                    ap[i]
                    for i, c in enumerate(self._class_names)
                    if c in self._novel_classes
                ]
            nAP = {iou: np.mean(x) for iou, x in novel_aps.items()}
            ret["bbox"].update(
                {"nAP": np.mean(list(nAP.values())), "nAP50": nAP[50], "nAP75": nAP[75]}
            )

        per_class_res = {self._class_names[i]: ap for i, ap in enumerate(aps[50])}
        self._logger.info("Per-class AP50:\n" + create_small_table(per_class_res))
        self._logger.info("Evaluation Result:\n" + create_small_table(ret["bbox"]))

        return ret

    def _format_res_table(self, data: dict):
        keys = ["AP", "AP50", "AP75", "AR", "AR50", "AR75"]
        keys += [f"b{c}" for c in ["AP", "AP50", "AP75", "AR", "AR50", "AR75"]]
        keys += [f"n{c}" for c in ["AP", "AP50", "AP75", "AR", "AR50", "AR75"]]

        table = f'┌{95 * "─"}┐\n'
        title = f"PASCAL VOC {2007 if self._is_2007 else 2012}"
        table += f"│{title:^95s}│\n"
        table += f'├{"─" * 31}{("┬" + "─" * 31) * 2}┤\n'
        table += f'│{"ALL":^31s}│{"BASE":^31s}│{"NOVEL":^31s}│\n'
        table += f'├{"┬".join(["─" * 7,] * 4)}{("┼" + "┬".join(["─" * 7,] * 4)) * 2}┤\n'
        for i in range(3):
            for k in keys[i : len(keys) : 3]:
                v = data.get(k, float("nan"))
                table += f"│{k:^7s}│{v:^7.2f}"
            table += "│\n"
        table += f'└───────{"┴───────" * 11}┘'
        return table

    @staticmethod
    def _calculate_ap(rec, pre, use_07_metric=False):
        if use_07_metric:
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec >= t) > 0:
                    ap += np.max(pre[rec >= t])
            ap /= 11.0
        else:
            mrec = np.asarray([0.0, *rec, 1.0])
            mpre = np.asarray([0.0, *pre, 0.0])
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum(mrec[i + 1] - mrec[i]) * mpre[i + 1]
        return ap

    @staticmethod
    @lru_cache(maxsize=None)
    def _parse_xml(filename):
        try:
            with PathManager.open(filename) as f:
                tree = ET.parse(f)
        except:
            tree = ET.parse(filename)

        objects = []
        for obj in tree.findall("object"):
            bbox = obj.find("bndbox")
            obj_struct = {
                "name": obj.find("name").text,
                "pose": obj.find("pose").text,
                "truncated": int(obj.find("truncated").text),
                "difficult": int(obj.find("difficult").text),
                "bbox": [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                ],
            }
            objects.append(obj_struct)
        return objects

    def _voc_eval(
        self,
        detections,
        annopath,
        imagesetfile,
        classname,
        ovthresh=0.5,
        use_07_metric=False,
    ):
        with PathManager.open(imagesetfile, "r") as f:
            lines = f.readlines()
            img_names = [x.strip() for x in lines]

        annos = {}
        for img_name in img_names:
            annos[img_name] = self._parse_xml(annopath.format(img_name=img_name))

        class_annos = {}
        npos = 0
        for img_name in img_names:
            anno = [obj for obj in annos[img_name] if obj["name"] == classname]
            bbox = np.array([x["bbox"] for x in anno])
            diff = np.array([x["difficult"] for x in anno]).astype(bool)
            npos += sum(~diff)
            class_annos[img_name] = {
                "bbox": bbox,
                "difficult": diff,
                "det": [
                    False,
                ]
                * len(anno),
            }

        if len(detections) > 0:
            image_ids, confidence, *bboxes = zip(*detections)
            confidence = np.asarray(confidence)
            bboxes = np.asarray(bboxes).T

            index = np.argsort(-confidence)
            bboxes = bboxes[index, :]
            image_ids = [image_ids[i] for i in index]
        else:
            image_ids = confidence = bboxes = []

        n = len(image_ids)
        tp = np.zeros(n)
        fp = np.zeros(n)
        for i in range(n):
            anno = class_annos[image_ids[i]]
            bbox = bboxes[i, :].astype(float)
            bbox_gt = anno["bbox"].astype(float)
            iou_max = -np.inf

            if bbox_gt.size > 0:
                ixmin = np.maximum(bbox_gt[:, 0], bbox[0])
                iymin = np.maximum(bbox_gt[:, 1], bbox[1])
                ixmax = np.minimum(bbox_gt[:, 2], bbox[2])
                iymax = np.minimum(bbox_gt[:, 3], bbox[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inter = iw * ih
                union = (
                    (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0)
                    + (bbox_gt[:, 2] - bbox_gt[:, 0] + 1.0)
                    * (bbox_gt[:, 3] - bbox_gt[:, 1] + 1.0)
                    - inter
                )

                ious = inter / union
                iou_max = np.max(ious)
                ind_max = np.argmax(ious)

            if iou_max > ovthresh:
                if not anno["difficult"][ind_max]:
                    if not anno["det"][ind_max]:
                        tp[i] = 1.0
                        anno["det"][ind_max] = True
                    else:
                        fp[i] = 1.0
            else:
                fp[i] = 1.0

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        pre = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self._calculate_ap(rec, pre, use_07_metric)
        return rec, pre, ap
