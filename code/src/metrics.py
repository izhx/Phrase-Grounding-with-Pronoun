from typing import Dict, List, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
from allennlp.training.metrics import Metric
from allennlp.nn.util import dist_reduce_sum


@Metric.register("average_dict")
class AverageDict(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self) -> None:
        self._total = defaultdict(float)
        self._count = defaultdict(int)

    def __call__(self, value_dict: Dict[str, torch.Tensor]):
        """
        # Parameters

        value_dict : `Dict[str, float]`
            The values to average.
        """
        for k, v in value_dict.items():
            self._count[k] += dist_reduce_sum(1)
            self._total[k] += dist_reduce_sum(float(list(self.detach_tensors(v))[0]))

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """
        metrics = dict()
        for k, v in self._count.items():
            metrics[k] = float(self._total[k] / v if v > 0 else 0.0)
        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._total = defaultdict(float)
        self._count = defaultdict(int)


def safe_recall(pred: float, gold: int) -> float:
    if gold == 0:
        return 0.0
    return float(pred) / gold


Span = Tuple[int, int]
TypedSpan = Tuple[str, Tuple[int, int]]
Result = Dict[Union[Span, TypedSpan], List[int]]


class PhraseGroundingRecall(Metric):
    """
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._num_gold = defaultdict(int)
        self._num_pred = {k: defaultdict(int) for k in (1, 5, 10)}

    def __call__(
        self,
        predictions: List[Result],
        gold_labels: List[Result],
        predicted_boxes: torch.Tensor,
        gold_boxes: List[np.ndarray],
    ) -> None:
        predicted_boxes = list(self.detach_tensors(predicted_boxes))[0].cpu().numpy()
        for i, (pred, gold, boxes) in enumerate(zip(predictions, gold_labels, gold_boxes)):
            _predicted_boxes = predicted_boxes[i]
            iou = box_iou(_predicted_boxes, boxes)
            for (kind, span), gold_ids in gold.items():
                self._num_gold[kind] += 1
                if span not in pred:
                    continue
                pred_ids = pred[span]
                match_flag = False
                for n, p in enumerate(pred_ids[:10]):  # boxes is sorted by confidence
                    for g in gold_ids:  # if the protocol is merge box, len(gold_ids) == 1
                        if iou[p, g] > self.iou_threshold:
                            if n < 1:
                                self._num_pred[1][kind] += 1
                            if n < 5:
                                self._num_pred[5][kind] += 1
                            self._num_pred[10][kind] += 1
                            match_flag = True
                            break
                    if match_flag:
                        break
        del predictions, gold_labels, predicted_boxes, gold_boxes

    def get_metric(self, reset: bool) -> Dict[str, float]:
        metric = dict()
        for k, v in self._num_pred.items():
            if len(v) == 0:
                continue
            for kind, num in v.items():
                metric[f'recall_{k}_{kind}'] = safe_recall(num, self._num_gold[kind])
            metric[f'recall_{k}'] = safe_recall(sum(v.values()), sum(self._num_gold.values()))
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        self._num_gold = defaultdict(int)
        self._num_pred = {k: defaultdict(int) for k in (1, 5, 10)}


# Bounding box utilities imported from torchvision and converted to numpy
def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.ndarray, boxes2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou
