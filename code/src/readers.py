from collections import defaultdict
from itertools import chain, permutations
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable
import json
import pickle
# from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from PIL import Image
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
# from transformers import AutoFeatureExtractor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.instance import Instance
from allennlp.data.fields import TensorField, TextField, MetadataField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer


logger = logging.getLogger(__name__)
Span = Tuple[int, int]


class CorefGroundBase(DatasetReader):
    def __init__(
        self,
        image_dir: str,
        only_nouns: bool = False,
        merge_boxes: bool = False,
        build_graph: bool = False,
        one_word_span: bool = False,
        no_span_edge: bool = False,
        no_cluster_edge: bool = False,
        pseudo_cluster: bool = False,
        add_gold_spans: bool = False,
        num_graph_relations: int = 5,  # exlcude the self-loop
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._image_dir = image_dir
        self._only_nouns = only_nouns
        self._merge_boxes = merge_boxes
        self._build_graph = build_graph
        self._one_word_span = one_word_span
        self._no_span_edge = no_span_edge
        self._no_cluster_edge = no_cluster_edge
        self._pseudo_cluster = pseudo_cluster
        self._add_gold_spans = add_gold_spans
        self._num_graph_relations = num_graph_relations
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from lines in: %s", file_path)
            for _, line in enumerate(data_file):
                obj = json.loads(line)
                yield self.text_to_instance(**obj)
                # break

    def basic_fields(
        self,  # type: ignore
        image: str,
        dialog: List[Dict[str, Any]],
        text_ref: List[Dict[str, Any]],
        clusters: List[List[int]],
        image_ref: Optional[List[Dict[str, Any]]] = None,
        predicted_clusters: List[List[Span]] = None,
        **kwargs
    ) -> Tuple:
        sentences = [sentence["text"] for sentence in dialog]
        flat_sentences_tokens = [Token(w) for s in sentences for w in s]
        text_field = TextField(flat_sentences_tokens, self._token_indexers)
        img = Image.open(f"{self._image_dir}{image}").convert('RGB')
        fields = {'text': text_field}

        cluster_to_spans, gold_clusters = defaultdict(list), list()
        for i, cluster in enumerate(clusters):
            spans = list()
            for index in cluster:
                ref = text_ref[index]
                offset = sum(len(s) for s in sentences[:ref['sentence_id']])
                span = (ref['start'] + offset, ref['end'] + offset)
                if self._only_nouns and ref['pronoun']:
                    continue
                typed_span = ("pronoun" if ref['pronoun'] else "phrase", span)
                spans.append(typed_span)
                cluster_to_spans[i].append(typed_span)
            gold_clusters.append(spans)
        else:
            del i, cluster, spans, index, ref, offset, span, typed_span

        if len(clusters) > len([_ for _ in gold_clusters if len(_) > 0]):
            raise Exception("Some clusters are empty")

        gold_spans = list(chain(*cluster_to_spans.values()))
        meta = {'gold_clusters': gold_clusters, 'spans': gold_spans, 'size': img.size}

        if self._build_graph:
            if self._pseudo_cluster and predicted_clusters is not None:
                _clusters = [[tuple(s) for s in c] for c in predicted_clusters]
                _spans, _typed = list(chain(*_clusters)), False
                if self._add_gold_spans:
                    _spans = _spans + [s for _, s in gold_spans if s not in _spans]
            else:
                _clusters, _spans, _typed = gold_clusters, gold_spans, True
            edge_index, edge_type = self.build_graph(
                len(text_field), _spans, _clusters, _typed
            )
            assert (edge_type >= self._num_graph_relations).sum() == 0
            fields['edge_index'] = TensorField(
                edge_index.transpose(1, 0), padding_value=-1  # (2, edge_num)
            )
            fields['edge_type'] = TensorField(edge_type, padding_value=-1)
            meta['edge_num'] = len(edge_type)
            if self._no_span_edge:
                node_span = list()
            else:
                node_span = [s[1] for s in _spans] if _typed else _spans
                if not self._one_word_span:
                    node_span = [s for s in node_span if s[0] != s[1]]
            meta['node_span'] = node_span

        if image_ref is not None:
            box_list, span_to_box_ids = list(), defaultdict(list)
            for ref in image_ref:
                target_spans = cluster_to_spans[ref['cluster_id']]
                if len(target_spans) > 0:
                    i = len(box_list)
                    for span in target_spans:
                        span_to_box_ids[span].append(i)
                    box_list.append(valid_crop_box(ref, img.size))
            else:
                del ref, target_spans, i, span

            if self._merge_boxes:
                box_list, span_to_box_ids = self.merge_box(box_list, span_to_box_ids)

            labels = np.zeros((len(box_list), len(gold_spans)), dtype=np.int64)
            for s, span in enumerate(gold_spans):
                for b in span_to_box_ids[span]:
                    labels[b, s] = 1
            else:
                del s, span, b

            meta.update(
                box_list=box_list, labels=labels, boxes=np.array(box_list),  # xyx2y2
                span_to_box_ids=span_to_box_ids
            )

        return fields, img, meta

    @staticmethod
    def merge_box(box_list, span_to_box_ids):
        ids_to_merged = dict()
        merged_boxes, merged_span_to_box_ids = list(), dict()
        for span, box_ids in span_to_box_ids.items():
            if len(box_ids) > 1:
                key = tuple(sorted(box_ids))
                if key in ids_to_merged:
                    box = merged_boxes[ids_to_merged[key]]
                else:
                    ids_to_merged[key] = len(merged_boxes)
                    boxes = np.array([box_list[i] for i in box_ids])
                    box = [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]
                    box = [float(b) for b in box]
            else:
                box = box_list[box_ids[0]]
            merged_span_to_box_ids[span] = [len(merged_boxes)]
            merged_boxes.append(box)
        return merged_boxes, merged_span_to_box_ids

    def build_graph(self, seq_len, spans, clusters, typed=True) -> Tuple[torch.Tensor, ...]:
        if typed:
            def end_point(span):
                return (span[1][0], span[1][1] + 1)
        else:
            def end_point(span):
                return (span[0], span[1] + 1)

        edge_index, edge_type = list(), list()
        for i in range(seq_len - 1):
            edge_index.append((i, i + 1))
            edge_type.append(0)  # next-word
            edge_index.append((i + 1, i))
            edge_type.append(1)  # last-word

        if not self._no_span_edge:
            span_to_id, last_big_span = dict(), None
            for span in spans:
                start, end = end_point(span)
                if not self._one_word_span and start == end - 1:
                    span_to_id[span] = start
                    continue
                if last_big_span is None:
                    i = seq_len
                else:
                    i = span_to_id[last_big_span] + 1
                last_big_span = span
                span_to_id[span] = i
                for j in range(start, end):
                    edge_index.append((i, j))
                    edge_type.append(2)  # span-word
                    edge_index.append((j, i))
                    edge_type.append(3)  # word-span
            if not self._no_cluster_edge:
                for cluster in clusters:
                    span_ids = [span_to_id[span] for span in cluster]
                    for e in permutations(span_ids, 2):
                        edge_index.append(e)
                        edge_type.append(4)  # in-cluster
        elif not self._no_cluster_edge:
            for cluster in clusters:
                word_ids = set(chain(*[
                    list(range(*end_point(span))) for span in spans
                ]))
                for e in permutations(word_ids, 2):
                    edge_index.append(e)
                    edge_type.append(2)  # in-cluster
        return torch.LongTensor(edge_index), torch.LongTensor(edge_type)


@DatasetReader.register("coref_ground_mdetr")
class CorefGroundMdetr(CorefGroundBase):
    def __init__(
        self,
        image_dir: str,
        timm_config: Dict[str, Any],
        max_length: int = 511,
        **kwargs,
    ) -> None:
        super().__init__(image_dir, **kwargs)
        self.max_length = max_length
        config = resolve_data_config(timm_config)
        self._transform = create_transform(**config)

    def text_to_instance(
        self,  # type: ignore
        image: str,
        dialog: List[Dict[str, Any]],
        text_ref: List[Dict[str, Any]],
        clusters: List[List[int]],
        image_ref: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Instance:
        fields, img, meta = self.basic_fields(
            image, dialog, text_ref, clusters, image_ref, **kwargs
        )
        tensor = self._transform(img)
        fields['samples'] = TensorField(tensor)

        if image_ref is not None:
            box_list, labels = meta.pop('box_list'), meta.pop('labels')
            boxes = torch.tensor(box_list, dtype=torch.float)  # xyx2y2
            # The target boxes are expected in format (center_x, center_y, w, h)
            boxes[:, 2:].sub_(boxes[:, :2])
            boxes[:, 0].add_(boxes[:, 2] / 2).div_(img.width)
            boxes[:, 1].add_(boxes[:, 3] / 2).div_(img.height)
            boxes[:, 2:].div_(torch.tensor(img.size).unsqueeze(0))
            positive_map = torch.zeros(len(box_list), self.max_length + 1)
            for b, array in enumerate(labels):
                for s, item in enumerate(array):
                    if item == 1:
                        start, end = meta['spans'][s][1]
                        positive_map[b, start: end + 1] = 1
            positive_map = positive_map / (positive_map.sum(-1, keepdim=True) + 1e-6)
            fields['targets'] = MetadataField({
                "boxes": boxes, "labels": labels, "positive_map": positive_map
            })

        fields['metadata'] = MetadataField(meta)
        return Instance(fields)


def valid_crop_box(box, size):
    box = [
        # int(Decimal(str(box[k])).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
        box[k]
        for k in "xywh"
    ]
    box[2], box[3] = box[0] + box[2], box[1] + box[3]
    for i in (0, 1):
        if box[i] < 0:
            box[i] = 0
        if box[i + 2] > size[i]:
            box[i + 2] = size[i]
    return tuple(box)  # xyx2y2


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.
    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
