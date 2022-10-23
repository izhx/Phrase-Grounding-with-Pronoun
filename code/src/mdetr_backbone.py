from typing import Dict, List, Optional, Tuple
import logging
from argparse import Namespace

import torch
from torch import Tensor
from allennlp.data import TextFieldTensors
from allennlp.data.fields import MetadataField
from allennlp.modules import Backbone
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder

from .mdetr.mdetr import MDETR, NestedTensor, SetCriterion, build
from .gnn import RelationalGnnSeq2SeqEncoder

logger = logging.getLogger(__name__)
Span = Tuple[int, int]
Result = Dict[Span, List[int]]


@Backbone.register("phrase_grounding_mdetr")
class MdetrBackbone(Backbone):
    def __init__(
        self,
        mdetr_file: str,
        max_length: int = 255,
        gnn: Optional[RelationalGnnSeq2SeqEncoder] = None,
        sub_token_mode: str = "first",
        **kwargs
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.args: Namespace = None
        self.mdetr: MDETR = None
        self.criterion: SetCriterion = None
        self.weight_dict = None
        self.build_mdetr(mdetr_file, **kwargs)
        self.embedder = PretrainedTransformerMismatchedEmbedder(
            self.args.text_encoder_type,
            max_length=max_length,
            load_weights=False,
            sub_token_mode=sub_token_mode
        )
        self.embedder._matched_embedder.transformer_model = self.mdetr.transformer.text_encoder
        self.mdetr.transformer.text_encoder = None
        logger.info("Move reberta weights of MDETR")
        self.gnn = gnn
        if gnn is None:
            self.gnn_position = 0
        elif gnn.in_channels == self.embedder.get_output_dim():
            self.gnn_position = 1
        elif gnn.in_channels == self.mdetr.transformer.d_model:
            self.gnn_position = 2
        else:
            raise ValueError("GNN in_channels %s is not matched", gnn.in_channels)

    def build_mdetr(self, mdetr_file: str, **kwargs) -> Namespace:
        checkpoint = torch.load(mdetr_file, map_location="cpu")
        self.args = checkpoint["args"]
        for k, v in kwargs.items():
            assert hasattr(self.args, k)
            setattr(self.args, k, v)
        model, criterion, weight_dict = build(self.args, self.max_length)
        if self.max_length != 255:  # 目前只有255的模型
            for k in list(checkpoint['model'].keys()):
                if k.startswith("class_embed"):
                    checkpoint['model'].pop(k)
        _ = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info("Load MDETR from %s. %s", mdetr_file, str(_))
        model.transformer.resizer.fc.bias = None
        self.mdetr = model
        self.criterion = criterion
        self.weight_dict = weight_dict

    def forward(
        self,
        samples: Tensor,
        text: TextFieldTensors,
        metadata: List[MetadataField],
        targets=None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        encoded_text = self.embedder(**text["roberta"])
        mask = text["roberta"]["mask"]

        if self.gnn_position == 1:
            encoded_text = self.gnn(encoded_text, mask, metadata, **kwargs)

        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory: torch.Tensor = self.mdetr.transformer.resizer(encoded_text)

        if self.gnn_position == 2:
            text_memory = self.gnn(text_memory, mask, metadata, **kwargs)

        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = mask.ne(1).bool()
        # Transpose memory because pytorch's attention expects sequence first
        text = text_attention_mask, text_memory.transpose(0, 1), None
        samples = NestedTensor.from_tensor_list(samples)
        memory_cache = self.mdetr(samples, text, encode_and_save=True)
        outputs = self.mdetr(samples, text, encode_and_save=False, memory_cache=memory_cache)

        predictions, pred_boxes = self.inference(metadata, **outputs)
        output_dict = {"predictions": predictions, "pred_boxes": pred_boxes}

        if targets is not None:
            positive_map = torch.cat([v["positive_map"] for v in targets])
            loss_dict = self.criterion(outputs, targets, positive_map)
            loss = sum(loss_dict[k] * w for k, w in self.weight_dict.items() if k in loss_dict)
            output_dict["loss"] = loss
            output_dict["loss_dict"] = {k: v for k, v in loss_dict.items() if k in self.weight_dict}

        return output_dict

    @torch.no_grad()
    def inference(
        self,
        metadata: List[MetadataField],
        pred_logits: Tensor,
        pred_boxes: Tensor,
        **kwargs
    ) -> Tuple[List[Result], Tensor]:
        def one(scores, spans) -> Result:
            span_to_box_ids = dict()
            for _, (start, end) in spans:
                max_scores, _ = scores[:, start: end + 1].max(dim=-1)
                _, indices = torch.sort(max_scores, descending=True)
                ids = indices.cpu().tolist()
                span_to_box_ids[(start, end)] = ids[:10]
            return span_to_box_ids

        prob = pred_logits.detach().cpu().softmax(-1)  # [batch_size, num_queries, (num_classes + 1)]
        predictions = [one(prob[i], m['spans']) for i, m in enumerate(metadata)]

        pred_boxes = pred_boxes.detach().cpu()  # (center_x, center_y, w, h)
        sizes = torch.tensor([m['size'] for m in metadata])
        # rescale to original image size
        pred_boxes.mul_(torch.cat([sizes, sizes], dim=-1).unsqueeze(1))
        # center_x, center_y to xy
        pred_boxes[..., 0].sub_(pred_boxes[..., 2] / 2)
        pred_boxes[..., 1].sub_(pred_boxes[..., 3] / 2)
        # wh -> x2y2
        pred_boxes[..., 2].add_(pred_boxes[..., 0])
        pred_boxes[..., 3].add_(pred_boxes[..., 1])

        return predictions, pred_boxes
