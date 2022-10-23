from typing import Dict, List

import torch

from allennlp.data import Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import Backbone
from allennlp.nn import InitializerApplicator

from src.metrics import AverageDict, PhraseGroundingRecall


@Model.register("phrase_grounding_wrapper")
class PhraseGroundWraper(Model):
    """
    This `Model` wraps common allen-style codes for the visual grounding task.
    Registered as a `Model` with name "visual_ground".

    # Parameters

    vocab : `Vocabulary`
    backbone : `Backbone`
        different visual grounding models.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        backbone: Backbone,
        iou_threshold: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._backbone = backbone
        self._recall = PhraseGroundingRecall(iou_threshold)
        self._average = AverageDict()
        initializer(self)

    def forward(
        self,
        metadata: List[MetadataField],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        text : `TextFieldTensors`
            From a `TextField`
        metadata : `List[MetadataField]`, optional (default = `None`)
            From a `MetadataField`

        # Returns

        An output dictionary consisting of:

            - `predictions` (`torch.FloatTensor`) :
                List of Dict[Span, List[int]], the int items are indices in pred_boxes.
            - `pred_boxes` (`torch.IntTensor`) :
                A tensor of shape `(batch_size, num_queries, 4)`,
                expected (x, y, x2, y2).
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        output_dict = self._backbone(metadata=metadata, **kwargs)
        if 'boxes' in metadata[0]:
            self._recall(
                output_dict["predictions"], [x['span_to_box_ids'] for x in metadata],
                output_dict["pred_boxes"], [x['boxes'] for x in metadata]
            )
            if "loss_dict" in output_dict:
                self._average(output_dict["loss_dict"])
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._average.get_metric(reset)
        metrics.update(self._recall.get_metric(reset))
        return metrics
