import json
import logging
from typing import Any, Dict, List, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp_models.coref.util import make_coref_instance

logger = logging.getLogger(__name__)


@DatasetReader.register("coref_my")
class MyCorefReader(DatasetReader):
    """
    For the cross-validation of C2f-SpanBERT.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.
    # Parameters
    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    remove_singleton_clusters : `bool`, optional (default = `False`)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them. Ontonotes shouldn't have these, and this option should be used for
        testing only.
    """

    def __init__(
        self,
        cross_no: int,  # 0-4
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        remove_singleton_clusters: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cross_no = cross_no
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._remove_singleton_clusters = remove_singleton_clusters

    def _read(self, file_path: str):
        parts = (0, 1772, 3543, 5315, 7086, 8858)
        start, end = parts[self.cross_no], parts[self.cross_no + 1]
        dev = file_path.endswith(".dev")
        with open(file_path.replace(".dev", "")) as file:
            for i, line in enumerate(file):
                hit = (start <= i < end)
                if dev ^ hit:
                    continue
                obj = json.loads(line)
                yield self.text_to_instance(**obj)

    def text_to_instance(
        self,  # type: ignore
        dialog: List[Dict[str, Any]],
        text_ref: List[Dict[str, Any]],
        clusters: List[List[int]],
        **kwargs
    ) -> Instance:
        sentences = [sentence["text"] for sentence in dialog]
        gold_clusters = span_clusters(clusters, text_ref, sentences)
        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            gold_clusters,
            remove_singleton_clusters=self._remove_singleton_clusters,
        )


def span_clusters(clusters, text_ref, sentences) -> List[List[Tuple[int, int]]]:
    gold_clusters = list()
    for cluster in clusters:
        spans = list()
        for index in cluster:
            ref = text_ref[index]
            offset = sum(len(s) for s in sentences[:ref['sentence_id']])
            span = (ref['start'] + offset, ref['end'] + offset)
            spans.append(span)
        gold_clusters.append(spans)

    return gold_clusters
