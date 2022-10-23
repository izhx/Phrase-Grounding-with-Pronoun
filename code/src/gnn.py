from typing import Any, Dict, List, Union

import torch
from allennlp.data.fields import MetadataField
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn import Activation
from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGATConv, RGCNConv

TYPE_TO_GNN = {
    "rgat": RGATConv,
    "rgcn": RGCNConv,
}


@Seq2SeqEncoder.register("rgnn")
class RelationalGnnSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Union[int, List[int]],
        num_layers: int,
        num_relations: int,
        pyg_kwargs: Dict[str, Any],
        activations: Union[Activation, List[Activation]] = torch.nn.ReLU(),
    ) -> None:
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        else:
            assert len(hidden_channels) == num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        else:
            assert len(activations) == num_layers
        activations[-1] = torch.nn.Identity()
        self._activations = torch.nn.ModuleList(activations)
        gnn_class = TYPE_TO_GNN[pyg_kwargs.pop('type')]
        self._layers = torch.nn.ModuleList([
            gnn_class(_in, _out, num_relations, **pyg_kwargs)
            for _in, _out in zip([in_channels] + hidden_channels, hidden_channels)
        ])
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_relations = num_relations

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        metadata: List[MetadataField],
        edge_index: torch.Tensor,  # (batch, 2, num_edges)
        edge_type: torch.Tensor  # (batch, num_edges)
    ) -> torch.Tensor:
        assert self.in_channels == input.size(-1)
        sequence_lengths = mask.sum(dim=1).tolist()

        # construct data for torch_geometric
        data_list = list()
        for i, (length, meta) in enumerate(zip(sequence_lengths, metadata)):
            if len(meta['node_span']) > 0:
                x = [
                    input[i, start: end + 1].mean(0, keepdim=True)
                    for start, end in meta['node_span']
                ]
                x = torch.cat([input[i, :length]] + x, dim=0)
            else:
                x = input[i, :length]
            e = meta['edge_num']
            data = Data(x, edge_index[i, :, :e], edge_type=edge_type[i, :e])
            data_list.append(data)
        else:
            del i, length, x, e, data
        batch: Batch = Batch.from_data_list(data_list)

        # GNN forward computations
        output = batch.x
        for layer, activation in zip(self._layers, self._activations):
            output = layer(output, batch.edge_index, batch.edge_type)
            output = activation(output)

        # reconstruct the sequence from the batched graph
        batch.x = output
        output = torch.zeros_like(input)
        for i, s in enumerate(sequence_lengths):
            output[i, :s] = batch.get_example(i).x[:s]

        return output

    def get_input_dim(self) -> int:
        return self.in_channels

    def get_output_dim(self) -> int:
        return self.hidden_channels[-1]

    def is_bidirectional(self) -> bool:
        return False
