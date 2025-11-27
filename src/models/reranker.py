from abc import abstractmethod

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from ..create_graph.graph_utils import LocalGraphInfo


class Reranker():
    @abstractmethod
    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        dim: int,
        nlayers: int,
        nheads: int,
        emb_dropout=0.2,
        feat_dropout=0.2,
        as_prob=False,
    ): ...

    @abstractmethod
    def forward(
        self,
        local_graphs: list[HeteroData | None],
        global_graph: HeteroData | None,
        query_emb: torch.Tensor,
        candidates: list[str],
        original_nodes: list[LocalGraphInfo | None],
    ) -> torch.Tensor:
        ...
