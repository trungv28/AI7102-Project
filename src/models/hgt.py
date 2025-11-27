import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from torch_geometric.typing import EdgeType, NodeType


class HGT(nn.Module):
    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        dim: int,
        nlayers: int,
        nheads: int,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dim = dim

        self.elayers = nn.ModuleList(
            [
                HGTConv(dim, dim, (self.node_types, self.edge_types), nheads)
                for _ in range(nlayers)
            ]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(nlayers)])
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(nlayers)]
        )

    def forward(
        self,
        nodes: dict[NodeType, torch.Tensor],
        edges: dict[EdgeType, torch.Tensor]
    ):
        # nodes: {t: (N_t, D)}, edges: {e: (2, E)}
        x = {
            t: nodes.get(t,
                torch.empty(0, self.dim, device=next(self.parameters()).device)
            )
            for t in self.node_types
        }
        e = {
            et: edges.get(et,
                torch.empty(2, 0, dtype=torch.long,
                    device=next(self.parameters()).device
                )
            )
            for et in self.edge_types
        }

        out = x
        for i, layer in enumerate(self.elayers):
            nxt: dict[NodeType, torch.Tensor] = layer(out, e)
            merged = {}
            for t in self.node_types:
                if t in out and t in nxt and out[t].size(0) > 0:
                    y = out[t] + nxt[t]
                    y = self.layernorms[i](y)
                    y = self.dropouts[i](y)
                    merged[t] = y
                else:
                    merged[t] = nxt.get(t, out.get(t,
                        torch.empty(
                            0, self.dim,
                            device=out[self.node_types[0]].device
                        )
                    ))
            out = merged
        return out

