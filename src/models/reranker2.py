import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from ..create_graph.graph_utils import LocalGraphInfo
from .hgt import HGT
from .pool import QueryPooler
from .reranker import Reranker


class DualReranker(nn.Module, Reranker):
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
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.dim = dim
        self.as_prob = as_prob

        # encoders + pooler
        self.enc = HGT(self.node_types, self.edge_types, dim, nlayers, nheads)
        self.pool = QueryPooler(dim, nheads)

        # normalization + dropout
        self.ln = nn.LayerNorm(dim)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.feat_dropout = nn.Dropout(feat_dropout)

        # MLP score head over rich similarity features
        # input = [rn, tvec, rn*tvec, |rn - tvec|]  -> 4*dim
        in_dim = dim * 4
        hidden = max(dim, 256)
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(feat_dropout),
            nn.Linear(hidden, hidden // 2, bias=False),
            nn.GELU(),
            nn.Dropout(feat_dropout),
            nn.Linear(hidden // 2, 1, bias=True),  # -> logit
        )

    def _split_hetero(self, g: HeteroData, device):
        nodes: dict[NodeType, torch.Tensor] = {}
        _default = torch.empty(0, self.dim, device=device)
        for t in self.node_types:
            if (
                t in g.node_types
                and hasattr(g[t], "x")
                and g[t].x is not None
            ):
                nodes[t] = g[t].x.to(device)
            else: nodes[t] = _default

        edges: dict[EdgeType, torch.Tensor] = {}
        _default = torch.empty(2, 0, dtype=torch.long, device=device)
        for et in self.edge_types:
            if (
                et in g.edge_types
                and hasattr(g[et], "edge_index")
                and g[et].edge_index is not None
            ):
                edges[et] = g[et].edge_index.to(device)
            else: edges[et] = _default

        return nodes, edges

    def _resolve_root(self, g: HeteroData, orig_info: LocalGraphInfo):
        # try by explicit (type, idx)
        rtype = orig_info.get("root_type")
        ridx = orig_info.get("root_idx")
        if rtype is not None and ridx is not None and rtype in g.node_types:
            return rtype, ridx

        # fallback by candidate URI
        cand_uri = orig_info.get("cand_uri")
        if cand_uri is not None:
            for t in g.node_types:
                if not hasattr(g[t], "uri"): continue
                j = g[t].uri.index(cand_uri)
                return t, j
        return None, None

    @staticmethod
    def _unit_norm(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def _score_pair(self, rn: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
        """
        rn   : (D,) normalized + LN + dropout
        tvec : (D,) normalized + LN + dropout
        returns scalar (logit) or prob depending on self.as_prob
        """
        # feature crafting
        mul = rn * tvec
        diff = (rn - tvec).abs()
        feat = torch.cat([rn, tvec, mul, diff], dim=-1)
        feat = self.feat_dropout(feat)
        logit = self.scorer(feat).squeeze(-1)
        return torch.sigmoid(logit) if self.as_prob else logit

    def forward(
        self,
        local_graphs: list[HeteroData | None],
        global_graph: HeteroData | None,
        query_emb: torch.Tensor,
        candidates: list[str],
        original_nodes: list[LocalGraphInfo | None],
    ):
        device = query_emb.device

        # Global side
        if global_graph is not None:
            g_nodes, g_edges = self._split_hetero(global_graph, device)
            g_embs: dict[NodeType, torch.Tensor] = self.enc(g_nodes, g_edges)
            tvec: torch.Tensor = self.pool(g_embs, query_emb)  # (D,)
        else:
            tvec = query_emb

        # normalize + LN + dropout for stability
        tvec = self._unit_norm(tvec).to(device)
        tvec = self.ln(tvec)
        tvec = self.emb_dropout(tvec)

        # --- Local side per candidate ---
        out: list[torch.Tensor] = []
        for g, info in zip(local_graphs, original_nodes):
            if g is None or info is None:
                out.append(torch.tensor(0.0, device=device))
                continue

            local_nodes, local_edges = self._split_hetero(g, device)
            local_enc = self.enc(local_nodes, local_edges)

            # query-guided pooling over local graph (same mechanism as global)
            rn: torch.Tensor = self.pool(local_enc, query_emb)

            # normalize + LN + dropout
            rn = self._unit_norm(rn)
            rn = self.ln(rn)
            rn = self.emb_dropout(rn)

            score = self._score_pair(rn, tvec)
            out.append(score)

        return torch.stack(out, dim=0) if out else torch.zeros(0, device=device)
    