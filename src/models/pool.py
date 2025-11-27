import torch
import torch.nn as nn
from torch_geometric.typing import NodeType


class QueryPooler(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            dim, nheads, batch_first=True, bias=False
        )
        self.lin = nn.Linear(dim * 2, dim, bias=False)

    def forward(
        self,
        node_embs: dict[NodeType, torch.Tensor],
        q_emb: torch.Tensor,
    ) -> torch.Tensor:
        # node_embs: {t: (N_t, D)}, q_emb: (D,)
        xs = [x for x in node_embs.values() if x is not None and x.size(0) > 0]
        if len(xs) == 0:
            # fallback: just return normalized query
            qn = q_emb / (q_emb.norm(p=2) + 1e-12)
            return qn

        K = torch.cat(xs, dim=0).unsqueeze(0)  # (1, N, D)
        Q = q_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        out, _ = self.mha(Q, K, K)  # (1, 1, D)
        g = out.squeeze(0).squeeze(0)  # (D,)
        tvec = self.lin(torch.cat([g, q_emb], dim=-1))
        tvec = tvec / (tvec.norm(p=2) + 1e-12)
        return tvec

