from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..helper import CorpusIndexBuilder, DatasetRow, GraphData, faiss_topk, get_device
from ..models.reranker2 import DualReranker
from .dataset import LegalGraphDataset
from .splits import SplitType


def collate_graph(batch_data: Collection[GraphData]):
    valid_data = [b for b in batch_data if len(b['local_graphs']) > 0]
    if not valid_data: return None

    return {
            'query_embs': torch.stack([b['query_emb'] for b in valid_data]),
            'labels_list': [b['labels'] for b in valid_data],
            'local_graphs_list': [b['local_graphs'] for b in valid_data],
            'global_graphs_list': [b['global_graph'] for b in valid_data],
            'candidates_list': [b['candidates'] for b in valid_data],
            'original_nodes_list': [b['original_nodes'] for b in valid_data]
        }


def avg_metrics(metrics: dict[str, list[float]]):
    out: dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = float(np.mean(v)) if len(v) > 0 else 0.0
    return out


def compute_metrics(
    cands: list[str],
    rels: list[str],
    recall_values: list[int] | None = None,
):
    if recall_values is None:
        recall_values = [1, 5, 10]

    m: dict[str, float] = {}
    R = set(rels)
    for k in recall_values:
        top_k = cands[:k]
        hit = len([c for c in top_k if c in R])
        m[f"recall@{k}"] = (hit / max(1, len(R))) if R else 0.0
        m[f"precision@{k}"] = (hit / k) if k > 0 else 0.0

    m["mrr@5"] = mrr_at(cands, rels, 5)
    m["mrr@10"] = mrr_at(cands, rels, 10)
    return m


def mrr_at(cands: list[str], rels: list[str], k: int):
    R = set(rels)
    for r, c in enumerate(cands[:k], 1):
        if c in R:
            return 1.0 / r
    return 0.0


def get_query_and_rels(row: DatasetRow):
    query = row.get("question", "").strip()
    rels = [str(u) for u in row.get("relevant_node_uris", [])]
    return query, rels


@dataclass
class Evaluator:
    model: SentenceTransformer
    uri2text: dict[str, str]
    retrieval_depth: int

    def __post_init__(self):
        self.uris = list(self.uri2text.keys())
        texts = list(self.uri2text.values())
        self.index = CorpusIndexBuilder(self.model, texts).build()
        self.device = get_device()


    def non_graph(self, split_rows: list[DatasetRow], split: SplitType):
        metrics: dict[str, list[float]] = {}
        for row in tqdm(split_rows, desc=f"eval {split}"):
            q, rels = get_query_and_rels(row)
            if not q or not rels: continue

            D, I = faiss_topk(
                [q],
                self.model,
                self.index,
                k = self.retrieval_depth,
            )
            cands: list[str] = [self.uris[i] for i in I[0]]

            m = compute_metrics(cands, rels)
            for k, v in m.items():
                metrics.setdefault(k, []).append(v)

        return avg_metrics(metrics)


    def base(
        self,
        rows: list[DatasetRow],
        split: SplitType,
        ds: LegalGraphDataset,
    ):
        metrics: dict[str, list[float]] = {}
        for i in tqdm(range(len(ds)), desc=f"gnn {split}"):
            qid = ds.query_ids[i]
            q, rels = get_query_and_rels(rows[qid])
            if not q or not rels: continue

            D, I = faiss_topk(
                [q],
                self.model,
                self.index,
                k = self.retrieval_depth,
            )
            cands: list[str] = [self.uris[i] for i in I[0]]

            m = compute_metrics(cands, rels)
            for k, v in m.items():
                metrics.setdefault(k, []).append(v)

        return avg_metrics(metrics)

    @torch.no_grad()
    def hgt(
        self,
        rows: list[DatasetRow],
        split: SplitType,
        ds: LegalGraphDataset,
        reranker: DualReranker,
    ):
        metrics: dict[str, list[float]] = {}
        for i in tqdm(range(len(ds)), desc=f"gnn {split}"):
            qid = ds.query_ids[i]
            q, rels = get_query_and_rels(rows[qid])
            if not q or not rels: continue

            D, I = faiss_topk(
                [q],
                self.model,
                self.index,
                k = self.retrieval_depth,
            )
            cands: list[str] = [self.uris[i] for i in I[0]]

            fallback_scores: list[float] = ((D[0] + 1.0) * 0.5).tolist()
            cand2score = {u: s for u, s in zip(cands, fallback_scores)}

            # If we have local graphs for a subset, overwrite their scores with GNN
            b = ds[i]
            if len(b["local_graphs"]) > 0:
                q_gnn = b["query_emb"].to(self.device)
                scores = reranker(
                    b["local_graphs"],
                    b["global_graph"],
                    q_gnn,
                    b["candidates"],
                    b["original_nodes"],
                )
                if scores.numel() > 0:
                    scores = scores.detach().cpu().numpy().tolist()
                    for u, sv in zip(b["candidates"], scores):
                        cand2score[u] = float(sv)

            m = compute_metrics(cands, rels)
            for k, v in m.items():
                metrics.setdefault(k, []).append(v)

        return avg_metrics(metrics)


    def bge(
        self,
        rows: list[DatasetRow],
        split: SplitType,
        ds: LegalGraphDataset,
        bge_path: str,
    ):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        metrics: dict[str, list[float]] = {}
        tokenizer = AutoTokenizer.from_pretrained(
            bge_path, local_files_only=True
        )
        bge_m = (
            AutoModelForSequenceClassification.from_pretrained(bge_path)
            .to(self.device).eval()
        )
        for i in tqdm(range(len(ds)), desc=f"bge {split}"):
            qid = ds.query_ids[i]
            q, rels = get_query_and_rels(rows[qid])
            if not q or not rels: continue

            D, I = faiss_topk(
                [q],
                self.model,
                self.index,
                k = self.retrieval_depth,
            )
            cands: list[str] = [self.uris[i] for i in I[0]]

            # Build pairs for BGE; fall back to name if no text
            pairs: list[tuple[str, str]] = []
            for u in cands:
                text = self.uri2text.get(u, "")
                if text and len(text.strip()) > 10:
                    pairs.append((q, text[:512]))
                else:
                    name = u.split("/")[-1]
                    pairs.append((q, f"Legal document: {name}"))

            # Score with BGE and rerank
            with torch.no_grad():
                inp = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                scores: torch.Tensor = (
                    bge_m(**inp, return_dict=True).logits.view(-1).float()
                )
                scores = torch.sigmoid(scores)

            indexes = torch.argsort(scores, descending=True).cpu().numpy().tolist()
            cands = [cands[i] for i in indexes]

            m = compute_metrics(cands, rels)
            for k, v in m.items():
                metrics.setdefault(k, []).append(v)

        return avg_metrics(metrics)
