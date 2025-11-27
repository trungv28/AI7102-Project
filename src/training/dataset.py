import json
import random
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ..create_graph.store import QueryMeta, Store
from ..helper import GraphData, LocalGraphInfo, load_dataset
from .splits import SplitData, SplitType


class QueryData(TypedDict):
    meta: QueryMeta
    query_emb: torch.Tensor
    local_graphs: list[HeteroData | None]
    global_graph: HeteroData | None
    original_nodes: list[LocalGraphInfo | None]


class LegalGraphDataset(Dataset):
    def __init__(
        self,
        storage_dir: str,
        jsonl_file: str,
        split: SplitType = "train",
        max_samples: int | None = None,
        splits_path: str = "dataset/splits.json",
    ):
        self.storage_dir = Path(storage_dir)

        self.jsonl_data = load_dataset(jsonl_file)

        self.store = Store(storage_dir)
        self.index = self.store.idx

        available_qids: list[int] = []
        for qid_str in self.index.queries:
            qid = int(qid_str)
            if (
                qid < len(self.jsonl_data)
                and (self.storage_dir / f"query_{qid}").exists()
            ):
                available_qids.append(qid)

        if (path := Path(splits_path)).exists():
            with path.open() as f:
                groups: SplitData = json.load(f)
            wanted = set(groups[split])
            self.query_ids = [q for q in available_qids if q in wanted]
        else:
            random.shuffle(available_qids)
            split_point = int(0.8 * len(available_qids))
            self.query_ids = (
                available_qids[:split_point]
                if split == "train"
                else available_qids[split_point:]
            )

        self.query_ids = self.query_ids[:max_samples]

        self.all_node_types = list(self.index.ntypes)
        self.all_edge_types = list(self.index.etypes)

    def _load_query_data(self, qid: int):
        q_dir = self.storage_dir / f"query_{qid}"
        with (q_dir / "meta.json").open() as f:
            meta: QueryMeta = json.load(f)

        qv = self.store.load_q(qid)
        if not isinstance(qv, torch.Tensor):
            # should not reach here !
            qv = torch.from_numpy(qv).float()

        local_graphs: list[HeteroData | None] = []
        original_nodes: list[LocalGraphInfo | None] = []
        for i in range(meta['num_locals']):
            g, oi = self.store.load_graph(qid, str(i))
            local_graphs.append(g)
            original_nodes.append(oi)

        global_graph = None
        if meta['global_file']:
            global_graph, _ = self.store.load_graph(qid, 'global')

        ret: QueryData = {
            'meta': meta,
            'query_emb': qv,
            'local_graphs': local_graphs,
            'global_graph': global_graph,
            'original_nodes': original_nodes
        }
        return ret

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx: int):
        qid = self.query_ids[idx]
        graph_data = self._load_query_data(qid)
        jsonl_item = self.jsonl_data[qid]

        relevant_uris = set(jsonl_item.get('relevant_node_uris', []))
        candidates = graph_data['meta']['cands']
        labels = [1.0 if c in relevant_uris else 0.0 for c in candidates]

        ret: GraphData = {
            'query_emb': graph_data['query_emb'],
            'local_graphs': graph_data['local_graphs'],
            'global_graph': graph_data['global_graph'],
            'original_nodes': graph_data['original_nodes'],
            'labels': torch.tensor(labels, dtype=torch.float),
            'candidates': candidates,
            'relevant_uris': list(relevant_uris)
        }
        return ret

