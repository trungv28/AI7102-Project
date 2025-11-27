import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Collection, TypedDict

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from ..helper import LocalGraphInfo


@dataclass
class _Resource:
    idx_f: Path
    emb_f: Path

    idx_map: dict[str, int]
    emb: np.memmap
    after: int


class EmbStore:
    def __init__(self, dir: str, *, dim=768, step=2000, init_cap=1111):
        self.dir = Path(dir)
        self.dir.mkdir(exist_ok=True)

        self.dim = dim
        self.step = step
        self.init_cap = init_cap

        idx_f = self.dir / "embedding_index.json"
        emb_f = self.dir / "embeddings.dat"
        self.data = self._init_index_and_embeding(idx_f, emb_f)

        qidx_f = self.dir / "query_index.json"
        qemb_f = self.dir / "query_embeddings.dat"
        self.qdata = self._init_index_and_embeding(qidx_f, qemb_f)


    def _init_index_and_embeding(self, idx_f: Path, emb_f: Path) -> _Resource:
        if idx_f.exists():
            with idx_f.open() as f:
                idx_map: dict[str, int] = json.load(f)

            n = len(idx_map)
            cap = (n + 50) if n > 0 else self.init_cap
            mode = "r+" if emb_f.exists() else "w+"
            emb = np.memmap(
                emb_f,
                dtype=np.float32,
                mode=mode,
                shape=(cap, self.dim),
            )
            after = n
        else:
            idx_map = {}
            emb = np.memmap(
                emb_f,
                dtype=np.float32,
                mode="w+",
                shape=(self.init_cap, self.dim),
            )
            after = 0

        return _Resource(idx_f, emb_f, idx_map, emb, after)

    def _get_data(self, *, for_q: bool):
        return self.qdata if for_q else self.data

    def _grow(self, data: _Resource):
        old: int = data.emb.shape[0]
        new = old + self.step
        tmp_file = Path(f"{data.emb_f}.tmp")
        tmp = np.memmap(
            tmp_file,
            dtype=np.float32,
            mode="w+",
            shape=(new, self.dim),
        )
        tmp[:old] = data.emb[:old]

        tmp_file.replace(data.emb_f)
        data.emb = np.memmap(
            data.emb_f,
            dtype=np.float32,
            mode="r+",
            shape=(new, self.dim),
        )


    def add(self, u: str, v: np.ndarray, *, for_q=False):
        data = self._get_data(for_q=for_q)

        if u not in data.idx_map:
            if data.after >= data.emb.shape[0]: self._grow(data)

            data.idx_map[u] = data.after
            data.emb[data.after] = v.astype(np.float32)
            data.after += 1

        return data.idx_map[u]


    def get(self, uri: str, *, for_q=False):
        data = self._get_data(for_q=for_q)
        return data.emb[data.idx_map[uri]] if uri in data.idx_map else None


    def gets(self, uris: Collection[str]) -> tuple[np.ndarray, list[str]]:
        indexes: list[int] = []
        keep_uris: list[str] = []

        for u in uris:
            idx = self.data.idx_map.get(u)
            if idx is None: continue
            indexes.append(idx)
            keep_uris.append(u)

        return (
            (self.data.emb[indexes], keep_uris)
            if indexes
            else (np.array([]), [])
        )

    def flush(self):
        self.data.emb.flush()
        with self.data.idx_f.open('w') as f:
            json.dump(self.data.idx_map, f)

        self.qdata.emb.flush()
        with self.qdata.idx_f.open('w') as f:
            json.dump(self.qdata.idx_map, f)


class QueryInfo(TypedDict):
    question: str
    num_cands: int
    num_locals: int
    has_global: bool
    dir: str

@dataclass
class StoreIdx:
    queries: dict[str, QueryInfo] = field(default_factory=dict)
    total: int = 0
    ntypes: set[NodeType] = field(default_factory=set)
    etypes: set[EdgeType] = field(default_factory=set)


class SaveGraph(TypedDict):
    ntypes: list[NodeType]
    etypes: list[EdgeType]
    node_uris: dict[NodeType, list[str]]
    edges: dict[str, list[float]]


class QueryMeta(TypedDict):
    qid: int
    question: str
    cands: list[str]
    labels: list[int]
    rel_uris: list[str]
    local_files: list[str]
    global_file: str | None
    original_node_files: list[str]
    num_locals: int


class Store:
    """Graphs + meta storage """

    def __init__(self, base: str):
        self.base = Path(base)
        self.base.mkdir(exist_ok=True)

        self.idx_f = self.base / "index.json"
        self.store = EmbStore(base)

        if self.idx_f.exists():
            with self.idx_f.open() as f:
                idx_load = json.load(f)

            idx_load["ntypes"] = set(idx_load['ntypes'])
            idx_load["etypes"] = set(tuple(et) for et in idx_load["etypes"])
            self.idx = StoreIdx(**idx_load)
        else:
            self.idx = StoreIdx()

    def save_graph(self, g: HeteroData, path: Path):
        save_obj: SaveGraph = {
            "ntypes": list(g.node_types),
            "etypes": g.edge_types,
            "node_uris": {},
            "edges": {},
        }
        for nt in g.node_types:
            if hasattr(g[nt], "uri"):
                save_obj["node_uris"][nt] = g[nt].uri

        for et in g.edge_types:
            if hasattr(g[et], "edge_index"):
                save_obj["edges"][str(et)] = g[et].edge_index.tolist()

        with open(path, "w") as f:
            json.dump(save_obj, f)

    def save_orig(self, info: LocalGraphInfo, path: Path):
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def save_idx(self):
        j = deepcopy(asdict(self.idx))
        j["ntypes"] = list(self.idx.ntypes)
        j["etypes"] = list(self.idx.etypes)
        with open(self.idx_f, "w") as f:
            json.dump(j, f)

    def save(
        self,
        qid: int,
        qtext: str,
        q_emb: torch.Tensor,
        cands: list[str],
        labels: list[int],
        rel_uris: list[str],
        local_graphs: list[HeteroData],
        local_graph_infos: list[LocalGraphInfo],
        G: HeteroData | None,
    ):
        qdir = self.base / f"query_{qid}"
        qdir.mkdir(exist_ok=True)

        self.store.add(str(qid), q_emb.cpu().numpy(), for_q=True)

        local_struct_files = []
        orig_node_files = []
        for i, (g, info) in enumerate(zip(local_graphs, local_graph_infos)):
            lf = f"local_{i}_structure.json"
            of = f"original_node_{i}.json"
            self.save_graph(g, qdir / lf)
            self.save_orig(info, qdir / of)

            local_struct_files.append(lf)
            orig_node_files.append(of)

            self.idx.ntypes.update(g.node_types)
            self.idx.etypes.update(g.edge_types)

        global_struct_file = None
        if G is not None:
            global_struct_file = "global_structure.json"
            self.save_graph(G, qdir / global_struct_file)

        meta: QueryMeta = {
            "qid": qid,
            "question": qtext,
            "cands": cands,
            "labels": labels,
            "rel_uris": rel_uris,
            "local_files": local_struct_files,
            "global_file": global_struct_file,
            "original_node_files": orig_node_files,
            "num_locals": len(local_graphs),
        }
        with open(qdir / "meta.json", "w") as f:
            json.dump(meta, f)

        self.idx.queries[str(qid)] = {
            "question": qtext,
            "num_cands": len(cands),
            "num_locals": len(local_struct_files),
            "has_global": global_struct_file is not None,
            "dir": str(qdir.relative_to(self.base)),
        }

        self.idx.total += 1
        if (self.idx.total % 10) == 0: self.store.flush()

        self.save_idx()

    def _load_graph(self, path: Path):
        if not path.exists(): return None

        with path.open() as f:
            save_obj: SaveGraph = json.load(f)

        g = HeteroData()
        for nt in save_obj["ntypes"]:
            uris = save_obj['node_uris'].get(nt)
            if uris is None: continue

            batch, keep = self.store.gets(uris)
            if len(keep) > 0:
                g[nt].x = torch.from_numpy(batch).float()
                g[nt].uri = keep

        for et_str, edge in save_obj["edges"].items():
            et: EdgeType = tuple(eval(et_str))
            if edge:
                g[et].edge_index = torch.tensor(edge, dtype=torch.long)
        return g

    def load_graph(
        self, qid: int, kind="global"
    ) -> tuple[HeteroData | None, LocalGraphInfo | None]:
        qdir = self.base / f"query_{qid}"
        if kind == "global":
            p = qdir / "global_structure.json"
            return self._load_graph(p), None

        p = qdir / f"local_{kind}_structure.json"
        g = self._load_graph(p)

        oi = None
        op = qdir / f"original_node_{kind}.json"
        if op.exists():
            with open(op, "r") as f:
                oi = json.load(f)
        return g, oi

    def load_q(self, qid: int):
        return torch.from_numpy(self.store.get(str(qid), for_q=True)).float()

    def close(self):
        self.store.flush()
