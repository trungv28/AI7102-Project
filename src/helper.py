from dataclasses import dataclass
import json
import pickle
import faiss
import random
from typing import NamedTuple, TypeVar, TypedDict

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from numpy.typing import NDArray
from rdflib.term import URIRef
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

T = TypeVar('T')

class LocalGraphInfo(TypedDict):
    cand_uri: str
    doc_root: str
    cand_in_graph: bool
    added_docs: list[str]
    root_type: str | None
    root_idx: int | None


class DatasetRow(TypedDict):
    question: str
    relevant_node_uris: list[str]


class _URIRefInfo(TypedDict):
    text: str
    file_path: str
PklDbRow = tuple[URIRef, _URIRefInfo]


@dataclass
class CorpusIndexBuilder:
    """Faiss Index builder for a corpus"""
    model: SentenceTransformer
    texts: list[str]
    batch_size: int = 64

    def build(self) -> faiss.IndexFlatIP:
        emb = self.encode_corpus()
        dim = emb.shape[1] if emb.ndim == 2 else 768
        cpu_idx = faiss.IndexFlatIP(dim)
        if faiss.get_num_gpus() > 0:
            # raise RuntimeError('not supported for faiss gpu yet')
            gpu_idx = faiss.index_cpu_to_all_gpus(cpu_idx)
            if emb.size > 0: gpu_idx.add(emb)
            cpu_idx = faiss.index_gpu_to_cpu(gpu_idx)
        else:
            if emb.size > 0: cpu_idx.add(emb)  # pyright: ignore[reportCallIssue]
        return cpu_idx

    def encode_corpus(self) -> NDArray[np.float32]:
        vecs: list[NDArray[np.float32]] = []
        for i in tqdm(range(0, len(self.texts), self.batch_size)):
            chunk = self.texts[i : i + self.batch_size]
            v = self.model.encode(
                chunk,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
                trust_remote_code=True,
            )
            vecs.append(v.astype(np.float32))
        if len(vecs) == 0:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack(vecs)


class GraphData(TypedDict):
    query_emb: torch.Tensor
    local_graphs: list[HeteroData | None]
    global_graph: HeteroData | None
    original_nodes: list[LocalGraphInfo | None]
    labels: torch.Tensor
    candidates: list[str]
    relevant_uris: list[str]


def set_seed(seed=11):
    """Set random seed for all possible libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_mem_status():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"gpu mem  alloc: {a:.2f} GB   reserv: {r:.2f} GB")


_device: torch.device | None = None
def get_device():
    if _device: return _device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset_file: str) -> list[DatasetRow]:
    with open(dataset_file) as f:
        rows = [json.loads(line) for line in f]
    return rows

def load_meta_raw(meta_file: str) -> list[PklDbRow]:
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    return meta

class ExtractedMeta(NamedTuple):
    uri2text: dict[str, str]
    uri2file: dict[str, str]

def extract_meta(meta_file: str):
    meta = load_meta_raw(meta_file)
    uri2text = {str(u): str(m["text"]) for u, m in meta}
    uri2file = {str(u): str(m["file_path"]) for u, m in meta}
    del meta
    return ExtractedMeta(uri2text, uri2file)


def faiss_topk(
    qs: list[str], model: SentenceTransformer, faiss_idx, k=5
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    # output shapes: N x K, distances and lables of top k resp.
    qv = model.encode(
        qs,
        convert_to_tensor=True,
        normalize_embeddings=True,
        trust_remote_code=True,
    )
    D, I = faiss_idx.search(qv.cpu().numpy(), k)
    return D, I

def get_embedder(model_path: str):
    return SentenceTransformer(model_path, trust_remote_code=True)
