import gc
import random
from pathlib import Path

import faiss
import torch
import typer
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from tqdm import tqdm

from ..helper import (
    LocalGraphInfo,
    faiss_topk,
    load_dataset,
    extract_meta,
    set_seed,
)
from .graph_utils import GlobalGraphBuilder, LocalGraphBuilder
from .store import Store


def main(
    dataset_file: str,
    max_samples: int | None = None,
    max_cands: int = 50,
    out_dir: str = "dataset/legal_graphs2",
    add_root_edge: bool = False,
    root_rel: str = "root_to",
    log_k: int = 3,
):
    # LOAD RESOURCES
    logger.info("load document to filepath mapping")
    doc_folders = [
        "dataset/ttldataAJZ",
        "dataset/ttldataDEC",
        "dataset/ttldataDOC",
    ]
    doc2file: dict[str, str] = {}
    for d in doc_folders:
        for p in Path(d).rglob("*.ttl"):
            doc2file[p.stem] = str(p)

    ## database
    logger.info("load uri list and its text, file mapping from database.pkl")
    uri2text, uri2file = extract_meta("dataset/vectordb_files/database11.pkl")
    uri_list = list(uri2file)

    logger.info("load faiss index")
    faiss_idx = faiss.read_index("dataset/vectordb_files/vectordb11.bin")

    logger.info("load embedding model")
    model_name = "distilbert/distilbert-base-uncased"
    model = SentenceTransformer(model_name, trust_remote_code=True)

    ## queries dataset
    rows = load_dataset(dataset_file)

    idx_rows = [(i, x) for i, x in enumerate(rows)]
    del rows
    if max_samples:
        sample_size = min(len(idx_rows), max_samples)
        idx_rows = random.sample(idx_rows, sample_size)

    # ============================================================
    logger.info("begin building graphs")
    set_seed()
    S = Store(out_dir)
    LGB = LocalGraphBuilder(
        uri2file,
        uri2text,
        doc2file,
        model,
        S.store,
        add_star=add_root_edge,
        root_rel=root_rel,
    )

    shown = 0
    for qid, item in tqdm(idx_rows, desc="build"):
        q: str = item["question"]

        _, I = faiss_topk([q], model, faiss_idx, k=max_cands)
        cand_uris: list[str] = [uri_list[i] for i in I[0]]

        # build local graphs
        local_graph_list: list[HeteroData] = []
        local_graph_infos: list[LocalGraphInfo] = []
        valid_cand_uris: list[str] = []

        for uri in cand_uris:
            g, info = LGB.build(uri)
            if g is None or info is None:
                continue

            local_graph_list.append(g)
            local_graph_infos.append(info)
            valid_cand_uris.append(uri)

        # build global graph
        drop_relations = {root_rel}
        G = GlobalGraphBuilder(
            local_graph_list, drop_relations=drop_relations
        ).build()

        # store save
        rel_set = set(item["relevant_node_uris"])
        labels = [int(uri in rel_set) for uri in valid_cand_uris]
        qv = model.encode(
            [q],
            convert_to_tensor=True,
            trust_remote_code=True,
        ).squeeze(0)

        S.save(
            qid, q, qv,
            valid_cand_uris,
            labels,
            list(rel_set),
            local_graph_list,
            local_graph_infos,
            G,
        )

        # light logging
        if shown < log_k:
            print("\n=== example ===")
            print("qid:", qid)
            print("q:", q[:160].replace("\n", " "))
            print(
                "num_cands:", len(valid_cand_uris),
                "num_locals:", len(local_graph_list),
                "has_global:", G is not None,
            )
            for j, (uri, info) in enumerate(
                zip(valid_cand_uris[:3], local_graph_infos[:3])
            ):
                print(f"  cand[{j}]:", uri.split("/")[-1])
                print("   - cand_in_graph:", info["cand_in_graph"])
                print("   - doc_root:", info["doc_root"].split("/")[-1])
                print(
                    "   - added_1hop_docs:",
                    [x.split("/")[-1] for x in info["added_docs"]],
                )
                print(
                    "   - root_type:", info["root_type"],
                    "root_idx:", info["root_idx"],
                )
            shown += 1

        # cleanup memory
        del local_graph_list, local_graph_infos, G, qv
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    S.close()


if __name__ == "__main__":
    typer.run(main)

