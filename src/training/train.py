import json
import os
import random
from typing import Literal

import torch
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import typer

from ..helper import (
    extract_meta,
    get_device,
    load_dataset,
    log_mem_status,
    set_seed,
    get_embedder,
)
from ..models.reranker import Reranker
from ..models.reranker2 import DualReranker
from .bge_distill import run_bge_multiround_pipeline
from .base_distill import run_multiround_pipeline
from .dataset import LegalGraphDataset
from .loss import ContrastiveLoss, InfoNCELoss
from .splits import SplitData, create_splits
from .train_loop import TrainingLoop
from .train_utils import (
    Evaluator,
    collate_graph,
)

BGE_PATH = "models/bge-reranker-v2-m3"
SBERT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEVICE = get_device()

def main(
    task: Literal[
        "gnn_dual",
        "base_retr",
        "base_teacher",
        "bge_teacher"
    ] = "gnn_dual",
    storage_dir: str = "dataset/legal_graphs2",
    jsonl_file: str = "dataset/dataset.jsonl",
    meta_pkl: str = "dataset/vectordb_files/database11.pkl",
    epochs: int = 30,
    bs: int = 32,
    lr: float = 3e-4,
    seed: int = 11,
    patience: int = 6,
    warmup_ratio: float = 0.1,
    weight_decay: float = 1e-2,
    opt_lr: float = 1e-4,
    loss_name: Literal["contrastive", "infonce", "bce"] = "bce",
    retrieval_depth: int = 100,
):
    set_seed(seed)

    print("===== config =====")
    print("task:", task)
    print("device:", DEVICE)
    print("seed:", seed)
    print("storage_dir:", storage_dir)
    print("jsonl_file:", jsonl_file)
    print("meta_pkl:", meta_pkl)
    print("epochs:", epochs)
    print("bs:", bs)
    print("lr:", lr)
    print("==================")

    splits_path = "dataset/splits.json"
    create_splits(jsonl_file, splits_path)

    rows = load_dataset(jsonl_file)
    with open(splits_path, "r") as f:
        groups: SplitData = json.load(f)

    train_rows = [rows[i] for i in groups.get("train", [])]
    val_rows = [rows[i] for i in groups.get("val", [])]
    test_rows = [rows[i] for i in groups.get("test", [])]

    if task == "base_retr":
        print("training base retriever (MultipleNegativesRankingLoss)")
        uri2text, _ = extract_meta(meta_pkl)

        train_pairs: list[InputExample] = []
        for row in train_rows:
            q = row.get("question", "").strip()
            rels = [u for u in row.get("relevant_node_uris", [])]
            if not q or not rels:
                continue
            uri = random.choice(rels)
            pos = uri2text.get(uri, uri)
            train_pairs.append(InputExample(texts=[q, pos]))
        print("train pairs:", len(train_pairs))

        model = get_embedder(SBERT_MODEL)
        loss = losses.MultipleNegativesRankingLoss(model)
        loader = DataLoader(
            train_pairs,  # pyright: ignore[reportArgumentType]
            batch_size=max(1, bs),
            shuffle=True,
            drop_last=True,
        )
        warmup = int(0.1 * (len(loader) * max(1, epochs)))
        OUT_DIR = "weight/base_retriever_st"
        os.makedirs(OUT_DIR, exist_ok=True)
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=warmup,
            use_amp=True,
            output_path=OUT_DIR,
        )
        model = get_embedder(OUT_DIR)

        evaluator = Evaluator(model, uri2text, retrieval_depth=retrieval_depth)

        val_metrics = evaluator.non_graph(val_rows, "val")
        test_metrics = evaluator.non_graph(test_rows, "test")

        for metric in val_metrics:
            print(
                "{:<12} {:<8.4f} {:<8.4f}".format(
                    metric,
                    val_metrics[metric],
                    test_metrics[metric],
                )
            )

    elif task == "base_teacher":
        print("teacher-student distillation")
        out_dir = "weight/multiround_gte_ce"
        topk = 30
        neg_per_q = 20
        encode_bs = 64
        fit_bs = 8
        ce_bs = 8

        run_multiround_pipeline(
            jsonl_file=jsonl_file,
            meta_pkl=meta_pkl,
            sbert_model=SBERT_MODEL,
            ce_path=BGE_PATH,
            out_dir=out_dir,
            encode_bs=encode_bs,
            topk=topk,
            neg_per_q=neg_per_q,
            emb_epochs_r1=3,
            emb_epochs_teacher=3,
            ce_epochs_r1_3=3,
            fit_bs=fit_bs,
            ce_bs=ce_bs,
        )

        final_dir = os.path.join(out_dir, "round_5_embedder")
        model = get_embedder(final_dir)

        uri2text, _ = extract_meta(meta_pkl)
        evaluator = Evaluator(model, uri2text, retrieval_depth=retrieval_depth)

        val_metrics = evaluator.non_graph(val_rows, "val")
        test_metrics = evaluator.non_graph(test_rows, "test")

        for metric in val_metrics:
            print(
                "{:<12} {:<8.4f} {:<8.4f}".format(
                    metric,
                    val_metrics[metric],
                    test_metrics[metric],
                )
            )

    elif task == "bge_teacher":
        print("teacher-student distillation for bge")
        out_dir = "weight/multiround_bge_ce"
        topk = 30
        neg_per_q = 20
        encode_bs = 64
        fit_bs = 8
        ce_bs = 8

        run_bge_multiround_pipeline(
            jsonl_file=jsonl_file,
            meta_pkl=meta_pkl,
            sbert_model=SBERT_MODEL,
            ce_path=BGE_PATH,
            out_dir=out_dir,
            encode_bs=encode_bs,
            topk=topk,
            neg_per_q=neg_per_q,
            emb_epochs_r1=3,
            emb_epochs_teacher=3,
            ce_epochs_r1_3=3,
            fit_bs=fit_bs,
            ce_bs=ce_bs,
        )

        final_dir = os.path.join(out_dir, "round_5_embedder")
        model = get_embedder(final_dir)

        uri2text, _ = extract_meta(meta_pkl)
        evaluator = Evaluator(model, uri2text, retrieval_depth=retrieval_depth)

        val_metrics = evaluator.non_graph(val_rows, "val")
        test_metrics = evaluator.non_graph(test_rows, "test")

        for metric in val_metrics:
            print(
                "{:<12} {:<8.4f} {:<8.4f}".format(
                    metric,
                    val_metrics[metric],
                    test_metrics[metric],
                )
            )

    else:
        print("GNN / HGT")
        layers = 3
        heads = 8

        # Datasets and loaders (unchanged)
        train_ds = LegalGraphDataset(storage_dir, jsonl_file, "train")
        val_ds = LegalGraphDataset(storage_dir, jsonl_file, "val")
        test_ds = LegalGraphDataset(storage_dir, jsonl_file, "test")

        print("node_types:", train_ds.all_node_types)
        print("edge_types:", train_ds.all_edge_types)

        train_loader = DataLoader(
            train_ds,
            batch_size=max(1, bs // 2),
            shuffle=True,
            collate_fn=collate_graph,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_graph,
        )

        if task == "gnn_dual":
            RerankerType, save_name = DualReranker, "gnn_dual"
        else:
            return print('GNN task not supported: ', task)

        print("reranker:", RerankerType.__name__)
        print("layers:", layers)
        print("heads:", heads)

        # Loss
        if loss_name == "contrastive":
            criterion = ContrastiveLoss()
        elif loss_name == "infonce":
            criterion = InfoNCELoss()
        else:
            criterion = None

        assert issubclass(RerankerType, Reranker)
        reranker = RerankerType(
            train_ds.all_node_types,
            train_ds.all_edge_types,
            dim=768,
            nlayers=layers,
            nheads=heads,
        ).to(DEVICE)

        loop = TrainingLoop(
            reranker,
            train_loader,
            val_loader,
            device=DEVICE,
            lr=opt_lr,
            save_name=save_name,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            patience=patience,
            criterion=criterion,
        )
        best_path = loop.train(n_epochs=epochs, eval_interval=1)

        ck = torch.load(best_path, map_location=DEVICE)
        reranker.load_state_dict(ck["model_state_dict"])
        reranker.eval()
        log_mem_status()

        model = get_embedder(SBERT_MODEL)
        uri2text, _ = extract_meta(meta_pkl)

        evaluator = Evaluator(model, uri2text, retrieval_depth)

        base_metrics = evaluator.base(rows, "test", test_ds)
        hgt_metrics = evaluator.hgt(rows, "test", test_ds, reranker)
        bge_metrics = evaluator.bge(rows, "test", test_ds, BGE_PATH)

        print("\nmetric baseline gnn bge (test)")
        print("---------------------------")
        for m in [
            "recall@1",
            "recall@5",
            "recall@10",
            "precision@1",
            "precision@5",
            "precision@10",
            "mrr@5",
            "mrr@10",
        ]:
            print("{:<12} {:<8.4f} {:<8.4f} {:<8.4f}".format(
                base_metrics[m],
                hgt_metrics[m],
                bge_metrics[m],
            ))

        with open("eval_results.json", "w") as f:
            json.dump(
                {
                    "baseline": base_metrics,
                    "gnn": hgt_metrics,
                    "bge": bge_metrics,
                },
                f,
                indent=2,
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gnn_dual", choices=["gnn_dual", "base_retr", "base_teacher", "bge_teacher"])
    parser.add_argument("--storage_dir", type=str, default="dataset/legal_graphs11")
    parser.add_argument("--jsonl_file", type=str, default="dataset/dataset.jsonl")
    parser.add_argument("--meta_pkl", type=str, default="dataset/vectordb_files/database11.pkl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--opt_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--loss_name", type=str, default="bce", choices=["contrastive", "infonce", "bce"])
    parser.add_argument("--retrieval_depth", type=int, default=100)

    args = parser.parse_args()

    main(
        task=args.task,
        storage_dir=args.storage_dir,
        jsonl_file=args.jsonl_file,
        meta_pkl=args.meta_pkl,
        epochs=args.epochs,
        bs=args.bs,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        opt_lr=args.opt_lr,
        loss_name=args.loss_name,
        retrieval_depth=args.retrieval_depth,
    )