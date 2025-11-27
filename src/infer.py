import random
import typer
import json
import faiss
from .training.splits import SplitData

from .helper import (
    CorpusIndexBuilder,
    extract_meta,
    faiss_topk,
    get_device,
    get_embedder,
    load_dataset,
    set_seed,
)


def main(
    jsonl_file: str = "dataset/dataset.jsonl",
    splits_path: str = "dataset/splits.json",
    n_demo: int = 5,
    outfile: str = "demo.json",
    meta_pkl: str = "dataset/vectordb_files/database11.pkl",
    model_id: str = "weight/base_teacher/round_4",
    encode_bs: int = 64,
    topk: int = 3,
    seed: int = 22,
    idx_path: str | None = "dataset/vectordb_files/vectordb11.bin"
):
    DEVICE = get_device()
    set_seed(seed)

    model = get_embedder(model_id).to(DEVICE)

    uri2text, _ = extract_meta(meta_pkl)
    texts = list(uri2text.values())

    if idx_path is None:
        faiss_idx = CorpusIndexBuilder(model, texts, encode_bs).build()
    else:
        faiss_idx = faiss.read_index(idx_path)

    rows = load_dataset(jsonl_file)
    with open(splits_path, "r") as f:
        groups: SplitData = json.load(f)

    test_rows = [rows[i] for i in groups.get("test", [])]
    test_samples = random.sample(test_rows, n_demo)

    queries = [row["question"] for row in test_samples]
    _, I = faiss_topk(queries, model, faiss_idx, k=topk)

    demo_data = []
    for q, cand_indexes in zip(queries, I):
        cand_texts: list[str] = [texts[i] for i in cand_indexes]
        demo_data.append({"input": q, "output": cand_texts})

    with open(outfile, "w") as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    typer.run(main)