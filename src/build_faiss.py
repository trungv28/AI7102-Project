import faiss
import numpy as np

from .helper import extract_meta, get_device, get_embedder


def build_with_model(
    model_dir: str,
    meta_pkl: str,
    out_path: str,
    batch=64,
):
    uri2text, _ = extract_meta(meta_pkl)
    texts = list(uri2text.values())

    model = get_embedder(model_dir).to(get_device())
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch,
        show_progress_bar=True,
    ).astype(np.float32)
    d = emb.shape[1]
    index = faiss.IndexHNSWFlat(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(faiss.index_cpu_to_all_gpus(index))
    index.add(emb)
    faiss.write_index(index, out_path)
    return out_path

if __name__ == "__main__":
    build_with_model(
        "weight/base_teacher/round_4",
        "dataset/vectordb_files/database11.pkl",
        "dataset/vectordb_files/vectordb11.bin",
    )
