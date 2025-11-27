from functools import partial
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from ..helper import (
    CorpusIndexBuilder,
    DatasetRow,
    extract_meta,
    faiss_topk,
    get_device,
    get_embedder,
    load_dataset,
)
from .train_utils import get_query_and_rels

DEVICE = get_device()

# --------------------- Scoring functions ----------------------
MinedPair = list[str] # 2 elements only
LabeledMinedPair = tuple[MinedPair, bool]

def mine_pairs(
    embedder: SentenceTransformer,
    faiss_index,
    uris: list[str],
    uri2text: dict[str, str],
    rows: list[DatasetRow],
    topk=30,
    neg_per_q=20,
):
    mined: list[LabeledMinedPair] = []
    for row in tqdm(rows, desc=f"Mining pairs (top{topk}/{neg_per_q} negs)"):
        q, pos_uris = get_query_and_rels(row)
        if not pos_uris: continue

        D, I = faiss_topk([q], embedder, faiss_index, k=topk)
        cands: list[str] = [uris[i] for i in I[0]]
        neg_uris = [u for u in cands if u not in pos_uris][:neg_per_q]
        for u in pos_uris:
            mined.append(([q, uri2text.get(u, u)], True))
        for u in neg_uris:
            mined.append(([q, uri2text.get(u, u)], False))
    return mined

def score_pairs_embedder_cos(
    embedder: SentenceTransformer,
    pairs: list[MinedPair],
    batch_size: int,
) -> NDArray[np.float32]:
    out = []
    for i in tqdm(
        range(0, len(pairs), batch_size),
        desc="Scoring pairs with embedder cosine",
    ):
        chunk = pairs[i : i + batch_size]
        qs = [str(a) for a, b in chunk]
        ds = [str(b) for a, b in chunk]
        qv = embedder.encode(
            qs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            trust_remote_code=True,
        )
        dv = embedder.encode(
            ds,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            trust_remote_code=True,
        )
        out.append((qv * dv).sum(axis=1).astype(np.float32))
    return (
        np.concatenate(out, axis=0)
        if out else np.zeros((0,), dtype=np.float32)
    )


def score_pairs_ce(
    cross_encoder,
    pairs: list[MinedPair],
    tokenizer,
    batch_size: int,
) -> NDArray[np.float32]:
    out = []
    for i in tqdm(
        range(0, len(pairs), batch_size),
        desc="Scoring pairs with CE",
    ):
        chunk = pairs[i : i + batch_size]
        q = [str(a) for a, b in chunk]
        d = [str(b) for a, b in chunk]

        with torch.no_grad():
            inp = tokenizer(
                q,
                text_pair=d,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(DEVICE)
            logits = (
                cross_encoder(**inp, return_dict=True)
                .logits.view(-1).float().cpu().numpy()
            )
        out.append(logits)
    return (
        np.concatenate(out, axis=0)
        if out else np.zeros((0,), dtype=np.float32)
    )


# --------------------- CE training ----------------------------
class CEPairDataset(Dataset):
    def __init__(self, tokenizer, pairs: list[LabeledMinedPair]):
        self.tokenizer = tokenizer
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        (q, d), y = self.pairs[i]
        return q, d, float(y)

    def collate(self, batch: list[tuple[str, str, float]]):
        q = [b[0] for b in batch]
        d = [b[1] for b in batch]
        y = [b[2] for b in batch]
        y = torch.tensor(y, dtype=torch.float32)
        enc = self.tokenizer(
            q,
            text_pair=d,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return {**enc, "labels": y}


def train_cross_encoder(
    cross_encoder,
    labeled_pairs: list[LabeledMinedPair],
    tokenizer,
    epochs=3,
    bs=32,
    lr=2e-5,
    wd=0.01,
):
    ds = CEPairDataset(tokenizer, labeled_pairs)
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=ds.collate,
        drop_last=False,
    )

    cross_encoder.train().to(DEVICE)
    bce = torch.nn.BCEWithLogitsLoss()

    optimizer = AdamW(cross_encoder.parameters(), lr=lr, weight_decay=wd)
    total_steps = max(1, epochs * len(dl))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    for ep in tqdm(range(epochs), desc="CE epochs"):
        running = 0.0
        for step, batch in enumerate(
            tqdm(
                dl,
                desc=f"CE train (e{ep + 1}/{epochs})",
                leave=False,
            )
        ):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = cross_encoder(**batch, return_dict=True)
            loss = bce(out.logits.view(-1), batch["labels"])
            loss.backward()

            clip_grad_norm_(cross_encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            cross_encoder.zero_grad(set_to_none=True)

            running += loss.item()
            if (step + 1) % 10 == 0:
                tqdm.write(
                    "[CE e{}] step {}/{} - loss: {}".format(
                        ep + 1, step + 1, len(dl), running
                    )
                )
                running = 0.0
    cross_encoder.eval()
    return cross_encoder


# --------------------- Main 5-round pipeline -------------------
def run_multiround_pipeline(
    jsonl_file: str,
    meta_pkl: str,
    sbert_model="Alibaba-NLP/gte-multilingual-base",
    ce_path="models/bge-reranker-v2-m3",
    out_dir="weight/multiround",
    encode_bs=256,
    topk=30,
    neg_per_q=20,

    emb_epochs_r1=1,
    emb_epochs_teacher=1,
    ce_epochs_r1_3=3,
    fit_bs=64,
    ce_bs=32,

):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    uri2text, _ = extract_meta(meta_pkl)
    uris = list(uri2text.keys())
    texts = list(uri2text.values())

    rows = load_dataset(jsonl_file)
    train_rows = [r for r in rows if r.get("split", "train") == "train"]

    # partial functions
    mine_pairs_part = partial(
        mine_pairs,
        uris = uris,
        uri2text = uri2text,
        rows = train_rows,
        topk = topk,
        neg_per_q = neg_per_q,
    )

    DataLoaderPart = partial(
        DataLoader,
        batch_size=fit_bs,
        shuffle=True,
        drop_last=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ce_path, local_files_only=True)
    train_cross_encoder_part = partial(
        train_cross_encoder,
        tokenizer=tokenizer,
        epochs=ce_epochs_r1_3,
        bs=ce_bs,
    )

    # FULL 5-ROUND PIPELINE
    embedder = SentenceTransformer(sbert_model, trust_remote_code=True)
    cross_encoder = (
        AutoModelForSequenceClassification.from_pretrained(
            ce_path, local_files_only=True
        ).to(DEVICE).eval()
    )

    # -------- Round 1: train embedder on gold; train CE on gold ----------
    r = 1
    print(f"\n[Round {r}] Train embedder (gold) → mine → train CE (gold)")
    index = CorpusIndexBuilder(embedder, texts, encode_bs).build()
    mined = mine_pairs_part(embedder, index)

    # Train embedder on gold labels
    st_examples = [
        InputExample(texts=p, label=float(y)) for (p, y) in mined
    ]
    loss = losses.CosineSimilarityLoss(model=embedder)
    loader = DataLoaderPart(st_examples)  # pyright: ignore[reportArgumentType]
    warm = int(0.05 * len(loader) * emb_epochs_r1)
    save_rx = out_dir / f"round_{r}_embedder"
    embedder.fit(
        train_objectives=[(loader, loss)],
        epochs=emb_epochs_r1,
        warmup_steps=warm,
        use_amp=True,
        output_path=str(save_rx),
    )
    embedder = get_embedder(str(save_rx))

    # Re-mine with updated embedder and train CE on **gold**
    index = CorpusIndexBuilder(embedder, texts, encode_bs).build()
    mined = mine_pairs_part(embedder, index)
    cross_encoder = train_cross_encoder_part(cross_encoder, mined)
    torch.save(cross_encoder.state_dict(), str(out_dir / f"round_{r}_ce.pt"))

    # -------- Rounds 2–3: teacher-guided with CE scores ----------
    for r in [2,3]:
        print(f"\n[Round {r}] CE labels → fine-tune embedder; CE retrained on gold")
        index = CorpusIndexBuilder(embedder, texts, encode_bs).build()
        mined = mine_pairs_part(embedder, index)
        mined = [p for (p, _) in mined]

        logits = score_pairs_ce(
            cross_encoder,
            mined,
            tokenizer,
            batch_size=128,
        )
        probs = 1 / (1 + np.exp(-logits))
        st_examples = [
            InputExample(texts=p, label=float(l)) for p, l in zip(mined, probs)
        ]
        loss = losses.CosineSimilarityLoss(model=embedder)
        loader = DataLoaderPart(st_examples)  # pyright: ignore[reportArgumentType]
        warm = int(0.05 * len(loader) * emb_epochs_teacher)

        save_rx = out_dir / f"round_{r}_embedder"
        embedder.fit(
            train_objectives=[(loader, loss)],
            epochs=emb_epochs_teacher,
            warmup_steps=warm,
            use_amp=True,
            output_path=str(save_rx),
        )
        embedder = get_embedder(str(save_rx))

        index = CorpusIndexBuilder(embedder, texts, encode_bs).build()
        mined = mine_pairs_part(embedder, index)
        cross_encoder = train_cross_encoder_part(cross_encoder, mined)
        torch.save(cross_encoder.state_dict(), str(out_dir / f"round_{r}_ce.pt"))

    # -------- Rounds 4–5: student self-improvement with embedder scores ----------
    for r in [4,5]:
        print(f"\n[Round {r}] Self-training with embedder cosine scores")
        index = CorpusIndexBuilder(embedder, texts, encode_bs).build()
        mined = mine_pairs_part(embedder, index)
        mined = [p for (p, _) in mined]

        cos = score_pairs_embedder_cos(embedder, mined, batch_size=128)
        soft = (cos + 1.0) / 2.0
        st_examples = [
            InputExample(texts=p, label=float(l)) for p, l in zip(mined, soft)
        ]

        loss = losses.CosineSimilarityLoss(model=embedder)
        loader = DataLoaderPart(st_examples)  # pyright: ignore[reportArgumentType]
        warm = int(0.05 * len(loader))
        save_rx = out_dir / f"round_{r}_embedder"
        embedder.fit(
            train_objectives=[(loader, loss)],
            epochs=emb_epochs_teacher,
            warmup_steps=warm,
            use_amp=True,
            output_path=str(save_rx),
        )
        embedder = get_embedder(str(save_rx))

    return str(out_dir)

