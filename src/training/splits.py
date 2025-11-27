import json
import random
from pathlib import Path
from typing import Literal
import typer
from loguru import logger

from ..helper import load_dataset

SplitType = Literal["train", "test", "val"]
SplitData = dict[SplitType, list[int]]


def create_splits(
    jsonl_file: str,
    out_file: str = "dataset/splits.json",
    train: float = 0.8,
    val: float = 0.1,
    seed: int = 11,
    force: bool = False
):
    out_path = Path(out_file)
    if out_path.exists() and not force:
        return logger.warning(f"split file {out_path} already exists !")

    rows = load_dataset(jsonl_file)

    groups: SplitData = {"train": [], "val": [], "test": []}

    if any("split" in r for r in rows):
        logger.warning("why 'split' exists in dataset row ?")
        for i, r in enumerate(rows):
            s = r.get("split", "train")
            if s not in groups:
                s = "train"
            groups[s].append(i)
    else:
        indexes = list(range(len(rows)))
        random.Random(seed).shuffle(indexes)

        n_idx = len(indexes)
        n_train = int(train * n_idx)
        n_val = int(val * n_idx)
        groups["train"] = indexes[:n_train]
        groups["val"] = indexes[n_train : n_train + n_val]
        groups["test"] = indexes[n_train + n_val :]

    with out_path.open("w") as f:
        json.dump(groups, f, indent=2)
    logger.success(f'wrote split file {out_path}')


if __name__ == "__main__":
    typer.run(create_splits)

