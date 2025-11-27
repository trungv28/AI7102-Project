import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.reranker import Reranker


class TrainingLoop:
    def __init__(
        self,
        model: Reranker,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        save_name: str = "gnn",
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        patience: int = 6,
        early_metric: str = "val_loss",
        criterion: nn.Module | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_name = save_name

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.early_metric = early_metric
        self.criterion = criterion or nn.BCEWithLogitsLoss()

    def _build_scheduler(self, total_steps: int):
        """Cosine decay with linear warmup."""
        warmup = max(1, int(self.warmup_ratio * total_steps))

        def lr_lambda(step: int):
            if step < warmup:
                return float(step) / max(1, warmup)
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        val_loss, n_batches = 0.0, 0

        for B in self.val_loader:
            # see .train_utils.collate_graph for details
            if B is None:
                continue

            for i in range(len(B["query_embs"])):
                q = B["query_embs"][i].to(self.device)
                labels = B["labels_list"][i].to(self.device)
                locals_ = B["local_graphs_list"][i]
                global_ = B["global_graphs_list"][i]
                cands = B["candidates_list"][i]
                origs = B["original_nodes_list"][i]

                if len(locals_) == 0:
                    continue

                s = self.model(locals_, global_, q, cands, origs)
                if s.numel() == 0 or s.shape[0] != labels.shape[0]:
                    continue

                loss = self.criterion(s, labels.float())
                val_loss += float(loss)
                n_batches += 1

        return val_loss / max(1, n_batches)

    def train(self, n_epochs: int = 10, eval_interval: int = 1) -> str:
        # total optimizer steps ~= number of per-query updates across all epochs
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = n_epochs * steps_per_epoch
        scheduler = self._build_scheduler(total_steps)

        best_metric = float("inf")
        bad_epochs = 0
        best_path = f"weight/{self.save_name}_best.pt"
        global_step = 0

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            running = 0.0
            n_batches = 0

            for B in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
            ):
                if B is None:
                    continue

                for i in range(len(B["query_embs"])):
                    q = B["query_embs"][i].to(self.device)
                    labs = B["labels_list"][i].to(self.device)
                    locals_ = B["local_graphs_list"][i]
                    global_ = B["global_graphs_list"][i]
                    cands = B["candidates_list"][i]
                    origs = B["original_nodes_list"][i]

                    if len(locals_) == 0:
                        continue

                    self.optimizer.zero_grad(set_to_none=True)

                    s = self.model(locals_, global_, q, cands, origs)
                    if s.numel() == 0 or s.shape[0] != labs.shape[0]:
                        continue

                    loss = self.criterion(s, labs.float())
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    scheduler.step()

                    running += float(loss)
                    n_batches += 1
                    global_step += 1

            avg_train = running / max(1, n_batches)
            print(f"epoch {epoch}/{n_epochs} - train_loss: {avg_train:.4f}")

            # validate on schedule
            do_eval = (epoch % eval_interval == 0) or (epoch == n_epochs)
            if not do_eval:
                continue

            val_loss = self._validate()
            print(f"epoch {epoch}/{n_epochs} - val_loss: {val_loss:.4f}")

            current = val_loss
            is_better = current < best_metric
            if is_better:
                best_metric = current
                bad_epochs = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "best_metric": best_metric,
                        "epoch": epoch,
                    },
                    best_path,
                )
                print(f"  ✓ saved best to {best_path}")
            else:
                bad_epochs += 1
                print(f"  no improvement ({bad_epochs}/{self.patience})")
                if bad_epochs >= self.patience:
                    print("  early stopping")
                    break

        return best_path

