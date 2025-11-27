import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, scores, labels):
        pos = labels == 1
        if pos.sum() == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        z = scores / self.t
        num = torch.exp(z[pos]).sum()
        den = torch.exp(z).sum()
        return -torch.log(num / (den + 1e-12) + 1e-12)


class InfoNCELoss(nn.Module):
    def __init__(self, init_t=0.07):
        super().__init__()
        self.log_t = nn.Parameter(torch.log(torch.tensor(init_t)))

    def forward(self, scores, labels):
        pos = labels > 0.5
        if pos.sum() == 0:
            return scores.sum() * 0.0
        t = self.log_t.exp().clamp(1e-3, 100.0)
        z = (scores - scores.max()) / t
        log_den = torch.logsumexp(z, dim=0)
        log_num = torch.logsumexp(z[pos], dim=0)
        return -(log_num - log_den)


class BgeM3Loss(nn.Module):
    def __init__(self, model: SentenceTransformer, init_t: float = 0.07):
        super().__init__()
        self.model = model
        self.infonce = InfoNCELoss(init_t=init_t)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, sentence_features, labels):
        reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        emb_q, emb_d = reps
        emb_q = F.normalize(emb_q, p=2, dim=1)
        emb_d = F.normalize(emb_d, p=2, dim=1)

        labels = labels.view(-1).to(emb_q.device).float()
        scores = (emb_q * emb_d).sum(dim=1)

        hard = (labels > 0.5).float()
        loss_infonce = self.infonce(scores, hard)
        loss_bce = self.bce(scores, labels)

        return 0.5 * (loss_infonce + loss_bce)
