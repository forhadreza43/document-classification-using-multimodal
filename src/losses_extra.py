import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_squared_l2(z: torch.Tensor) -> torch.Tensor:
    z2 = (z * z).sum(dim=1, keepdim=True)
    dist2 = z2 + z2.t() - 2.0 * (z @ z.t())
    return torch.clamp(dist2, min=0.0)

def l2norm(h: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(h, p=2, dim=1, eps=eps)

# (5) CE
class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, h, labels):
        return self.ce(logits, labels)

# (1) Margin (fixed margin)
class FixedMarginLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, lam: float, eps: float = 1e-12):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, h, labels):
        ce_loss = self.ce(logits, labels)
        z = l2norm(h, self.eps)
        B, D = z.shape
        dist2 = pairwise_squared_l2(z)

        labels_col = labels.view(-1, 1)
        same = (labels_col == labels_col.t())
        eye = torch.eye(B, dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye
        neg_mask = ~same

        if pos_mask.sum() == 0:
            raise RuntimeError("No positive pairs in batch. Check sampler/min_per_class.")

        xi = torch.tensor(self.alpha, device=labels.device, dtype=dist2.dtype)

        L_pos = dist2[pos_mask].mean()
        L_neg = F.relu(xi - dist2[neg_mask]).mean() if neg_mask.sum() > 0 else torch.zeros((), device=labels.device)
        L_margin = (self.beta * L_pos + L_neg) / float(D)

        return ce_loss + self.lam * L_margin

# (3) SCL (Supervised Contrastive Loss) + CE
class SCLLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, lam: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.tau = temperature
        self.lam = lam
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, h, labels):
        ce_loss = self.ce(logits, labels)
        z = l2norm(h, self.eps)
        B = z.size(0)

        sim = (z @ z.t()) / self.tau
        sim = sim - torch.max(sim, dim=1, keepdim=True).values

        labels_col = labels.view(-1, 1)
        mask = (labels_col == labels_col.t()).float()
        eye = torch.eye(B, device=labels.device)
        mask = mask * (1.0 - eye)

        exp_sim = torch.exp(sim) * (1.0 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)

        pos_count = mask.sum(dim=1)
        scl = -(mask * log_prob).sum(dim=1) / (pos_count + self.eps)
        scl = scl.mean()

        return ce_loss + self.lam * scl

# (4) Weighted CE
class WeightedCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, h, labels):
        return self.ce(logits, labels)