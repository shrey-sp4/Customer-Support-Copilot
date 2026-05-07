"""Custom triage losses: boundary-margin loss for confident triage decisions.

Loss formulation:
    z = model logits (batch, 3)
    p = softmax(z)
    z_correct  = logit of gold class
    z_wrong_max = maximum logit among wrong classes
    L_boundary = softplus(z_wrong_max - z_correct + mu)
    L_total = L_CE + lambda_boundary * L_boundary

KB proximity regularization (optional):
    If provided, L_KB penalizes ANSWER predictions when nearest_chunk_sim is very low.
    This is implemented as a soft regularization on the logit distribution, not on frozen embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BoundaryAwareLoss(nn.Module):
    """
    Combines cross-entropy with a margin-based boundary loss that rewards
    confident, well-separated triage decisions.
    """

    def __init__(
        self,
        mu: float = 0.15,
        lambda_boundary: float = 0.6,
        lambda_kb: float = 0.0,
    ):
        super().__init__()
        self.mu              = mu
        self.lambda_boundary = lambda_boundary
        self.lambda_kb       = lambda_kb
        self.ce_loss         = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,           # (batch, num_classes)
        labels: torch.Tensor,           # (batch,) int
        nearest_chunk_sim: Optional[torch.Tensor] = None,  # (batch,) float in [0,1]
    ) -> torch.Tensor:
        # --- Cross-entropy loss ---
        L_CE = self.ce_loss(logits, labels)

        # --- Boundary margin loss ---
        # For each sample: softplus(max_wrong_logit - correct_logit + mu)
        batch_size, num_classes = logits.shape
        # correct logit for each sample
        correct_logits = logits[torch.arange(batch_size), labels]           # (batch,)
        # mask out correct class
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False
        wrong_logits = logits.masked_fill(~mask, float("-inf"))
        z_wrong_max  = wrong_logits.max(dim=-1).values                      # (batch,)

        L_boundary = F.softplus(z_wrong_max - correct_logits + self.mu).mean()

        # --- KB proximity regularization (optional) ---
        L_KB = torch.tensor(0.0, device=logits.device)
        if self.lambda_kb > 0.0 and nearest_chunk_sim is not None:
            # Penalize high ANSWER confidence when chunk sim is low
            # L_KB = mean(p_ANSWER * (1 - nearest_chunk_sim)) for near-zero sim
            p = F.softmax(logits, dim=-1)
            p_answer = p[:, 0]  # ANSWER is class 0
            low_sim  = (1.0 - nearest_chunk_sim).clamp(0.0, 1.0)
            L_KB     = (p_answer * low_sim).mean()

        L_total = L_CE + self.lambda_boundary * L_boundary + self.lambda_kb * L_KB
        return L_total, {
            "L_CE":       L_CE.item(),
            "L_boundary": L_boundary.item(),
            "L_KB":       L_KB.item(),
        }
