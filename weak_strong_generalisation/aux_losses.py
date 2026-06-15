"""Auxiliary loss variants for weak-to-strong generalization experiments.

Every variant shares one signature so the training driver can swap them freely:

    fn(logits, soft_labels, alpha, state, cfg) -> (loss, metrics)

    logits      : [B, 2]  strong-student logits at the last real token
    soft_labels : [B, 2]  weak-teacher predictive distribution (the supervision)
    alpha       : float    current weight on the auxiliary term (warmed up 0 -> alpha_max)
    state       : dict      persists across steps (used for EMA threshold tracking)
    cfg         : AuxConfig  hyperparameters

All variants collapse to the plain weak-imitation CE when alpha == 0, so the
alpha warmup gives every method an identical starting point. The first term is
always CE(student, weak_labels); the variants differ only in the auxiliary term.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AuxConfig:
    prior: float = 0.5        # target positive rate for the "prior" threshold
    tau: float = 0.90         # confidence cutoff for "confmask"
    ema: float = 0.9          # momentum for the running threshold in "prior"


def _soft_ce(logits, soft_labels):
    """Cross-entropy against a probability-distribution target (the weak labels)."""
    return F.cross_entropy(logits, soft_labels)


# ------------------------------------------------------------------
# Variants
# ------------------------------------------------------------------
def loss_naive(logits, soft_labels, alpha, state, cfg):
    """Baseline: imitate the weak teacher, no auxiliary term."""
    return _soft_ce(logits, soft_labels), {}


def loss_half(logits, soft_labels, alpha, state, cfg):
    """Paper Eq.1 as originally reproduced: hard self-target with a fixed 50/50
    threshold (exactly half the batch labelled class 1). Kept for comparison —
    this is the variant suspected of injecting bias on imbalanced data."""
    weak = _soft_ce(logits, soft_labels)
    p1 = F.softmax(logits, dim=-1)[:, 1]
    hard = torch.zeros_like(p1, dtype=torch.long)
    k = p1.numel() // 2
    if k > 0:
        hard[torch.topk(p1, k).indices] = 1
    conf = F.cross_entropy(logits, hard)
    return (1 - alpha) * weak + alpha * conf, {"pos_rate": float(hard.float().mean())}


def loss_prior(logits, soft_labels, alpha, state, cfg):
    """Prior-corrected self-bootstrapping (paper Eq.1, Appendix A.4 footnote).

    The hardening threshold is set to the cfg.prior quantile of the batch's
    positive-class probability (so the *prior* fraction of examples are labelled
    class 1), and smoothed with an EMA across steps to beat the noise of tiny
    physical batches. This removes the 50/50 bias that hurts on imbalanced data.
    """
    weak = _soft_ce(logits, soft_labels)
    p1 = F.softmax(logits, dim=-1)[:, 1].float().detach()

    # Cutoff such that `prior` fraction of the batch sit above it.
    q = torch.quantile(p1, 1.0 - cfg.prior)
    t = state.get("t_ema", q)
    t = cfg.ema * t + (1.0 - cfg.ema) * q
    state["t_ema"] = t.detach()

    hard = (p1 > t).long()
    conf = F.cross_entropy(logits, hard)
    return (1 - alpha) * weak + alpha * conf, {
        "threshold": float(t),
        "pos_rate": float(hard.float().mean()),
    }


def loss_entropy(logits, soft_labels, alpha, state, cfg):
    """Conditional entropy minimization (Grandvalet & Bengio, 2004).

    Threshold-free, soft version of the confidence loss: instead of hardened
    pseudo-labels we directly minimize the entropy of the student's predictions.
    No class-balance assumption, so it is immune to the imbalance problem.
    """
    weak = _soft_ce(logits, soft_labels)
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean()
    return (1 - alpha) * weak + alpha * ent, {"entropy": float(ent)}


def loss_confmask(logits, soft_labels, alpha, state, cfg):
    """Confidence-masked self-training (FixMatch-style, Sohn et al., 2020).

    The self-training term is applied only to examples the student is already
    confident about (max prob > tau), so it never reinforces uncertain — and
    likely weak-error-driven — predictions.
    """
    weak = _soft_ce(logits, soft_labels)
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    mask = conf > cfg.tau
    if mask.any():
        self_loss = F.cross_entropy(logits[mask], pred[mask])
    else:
        self_loss = logits.new_zeros(())
    return (1 - alpha) * weak + alpha * self_loss, {"masked_frac": float(mask.float().mean())}


AUX_LOSSES = {
    "naive": loss_naive,
    "half": loss_half,
    "prior": loss_prior,
    "entropy": loss_entropy,
    "confmask": loss_confmask,
}
