"""Weak-to-strong training driver with selectable auxiliary loss.

Runs the full 4-phase pipeline (weak teacher -> soft labels -> strong student
-> ceiling) exactly like weak_to_strong_aux.py, but lets you pick the auxiliary
loss for the student phase via --aux-loss. This is the experiment harness for
RCA'ing the negative-PGR regression and comparing loss variants head to head.

Examples:
    python train_variants.py --aux-loss prior
    python train_variants.py --aux-loss entropy --alpha-max 0.5
    python train_variants.py --aux-loss confmask --tau 0.9 --sanity-check --no-wandb
"""

import argparse
import gc
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from aux_losses import AUX_LOSSES, AuxConfig
from weak_to_strong import (
    GRAD_ACCUM_STEPS,
    LR,
    MAX_GRAD_NORM,
    STRONG_BATCH_SIZE,
    STRONG_MODEL_ID,
    WANDB_PROJECT,
    WEAK_MODEL_ID,
    evaluate,
    load_classifier,
    prepare_data,
    setup_wandb,
    train_model,  # reused for the GT phases (weak teacher + ceiling)
)
from transformers import AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

SEED = 42
SAVE_ROOT = "./saved_models/variants"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_student(model, dataloader, loss_name, cfg, alpha_max, warmup_frac, desc, log_wandb=False):
    """Train the strong student with the selected auxiliary loss."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = AUX_LOSSES[loss_name]
    state = {}

    num_steps = len(dataloader)
    warmup_steps = max(1, int(warmup_frac * num_steps))
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc=desc)
    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        soft_labels = batch["soft_label"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]

        # naive has no auxiliary term; everyone else warms alpha up linearly.
        alpha = 0.0 if loss_name == "naive" else alpha_max * min(1.0, step / warmup_steps)
        loss, metrics = loss_fn(logits, soft_labels, alpha, state, cfg)

        (loss / GRAD_ACCUM_STEPS).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == num_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        progress.set_postfix({"loss": f"{loss.item():.4f}", "alpha": f"{alpha:.2f}"})
        if log_wandb:
            wandb.log({f"{desc}/loss": loss.item(), f"{desc}/alpha": alpha,
                       **{f"{desc}/{k}": v for k, v in metrics.items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux-loss", choices=list(AUX_LOSSES), default="prior",
                        help="Auxiliary loss for the student phase")
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--warmup-frac", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.90, help="confidence cutoff (confmask)")
    parser.add_argument("--prior", type=float, default=None,
                        help="positive-class prior for the 'prior' threshold; "
                             "default = estimated from weak predictions")
    parser.add_argument("--ema", type=float, default=0.9, help="EMA momentum for the prior threshold")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = os.path.join(SAVE_ROOT, args.aux_loss)
    os.makedirs(save_dir, exist_ok=True)

    use_wandb = setup_wandb(
        enabled=not args.no_wandb,
        name=f"{args.aux_loss}-w2s" + ("-sanity" if args.sanity_check else ""),
        config={
            "method": args.aux_loss,
            "weak_model": WEAK_MODEL_ID,
            "strong_model": STRONG_MODEL_ID,
            "alpha_max": args.alpha_max,
            "warmup_frac": args.warmup_frac,
            "tau": args.tau,
            "ema": args.ema,
            "seed": args.seed,
            "sanity_check": args.sanity_check,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(WEAK_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dl_weak, dl_strong_inf, dl_test, ds_strong, collator = prepare_data(
        tokenizer, is_sanity_check=args.sanity_check, seed=args.seed
    )

    # ---- Phase 1: Weak Teacher (ground truth) ----
    model_weak = load_classifier(WEAK_MODEL_ID, tokenizer)
    train_model(model_weak, dl_weak, desc="Training Weak Teacher", use_soft_labels=False, log_wandb=use_wandb)
    acc_weak = evaluate(model_weak, dl_test, desc="Eval Weak Teacher")
    print(f"--> Weak Teacher Accuracy: {acc_weak * 100:.2f}%\n")

    # ---- Phase 2: Soft-label generation ----
    model_weak.eval()
    weak_soft_labels = []
    print("Generating weak soft-probabilities...")
    with torch.no_grad():
        for batch in tqdm(dl_strong_inf, desc="Inference Pass"):
            input_ids = batch["input_ids"].to(model_weak.device)
            attention_mask = batch["attention_mask"].to(model_weak.device)
            outputs = model_weak(input_ids=input_ids, attention_mask=attention_mask)
            seq_lengths = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            weak_soft_labels.extend(torch.softmax(logits, dim=-1).cpu().tolist())

    ds_strong = ds_strong.add_column("soft_label", weak_soft_labels)
    ds_strong.set_format("torch", columns=["input_ids", "attention_mask", "label_int", "soft_label"])
    gen = torch.Generator().manual_seed(args.seed)
    dl_strong_train = DataLoader(ds_strong, batch_size=STRONG_BATCH_SIZE, shuffle=True,
                                 collate_fn=collator, generator=gen)

    # Positive-class prior, estimated from the weak teacher's own predictions
    # (paper A.4 footnote: compute the threshold prior from weak predictions).
    weak_probs = torch.tensor(weak_soft_labels)
    est_prior = float((weak_probs.argmax(dim=-1) == 1).float().mean())
    prior = args.prior if args.prior is not None else est_prior
    cfg = AuxConfig(prior=prior, tau=args.tau, ema=args.ema)
    print(f"Estimated weak positive-rate prior: {est_prior:.3f} (using {prior:.3f})")

    del model_weak
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Phase 3: Strong Student (weak labels + selected aux loss) ----
    set_seed(args.seed)  # re-seed so head init matches across variants
    model_student = load_classifier(STRONG_MODEL_ID, tokenizer)
    print(f"\nTraining 7B Student with aux loss = '{args.aux_loss}'...")
    train_student(model_student, dl_strong_train, args.aux_loss, cfg,
                  args.alpha_max, args.warmup_frac,
                  desc="Training Student", log_wandb=use_wandb)
    acc_w2s = evaluate(model_student, dl_test, desc="Eval Student")
    print(f"--> Weak-to-Strong Accuracy: {acc_w2s * 100:.2f}%\n")
    if not args.sanity_check:
        model_student.save_pretrained(f"{save_dir}/strong_student", safe_serialization=False)
    del model_student
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Phase 4: Ceiling (ground truth) ----
    set_seed(args.seed)
    model_ceiling = load_classifier(STRONG_MODEL_ID, tokenizer)
    print("\nTraining 7B Ceiling on GROUND TRUTH...")
    train_model(model_ceiling, dl_strong_train, desc="Training Ceiling", use_soft_labels=False, log_wandb=use_wandb)
    acc_ceil = evaluate(model_ceiling, dl_test, desc="Eval Ceiling")
    print(f"--> Ceiling Accuracy: {acc_ceil * 100:.2f}%\n")
    if not args.sanity_check:
        model_ceiling.save_pretrained(f"{save_dir}/ceiling_model", safe_serialization=False)
    del model_ceiling
    torch.cuda.empty_cache()

    # ---- Results ----
    print("\n" + "=" * 45)
    print(f"RESULTS  (aux loss = {args.aux_loss}, seed = {args.seed})")
    print("=" * 45)
    print(f"Weak Teacher (1.5B):       {acc_weak * 100:.2f}%")
    print(f"Strong Student (7B + W2S): {acc_w2s * 100:.2f}%")
    print(f"Ceiling Model (7B + GT):   {acc_ceil * 100:.2f}%")
    print("-" * 45)

    gap = acc_ceil - acc_weak
    pgr = (acc_w2s - acc_weak) / gap * 100 if gap > 0 else None
    if pgr is not None:
        print(f"Performance Gap Recovered (PGR): {pgr:.2f}%")
    else:
        print("PGR undefined: ceiling did not beat the weak teacher.")
    print("=" * 45)

    if use_wandb:
        summary = {"acc_weak": acc_weak, "acc_w2s": acc_w2s, "acc_ceiling": acc_ceil}
        if pgr is not None:
            summary["pgr"] = pgr
        wandb.log(summary)
        wandb.summary.update(summary)
        wandb.finish()
