import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import gc
import os
import wandb
import random
import numpy as np

# ==========================================
# Configuration & Setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VRAM Optimized Batch Sizes for 9B Student
WEAK_BATCH_SIZE = 32
STRONG_BATCH_SIZE = 2        # REDUCED from 4/16 to fit 9B parameters on standard GPU VRAM
GRAD_ACCUM_STEPS = 8         # 2 physical batches * 8 steps = 16 effective batch size
SAVE_DIR = "./saved_models/aux"

# Training hyperparameters
LR = 1e-5
MAX_GRAD_NORM = 1.0

# Auxiliary confidence loss (paper Sec. 4.3.2 / Appendix A.4, Eq. 1):
#   L = (1 - a) * CE(f(x), f_w(x)) + a * CE(f(x), f_hat_t(x))
# where a is warmed up linearly from 0 to ALPHA_MAX over the first WARMUP_FRAC of
# training, and f_hat_t are the student's own predictions hardened with an
# adaptive threshold (exactly half the batch labelled class 1 -> class-balance prior).
ALPHA_MAX = 0.5      # paper uses 0.5, or 0.75 for the very largest students
WARMUP_FRAC = 0.2    # fraction of steps over which alpha ramps 0 -> ALPHA_MAX

# Weights & Biases
WANDB_PROJECT = "weak-to-strong-generalization"

# Reproducibility
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=SEED):
    """Seed all RNGs (Python, NumPy, torch CPU+CUDA) for reproducible runs.

    The data splits are fixed index slices (already deterministic); this seeds
    the stochastic parts — classifier-head init and DataLoader shuffling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# Data Processing Pipeline
# ==========================================
def prepare_data(tokenizer_weak, tokenizer_strong, is_sanity_check=False, seed=SEED):
    print("Loading and formatting dataset...")
    dataset = load_dataset("google/boolq")
    
    def format_and_label(example):
        prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}?\nAnswer:"
        return {"prompt": prompt, "label_int": 1 if example['answer'] else 0}
        
    dataset = dataset.map(format_and_label)
    
    # Tokenize separately for weak and strong models to prevent tokenizer mismatch bugs
    def tokenize_weak_func(example):
        return tokenizer_weak(example["prompt"], truncation=True, max_length=512)
        
    def tokenize_strong_func(example):
        return tokenizer_strong(example["prompt"], truncation=True, max_length=512)
        
    tokenized_weak = dataset.map(tokenize_weak_func, batched=True, remove_columns=["passage", "question", "answer", "prompt"])
    tokenized_weak.set_format("torch", columns=["input_ids", "attention_mask", "label_int"])
    
    tokenized_strong = dataset.map(tokenize_strong_func, batched=True, remove_columns=["passage", "question", "answer", "prompt"])
    tokenized_strong.set_format("torch", columns=["input_ids", "attention_mask", "label_int"])
    
    if is_sanity_check:
        print("SANITY CHECK MODE: Restricting data arrays to micro-slices.")
        ds_weak_train = tokenized_weak['train'].select(range(0, 32))
        ds_strong_inf = tokenized_weak['train'].select(range(32, 64))
        ds_strong_train = tokenized_strong['train'].select(range(32, 64))
        ds_test_weak = tokenized_weak['validation'].select(range(0, 32))
        ds_test_strong = tokenized_strong['validation'].select(range(0, 32))
    else:
        # Balanced slice split following the spirit of the paper
        ds_weak_train = tokenized_weak['train'].select(range(0, 4000))
        ds_strong_inf = tokenized_weak['train'].select(range(4000, 8000))
        ds_strong_train = tokenized_strong['train'].select(range(4000, 8000))
        ds_test_weak = tokenized_weak['validation']
        ds_test_strong = tokenized_strong['validation']
        
    collator_weak = DataCollatorWithPadding(tokenizer=tokenizer_weak)
    collator_strong = DataCollatorWithPadding(tokenizer=tokenizer_strong)
    
    gen_weak = torch.Generator().manual_seed(seed)
    dl_weak = DataLoader(ds_weak_train, batch_size=WEAK_BATCH_SIZE, shuffle=True, collate_fn=collator_weak, generator=gen_weak)
    dl_strong_inf = DataLoader(ds_strong_inf, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator_weak)
    dl_test_weak = DataLoader(ds_test_weak, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator_weak)
    dl_test_strong = DataLoader(ds_test_strong, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator_strong)
    
    return dl_weak, dl_strong_inf, dl_test_weak, dl_test_strong, ds_strong_train, collator_strong

# ==========================================
# Model Architecture Engineering
# ==========================================
def load_classifier(model_id, tokenizer):
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Enable Gradient Checkpointing to dynamically clear layer activations
    model.gradient_checkpointing_enable() 
    
    YES_ID = tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO_ID = tokenizer.encode("No", add_special_tokens=False)[0]
    
    hidden_size = model.config.hidden_size
    orig_head = model.lm_head
    
    with torch.no_grad():
        yes_weights = orig_head.weight[YES_ID].clone()
        no_weights = orig_head.weight[NO_ID].clone()
        
    # Fix multi-GPU device placement: use orig_head device
    device = orig_head.weight.device
    classifier = nn.Linear(hidden_size, 2, bias=False, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        classifier.weight[0] = no_weights.to(device)
        classifier.weight[1] = yes_weights.to(device)
        
    model.lm_head = classifier
    model.config.vocab_size = 2
    
    # Fix configuration warnings
    model.config.bos_token_id = None
    model.config.eos_token_id = None
    
    # We replaced the (tied) lm_head with a fresh 2-class head; break the tie so
    # save_pretrained doesn't choke on shared-tensor serialization.
    model.config.tie_word_embeddings = False
    return model

# ==========================================
# Isolated Training Mechanics
# ==========================================
def train_model(model, dataloader, desc, use_soft_labels=False, is_student=False, freeze_backbone=False, log_wandb=False):
    model.train()
    
    if freeze_backbone:
        print("Freezing model backbone (Linear Probing mode)...")
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier head
        for param in model.lm_head.parameters():
            param.requires_grad = True
            
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=LR)
    else:
        optimizer = AdamW(model.parameters(), lr=LR)
        
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    num_steps = len(dataloader)
    warmup_steps = max(1, int(WARMUP_FRAC * num_steps))

    progress = tqdm(dataloader, desc=desc)
    for step, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]

        alpha = 0.0
        # Auxiliary confidence loss ONLY runs in the student (weak-supervision) phase.
        if use_soft_labels and is_student:
            # Linear warmup of alpha: 0 -> ALPHA_MAX over the first WARMUP_FRAC of training.
            alpha = ALPHA_MAX * min(1.0, step / warmup_steps)

            # 1. Weak imitation term: CE against the teacher's soft probability targets.
            # Fix multi-GPU: move targets to logits device
            soft_targets = batch['soft_label'].to(logits.device)
            loss_weak = criterion(logits, soft_targets)

            # 2. Confidence term: CE against the student's OWN hardened predictions.
            #    Adaptive threshold t -> label exactly half the batch as class 1, which
            #    bakes in the class-balance prior and prevents collapse to one class.
            with torch.no_grad():
                p1 = F.softmax(logits, dim=-1)[:, 1]
                hard = torch.zeros_like(p1, dtype=torch.long)
                k = p1.size(0) // 2
                if k > 0:
                    hard[torch.topk(p1, k).indices] = 1
            loss_conf = criterion(logits, hard)

            # Convex combination (paper Eq. 1).
            loss = (1.0 - alpha) * loss_weak + alpha * loss_conf
        else:
            # Clean Ground Truth CrossEntropy for Weak Teacher and Max Ceiling.
            # Fix multi-GPU: move targets to logits device
            targets = batch['label_int'].to(logits.device)
            loss = criterion(logits, targets)

        # Scale for gradient accumulation, then accumulate.
        (loss / GRAD_ACCUM_STEPS).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == num_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        progress.set_postfix({'loss': f"{loss.item():.4f}", 'alpha': f"{alpha:.2f}"})
        if log_wandb:
            wandb.log({f"{desc}/loss": loss.item(), f"{desc}/alpha": alpha})

# ==========================================
# Evaluation Framework
# ==========================================
def evaluate(model, dataloader, desc):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            seq_lengths = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            
            predictions = torch.argmax(logits, dim=-1)
            
            # Dynamically move targets to predictions device
            targets = batch['label_int'].to(predictions.device)
            
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / total

# ==========================================
# Weights & Biases setup (token-based, no interactive login)
# ==========================================
def setup_wandb(enabled, name, config):
    """Initialise W&B without ever triggering an interactive login.

    Auth is via the WANDB_API_KEY environment variable (export it before
    running, e.g. `export WANDB_API_KEY=...`). If the key is missing we fall
    back to offline mode so runs are still recorded locally and can be pushed
    later with `wandb sync`. Returns True only if logging is active.
    """
    if not enabled:
        return False
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set -> running W&B in OFFLINE mode "
              "(sync later with `wandb sync`). Pass --no-wandb to disable.")
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=WANDB_PROJECT, name=name, config=config)
    return True


# ==========================================
# Main Orchestration Loop
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", type=str, default="Qwen/Qwen3.5-2B-Instruct", help="Weak supervisor model ID or local path")
    parser.add_argument("--strong-model", type=str, default="Qwen/Qwen3.5-9B-Instruct", help="Strong student model ID or local path")
    parser.add_argument("--hard-labels", action="store_true", help="Supervise student with hard thresholded labels instead of soft probabilities")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling factor for soft label logits")
    parser.add_argument("--linear-probe", action="store_true", help="Run latent representation linear probing on frozen strong backbone")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    use_wandb = setup_wandb(
        enabled=not args.no_wandb,
        name="aux-w2s" + ("-sanity" if args.sanity_check else ""),
        config={
            "method": "auxiliary_confidence",
            "weak_model": args.weak_model,
            "strong_model": args.strong_model,
            "weak_batch_size": WEAK_BATCH_SIZE,
            "strong_batch_size": STRONG_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "lr": LR,
            "max_grad_norm": MAX_GRAD_NORM,
            "alpha_max": ALPHA_MAX,
            "warmup_frac": WARMUP_FRAC,
            "hard_labels": args.hard_labels,
            "temperature": args.temperature,
            "linear_probe": args.linear_probe,
            "seed": args.seed,
            "sanity_check": args.sanity_check,
        },
    )

    # 1. Setup Tokenizers
    tokenizer_weak = AutoTokenizer.from_pretrained(args.weak_model)
    tokenizer_weak.pad_token = tokenizer_weak.eos_token
    tokenizer_weak.padding_side = "right"   # last-token readout assumes right padding

    tokenizer_strong = AutoTokenizer.from_pretrained(args.strong_model)
    tokenizer_strong.pad_token = tokenizer_strong.eos_token
    tokenizer_strong.padding_side = "right"

    dl_weak, dl_strong_inf, dl_test_weak, dl_test_strong, ds_strong_train, collator_strong = prepare_data(
        tokenizer_weak, tokenizer_strong, is_sanity_check=args.sanity_check, seed=args.seed
    )

    # ------------------------------------------
    # Phase 1: Train Weak Teacher (GT Only)
    # ------------------------------------------
    set_seed(args.seed)
    model_weak = load_classifier(args.weak_model, tokenizer_weak)
    train_model(model_weak, dl_weak, desc="Training Weak Teacher", use_soft_labels=False, is_student=False, log_wandb=use_wandb)
    acc_weak = evaluate(model_weak, dl_test_weak, desc="Eval Weak Teacher")
    print(f"--> Clean Weak Teacher Accuracy: {acc_weak * 100:.2f}%\n")

    if not args.sanity_check:
        model_weak.save_pretrained(f"{SAVE_DIR}/weak_teacher", safe_serialization=False)
    
    # ------------------------------------------
    # Phase 2: Soft Label Generation
    # ------------------------------------------
    model_weak.eval()
    weak_soft_labels = []
    print("Generating weak soft-probabilities...")
    with torch.no_grad():
        for batch in tqdm(dl_strong_inf, desc="Inference Pass"):
            input_ids = batch['input_ids'].to(model_weak.device)
            attention_mask = batch['attention_mask'].to(model_weak.device)
            outputs = model_weak(input_ids=input_ids, attention_mask=attention_mask)
            seq_lengths = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            
            # Apply temperature scaling
            scaled_logits = logits / args.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Optionally convert to hard targets
            if args.hard_labels:
                hard_label_idx = torch.argmax(probs, dim=-1)
                probs = F.one_hot(hard_label_idx, num_classes=2).float()
                
            weak_soft_labels.extend(probs.cpu().tolist())

    ds_strong_train = ds_strong_train.add_column("soft_label", weak_soft_labels)
    ds_strong_train.set_format("torch", columns=["input_ids", "attention_mask", "label_int", "soft_label"])
    
    gen_strong = torch.Generator().manual_seed(args.seed)
    dl_strong_train = DataLoader(ds_strong_train, batch_size=STRONG_BATCH_SIZE, shuffle=True, collate_fn=collator_strong, generator=gen_strong)
    
    # Explicitly tear down teacher to guarantee empty VRAM
    del model_weak
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------
    # Phase 3: Strong Student (Weak Labels + Aux)
    # ------------------------------------------
    set_seed(args.seed)   # re-seed so the student head init is reproducible
    model_student = load_classifier(args.strong_model, tokenizer_strong)
    print("\nTraining 7B Student WITH Auxiliary Confidence Loss...")
    train_model(model_student, dl_strong_train, desc="Training Student (W2S + Aux)", use_soft_labels=True, is_student=True, log_wandb=use_wandb)
    acc_w2s = evaluate(model_student, dl_test_strong, desc="Eval Student")
    print(f"--> Weak-to-Strong Accuracy: {acc_w2s * 100:.2f}%\n")

    if not args.sanity_check:
        model_student.save_pretrained(f"{SAVE_DIR}/strong_student_aux", safe_serialization=False)
        
    del model_student
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------
    # Phase 4: Max Capability Ceiling (GT Only)
    # ------------------------------------------
    set_seed(args.seed)   # re-seed so the ceiling head init is reproducible
    model_ceiling = load_classifier(args.strong_model, tokenizer_strong)
    print("\nTraining 7B Ceiling on GROUND TRUTH...")
    train_model(model_ceiling, dl_strong_train, desc="Training Ceiling", use_soft_labels=False, is_student=False, log_wandb=use_wandb)
    acc_ceil = evaluate(model_ceiling, dl_test_strong, desc="Eval Ceiling")
    print(f"--> Clean Ceiling Accuracy: {acc_ceil * 100:.2f}%\n")

    if not args.sanity_check:
        model_ceiling.save_pretrained(f"{SAVE_DIR}/ceiling_model", safe_serialization=False)
        
    del model_ceiling
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------
    # Phase 5: Latent Probing (Linear Probing Elicitation Test)
    # ------------------------------------------
    acc_probe = None
    if args.linear_probe:
        print("\n" + "-"*45)
        print("PHASE 5: LATENT PROBING (LINEAR PROBING)")
        print("-"*45)
        set_seed(args.seed)   # re-seed so probe head init is reproducible
        model_probe = load_classifier(args.strong_model, tokenizer_strong)
        train_model(model_probe, dl_strong_train, desc="Training Linear Probe", use_soft_labels=False, is_student=False, freeze_backbone=True, log_wandb=use_wandb)
        acc_probe = evaluate(model_probe, dl_test_strong, desc="Eval Linear Probe")
        print(f"--> Linear Probe Accuracy: {acc_probe * 100:.2f}%\n")
        
        del model_probe
        torch.cuda.empty_cache()

    # ------------------------------------------
    # Final Metric Calculations
    # ------------------------------------------
    print("\n" + "="*45)
    print("FINAL VALIDATED RESULTS (AUXILIARY ENGINE)")
    print("="*45)
    print(f"Weak Teacher:             {acc_weak * 100:.2f}%")
    print(f"Strong Student (Aux):     {acc_w2s * 100:.2f}%")
    print(f"Ceiling Model (GT):       {acc_ceil * 100:.2f}%")
    if acc_probe is not None:
        print(f"Linear Probe (GT):        {acc_probe * 100:.2f}%")
    print("-" * 45)
    
    gap = acc_ceil - acc_weak
    recovered = acc_w2s - acc_weak

    pgr = None
    if gap > 0:
        pgr = (recovered / gap) * 100
        print(f"Performance Gap Recovered (PGR): {pgr:.2f}%")
    else:
        print("PGR Calculation Paused: Ceiling didn't beat Weak Teacher baseline.")
    print("="*45)

    if use_wandb:
        summary = {
            "acc_weak": acc_weak,
            "acc_w2s": acc_w2s,
            "acc_ceiling": acc_ceil,
        }
        if acc_probe is not None:
            summary["acc_probe"] = acc_probe
        if pgr is not None:
            summary["pgr"] = pgr
        wandb.log(summary)
        wandb.summary.update(summary)
        wandb.finish()