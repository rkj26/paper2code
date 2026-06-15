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
import random

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

# ==========================================
# Configuration & Setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEAK_MODEL_ID = "Qwen/Qwen2.5-1.5B"
STRONG_MODEL_ID = "Qwen/Qwen2.5-7B"

# A100 Optimized Batch Sizes
WEAK_BATCH_SIZE = 32
STRONG_BATCH_SIZE = 4        # REDUCED from 16 to save VRAM
GRAD_ACCUM_STEPS = 4         # 4 physical batches * 4 steps = 16 effective batch size

# Training hyperparameters
LR = 1e-5
MAX_GRAD_NORM = 1.0

# Weights & Biases
WANDB_PROJECT = "weak-to-strong-generalization"

# Reproducibility
SEED = 42

SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed=SEED):
    """Seed all RNGs (Python, NumPy, torch CPU+CUDA) for reproducible runs.

    Note: the data splits are fixed index slices and already deterministic; this
    seeds the stochastic parts — classifier-head init and DataLoader shuffling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# Core Functions
# ==========================================
def prepare_data(tokenizer, is_sanity_check=False, seed=SEED):
    print("Loading and tokenizing dataset...")
    dataset = load_dataset("google/boolq")
    
    def format_and_label(example):
        prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}?\nAnswer:"
        return {"prompt": prompt, "label_int": 1 if example['answer'] else 0}
        
    dataset = dataset.map(format_and_label)
    
    def tokenize_func(example):
        return tokenizer(example["prompt"], truncation=True, max_length=512)
        
    tokenized = dataset.map(tokenize_func, batched=True, remove_columns=["passage", "question", "answer", "prompt"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label_int"])
    
    if is_sanity_check:
        print("SANITY CHECK MODE: Using microscopic datasets.")
        ds_weak = tokenized['train'].select(range(0, 32))
        ds_strong = tokenized['train'].select(range(32, 64))
        ds_test = tokenized['validation'].select(range(0, 32))
    else:
        ds_weak = tokenized['train'].select(range(0, 4000))
        ds_strong = tokenized['train'].select(range(4000, 8000))
        ds_test = tokenized['validation']
        
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    gen = torch.Generator().manual_seed(seed)   # reproducible shuffle order
    dl_weak = DataLoader(ds_weak, batch_size=WEAK_BATCH_SIZE, shuffle=True, collate_fn=collator, generator=gen)
    dl_strong_inf = DataLoader(ds_strong, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator)
    dl_test = DataLoader(ds_test, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator)

    return dl_weak, dl_strong_inf, dl_test, ds_strong, collator

def load_classifier(model_id, tokenizer):
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    # --- ADD THIS LINE ---
    model.gradient_checkpointing_enable() 
    # ---------------------
    
    YES_ID = tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO_ID = tokenizer.encode("No", add_special_tokens=False)[0]
    
    hidden_size = model.config.hidden_size
    orig_head = model.lm_head
    
    with torch.no_grad():
        yes_weights = orig_head.weight[YES_ID].clone()
        no_weights = orig_head.weight[NO_ID].clone()
        
    classifier = nn.Linear(hidden_size, 2, bias=False, dtype=torch.bfloat16).to(model.device)
    with torch.no_grad():
        classifier.weight[0] = no_weights
        classifier.weight[1] = yes_weights
        
    model.lm_head = classifier
    model.config.vocab_size = 2
    # We replaced the (tied) lm_head with a fresh 2-class head; break the tie so
    # save_pretrained doesn't choke on shared-tensor serialization.
    model.config.tie_word_embeddings = False
    return model

def train_model(model, dataloader, desc, use_soft_labels=False, log_wandb=False):
    model.train()
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    num_steps = len(dataloader)
    progress = tqdm(dataloader, desc=desc)
    optimizer.zero_grad()

    for step, batch in enumerate(progress):

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        targets = batch['soft_label'].to(model.device) if use_soft_labels else batch['label_int'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]

        loss = criterion(logits, targets)

        # Scale for gradient accumulation, then accumulate.
        (loss / GRAD_ACCUM_STEPS).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == num_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()   # <-- the bug fix: clear grads every optimizer step

        progress.set_postfix({'loss': f"{loss.item():.4f}"})
        if log_wandb:
            wandb.log({f"{desc}/loss": loss.item()})

def evaluate(model, dataloader, desc):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            targets = batch['label_int'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            seq_lengths = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            
            predictions = torch.argmax(logits, dim=-1)
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
    if wandb is None:
        print("wandb not installed (`pip install wandb`) -> logging disabled.")
        return False
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set -> running W&B in OFFLINE mode "
              "(sync later with `wandb sync`). Pass --no-wandb to disable.")
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=WANDB_PROJECT, name=name, config=config)
    return True


# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run quickly on tiny data to test for bugs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    use_wandb = setup_wandb(
        enabled=not args.no_wandb,
        name="naive-w2s" + ("-sanity" if args.sanity_check else ""),
        config={
            "method": "naive",
            "weak_model": WEAK_MODEL_ID,
            "strong_model": STRONG_MODEL_ID,
            "weak_batch_size": WEAK_BATCH_SIZE,
            "strong_batch_size": STRONG_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "lr": LR,
            "max_grad_norm": MAX_GRAD_NORM,
            "sanity_check": args.sanity_check,
        },
    )

    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(WEAK_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # last-token readout assumes right padding
    dl_weak, dl_strong_inf, dl_test, ds_strong, collator = prepare_data(tokenizer, is_sanity_check=args.sanity_check, seed=args.seed)

    # 2. Phase 1: Weak Teacher
    model_weak = load_classifier(WEAK_MODEL_ID, tokenizer)
    train_model(model_weak, dl_weak, desc="Training Weak Teacher", log_wandb=use_wandb)
    acc_weak = evaluate(model_weak, dl_test, desc="Eval Weak Teacher")
    print(f"--> Weak Teacher Accuracy: {acc_weak:.4f}\n")

    if not args.sanity_check:
        model_weak.save_pretrained(f"{SAVE_DIR}/weak_teacher", safe_serialization=False)
    
    # 3. Phase 2: Generate Weak Labels
    model_weak.eval()
    weak_soft_labels = []
    print("Generating weak labels...")
    with torch.no_grad():
        for batch in tqdm(dl_strong_inf, desc="Inferencing"):
            input_ids = batch['input_ids'].to(model_weak.device)
            attention_mask = batch['attention_mask'].to(model_weak.device)
            outputs = model_weak(input_ids=input_ids, attention_mask=attention_mask)
            seq_lengths = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            probs = F.softmax(logits, dim=-1)
            weak_soft_labels.extend(probs.cpu().tolist())

    del model_weak
    torch.cuda.empty_cache()
    gc.collect()
    
    ds_strong = ds_strong.add_column("soft_label", weak_soft_labels)
    ds_strong.set_format("torch", columns=["input_ids", "attention_mask", "label_int", "soft_label"])
    gen_strong = torch.Generator().manual_seed(args.seed)
    dl_strong_train = DataLoader(ds_strong, batch_size=STRONG_BATCH_SIZE, shuffle=True, collate_fn=collator, generator=gen_strong)



    # 4. Phase 3: Strong Student (Weak-to-Strong)
    set_seed(args.seed)   # re-seed so the student head init is reproducible
    model_student = load_classifier(STRONG_MODEL_ID, tokenizer)
    train_model(model_student, dl_strong_train, desc="Training Student (W2S)", use_soft_labels=True, log_wandb=use_wandb)
    acc_w2s = evaluate(model_student, dl_test, desc="Eval Student")
    print(f"--> Weak-to-Strong Accuracy: {acc_w2s:.4f}\n")

    if not args.sanity_check:
        model_student.save_pretrained(f"{SAVE_DIR}/strong_student", safe_serialization=False)
        
    del model_student
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Phase 4: Max Capability Ceiling
    set_seed(args.seed)   # re-seed so the ceiling head init is reproducible
    model_ceiling = load_classifier(STRONG_MODEL_ID, tokenizer)
    train_model(model_ceiling, dl_strong_train, desc="Training Ceiling", use_soft_labels=False, log_wandb=use_wandb)
    acc_ceil = evaluate(model_ceiling, dl_test, desc="Eval Ceiling")
    print(f"--> Ceiling Accuracy: {acc_ceil:.4f}\n")

    if not args.sanity_check:
        model_ceiling.save_pretrained(f"{SAVE_DIR}/ceiling_model", safe_serialization=False)
        
    del model_ceiling
    torch.cuda.empty_cache()

    # 6. Final Results (PGR)
    print("\n" + "="*40)
    print("FINAL EXPERIMENT RESULTS")
    print("="*40)
    print(f"Weak Teacher (1.5B):       {acc_weak * 100:.2f}%")
    print(f"Strong Student (7B + W2S): {acc_w2s * 100:.2f}%")
    print(f"Ceiling Model (7B + GT):   {acc_ceil * 100:.2f}%")
    print("-" * 40)
    
    gap = acc_ceil - acc_weak
    recovered = acc_w2s - acc_weak

    pgr = None
    if gap > 0:
        pgr = (recovered / gap) * 100
        print(f"Performance Gap Recovered (PGR): {pgr:.2f}%")
    else:
        print("PGR Error: Ceiling did not outperform Weak Teacher.")
    print("="*40)

    if use_wandb:
        summary = {
            "acc_weak": acc_weak,
            "acc_w2s": acc_w2s,
            "acc_ceiling": acc_ceil,
        }
        if pgr is not None:
            summary["pgr"] = pgr
        wandb.log(summary)
        wandb.summary.update(summary)
        wandb.finish()