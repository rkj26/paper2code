import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup
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
        print("SANITY CHECK MODE: Using microscopic datasets.")
        ds_weak_train = tokenized_weak['train'].select(range(0, 32))
        ds_strong_inf = tokenized_weak['train'].select(range(32, 64))
        ds_strong_train = tokenized_strong['train'].select(range(32, 64))
        ds_test_weak = tokenized_weak['validation'].select(range(0, 32))
        ds_test_strong = tokenized_strong['validation'].select(range(0, 32))
    else:
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

def load_classifier(model_id, tokenizer):
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
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

def train_model(model, dataloader, desc, use_soft_labels=False, freeze_backbone=False, log_wandb=False):
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

    num_batches = len(dataloader)
    total_training_steps = (num_batches + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    num_warmup_steps = int(0.1 * total_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    progress = tqdm(dataloader, desc=desc)
    optimizer.zero_grad()

    for step, batch in enumerate(progress):

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]

        # Dynamically move targets to the logits device to support multi-GPU setups
        targets = batch['soft_label'].to(logits.device) if use_soft_labels else batch['label_int'].to(logits.device)

        loss = criterion(logits, targets)

        # Scale for gradient accumulation, then accumulate.
        (loss / GRAD_ACCUM_STEPS).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress.set_postfix({'loss': f"{loss.item():.4f}"})
        if log_wandb:
            wandb.log({f"{desc}/loss": loss.item(), f"{desc}/lr": scheduler.get_last_lr()[0]})

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
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", type=str, default="Qwen/Qwen3.5-2B-Instruct", help="Weak supervisor model ID or local path")
    parser.add_argument("--strong-model", type=str, default="Qwen/Qwen3.5-9B-Instruct", help="Strong student model ID or local path")
    parser.add_argument("--hard-labels", action="store_true", help="Supervise student with hard thresholded labels instead of soft probabilities")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling factor for soft label logits")
    parser.add_argument("--linear-probe", action="store_true", help="Run latent representation linear probing on frozen strong backbone")
    parser.add_argument("--sanity-check", action="store_true", help="Run quickly on tiny data to test for bugs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    use_wandb = setup_wandb(
        enabled=not args.no_wandb,
        name="naive-w2s" + ("-sanity" if args.sanity_check else ""),
        config={
            "method": "naive",
            "weak_model": args.weak_model,
            "strong_model": args.strong_model,
            "weak_batch_size": WEAK_BATCH_SIZE,
            "strong_batch_size": STRONG_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "lr": LR,
            "max_grad_norm": MAX_GRAD_NORM,
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

    # 2. Phase 1: Weak Teacher
    set_seed(args.seed)
    model_weak = load_classifier(args.weak_model, tokenizer_weak)
    train_model(model_weak, dl_weak, desc="Training Weak Teacher", log_wandb=use_wandb)
    acc_weak = evaluate(model_weak, dl_test_weak, desc="Eval Weak Teacher")
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
            
            # Apply temperature scaling
            scaled_logits = logits / args.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Optionally convert to hard targets
            if args.hard_labels:
                hard_label_idx = torch.argmax(probs, dim=-1)
                probs = F.one_hot(hard_label_idx, num_classes=2).float()
                
            weak_soft_labels.extend(probs.cpu().tolist())

    del model_weak
    torch.cuda.empty_cache()
    gc.collect()
    
    ds_strong_train = ds_strong_train.add_column("soft_label", weak_soft_labels)
    ds_strong_train.set_format("torch", columns=["input_ids", "attention_mask", "label_int", "soft_label"])
    
    gen_strong = torch.Generator().manual_seed(args.seed)
    dl_strong_train = DataLoader(ds_strong_train, batch_size=STRONG_BATCH_SIZE, shuffle=True, collate_fn=collator_strong, generator=gen_strong)

    # 4. Phase 3: Strong Student (Weak-to-Strong)
    set_seed(args.seed)   # re-seed so the student head init is reproducible
    model_student = load_classifier(args.strong_model, tokenizer_strong)
    train_model(model_student, dl_strong_train, desc="Training Student (W2S)", use_soft_labels=True, log_wandb=use_wandb)
    acc_w2s = evaluate(model_student, dl_test_strong, desc="Eval Student")
    print(f"--> Weak-to-Strong Accuracy: {acc_w2s:.4f}\n")

    if not args.sanity_check:
        model_student.save_pretrained(f"{SAVE_DIR}/strong_student", safe_serialization=False)
        
    del model_student
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Phase 4: Max Capability Ceiling
    set_seed(args.seed)   # re-seed so the ceiling head init is reproducible
    model_ceiling = load_classifier(args.strong_model, tokenizer_strong)
    train_model(model_ceiling, dl_strong_train, desc="Training Ceiling", use_soft_labels=False, log_wandb=use_wandb)
    acc_ceil = evaluate(model_ceiling, dl_test_strong, desc="Eval Ceiling")
    print(f"--> Ceiling Accuracy: {acc_ceil:.4f}\n")

    if not args.sanity_check:
        model_ceiling.save_pretrained(f"{SAVE_DIR}/ceiling_model", safe_serialization=False)
        
    del model_ceiling
    torch.cuda.empty_cache()
    gc.collect()

    # 6. Phase 5: Latent Probing (Linear Probing Elicitation Test)
    acc_probe = None
    if args.linear_probe:
        print("\n" + "-"*40)
        print("PHASE 5: LATENT PROBING (LINEAR PROBING)")
        print("-"*40)
        set_seed(args.seed)   # re-seed so probe head init is reproducible
        model_probe = load_classifier(args.strong_model, tokenizer_strong)
        train_model(model_probe, dl_strong_train, desc="Training Linear Probe", use_soft_labels=False, freeze_backbone=True, log_wandb=use_wandb)
        acc_probe = evaluate(model_probe, dl_test_strong, desc="Eval Linear Probe")
        print(f"--> Linear Probe Accuracy: {acc_probe:.4f}\n")
        
        del model_probe
        torch.cuda.empty_cache()

    # 7. Final Results (PGR)
    print("\n" + "="*40)
    print("FINAL EXPERIMENT RESULTS")
    print("="*40)
    print(f"Weak Teacher:             {acc_weak * 100:.2f}%")
    print(f"Strong Student (W2S):     {acc_w2s * 100:.2f}%")
    print(f"Ceiling Model (GT):       {acc_ceil * 100:.2f}%")
    if acc_probe is not None:
        print(f"Linear Probe (GT):        {acc_probe * 100:.2f}%")
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
        if acc_probe is not None:
            summary["acc_probe"] = acc_probe
        if pgr is not None:
            summary["pgr"] = pgr
        wandb.log(summary)
        wandb.summary.update(summary)
        wandb.finish()