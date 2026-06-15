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

# ==========================================
# Configuration & Setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEAK_MODEL_ID = "Qwen/Qwen2.5-1.5B"
STRONG_MODEL_ID = "Qwen/Qwen2.5-7B"

# Memory & Throughput Optimized for A100 80GB
WEAK_BATCH_SIZE = 32
STRONG_BATCH_SIZE = 4        
GRAD_ACCUM_STEPS = 4         
SAVE_DIR = "./saved_models/aux"

# OpenAI Paper Hyperparameters for Auxiliary Loss
ALPHA = 1.0  # Weight of the auxiliary confidence loss term

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# Data Processing Pipeline
# ==========================================
def prepare_data(tokenizer, is_sanity_check=False):
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
        print("SANITY CHECK MODE: Restricting data arrays to micro-slices.")
        ds_weak = tokenized['train'].select(range(0, 32))
        ds_strong = tokenized['train'].select(range(32, 64))
        ds_test = tokenized['validation'].select(range(0, 32))
    else:
        # Balanced slice split following the spirit of the paper
        ds_weak = tokenized['train'].select(range(0, 4000))
        ds_strong = tokenized['train'].select(range(4000, 8000))
        ds_test = tokenized['validation']
        
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dl_weak = DataLoader(ds_weak, batch_size=WEAK_BATCH_SIZE, shuffle=True, collate_fn=collator)
    dl_strong_inf = DataLoader(ds_strong, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator)
    dl_test = DataLoader(ds_test, batch_size=WEAK_BATCH_SIZE, shuffle=False, collate_fn=collator)
    
    return dl_weak, dl_strong_inf, dl_test, ds_strong, collator

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
        
    classifier = nn.Linear(hidden_size, 2, bias=False, dtype=torch.bfloat16).to(model.device)
    with torch.no_grad():
        classifier.weight[0] = no_weights
        classifier.weight[1] = yes_weights
        
    model.lm_head = classifier
    model.config.vocab_size = 2
    return model

# ==========================================
# Isolated Training Mechanics
# ==========================================
def train_model(model, dataloader, desc, use_soft_labels=False, is_student=False):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    
    progress = tqdm(dataloader, desc=desc)
    for step, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
        
        # Rigorous conditional check ensuring auxiliary calculations ONLY run on Student phase
        if use_soft_labels and is_student:
            # 1. Base weak imitation loss (against the teacher's probability distributions)
            targets = batch['soft_label'].to(model.device)
            loss_weak = criterion(logits, targets)
            
            # 2. Auxiliary confidence loss (forcing student to trust its own highest confidence predictions)
            probs = F.softmax(logits, dim=-1)
            confident_pseudo_labels = torch.argmax(probs, dim=-1)
            loss_conf = criterion(logits, confident_pseudo_labels)
            
            # Combined Loss Equation (OpenAI Paper Section 4.1)
            loss = loss_weak + ALPHA * loss_conf
        else:
            # Clean Ground Truth CrossEntropy for Weak Teacher and Max Ceiling
            targets = batch['label_int'].to(model.device)
            loss = criterion(logits, targets)
            
        # Scale for Gradient Accumulation
        loss = loss / GRAD_ACCUM_STEPS 
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
        progress.set_postfix({'loss': f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})

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
            targets = batch['label_int'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            seq_lengths = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(input_ids.size(0)), seq_lengths]
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / total

# ==========================================
# Main Orchestration Loop
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(WEAK_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    dl_weak, dl_strong_inf, dl_test, ds_strong, collator = prepare_data(tokenizer, is_sanity_check=args.sanity_check)
    
    # ------------------------------------------
    # Phase 1: Train Weak Teacher (GT Only)
    # ------------------------------------------
    model_weak = load_classifier(WEAK_MODEL_ID, tokenizer)
    train_model(model_weak, dl_weak, desc="Training Weak Teacher", use_soft_labels=False, is_student=False)
    acc_weak = evaluate(model_weak, dl_test, desc="Eval Weak Teacher")
    print(f"--> Clean Weak Teacher Accuracy: {acc_weak * 100:.2f}%\n")
    
    if not args.sanity_check:
        model_weak.save_pretrained(f"{SAVE_DIR}/weak_teacher")
    
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
            probs = F.softmax(logits, dim=-1)
            weak_soft_labels.extend(probs.cpu().tolist())

    ds_strong = ds_strong.add_column("soft_label", weak_soft_labels)
    ds_strong.set_format("torch", columns=["input_ids", "attention_mask", "label_int", "soft_label"])
    dl_strong_train = DataLoader(ds_strong, batch_size=STRONG_BATCH_SIZE, shuffle=True, collate_fn=collator)
    
    # Explicitly tear down teacher to guarantee empty VRAM
    del model_weak
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------
    # Phase 3: Strong Student (Weak Labels + Aux)
    # ------------------------------------------
    model_student = load_classifier(STRONG_MODEL_ID, tokenizer)
    print("\nTraining 7B Student WITH Auxiliary Confidence Loss...")
    train_model(model_student, dl_strong_train, desc="Training Student (W2S + Aux)", use_soft_labels=True, is_student=True)
    acc_w2s = evaluate(model_student, dl_test, desc="Eval Student")
    print(f"--> Weak-to-Strong Accuracy: {acc_w2s * 100:.2f}%\n")
    
    if not args.sanity_check:
        model_student.save_pretrained(f"{SAVE_DIR}/strong_student_aux")
        
    del model_student
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------
    # Phase 4: Max Capability Ceiling (GT Only)
    # ------------------------------------------
    model_ceiling = load_classifier(STRONG_MODEL_ID, tokenizer)
    print("\nTraining 7B Ceiling on GROUND TRUTH...")
    train_model(model_ceiling, dl_strong_train, desc="Training Ceiling", use_soft_labels=False, is_student=False)
    acc_ceil = evaluate(model_ceiling, dl_test, desc="Eval Ceiling")
    print(f"--> Clean Ceiling Accuracy: {acc_ceil * 100:.2f}%\n")
    
    if not args.sanity_check:
        model_ceiling.save_pretrained(f"{SAVE_DIR}/ceiling_model")
        
    del model_ceiling
    torch.cuda.empty_cache()

    # ------------------------------------------
    # Final Metric Calculations
    # ------------------------------------------
    print("\n" + "="*45)
    print("FINAL VALIDATED RESULTS (AUXILIARY ENGINE)")
    print("="*45)
    print(f"Weak Teacher (1.5B):       {acc_weak * 100:.2f}%")
    print(f"Strong Student (7B + Aux): {acc_w2s * 100:.2f}%")
    print(f"Ceiling Model (7B + GT):   {acc_ceil * 100:.2f}%")
    print("-" * 45)
    
    gap = acc_ceil - acc_weak
    recovered = acc_w2s - acc_weak
    
    if gap > 0:
        pgr = (recovered / gap) * 100
        print(f"Performance Gap Recovered (PGR): {pgr:.2f}%")
    else:
        print("PGR Calculation Paused: Ceiling didn't beat Weak Teacher baseline.")
    print("="*45)