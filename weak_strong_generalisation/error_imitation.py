"""Weak-error imitation metric — does the student copy the teacher's mistakes?

The whole point of the auxiliary confidence loss is to make the strong student
override the weak supervisor's errors instead of imitating them. This script
measures that directly on the test set, for one or more students, against the
weak teacher:

  imitation_rate     : among examples the WEAK teacher gets wrong, the fraction
                       where the student copies the weak (wrong) answer. LOWER is
                       better — a high value is the "over-imitation" failure mode.
  acc_on_weak_wrong  : student accuracy on the weak-wrong subset (= 1 - imitation
                       for binary). This is the knowledge the student adds.
  acc_on_weak_right  : student accuracy where the weak teacher was already right.

If the aux student's imitation_rate is HIGHER than the naive student's, the aux
loss is reinforcing weak errors rather than correcting them — a clean RCA signal.

    python error_imitation.py \
        --weak rkj26/w2s-weak-teacher \
        --student naive=rkj26/w2s-naive-student \
        --student aux=rkj26/w2s-aux-student
"""

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

SEED = 42
WEAK_MODEL_ID = "Qwen/Qwen2.5-1.5B"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_boolq_val(tokenizer):
    ds = load_dataset("google/boolq")["validation"]

    def fmt(e):
        return {"prompt": f"Passage: {e['passage']}\nQuestion: {e['question']}?\nAnswer:",
                "label_int": 1 if e["answer"] else 0}

    ds = ds.map(fmt)
    ds = ds.map(lambda e: tokenizer(e["prompt"], truncation=True, max_length=512),
                batched=True, remove_columns=["passage", "question", "answer", "prompt"])
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label_int"])
    return ds


@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        idx = torch.arange(input_ids.size(0))
        seq = attention_mask.sum(dim=1) - 1
        logits = out.logits[idx, seq]
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["label_int"].numpy())
    return np.concatenate(preds), np.concatenate(labels)


def imitation_metrics(weak_pred, student_pred, y):
    weak_wrong = weak_pred != y
    weak_right = ~weak_wrong
    m = {
        "student_acc": float((student_pred == y).mean()),
        "weak_acc": float((weak_pred == y).mean()),
        "frac_weak_wrong": float(weak_wrong.mean()),
    }
    if weak_wrong.any():
        m["imitation_rate"] = float((student_pred[weak_wrong] == weak_pred[weak_wrong]).mean())
        m["acc_on_weak_wrong"] = float((student_pred[weak_wrong] == y[weak_wrong]).mean())
    if weak_right.any():
        m["acc_on_weak_right"] = float((student_pred[weak_right] == y[weak_right]).mean())
    return m


def load_model(location):
    return AutoModelForCausalLM.from_pretrained(location, torch_dtype=torch.bfloat16, device_map="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak", required=True, help="weak teacher checkpoint (HF id or local path)")
    parser.add_argument("--student", action="append", required=True, metavar="name=location",
                        help="student checkpoint (repeatable)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    students = dict(s.split("=", 1) for s in args.student)

    tokenizer = AutoTokenizer.from_pretrained(WEAK_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dl = DataLoader(load_boolq_val(tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    print(f"=== Weak teacher ({args.weak}) ===")
    weak_model = load_model(args.weak)
    weak_pred, y = predict(weak_model, dl)
    del weak_model
    torch.cuda.empty_cache()

    rows = []
    for name, location in students.items():
        print(f"\n=== Student '{name}' ({location}) ===")
        model = load_model(location)
        student_pred, y2 = predict(model, dl)
        del model
        torch.cuda.empty_cache()
        assert np.array_equal(y, y2), "label order mismatch (shuffle leaked in?)"
        rows.append((name, imitation_metrics(weak_pred, student_pred, y)))

    cols = ["student_acc", "weak_acc", "imitation_rate", "acc_on_weak_wrong", "acc_on_weak_right"]
    print("\n" + "=" * 78)
    print("WEAK-ERROR IMITATION  (lower imitation_rate = better correction)")
    print("=" * 78)
    print("student".ljust(12) + "".join(c.replace("_", " ").rjust(20) for c in cols))
    for name, m in rows:
        print(name.ljust(12) + "".join(
            (f"{m[c] * 100:.2f}%" if c in m else "-").rjust(20) for c in cols))
