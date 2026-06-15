"""Linear probe for RCA'ing weak-to-strong (regression in) PGR.

Freezes a model's transformer body and trains a fresh logistic-regression head
on GROUND-TRUTH labels over the frozen last-token hidden states, at multiple
layers. This disentangles two failure modes:

  * probe stays high but the model's own head is worse  -> the *decision boundary*
    was corrupted by the (aux) loss; the knowledge is still in the features.
  * probe accuracy itself drops                          -> *representations*
    degraded (collapse); the loss damaged the features.

Pass any mix of checkpoints (HF repo id OR local path) as name=location:

    python linear_probe.py \
        --model pretrained=Qwen/Qwen2.5-7B \
        --model naive=rkj26/w2s-naive-student \
        --model aux=rkj26/w2s-aux-student \
        --probe-train-size 1500
"""

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

SEED = 42
WEAK_MODEL_ID = "Qwen/Qwen2.5-1.5B"   # tokenizer source (matches training)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_boolq_split(tokenizer, split, sel=None):
    ds = load_dataset("google/boolq")[split]
    if sel is not None:
        ds = ds.select(range(*sel))

    def fmt(e):
        return {"prompt": f"Passage: {e['passage']}\nQuestion: {e['question']}?\nAnswer:",
                "label_int": 1 if e["answer"] else 0}

    ds = ds.map(fmt)
    ds = ds.map(lambda e: tokenizer(e["prompt"], truncation=True, max_length=512),
                batched=True, remove_columns=["passage", "question", "answer", "prompt"])
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label_int"])
    return ds


@torch.no_grad()
def extract_features(model, dataloader, layers):
    """Return {layer: [N, H] float16} of last-token hidden states, plus labels."""
    model.eval()
    feats = {l: [] for l in layers}
    labels = []
    for batch in tqdm(dataloader, desc="Extracting"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        idx = torch.arange(input_ids.size(0))
        seq = attention_mask.sum(dim=1) - 1
        for l in layers:
            v = out.hidden_states[l][idx, seq].float().cpu().numpy().astype(np.float16)
            feats[l].append(v)
        labels.append(batch["label_int"].numpy())
    feats = {l: np.concatenate(v) for l, v in feats.items()}
    return feats, np.concatenate(labels)


def load_model(location):
    return AutoModelForCausalLM.from_pretrained(location, torch_dtype=torch.bfloat16, device_map="auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="name=location",
                        help="Checkpoint to probe (repeatable). location = HF repo id or local path.")
    parser.add_argument("--probe-train-size", type=int, default=1500)
    parser.add_argument("--probe-train-range", type=int, nargs=2, default=[8000, 9427],
                        help="BoolQ train slice for probe fitting (disjoint from weak/strong splits)")
    parser.add_argument("--layer-stride", type=int, default=2,
                        help="Probe every Nth layer (1 = all layers)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--csv", default="probe_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoints = dict(m.split("=", 1) for m in args.model)

    tokenizer = AutoTokenizer.from_pretrained(WEAK_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ds_train = load_boolq_split(tokenizer, "train", sel=tuple(args.probe_train_range))
    if args.probe_train_size and args.probe_train_size < len(ds_train):
        ds_train = ds_train.select(range(args.probe_train_size))
    ds_test = load_boolq_split(tokenizer, "validation")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    results = {}   # name -> {layer: test_acc}
    for name, location in checkpoints.items():
        print(f"\n=== Probing '{name}'  ({location}) ===")
        model = load_model(location)
        n_hidden = model.config.num_hidden_layers + 1  # +1 for the embedding layer
        layers = sorted(set(list(range(0, n_hidden, args.layer_stride)) + [n_hidden - 1]))

        f_tr, y_tr = extract_features(model, dl_train, layers)
        f_te, y_te = extract_features(model, dl_test, layers)
        del model
        torch.cuda.empty_cache()

        results[name] = {}
        for l in layers:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed),
            )
            clf.fit(f_tr[l].astype(np.float32), y_tr)
            acc = clf.score(f_te[l].astype(np.float32), y_te)
            results[name][l] = acc
            print(f"  layer {l:>3}: probe test acc = {acc * 100:.2f}%")

    # ---- Summary table ----
    all_layers = sorted({l for r in results.values() for l in r})
    header = "layer," + ",".join(results)
    print("\n" + "=" * 60)
    print("LINEAR PROBE (ground-truth) ACCURACY BY LAYER")
    print("=" * 60)
    print(header.replace(",", "\t"))
    rows = [header]
    for l in all_layers:
        cells = [f"{results[n].get(l, float('nan')) * 100:.2f}" for n in results]
        print(f"{l}\t" + "\t".join(cells))
        rows.append(f"{l}," + ",".join(cells))
    with open(args.csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nSaved -> {args.csv}")
