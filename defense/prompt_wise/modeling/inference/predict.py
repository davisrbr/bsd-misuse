#!/usr/bin/env python3
"""
inference_with_roc.py

Extended script: runs inference, writes CSV, computes accuracy, and generates an ROC curve plot.

Usage:
    python inference_with_roc.py --model_dir ... --platt_path ... --safe_json ... --unsafe_json ... \
        --pi_train ... --pi_star ... --batch_size ... --max_length ... --output_csv predictions.csv

Results:
  - predictions.csv: CSV file with columns query,true_label,pred_label,logit_safe,logit_unsafe,p_train,p_star,question_id
  - roc_curve.png: ROC curve plot saved to working directory
"""
import argparse
import json
import math
import csv

import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def load_and_flatten(path: str, label: int) -> pd.DataFrame:
    """
    Load a JSON list of entries with 'decompositions',
    flatten into one row per step with the given label.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    for entry in entries:
        for step in entry.get("decompositions", {}).values():
            records.append({"query": step, "true_label": label})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  type=str, required=True,
                        help="Path to the finetuned classification model directory")
    parser.add_argument("--platt_path", type=str, required=True,
                        help="Path to saved Platt-scaler (.joblib)")
    parser.add_argument("--safe_json",   type=str, required=True,
                        help="Safe decomps JSON (test set)")
    parser.add_argument("--unsafe_json", type=str, required=True,
                        help="Unsafe decomps JSON (test set)")
    parser.add_argument("--pi_train",    type=float, required=True,
                        help="Unsafe prior in training data, e.g. 32/832")
    parser.add_argument("--pi_star",     type=float, required=True,
                        help="Desired deployment unsafe prior, e.g. 1/1000")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--max_length",  type=int, default=256)
    parser.add_argument("--output_csv",  type=str, default="predictions.csv")
    args = parser.parse_args()

    # 1) Load test data
    df_safe   = load_and_flatten(args.safe_json,   label=0)
    df_unsafe = load_and_flatten(args.unsafe_json, label=1)
    df = pd.concat([df_safe, df_unsafe], ignore_index=True)

    # 2) Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 3) Load Platt scaler
    platt = joblib.load(args.platt_path)

    # 4) Inference loop
    records = []
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i:i+args.batch_size]
        enc = tokenizer(
            batch["query"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits                  # [B,2]
            log_probs = F.log_softmax(logits, dim=-1)     # [B,2]

        logits_np = logits.cpu().numpy()
        queries = batch["query"].tolist()
        true_labels = batch["true_label"].tolist()

        for (l0, l1), true_lbl, q in zip(logits_np, true_labels, queries):
            raw_unsafe = l1
            # Platt scaling -> p_train
            p_train = platt.predict_proba([[raw_unsafe]])[0,1]
            logit_train = math.log(p_train / (1 - p_train))
            # Prior shift -> p_star
            shift = math.log((args.pi_star * (1 - args.pi_train)) /
                             (args.pi_train * (1 - args.pi_star)))
            logit_star = logit_train + shift
            p_star = 1 / (1 + math.exp(-logit_star))
            pred_label = 1 if p_star >= 0.5 else 0

            records.append({
                "query": q,
                "true_label": true_lbl,
                "pred_label": pred_label,
                "logit_safe": f"{l0:.6f}",
                "logit_unsafe": f"{l1:.6f}",
                "p_train": f"{p_train:.6f}",
                "p_star": f"{p_star:.6f}"})

    # 5) Write CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # 6) Load for metrics
    df_out = pd.DataFrame(records)
    y_true = df_out["true_label"].astype(int)
    y_scores = df_out["p_star"].astype(float)

    # 7) Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 8) Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{args.output_csv}.png')
    plt.close()

    # 9) Compute accuracy
    total = len(df_out)
    correct = (df_out["true_label"] == df_out["pred_label"]).sum()
    acc = correct / total * 100 if total > 0 else 0.0

    # 10) Print results
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Predictions saved to {args.output_csv}")
    print(f"ROC curve plot saved to roc_curve_{args.output_csv}.png")


if __name__ == "__main__":
    main()
