#!/usr/bin/env python3
"""
inference_with_roc.py

Extended script: runs inference, writes CSV, computes accuracy, and generates an ROC curve plot.

Usage:
    python inference_with_roc.py --model_dir ... --platt_path ... --safe_json ... --unsafe_json ... \
        --pi_train ... --pi_star ... --batch_size ... --max_length ... --output_csv predictions.csv

    # Use original Llama-Guard instead of fine-tuned model:
    python inference_with_roc.py --use_finetuned False --safe_json ... --unsafe_json ... \
        --batch_size ... --max_length ... --output_csv predictions.csv
"""
import argparse
import json
import math
import csv
from pathlib import Path

import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def load_and_flatten(path: str, label: int, model_filter: str = None) -> pd.DataFrame:
    """
    Load data from HuggingFace dataset instead of local JSON.
    Download the BSD dataset from https://huggingface.co/datasets/BrachioLab/BSD
    """
    raise NotImplementedError(
        "Please download the BSD dataset from https://huggingface.co/datasets/BrachioLab/BSD\n"
        "This requires requesting access through the HuggingFace form.\n"
        "Update this function to load from your downloaded dataset."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_finetuned", type=str, default="True", 
                        help="Whether to use the finetuned model (True) or Llama-Guard-3-8B (False)")
    parser.add_argument("--model_dir", type=str, 
                        help="Path to the finetuned classification model directory")
    parser.add_argument("--platt_path", type=str, 
                        help="Path to saved Platt-scaler (.joblib)")
    parser.add_argument("--safe_json", type=str, required=True,
                        help="Safe-format JSON list")
    parser.add_argument("--unsafe_json", type=str, required=True,
                        help="Unsafe-format JSON list")
    parser.add_argument("--harmful_model_filter", type=str, default="openai/o1-preview-2024-09-12",
                        help="Only select data from this model from the harmful dataset")
    parser.add_argument("--pi_train", type=float, 
                        help="Unsafe prior in training data, e.g. 32/832")
    parser.add_argument("--pi_star", type=float, 
                        help="Desired deployment unsafe prior, e.g. 1/1000")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    # Determine whether to use fine-tuned model
    use_finetuned = args.use_finetuned.lower() == "true"
    
    # If not using fine-tuned model, use Llama-Guard-3-8B
    if not use_finetuned:
        args.model_dir = "meta-llama/Llama-Guard-3-8B"
        print(f"Using base Llama-Guard-3-8B model without fine-tuning")
    else:
        # Validate required arguments for fine-tuning mode
        required_args = ["model_dir", "platt_path", "pi_train", "pi_star"]
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(f"--use_finetuned=True requires arguments: {', '.join(missing_args)}")

    # 1) Load test data
    # If you have benign/harmful reversed, adjust the label values here
    # Assumption: safe_json contains benign data (label=0), unsafe_json contains harmful data (label=1)
    df_safe = load_and_flatten(args.safe_json, label=0)
    
    # Only select data from specific model from harmful dataset
    df_unsafe = load_and_flatten(args.unsafe_json, label=1, model_filter=args.harmful_model_filter)
    
    df = pd.concat([df_safe, df_unsafe], ignore_index=True)
    print(f"Combined dataset: {len(df_safe)} safe + {len(df_unsafe)} unsafe = {len(df)} total")

    # Rest of the code remains unchanged...
    
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

    # 3) Load Platt scaler (only in fine-tuning mode)
    platt = None
    if use_finetuned:
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
            logits = model(**enc).logits   # [B,2]
            log_probs = F.log_softmax(logits, dim=-1)

        logits_np = logits.cpu().numpy()
        queries = batch["query"].tolist()
        true_labels = batch["true_label"].tolist()

        # Ensure all fields are strings before adding records to records list
        for (l0, l1), true_lbl, q in zip(logits_np, true_labels, queries):
            raw_unsafe = l1
            
            # Ensure query is string type
            if q is None:
                q = ""  # Handle None values
            else:
                q = str(q)  # Ensure conversion to string
            
            if use_finetuned:
                # When using fine-tuned model: Platt scaling -> prior shift
                p_train = platt.predict_proba([[raw_unsafe]])[0,1]
                logit_train = math.log(p_train / (1 - p_train))
                shift = math.log((args.pi_star * (1 - args.pi_train)) /
                                (args.pi_train * (1 - args.pi_star)))
                logit_star = logit_train + shift
                p_star = 1 / (1 + math.exp(-logit_star))
                pred_label = 1 if p_star >= 0.5 else 0
            else:
                # When using original Llama-Guard: directly use softmax probabilities
                probs = F.softmax(torch.tensor([l0, l1]), dim=0).numpy()
                p_star = probs[1]  # Unsafe probability
                p_train = p_star   # Maintain consistency, keep output format same
                pred_label = 1 if p_star >= 0.5 else 0

            records.append({
                "query": q,
                "true_label": str(true_lbl),
                "pred_label": str(pred_label),
                "logit_safe": f"{l0:.6f}",
                "logit_unsafe": f"{l1:.6f}",
                "p_train": f"{p_train:.6f}",
                "p_star": f"{p_star:.6f}"
            })

    # 5) Write CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # 6) Compute and save ROC curve
    df_out = pd.DataFrame(records)
    y_true = df_out["true_label"].astype(int)
    y_scores = df_out["p_star"].astype(float)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{Path(args.output_csv).stem}.png")
    plt.close()

# 7) Print summary with confusion matrix
    total = len(df_out)
    correct = (df_out["true_label"] == df_out["pred_label"]).sum()
    acc = correct / total * 100 if total > 0 else 0.0
    
    # Create confusion matrix
    tp = ((df_out["true_label"] == "1") & (df_out["pred_label"] == "1")).sum()
    fp = ((df_out["true_label"] == "0") & (df_out["pred_label"] == "1")).sum()
    tn = ((df_out["true_label"] == "0") & (df_out["pred_label"] == "0")).sum()
    fn = ((df_out["true_label"] == "1") & (df_out["pred_label"] == "0")).sum()
    
    # Calculate accuracy for harmful and benign separately
    harmful_total = tp + fn
    harmful_correct = tp
    harmful_acc = harmful_correct / harmful_total * 100 if harmful_total > 0 else 0.0
    
    benign_total = tn + fp
    benign_correct = tn
    benign_acc = benign_correct / benign_total * 100 if benign_total > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX:")
    print("                 Predicted")
    print("              Benign  Harmful")
    print(f"Actual Benign   {tn:5d}    {fp:5d}")
    print(f"     Harmful    {fn:5d}    {tp:5d}")
    print("=" * 50)
    
    print("\nACCURACY SUMMARY:")
    print(f"Overall   - Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%")
    print(f"Harmful   - Total: {harmful_total}, Correct: {harmful_correct}, Accuracy: {harmful_acc:.2f}%")
    print(f"Benign    - Total: {benign_total}, Correct: {benign_correct}, Accuracy: {benign_acc:.2f}%")
    
    # Calculate other metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\nCLASSIFICATION METRICS:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("=" * 50)
    
    # Save detailed statistics to file
    stats_file = Path(args.output_csv).with_suffix('.stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Overall Accuracy: {acc:.2f}%\n")
        f.write(f"Harmful Accuracy: {harmful_acc:.2f}%\n")
        f.write(f"Benign Accuracy: {benign_acc:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TN: {tn}, FP: {fp}\n")
        f.write(f"FN: {fn}, TP: {tp}\n")
    
    print(f"\nOutput files:")
    print(f"Predictions → {args.output_csv}")
    print(f"Statistics  → {stats_file}")
    print(f"ROC plot    → roc_curve_{Path(args.output_csv).stem}.png")

if __name__ == "__main__":
    main()