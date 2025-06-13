#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Re-run predictions on existing prediction.csv")
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Path to existing prediction.csv file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--platt_path", type=str, 
                        help="Path to platt_scaler.joblib (if not in checkpoint dir)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-Guard-3-8B",
                        help="Base model name")
    parser.add_argument("--pi_star", type=float, default=1/1000,
                        help="Target prior for deployment")
    parser.add_argument("--pi_train", type=float, default=32/832,
                        help="Training prior")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--output_csv", type=str, default="repredictions.csv",
                        help="Output CSV file name")
    parser.add_argument("--use_base", action="store_true",
                        help="Use base model instead of finetuned")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Read input CSV file
    print(f"Loading CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Check required columns
    required_columns = ['query', 'label']
    for col in required_columns:
        if col not in df.columns:
            parser.error(f"Missing required column: {col}")
    
    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 3) Load model
    if args.use_base:
        print("Loading base Llama-Guard-3-8B model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=2,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
    else:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=2,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    # 4) Load Platt scaler (if using finetuned model)
    platt = None
    if not args.use_base:
        platt_path = args.platt_path
        if not platt_path:
            # Try to find from checkpoint directory
            platt_path = os.path.join(args.checkpoint_path, "platt_scaler.joblib")
            if not os.path.exists(platt_path):
                # Try to find from checkpoint's parent directory
                platt_path = os.path.join(os.path.dirname(args.checkpoint_path), "platt_scaler.joblib")
        
        if os.path.exists(platt_path):
            print(f"Loading Platt scaler from: {platt_path}")
            platt = joblib.load(platt_path)
        else:
            parser.error(f"Platt scaler not found. Please specify --platt_path")
    
    # 5) Run predictions
    print("Running predictions...")
    queries = df["query"].tolist()
    all_logits = []
    
    for i in range(0, len(queries), args.batch_size):
        batch = queries[i:i+args.batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = model(**enc).logits
            all_logits.append(logits.cpu().numpy())
        
        del enc, logits
        torch.cuda.empty_cache()
    
    logits_all = np.concatenate(all_logits, axis=0)
    
    # 6) Calculate prediction probabilities
    if args.use_base:
        # Base model: directly use softmax
        probs = F.softmax(torch.tensor(logits_all), dim=-1).numpy()
        p_star = probs[:, 1]  # unsafe probability
        p_train = p_star      # maintain consistency
    else:
        # Finetuned model: use Platt scaling and prior shift
        raw_unsafe_logits = logits_all[:, 1]
        p_train = platt.predict_proba(raw_unsafe_logits.reshape(-1, 1))[:, 1]
        
        # Prior shift
        shift = np.log((args.pi_star * (1 - args.pi_train)) / 
                      (args.pi_train * (1 - args.pi_star)))
        logit_train = np.log(p_train / (1 - p_train))
        logit_star = logit_train + shift
        p_star = 1 / (1 + np.exp(-logit_star))
    
    # New prediction labels
    pred_labels = (p_star >= 0.5).astype(int)
    
    # 7) Save new prediction results
    df_out = df.copy()
    df_out["pred_label"] = pred_labels
    df_out["logit_safe"] = logits_all[:, 0]
    df_out["logit_unsafe"] = logits_all[:, 1]
    df_out["p_train"] = p_train
    df_out["p_star"] = p_star
    
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")
    
    # 8) Calculate and print evaluation metrics
    if "label" in df.columns:
        true_labels = df["label"].astype(int).values
        
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, zero_division=0)
        rec = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n" + "="*50)
        print("CONFUSION MATRIX")
        print("="*50)
        print("                   Predicted")
        print("                 Safe    Unsafe")
        print(f"Actual  Safe   {tn:5d}    {fp:5d}")
        print(f"       Unsafe  {fn:5d}    {tp:5d}")
        print("="*50)
        
        # Calculate accuracy for each category separately
        safe_total = tn + fp
        unsafe_total = fn + tp
        safe_acc = tn / safe_total * 100 if safe_total > 0 else 0
        unsafe_acc = tp / unsafe_total * 100 if unsafe_total > 0 else 0
        
        print("\nPER-CLASS ACCURACY:")
        print(f"Safe:   {safe_acc:.2f}% ({tn}/{safe_total})")
        print(f"Unsafe: {unsafe_acc:.2f}% ({tp}/{unsafe_total})")
        print("="*50)
        
        # Additional statistical information
        print("\nADDITIONAL STATISTICS:")
        print(f"Total Safe:   {safe_total}")
        print(f"Total Unsafe: {unsafe_total}")
        print(f"Ratio:        {unsafe_total/(safe_total+unsafe_total)*100:.2f}% unsafe")
        print("="*50)
        
        # Draw Confusion Matrix heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Safe', 'Unsafe'],
                    yticklabels=['Safe', 'Unsafe'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_path = Path(args.output_csv).with_suffix('.confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"\nSaved confusion matrix plot to: {cm_path}")
        
        # ROC curve
        fpr, tpr, _ = roc_curve(true_labels, p_star)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        out_path = Path(args.output_csv).with_suffix('.roc.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved ROC curve to: {out_path}")
        
        # PR curve
        precision, recall, _ = precision_recall_curve(true_labels, p_star)
        ap = average_precision_score(true_labels, p_star)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.legend(loc="upper right")
        plt.tight_layout()
        out_path = Path(args.output_csv).with_suffix('.pr.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved PR curve to: {out_path}")

if __name__ == "__main__":
    main()