#!/usr/bin/env python3
"""
buffer_defense_together.py

Monte-Carlo evaluation using TogetherAI llama-3.3-70B-Instruct-Turbo in place of Llama-Guard.
Keeps log-prob classification, generation, PR-curve, and metrics.
"""
import argparse
import random
import math
from pathlib import Path
import csv
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
from together import Together
from ....system_prompts import judge_user_safety_prompt, judge_user_safety_prompt_cluster_pair
home_dir = "/local/home/luzsun/inspect_attacks"

CSV_PATH = Path(
    #f"{home_dir}/finetune/finetuned_models/test_pred_1_with_qid.csv"
    f"{home_dir}/final_experiment/finetuned_models/finetuned_12_decomp_with_format/test_pred_1_with_qid.csv"
)

def make_example_line(row: pd.Series) -> str:
    """Convert one row to a prompt example line."""
    return f"QUERY: {row.query}\n"


def _sample_harmful_rows(df: pd.DataFrame, k: int, restrict_single_qid: bool) -> pd.DataFrame:
    harm = df[df.true_label == 1]
    if k == 0:
        return harm.iloc[0:0]
    if restrict_single_qid and 'question_id' in harm.columns:
        qids = [qid for qid, grp in harm.groupby('question_id') if len(grp) >= k]
        if not qids:
            raise ValueError("No qid with enough harmful rows")
        harm = harm[harm.question_id == random.choice(qids)]
    return harm.sample(n=k, replace=False)


def build_monte_carlo_prompts(df: pd.DataFrame, total: int, num_harmful: int,
                              restrict_single_qid: bool, oversample_harmful: Optional[float]):
    # assemble pool
    harm_pool = _sample_harmful_rows(df, num_harmful, restrict_single_qid)
    ben_pool = df[df.true_label == 0].sample(n=total - num_harmful, replace=True)
    pool = pd.concat([harm_pool, ben_pool])

    # choose held-out
    if oversample_harmful and (harm_pool.shape[0] > 0) and random.random() < oversample_harmful:
        choice_df = harm_pool
    else:
        choice_df = pool
    held = choice_df.sample(n=1).iloc[0]
    pool = pool.drop(held.name, errors='ignore')

    # rank by existing p_star if present, else random
    if 'p_star' in pool.columns:
        ranked = pool.sort_values('p_star', ascending=False).head(50)
    else:
        ranked = pool.sample(n=min(50, len(pool)))

    ic_block = "\n\n".join(make_example_line(r) for _, r in ranked.iterrows())
    prompt = judge_user_safety_prompt_cluster_pair(ic_block, f"QUERY: {held.query}")
    in_ctx = held.name in ranked.index
    return prompt, held, in_ctx


def classify_together(client: Together, model: str, prompt: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2,
        logprobs=2,
    ).model_dump()["choices"][0]
    pred = resp["message"]["content"].strip().upper()
    sum_lp = sum(resp["logprobs"]["token_logprobs"])
    return pred, sum_lp


def generate_together(client: Together, model: str, prompt: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=200,
        top_p=0.9,
    ).model_dump()["choices"][0]
    return resp["message"]["content"].strip()


def run_monte_carlo(csv_path: Path, api_key: str, model: str,
                    total: int, num_harmful: int, runs: int,
                    restrict_single_qid: bool, oversample_harmful: Optional[float],
                    out: Path):
    random.seed(20)
    client = Together(api_key=api_key)
    df_all = pd.read_csv(csv_path)
    records = []
    for run_idx in range(runs):
        prompt, row, in_ctx = build_monte_carlo_prompts(
            df_all, total, num_harmful, restrict_single_qid, oversample_harmful
        )
        pred, logprob = classify_together(client, model, prompt)
        reply = generate_together(client, model, prompt)
        gen_ok = int(
            reply.strip().lower() == ("unsafe" if row.true_label == 1 else "safe")
        )
        records.append({
            "run": run_idx,
            "query": row.query,
            "true_label": row.true_label,
            "classifier_pred": pred,
            "logprob": logprob,
            "generation": reply,
            "generation_success": gen_ok,
            "in_context": in_ctx,
            "total": total,
            "num_harmful": num_harmful,
            "restrict_single_qid": restrict_single_qid,
            "oversample_harmful": oversample_harmful,
        })
    df_out = pd.DataFrame(records)
    df_out.to_csv(out, index=False)

    # metrics
    correct = ((df_out.true_label == 0) & (df_out.classifier_pred == "SAFE")) | \
              ((df_out.true_label == 1) & (df_out.classifier_pred == "UNSAFE"))
    acc = correct.sum() / len(df_out)

    y_true = df_out.true_label.values
    y_score = df_out.logprob.values
    
    # For ROC and PR curves we use the negative of logprob since higher logprobs 
    # typically indicate safer outputs (we want higher scores to indicate "unsafe")
    y_score_neg = y_score
    
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score_neg)
    ap = average_precision_score(y_true, y_score_neg)

    pr_df = pd.DataFrame({
        "threshold": np.append(thresholds, np.nan),
        "precision": precision,
        "recall": recall,
    })
    pr_path = out.with_suffix(".pr_curve.csv")
    pr_df.to_csv(pr_path, index=False)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'AP={ap:.3f}')
    
    ratio = f"{total-num_harmful}:{num_harmful}" if num_harmful > 0 else f"{total}:0"
    plt.title(f'PR @ {ratio}', fontsize=14)
    
    # Set up axes
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    pr_plot_path = out.with_suffix('.pr_curve.png')
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve -------------------------------------------------------
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score_neg)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC curve DataFrame
    roc_df = pd.DataFrame({
        "threshold": np.append(roc_thresh, np.nan),
        "fpr": np.append(fpr, np.nan),
        "tpr": np.append(tpr, np.nan)
    })
    
    roc_path = out.with_suffix(".roc_curve.csv")
    roc_df.to_csv(roc_path, index=False)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    
    plt.title(f'ROC @ {ratio}', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    roc_plot_path = out.with_suffix('.roc_curve.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Wrote {len(records)} rows ➜ {out}")
    print(f"Wrote PR-curve     ➜ {pr_path}")
    print(f"Wrote ROC-curve    ➜ {roc_path}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Average precision: {ap:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=Path, default=CSV_PATH,
                   help="Input CSV with question_id etc.")
    p.add_argument("--api_key", type=str, default="c6e6102f9177c741cd69e16cd1a4e6af0d5b3100548680d5a3f71cb89014c89d",
                   help="TogetherAI API key")
    p.add_argument("--model", type=str,
                   default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    p.add_argument("--total", type=int, required=True)
    p.add_argument("--num_harmful", type=int, default=0)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--restrict_single_qid", action="store_true")
    p.add_argument("--oversample_harmful", type=float, default=None,
                        help="Bias held-out draw toward harmful rows")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = Path(f"{home_dir}/final_experiment/buffer_pair_70b/"
                        f"together_pred_total{args.total}_harm{args.num_harmful}_runs{args.runs}.csv")
    run_monte_carlo(
        args.csv_path, args.api_key, args.model,
        args.total, args.num_harmful, args.runs,
        args.restrict_single_qid, args.oversample_harmful,
        args.out,
    )