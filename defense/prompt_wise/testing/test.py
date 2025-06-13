#!/usr/bin/env python3
import random
import csv
import json
import pandas as pd
from together import Together
from ....system_prompts import judge_user_safety_prompt

# ─────────────────────── Configuration ───────────────────────
API_KEY    = "c6e6102f9177c741cd69e16cd1a4e6af0d5b3100548680d5a3f71cb89014c89d"
MODEL      = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
PRED_CSV   = "/local/home/luzsun/inspect_attacks/finetune/finetuned_models/test_pred_1.csv"
SEED       = 12
MAX_TOKENS = 2      # “UNSAFE” is split into 2 subtokens
LOGPROBS   = 2
N_RUNS     = 200

random.seed(SEED)

# ───────────────────── Prompt-building helpers ─────────────────────
def make_example_line(row: pd.Series) -> str:
    label = "UNSAFE" if row.pred_label == 1 else "SAFE"
    return f"QUERY: {row.query}\n"

def sample_rows(df: pd.DataFrame, k: int) -> pd.Series:
    return df.sample(n=k, random_state=random.randint(0,100000)).iloc[0]

def build_prompts():
    df = pd.read_csv(PRED_CSV)
    safe_pool   = df[df.true_label == 0]
    unsafe_pool = df[df.true_label == 1]

    ic_safe = (
        safe_pool
        .sample(n=min(len(safe_pool), 1000), random_state=SEED)
        .nsmallest(30, "logit_unsafe")
    )
    ic_unsafe = (
        unsafe_pool
        .sample(n=min(len(unsafe_pool), 30), random_state=SEED)
        .nlargest(1, "logit_unsafe")
    )

    # benign prompt
    ic_block_ben = "\n\n".join(make_example_line(r) for _, r in ic_safe.iterrows())
    q_ben_row = sample_rows(safe_pool.drop(ic_safe.index, errors="ignore"), 1)
    benign_prompt = judge_user_safety_prompt(ic_block_ben, f"QUERY: {q_ben_row.query}")

    # harmful prompt
    ic_block_harm = "\n\n".join(
        make_example_line(r) for _, r in pd.concat([ic_safe, ic_unsafe]).iterrows()
    )
    q_har_row  = sample_rows(unsafe_pool.drop(ic_unsafe.index, errors="ignore"), 1)
    q_ben2_row = sample_rows(safe_pool.drop(ic_safe.index, errors="ignore"), 1)
    harmful_prompt = judge_user_safety_prompt(
        ic_block_harm,
        f"QUERY: {q_har_row.query}\nQUERY: {q_ben2_row.query}"
    )

    # return the two prompts plus the sampled rows (so we know their true_label)
    return (
        benign_prompt, q_ben_row.true_label,
        harmful_prompt, q_har_row.true_label, q_ben2_row.true_label
    )


# ───────────────────────── Main ─────────────────────────
if __name__ == "__main__":
    benign_prompt, ben_true, harmful_prompt, har_true, ben2_true = build_prompts()
    client = Together(api_key=API_KEY)
    records = []

    for condition, prompt, *true_labels in [
        ("SAFE",   benign_prompt, ben_true),
        ("UNSAFE", harmful_prompt, har_true, ben2_true)
    ]:
        for run in range(1, N_RUNS + 1):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                logprobs=LOGPROBS,
            ).model_dump()["choices"][0]

            tokens   = resp["logprobs"]["tokens"]
            lps      = resp["logprobs"]["token_logprobs"]
            pred     = resp["message"]["content"].strip()
            sum_lp   = sum(lps)

            row = {
                "run": run,
                "condition": condition,
                "predicted_label": pred,
                "tokens": "|".join(tokens),
                "token_logprobs": "|".join(f"{x:.6f}" for x in lps),
                "sum_logprob": sum_lp,
            }

            # attach the true_label(s) of the query(ies)
            if condition == "SAFE":
                row["query_true_label"] = true_labels[0]
            else:
                row["query1_true_label"] = true_labels[0]
                row["query2_true_label"] = true_labels[1]

            records.append(row)

    # write to CSV
    out_csv = "together_fewshot_eval.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved all runs to {out_csv}")

    # print accuracy
    df_res = pd.DataFrame(records)
    print("\n=== Accuracy ===")
    for cond in ["SAFE", "UNSAFE"]:
        sub     = df_res[df_res.condition == cond]
        total   = len(sub)
        correct = (sub.predicted_label == sub.condition).sum()
        acc     = correct / total * 100 if total else 0.0
        print(f"{cond}: {correct}/{total} = {acc:.1f}%")