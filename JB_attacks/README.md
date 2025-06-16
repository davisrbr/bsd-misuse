### **Jailbreaking Attacks**

This directory contains code for **five jailbreak attacks** used in our paper.  
All experiments run through **`jailbreakings_main.py`**; command-line flags let you switch between:

| Method | `--method` value | Decomposition flag |
|--------|------------------|--------------------|
| Decomposition Attack | `plain` | `--decomposition` |
| Pair | `pair` | `--no-decomposition` |
| Adaptive Attack | `adaptive` | `--no-decomposition` |
| Adversarial Reasoning | `adv` | `--no-decomposition` |
| Crescendo | `crescendo` | `--no-decomposition` |

#### 1 Quick Start


1. Install required libraries: 
`pip install inspect_ai`

2. Set up your API keys:
```bash 
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export TOGETHER_API_KEY="your-together-key"
export GEMINI_API_KEY="your-gemini-key"
```

3. Run an attack (example: decomposition)
```bash
python jailbreakings_main.py --dataset wmdpr \
  --decomposer-models together/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --composer-models  together/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --answerer-models  together/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --max-iterations 3 --max-decompositions 3 --epochs 3 \
  --output-csv "new_results_decomp_wmdpr_${TIMESTAMP}.csv" \
  --log-dir   "new_logs_decomp_wmdpr_${TIMESTAMP}" \
  --experiment-tag "${EXPERIMENT_TAG}" --run-id "custom_run" \
  --method plain --decomposition
```

`config.py` is the single source for dataset paths and per-attack hyper-parameters.
You can change `--epochs` to specify how many times each question is run; the final score is the average across these runs.
To use a custom model, provide its path-- for example:

`--model-path-decomposer /Models/Qwen/qwen1e-6`


#### 2 Dataset access
The wmdpr dataset is restricted for safety reasons. Contact us to request access. You can use you custom dataset and point to it in `config.py`. 

#### 3 Method-Specific Commands

##### 3.1 Decomposition Attack
```bash
python jailbreakings_main.py --dataset wmdpr \
  --decomposer-models together/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --attacker-models na \
  --composer-models   together/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --answerer-models   together/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --max-iterations 3 --max-decompositions 3 --epochs 3 \
  --output-csv "new_results_decomp_wmdpr_${TIMESTAMP}.csv" \
  --log-dir   "new_logs_decomp_wmdpr_${TIMESTAMP}" \
  --experiment-tag "${EXPERIMENT_TAG}" --run-id "custom_run" \
  --method plain --decomposition
```


##### 3.2 Pair
Supported attacker models are listed in pair/config.py.
Example using qwen: 
```bash
python jailbreakings_main.py --dataset wmdpr \
  --decomposer-models na \
  --attacker-models qwen \
  --composer-models  na \
  --answerer-models openai/o1-preview-2024-09-12 \
  --max-iterations 3 --max-decompositions 3 --epochs 3 \
  --output-csv "results_pair_wmdpr_${TIMESTAMP}.csv" \
  --log-dir   "logs_pair_wmdpr_${TIMESTAMP}" \
  --experiment-tag "${EXPERIMENT_TAG}" --run-id "41-gen" \
  --method pair --no-decomposition
```

##### 3.3 Adaptive Attack (2-step)

1.	Generate attack strings

```bash
# inside adaptive/
jupyter notebook adaptive/string_generator.ipynb
# → writes the JSONL file referenced by PATH_TO_STRINGS_FILE
```

2.	Run the attack

```bash
python jailbreakings_main.py --dataset wmdpr \
  --decomposer-models na \
  --attacker-models na \
  --composer-models  na \
  --answerer-models openai/o1-preview-2024-09-12 \
  --max-iterations 3 --max-decompositions 3 --epochs 3 \
  --output-csv "results_adaptive_wmdpr_${TIMESTAMP}.csv" \
  --log-dir   "logs_adaptive_wmdpr_${TIMESTAMP}" \
  --experiment-tag "${EXPERIMENT_TAG}" --run-id "41-gen" \
  --method adaptive --no-decomposition
```


##### 3.4 Adversarial Reasoning and Crescendo
Both methods accept any attacker; specify it with `--attacker-models`.


Example for Adversarial Reasoning: 
```bash
python jailbreakings_main.py --dataset wmdpr \
  --decomposer-models na \
  --attacker-models together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo \
  --composer-models  na \
  --answerer-models openai/o1-preview-2024-09-12 \
  --max-iterations 3 --max-decompositions 3 --epochs 3 \
  --output-csv "results_adv_wmdpr_${TIMESTAMP}.csv" \
  --log-dir   "logs_adv_wmdpr_${TIMESTAMP}" \
  --experiment-tag "${EXPERIMENT_TAG}" --run-id "41-gen" \
  --method adv --no-decomposition
```

Example for Crescendo: 
```bash
python jailbreakings_main.py --dataset wmdpr \
  --decomposer-models na \
  --attacker-models together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
  --composer-models  na \
  --answerer-models openai/o1-preview-2024-09-12 \
  --max-iterations 3 --max-decompositions 3 --epochs 3 \
  --output-csv "results_crescendo_wmdpr_${TIMESTAMP}.csv" \
  --log-dir   "logs_crescendo_wmdpr_${TIMESTAMP}" \
  --experiment-tag "${EXPERIMENT_TAG}" --run-id "41-gen" \
  --method crescendo --no-decomposition
```

#### 4 Outputs
After an experiment finishes, you can browse the log directory with **Inspect-AI**:
```bash
inspect view start --log-dir PATH_TO_DIRECTORY
```

This launches a local web UI where you can filter prompts and debug failures.

