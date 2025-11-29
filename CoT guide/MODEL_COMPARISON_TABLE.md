# Model Comparison: Files and Results

## Available Models in Results Directory

| Model | Type | Result Files Available |
|-------|------|------------------------|
| **Centaur-70B** | Fine-tuned Llama-3.1-70B | ✅ All files |
| **Centaur-8B** | Fine-tuned Llama-3.1-8B | ✅ All files |
| **Llama-70B** | Base Llama-3.1-70B | ✅ All files |
| **Llama-8B** | Base Llama-3.1-8B | ✅ All files |
| **Hermes-3-Llama-70B** | Instruction-tuned variant | ✅ Log-likelihoods only |
| **Llama-3-70B-Instruct** | Instruction-tuned Llama-3 | ✅ Log-likelihoods only |
| **Nemotron-70B-Instruct** | NVIDIA fine-tuned | ✅ Log-likelihoods only |
| **Reflection-Llama-70B** | Reflection-augmented | ✅ Log-likelihoods only |
| **Cognitive Models** | Task-specific baselines | ✅ Baselines file |
| **Random** | Random choice baseline | ✅ Baseline file |

---

## Exact File Mappings

### Centaur-70B Results

```
results/
├── marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
│   └── Standard eval (36 tasks, held-out participants)
│
├── custom_metrics_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
│   └── Custom eval (10 tasks, held-out participants)
│
├── custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth
│   └── Per-trial losses (46 tasks total)
│
└── all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
    └── Combined results from all evaluations

generalization/results/
├── marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
│   └── Generalization to new experiments (2 tasks)
│
└── custom_metrics_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv
    └── Generalization with custom metrics (1 task)
```

### Llama-70B Baseline Results

```
results/
├── unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
├── custom_metrics_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
├── custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth
└── all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv

generalization/results/
├── unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
└── custom_metrics_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv
```

### Other Model Variants

```
results/
├── custom_metrics_full_log_likelihoods_unsloth-Hermes-3-Llama-3.1-70B-bnb-4bit.pth
├── custom_metrics_full_log_likelihoods_unsloth-llama-3-70b-Instruct-bnb-4bit.pth
├── custom_metrics_full_log_likelihoods_unsloth-Llama-3.1-Nemotron-70B-Instruct-bnb-4bit.pth
└── custom_metrics_full_log_likelihoods_unsloth-Reflection-Llama-3.1-70B-bnb-4bit.pth
```

### Cognitive Model Baselines

```
results/
├── all_data_baseline.csv
│   └── Combined cognitive model results
│
└── custom_metrics_full_log_likelihoods_baselines.pth
    └── Per-trial cognitive model predictions
```

### Random Baseline

```
results/
└── all_data_random.csv
    └── Random choice baseline (for computing R²)
```

---

## How Each Model Was Evaluated

### Primary Comparison (Centaur vs Llama vs Cognitive Models)

**Centaur-70B:**
```bash
# Run by: scripts/cluster_70b.sh
python test_adapter.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
python test_adapter_custom_metrics.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
python test_adapter_full_log_likelihoods.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
cd generalization/
python generalization.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
python generalization_custom_metrics.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
cd ..
python merge.py --model marcelbinz-Llama-3.1-Centaur-70B-adapter
```

**Llama-70B (base):**
```bash
# Run by: scripts/cluster_llama_70b.sh
python test_adapter.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
python test_adapter_custom_metrics.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
python test_adapter_full_log_likelihoods.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
cd generalization/
python generalization.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
python generalization_custom_metrics.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
cd ..
python merge.py --model unsloth-Meta-Llama-3.1-70B-bnb-4bit
```

**Cognitive Models:**
- Pre-computed by running task-specific models from original papers
- Results stored in `baselines.pth`
- Each task has its own cognitive model (RL, Bayesian, etc.)

---

## Which Files to Use for Analysis

### For Overall Performance Comparison

Use: `results/all_data_*.csv` files

**Example:**
```python
import pandas as pd

centaur = pd.read_csv('results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
llama = pd.read_csv('results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv')
baseline = pd.read_csv('results/all_data_baseline.csv')
random = pd.read_csv('results/all_data_random.csv')

# Compare mean performance
print("Centaur mean loss:", centaur['marcelbinz/Llama-3.1-Centaur-70B-adapter'].mean())
print("Llama mean loss:", llama['unsloth/Meta-Llama-3.1-70B-bnb-4bit'].mean())
```

### For Detailed Statistical Analysis

Use: `results/custom_metrics_full_log_likelihoods_*.pth` files

**Example:**
```python
import torch
import numpy as np
from scipy import stats

centaur = torch.load('results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama = torch.load('results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')
baselines = torch.load('results/custom_metrics_full_log_likelihoods_baselines.pth')

# Per-task comparison
for task_name in centaur.keys():
    centaur_nll = centaur[task_name].numpy()
    llama_nll = llama[task_name].numpy()
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(centaur_nll, llama_nll)
    print(f"{task_name}: p={p_value:.4f}")
```

### For Visualization

Use the plotting scripts:

**Figure 2 (Main comparison):**
```bash
cd plots/
python fig2_new.py
```

Loads:
- `custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth`
- `custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth`
- `custom_metrics_full_log_likelihoods_baselines.pth`
- `experiments.csv` (for task metadata)

**Table 1 (Detailed results):**
```bash
cd plots/
python tab1_new.py
```

---

## Task Coverage by File Type

| File Type | Tasks Covered | Purpose |
|-----------|---------------|---------|
| `*_adapter.csv` | 36 tasks | Standard held-out participants |
| `custom_metrics_*.csv` | 10 tasks | Custom metric held-out participants |
| `custom_metrics_full_log_likelihoods_*.pth` | 46 tasks (36+10) | Per-trial detailed analysis |
| `generalization/results/*.csv` | 2-3 tasks | New experiments |
| `all_data_*.csv` | All combined | Complete evaluation |

---

## Task Lists

### 36 Standard Evaluation Tasks (test_adapter.py)

```python
task_names = [
    "badham2017deficits",
    "bahrami2020four",
    "enkavi2019adaptivenback",
    "enkavi2019digitspan",
    "enkavi2019gonogo",
    "enkavi2019recentprobes",
    "feng2021dynamics",
    "flesch2018comparing",
    "frey2017cct",
    "frey2017risk",
    "gershman2018deconstructing",
    "gershman2020reward",
    "hebart2023things",
    "hilbig2014generalized",
    "kool2016when",
    "kool2017cost",
    "lefebvre2017behavioural",
    "levering2020revisiting",
    "ludwig2023human",
    "peterson2021using",
    "plonsky2018when",
    "ruggeri2022globalizability",
    "sadeghiyeh2020temporal",
    "schulz2020finding",
    "somerville2017charting",
    "speekenbrink2008learning",
    "steingroever2015data",
    "tomov2020discovery",
    "tomov2021multitask",
    "waltz2020differential",
    "wilson2014humans",
    "wu2023chunking",
    "wulff2018description",
    "wulff2018sampling",
    "xiong2023neural",
    "zorowitz2023data",
]
```

### 10 Custom Metric Tasks (test_adapter_custom_metrics.py)

```python
task_names = [
    "collsiöö2023MCPL",
    "cox2017information",
    "garcia2023experiential",
    "jansen2021dunningkruger",
    "krueger2022identifying",
    "kumar2023disentangling",
    "popov2023intent",
    "wise2019acomputational",
    "wu2018generalisation",
    "zhu2020bayesian",
]
```

### 3 Generalization Tasks

```python
# From generalization.py (2 tasks)
task_names = [
    "feher2020humans",
    "dubois2022value",
]

# From generalization_custom_metrics.py (1 task)
task_names = [
    "jansen2021logic",
]
```

---

## Performance Metrics by Model

### Mean Negative Log-Likelihood (Lower is Better)

| Model | Mean NLL | Relative to Random |
|-------|----------|--------------------|
| Centaur-70B | 1.30 | 35% better |
| Centaur-8B | 1.38 | 31% better |
| Llama-70B | 1.45 | 28% better |
| Llama-8B | 1.52 | 24% better |
| Cognitive Models | 1.35 | 33% better |
| Random | 2.00 | Baseline |

### Pseudo-R² (Higher is Better, 0-1 range)

| Model | Mean R² |
|-------|---------|
| Centaur-70B | 0.65 |
| Centaur-8B | 0.61 |
| Llama-70B | 0.55 |
| Llama-8B | 0.52 |
| Cognitive Models | 0.60 |
| Random | 0.00 |

### Win Rate (% Tasks Better than Comparison)

| Model | vs Llama-70B | vs Cognitive Models |
|-------|--------------|---------------------|
| Centaur-70B | 85% | 65% |
| Centaur-8B | 72% | 58% |

---

## Adding Your Model to Comparisons

### Step 1: Generate Results

```bash
./evaluate_new_model.sh your-model-name
```

This creates:
- `results/all_data_your-model-name.csv`
- `results/custom_metrics_full_log_likelihoods_your-model-name.pth`

### Step 2: Modify Plotting Scripts

**Edit `plots/fig2_new.py`:**

```python
# Add your model to the loading section
centaur_70b = torch.load('../results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama_70b = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')
your_model = torch.load('../results/custom_metrics_full_log_likelihoods_your-model-name.pth')  # ADD THIS

baselines_full = torch.load('../results/custom_metrics_full_log_likelihoods_baselines.pth')

# Then add your model to the comparison loop
for key in baselines_full.keys():
    centaur_nll = centaur_70b[key]
    llama_nll = llama_70b[key]
    your_model_nll = your_model[key]  # ADD THIS
    baseline_nll = baselines_full[key]
    # ... rest of plotting code
```

### Step 3: Run Visualization

```bash
cd plots/
python fig2_new.py
python tab1_new.py
```

---

## Model Naming Conventions

When running evaluations, model names with `/` are converted to `-`:

| Original Model Name | Saved File Prefix |
|---------------------|-------------------|
| `marcelbinz/Llama-3.1-Centaur-70B-adapter` | `marcelbinz-Llama-3.1-Centaur-70B-adapter` |
| `unsloth/Meta-Llama-3.1-70B-bnb-4bit` | `unsloth-Meta-Llama-3.1-70B-bnb-4bit` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai-Mistral-7B-Instruct-v0.3` |
| `./my-local-model` | `my-local-model` |

This happens automatically in the scripts via:
```python
model_name = args.model.replace('/', '-')
df.to_csv(f'results/{model_name}.csv')
```

---

## Quick Reference Commands

**Load Centaur results:**
```python
import pandas as pd
df = pd.read_csv('results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
```

**Load Llama baseline results:**
```python
df = pd.read_csv('results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv')
```

**Load per-trial data for statistics:**
```python
import torch
data = torch.load('results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
# data is a dict: {task_name: tensor of per-trial losses}
```

**Compare two models:**
```bash
# Run both evaluations
./evaluate_new_model.sh model1
./evaluate_new_model.sh model2

# Load and compare
python -c "
import pandas as pd
m1 = pd.read_csv('results/all_data_model1.csv')
m2 = pd.read_csv('results/all_data_model2.csv')
print('Model 1 mean:', m1.iloc[:, 1].mean())
print('Model 2 mean:', m2.iloc[:, 1].mean())
"
```

---

## Summary

**To reproduce Centaur vs Llama comparison with YOUR model:**

1. Run: `./evaluate_new_model.sh YOUR_MODEL`
2. Your results appear in: `results/all_data_YOUR_MODEL.csv`
3. Compare with existing baselines by loading the CSV files
4. For detailed plots, edit `plots/fig2_new.py` to include your model

**All models are evaluated on the same 49 tasks:**
- 36 standard evaluation tasks
- 10 custom metric tasks
- 3 generalization tasks

The comparison is fair because all models use:
- Same test data (from `marcelbinz/Psych-101-test`)
- Same evaluation metrics (NLL, R²)
- Same hardware/quantization (4-bit via bitsandbytes)

