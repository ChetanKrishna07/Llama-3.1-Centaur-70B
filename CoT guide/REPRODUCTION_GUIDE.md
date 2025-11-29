# Centaur vs Llama Comparison: Reproduction Guide

## Overview

This codebase implements **Centaur**, a foundation model for predicting human cognition. The experiments compare:
- **Centaur** (fine-tuned Llama 3.1 models on Psych-101 dataset)
- **Base Llama 3.1** models (70B and 8B versions)
- **Cognitive models** (task-specific baselines)
- **Random baselines**

The paper demonstrates that Centaur outperforms both base Llama models and traditional cognitive models in predicting human behavior across 160+ cognitive tasks.

---

## Experimental Setup

### 1. **Training (Fine-tuning)**

**Script:** `finetune.py`

Centaur is created by fine-tuning Llama 3.1 (70B or 8B) on the **Psych-101** dataset using LoRA (Low-Rank Adaptation).

**Training Command:**
```bash
python finetune.py \
  --seed 100 \
  --model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
  --max_seq_len 32768 \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --optim "adamw_8bit" \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --output_dir "centaur-output" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32
```

**Key Training Details:**
- **Dataset:** `marcelbinz/Psych-101` (HuggingFace dataset with 10M+ choices from 60K+ participants)
- **LoRA Config:** r=8, alpha=8, dropout=0
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Special Tokens:** Human choices are wrapped with `<<` and `>>` tokens
- **Hardware:** A100 80GB GPU, ~96 hours for 70B model

---

### 2. **Evaluation Pipeline**

The evaluation consists of **four main steps**:

#### **Step 1: Test on Held-out Participants (Standard Metric)**
**Script:** `test_adapter.py`

Evaluates on 36 tasks with held-out participants from the same experiments.

```bash
python test_adapter.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
```

**Tasks Evaluated:**
- badham2017deficits, bahrami2020four, enkavi2019adaptivenback, enkavi2019digitspan, etc.
- Total: 36 tasks (see lines 14-51 in `test_adapter.py`)

**Output:** `results/marcelbinz-Llama-3.1-Centaur-70B-adapter.csv`

---

#### **Step 2: Test on Held-out Participants (Custom Metric)**
**Script:** `test_adapter_custom_metrics.py`

Evaluates on 10 additional tasks using per-trial loss computation.

```bash
python test_adapter_custom_metrics.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
```

**Tasks Evaluated:**
- collsiöö2023MCPL, cox2017information, garcia2023experiential, etc.
- Total: 10 tasks (see lines 40-50 in `test_adapter_custom_metrics.py`)

**Output:** `results/custom_metrics_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv`

---

#### **Step 3: Generalization to New Experiments**
**Scripts:** 
- `generalization/generalization.py`
- `generalization/generalization_custom_metrics.py`

Tests generalization to completely new experiments not seen during training.

```bash
cd generalization/
python generalization.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
python generalization_custom_metrics.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
```

**Tasks Evaluated:**
- feher2020humans, dubois2022value, jansen2021logic
- Total: 3 experiments

**Output:** 
- `generalization/results/marcelbinz-Llama-3.1-Centaur-70B-adapter.csv`
- `generalization/results/custom_metrics_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv`

---

#### **Step 4: Full Log-Likelihood Analysis**
**Script:** `test_adapter_full_log_likelihoods.py`

Computes detailed per-trial log-likelihoods for all 46 tasks (36 + 10).

```bash
python test_adapter_full_log_likelihoods.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
```

**Output:** `results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth`

This file contains PyTorch tensors with per-trial losses for detailed statistical analysis.

---

#### **Step 5: Merge Results**
**Script:** `merge.py`

Combines all evaluation results into a single CSV file.

```bash
python merge.py --model marcelbinz-Llama-3.1-Centaur-70B-adapter
```

**Output:** `results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv`

This file contains:
- Results from held-out participants (standard + custom metrics)
- Results from generalization experiments
- Flags for `custom_metric` (True/False) and `unseen` (participants/experiments)

---

### 3. **Baseline Comparisons**

The paper compares Centaur against three baselines:

#### **A. Base Llama 3.1 (70B/8B)**
Run the exact same evaluation pipeline with the base model:

```bash
# 70B baseline
python test_adapter.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
python test_adapter_custom_metrics.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
python test_adapter_full_log_likelihoods.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
cd generalization/
python generalization.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
python generalization_custom_metrics.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit
cd ..
python merge.py --model unsloth-Meta-Llama-3.1-70B-bnb-4bit
```

**Results:** `results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv`

#### **B. Cognitive Models (Task-specific)**
Pre-computed cognitive model baselines are stored in:
- `results/custom_metrics_full_log_likelihoods_baselines.pth`
- `results/all_data_baseline.csv`

These are task-specific cognitive models from the original papers (e.g., reinforcement learning models, Bayesian models, etc.).

#### **C. Random Baseline**
Pre-computed random choice baseline:
- `results/all_data_random.csv`

---

## How to Reproduce with Another Model

### Example: Evaluating Mistral-7B-Instruct

Let's say you want to compare Centaur with **Mistral-7B-Instruct**.

#### **Option 1: Evaluate Pre-trained Model (No Fine-tuning)**

```bash
# Step 1: Evaluate on held-out participants
python test_adapter.py --model mistralai/Mistral-7B-Instruct-v0.3

# Step 2: Evaluate with custom metrics
python test_adapter_custom_metrics.py --model mistralai/Mistral-7B-Instruct-v0.3

# Step 3: Full log-likelihoods (for detailed analysis)
python test_adapter_full_log_likelihoods.py --model mistralai/Mistral-7B-Instruct-v0.3

# Step 4: Generalization tests
cd generalization/
python generalization.py --model mistralai/Mistral-7B-Instruct-v0.3
python generalization_custom_metrics.py --model mistralai/Mistral-7B-Instruct-v0.3
cd ..

# Step 5: Merge all results
python merge.py --model mistralai-Mistral-7B-Instruct-v0.3
```

**Output:** `results/all_data_mistralai-Mistral-7B-Instruct-v0.3.csv`

---

#### **Option 2: Fine-tune Your Model on Psych-101**

If you want to create a "Mistral-Centaur" version:

```bash
# Fine-tune Mistral on Psych-101
python finetune.py \
  --seed 100 \
  --model_name_or_path "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" \
  --max_seq_len 32768 \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --optim "adamw_8bit" \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --output_dir "mistral-centaur" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32

# Then evaluate the fine-tuned adapter
python test_adapter.py --model ./mistral-centaur
python test_adapter_custom_metrics.py --model ./mistral-centaur
python test_adapter_full_log_likelihoods.py --model ./mistral-centaur
cd generalization/
python generalization.py --model ./mistral-centaur
python generalization_custom_metrics.py --model ./mistral-centaur
cd ..
python merge.py --model mistral-centaur
```

---

### **Important Notes for Model Compatibility**

1. **Unsloth Support:** The scripts use `unsloth.FastLanguageModel`. For models not supported by unsloth, you'll need to modify the loading code:

```python
# Replace this:
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model,
    max_seq_length = 32768,
    dtype = None,
    load_in_4bit = True,
)

# With standard transformers:
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True  # requires bitsandbytes
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
```

2. **Special Tokens:** The evaluation assumes models can handle `<<` and `>>` tokens. Make sure to add them to the tokenizer if needed:

```python
tokenizer.add_tokens([" <<", ">>"])
model.resize_token_embeddings(len(tokenizer))
```

3. **Hardware Requirements:**
   - **70B models:** 80GB GPU (A100 recommended)
   - **8B models:** 24GB GPU (RTX 3090/4090, A5000)
   - **For CPU/MPS:** Use smaller models or reduce batch size

4. **Context Length:** Models need to support at least 8K tokens, ideally 32K for longer experiments.

---

## Visualization and Analysis

After collecting results from multiple models, you can compare them using the plotting scripts:

### **Main Comparison Plot**
```bash
cd plots/
python fig2_new.py  # Compares Centaur vs Llama vs Cognitive Models
```

This script loads:
- `results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth`
- `results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth`
- `results/custom_metrics_full_log_likelihoods_baselines.pth`

And produces visualizations showing which model performs best on each task.

### **Summary Statistics**
```bash
cd plots/
python tab1_new.py  # Generates LaTeX table with results
```

---

## Expected Results

Based on the paper's findings:

| Model | Mean NLL (lower is better) | R² Score (higher is better) |
|-------|---------------------------|------------------------------|
| **Centaur-70B** | ~1.30 | ~0.65 |
| **Llama-70B** | ~1.45 | ~0.55 |
| **Cognitive Models** | ~1.35 | ~0.60 |
| **Random** | ~2.00 | 0.00 |

Centaur should:
- Beat base Llama on ~80-90% of tasks
- Beat cognitive models on ~60-70% of tasks
- Show better generalization to new experiments

---

## Full Experiment Workflow Script

Here's a complete bash script to reproduce everything:

```bash
#!/bin/bash

MODEL_NAME="your-model-name"
MODEL_PATH="path/to/your/model"

# 1. Test on held-out participants
echo "Testing on held-out participants (standard metrics)..."
python test_adapter.py --model $MODEL_PATH

# 2. Test on held-out participants (custom metrics)
echo "Testing on held-out participants (custom metrics)..."
python test_adapter_custom_metrics.py --model $MODEL_PATH

# 3. Full log-likelihoods
echo "Computing full log-likelihoods..."
python test_adapter_full_log_likelihoods.py --model $MODEL_PATH

# 4. Generalization tests
echo "Testing generalization..."
cd generalization/
python generalization.py --model $MODEL_PATH
python generalization_custom_metrics.py --model $MODEL_PATH
cd ..

# 5. Merge results
echo "Merging results..."
python merge.py --model $MODEL_NAME

echo "Done! Results saved to results/all_data_${MODEL_NAME}.csv"
```

---

## Key Files Summary

| File | Purpose |
|------|---------|
| `finetune.py` | Fine-tune Llama on Psych-101 |
| `test_adapter.py` | Evaluate on 36 tasks (held-out participants) |
| `test_adapter_custom_metrics.py` | Evaluate on 10 tasks with custom loss |
| `test_adapter_full_log_likelihoods.py` | Detailed per-trial analysis (46 tasks) |
| `generalization/generalization.py` | Test on 2 new experiments |
| `generalization/generalization_custom_metrics.py` | Test on 1 new experiment (custom loss) |
| `merge.py` | Combine all results into one file |
| `experiments.csv` | Metadata about all 160 experiments |
| `results/` | All evaluation outputs |
| `plots/` | Visualization scripts |
| `scripts/cluster_*.sh` | SLURM cluster job scripts |

---

## Questions?

For issues or questions:
- Check the paper: https://arxiv.org/abs/2410.20268
- Model on HuggingFace: https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter
- Dataset: https://huggingface.co/datasets/marcelbinz/Psych-101

---

## Citation

```bibtex
@article{binz2024centaur,
  title={Centaur: a foundation model of human cognition},
  author={Binz, Marcel and Akata, Elif and Bethge, Matthias and ...},
  journal={Nature},
  year={2025},
  doi={10.1038/s41586-025-09215-4}
}
```

