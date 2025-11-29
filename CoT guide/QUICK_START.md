# Quick Start Guide: Understanding and Reproducing Centaur Experiments

## What is Centaur?

**Centaur** is a fine-tuned version of Llama 3.1 that predicts human behavior in cognitive tasks. It's trained on **Psych-101**, a dataset with 10M+ choices from 60K+ participants across 160 experiments.

**Key Innovation:** A single unified model that outperforms both:
- Base Llama models (not fine-tuned)
- Task-specific cognitive models (designed by domain experts)

---

## How Were the Experiments Performed?

### 1. **Training Centaur**

```bash
python finetune.py \
  --model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --output_dir "centaur-output"
```

**Details:**
- Uses LoRA (Low-Rank Adaptation) to fine-tune efficiently
- Trained on `marcelbinz/Psych-101` dataset from HuggingFace
- Takes ~48-96 hours on A100 80GB GPU
- Special tokens: Human choices wrapped with `<<` and `>>`

### 2. **Evaluating Centaur vs Llama**

The comparison involves **4 evaluation scripts**:

```bash
# Script 1: Test on 36 tasks (held-out participants)
python test_adapter.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter
python test_adapter.py --model unsloth/Meta-Llama-3.1-70B-bnb-4bit

# Script 2: Test on 10 tasks with custom metrics
python test_adapter_custom_metrics.py --model <MODEL>

# Script 3: Detailed per-trial analysis (all 46 tasks)
python test_adapter_full_log_likelihoods.py --model <MODEL>

# Script 4: Test generalization to new experiments
cd generalization/
python generalization.py --model <MODEL>
python generalization_custom_metrics.py --model <MODEL>
```

### 3. **Combining Results**

```bash
python merge.py --model marcelbinz-Llama-3.1-Centaur-70B-adapter
```

This creates: `results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv`

### 4. **Visualization**

```bash
cd plots/
python fig2_new.py  # Compare Centaur vs Llama vs Cognitive Models
python tab1_new.py  # Generate results table
```

---

## How to Reproduce with Your Own Model

### **Option A: Quick Evaluation (Pre-existing Model)**

Use the automated script:

```bash
./evaluate_new_model.sh mistralai/Mistral-7B-Instruct-v0.3
```

This runs all 6 steps automatically and produces:
- `results/all_data_mistralai-Mistral-7B-Instruct-v0.3.csv`

### **Option B: Manual Step-by-Step**

```bash
# 1. Evaluate on held-out participants
python test_adapter.py --model YOUR_MODEL
python test_adapter_custom_metrics.py --model YOUR_MODEL

# 2. Detailed analysis
python test_adapter_full_log_likelihoods.py --model YOUR_MODEL

# 3. Generalization tests
cd generalization/
python generalization.py --model YOUR_MODEL
python generalization_custom_metrics.py --model YOUR_MODEL
cd ..

# 4. Merge results
python merge.py --model YOUR_MODEL_NAME
```

### **Option C: Fine-tune Your Own "Centaur" Version**

```bash
# Fine-tune your model on Psych-101
python finetune.py \
  --model_name_or_path "your-base-model" \
  --output_dir "your-centaur-model"

# Then evaluate it
./evaluate_new_model.sh ./your-centaur-model
```

---

## Key Results Files

After evaluation, you'll have:

| File | Contains |
|------|----------|
| `results/all_data_<MODEL>.csv` | **Main results** - All tasks combined |
| `results/custom_metrics_full_log_likelihoods_<MODEL>.pth` | **Per-trial data** - For detailed statistics |
| `results/<MODEL>.csv` | Standard metrics (36 tasks) |
| `results/custom_metrics_<MODEL>.csv` | Custom metrics (10 tasks) |
| `generalization/results/<MODEL>.csv` | Generalization tests |

---

## Understanding the Results

### Metrics Used

1. **Negative Log-Likelihood (NLL)** - Lower is better
   - Measures prediction accuracy
   - Centaur: ~1.30, Llama: ~1.45

2. **Pseudo-R²** - Higher is better (0-1 range)
   - Formula: `R² = 1 - (NLL_model / NLL_random)`
   - Centaur: ~0.65, Llama: ~0.55

### Expected Performance

From the paper's findings:

```
Centaur-70B:
  - Beats base Llama-70B on ~85% of tasks
  - Beats cognitive models on ~65% of tasks
  - Mean NLL: 1.30 (lower is better)

Llama-70B (base):
  - Mean NLL: 1.45
  - No fine-tuning on Psych-101
  
Cognitive Models:
  - Mean NLL: 1.35
  - Task-specific models designed by experts
```

---

## Comparison Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Data                        │
│               Psych-101 (10M choices)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
┌───────────────┐                    ┌───────────────┐
│  Centaur-70B  │                    │  Llama-70B    │
│ (Fine-tuned)  │                    │ (Base Model)  │
└───────────────┘                    └───────────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │      Evaluation (46 tasks)           │
        │  - Held-out participants (36+10)     │
        │  - New experiments (3)               │
        └──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │         Results Comparison           │
        │  Centaur > Llama > Cognitive Models  │
        └──────────────────────────────────────┘
```

---

## Task Examples

The 160 experiments cover diverse cognitive domains:

### Memory Tasks
- `wilson2014humans` - Multi-armed bandits with memory
- `badham2017deficits` - Working memory

### Decision Making
- `kool2016when` - Model-based vs model-free RL
- `frey2017risk` - Risk preferences

### Reasoning
- `jansen2021logic` - Logical reasoning
- `hebart2023things` - Category learning

### Attention
- `enkavi2019gonogo` - Go/No-Go task
- `enkavi2019adaptivenback` - N-back task

---

## Hardware Requirements

| Model Size | GPU Memory | Recommended GPU | Evaluation Time |
|------------|------------|-----------------|-----------------|
| 70B (4-bit) | 80GB | A100 80GB | 2-4 hours |
| 8B (4-bit) | 24GB | RTX 3090/4090 | 1-2 hours |

**Note:** The scripts use 4-bit quantization via `unsloth` and `bitsandbytes` to reduce memory usage.

---

## Common Issues & Solutions

### Issue 1: Out of Memory
**Solution:** 
- Reduce `per_device_eval_batch_size` to 1 in the scripts
- Use gradient checkpointing
- Try 8B model instead of 70B

### Issue 2: Model Not Found
**Solution:**
- For HuggingFace models: Ensure you have access (some require approval)
- For local models: Use absolute path or `./relative/path`

### Issue 3: Unsloth Compatibility
**Solution:**
If your model isn't supported by unsloth, modify the loading code:

```python
# Change from:
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)

# To:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(...)
```

---

## Example: Comparing Centaur with Mistral

```bash
# 1. Evaluate Mistral (base, no fine-tuning)
./evaluate_new_model.sh mistralai/Mistral-7B-Instruct-v0.3

# 2. (Optional) Fine-tune Mistral on Psych-101
python finetune.py \
  --model_name_or_path "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" \
  --output_dir "mistral-centaur"

# 3. Evaluate fine-tuned version
./evaluate_new_model.sh ./mistral-centaur

# 4. Compare results
cd plots/
# Edit fig2_new.py to load your Mistral results
python fig2_new.py
```

---

## File Reference

**Core Scripts:**
- `finetune.py` - Train Centaur
- `test_adapter.py` - Evaluate 36 tasks
- `test_adapter_custom_metrics.py` - Evaluate 10 tasks (custom loss)
- `test_adapter_full_log_likelihoods.py` - Detailed per-trial analysis
- `merge.py` - Combine all results

**Helper Scripts:**
- `evaluate_new_model.sh` - **Automated evaluation pipeline** ⭐
- `run_minimal.py` - Minimal demo for testing

**Cluster Scripts:** (for SLURM clusters)
- `scripts/cluster_70b.sh` - Run Centaur-70B evaluation
- `scripts/cluster_llama_70b.sh` - Run Llama-70B baseline
- `scripts/cluster_train.sh` - Fine-tuning job

**Documentation:**
- `README.md` - Project overview
- `REPRODUCTION_GUIDE.md` - **Detailed reproduction guide** ⭐
- `EXPERIMENT_ARCHITECTURE.md` - **Architecture explanation** ⭐
- `QUICK_START.md` - This file

---

## Next Steps

1. **Read the paper:** https://www.nature.com/articles/s41586-025-09215-4
2. **Explore the dataset:** https://huggingface.co/datasets/marcelbinz/Psych-101
3. **Try the model:** https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter
4. **Run your first evaluation:** `./evaluate_new_model.sh <YOUR_MODEL>`

---

## Questions?

For detailed information, see:
- **REPRODUCTION_GUIDE.md** - Complete experimental details
- **EXPERIMENT_ARCHITECTURE.md** - System architecture and findings

Contact: marcel.binz@helmholtz-munich.de

