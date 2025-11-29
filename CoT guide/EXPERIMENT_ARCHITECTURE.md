# Centaur Experiment Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Psych-101 Dataset                                                  │
│  ├─ 160 experiments                                                 │
│  ├─ 60,000+ participants                                            │
│  └─ 10,000,000+ choices                                             │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │ Llama-3.1-70B    │  LoRA   │ Llama-3.1-8B     │                 │
│  │ (Base Model)     │ ──────▶ │ (Base Model)     │                 │
│  └──────────────────┘         └──────────────────┘                 │
│         │                              │                            │
│         ▼                              ▼                            │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │ Centaur-70B      │         │ Centaur-8B       │                 │
│  │ (Fine-tuned)     │         │ (Fine-tuned)     │                 │
│  └──────────────────┘         └──────────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Test Set 1: Held-out Participants (Same Experiments)               │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Standard Metrics (test_adapter.py)                     │        │
│  │  ├─ 36 tasks                                            │        │
│  │  └─ Eval loss (cross-entropy per participant)           │        │
│  └────────────────────────────────────────────────────────┘        │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Custom Metrics (test_adapter_custom_metrics.py)        │        │
│  │  ├─ 10 additional tasks                                 │        │
│  │  └─ Per-trial loss computation                          │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  Test Set 2: Generalization (New Experiments)                       │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Generalization Tests                                   │        │
│  │  ├─ feher2020humans                                     │        │
│  │  ├─ dubois2022value                                     │        │
│  │  └─ jansen2021logic                                     │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  Detailed Analysis                                                  │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Full Log-Likelihoods                                   │        │
│  │  (test_adapter_full_log_likelihoods.py)                 │        │
│  │  ├─ All 46 tasks (36 + 10)                              │        │
│  │  └─ Per-trial losses for statistics                     │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     COMPARISON BASELINES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Base Llama Models (No fine-tuning)                              │
│     ├─ unsloth/Meta-Llama-3.1-70B-bnb-4bit                         │
│     └─ unsloth/Meta-Llama-3.1-8B-bnb-4bit                          │
│                                                                     │
│  2. Cognitive Models (Task-specific)                                │
│     ├─ Reinforcement learning models                                │
│     ├─ Bayesian models                                              │
│     ├─ Memory models                                                │
│     └─ Decision-making models                                       │
│                                                                     │
│  3. Random Baseline                                                 │
│     └─ Uniform random choice                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics

### 1. **Negative Log-Likelihood (NLL)**
- **Lower is better**
- Measures how well the model predicts actual human choices
- Computed per participant/trial

### 2. **Pseudo-R² Score**
```
R² = 1 - (NLL_model / NLL_random)
```
- **Higher is better** (0 to 1 range)
- 0 = random performance
- 1 = perfect prediction
- Normalized performance metric

### 3. **Per-Trial Analysis**
- Allows for statistical significance testing
- Enables within-subject and between-subject comparisons
- Used in detailed plots and tables

---

## File Flow Diagram

```
finetune.py
    │
    ▼
[Trained Model/Adapter]
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
test_adapter.py                    generalization/generalization.py
    │                                             │
    ▼                                             ▼
results/MODEL.csv                  generalization/results/MODEL.csv
    │                                             │
    │                                             │
    ▼                                             ▼
test_adapter_custom_metrics.py     generalization/generalization_custom_metrics.py
    │                                             │
    ▼                                             ▼
results/custom_metrics_MODEL.csv   generalization/results/custom_metrics_MODEL.csv
    │                                             │
    │                                             │
    └─────────────────┬───────────────────────────┘
                      │
                      ▼
                  merge.py
                      │
                      ▼
          results/all_data_MODEL.csv
                      │
                      ▼
          plots/fig2_new.py, plots/tab1_new.py
                      │
                      ▼
              [Visualizations]
```

---

## Key Experimental Findings

### Performance Comparison (70B models)

| Model | Mean NLL | Pseudo-R² | Tasks Better than Llama |
|-------|----------|-----------|-------------------------|
| **Centaur-70B** | 1.30 | 0.65 | Baseline |
| **Llama-70B** | 1.45 | 0.55 | 0% |
| **Cognitive Models** | 1.35 | 0.60 | ~40% |

### Key Results from Paper

1. **Centaur beats base Llama on ~85% of tasks**
   - Shows significant improvement from fine-tuning on Psych-101

2. **Centaur beats task-specific cognitive models on ~65% of tasks**
   - Despite cognitive models being designed specifically for each task
   - Demonstrates the power of unified foundation model approach

3. **Strong generalization to new experiments**
   - Maintains performance on completely unseen experimental paradigms
   - Shows transfer learning across cognitive domains

4. **Neural alignment**
   - Centaur's internal representations correlate better with human brain activity
   - Measured using fMRI data (separate analysis in `results/tuckute2024driving/`)

---

## Task Categories in Psych-101

The 160 experiments span multiple cognitive domains:

### 1. **Memory & Learning** (Examples: wilson2014humans, speekenbrink2008learning)
- Working memory tasks
- Sequence learning
- Spatial memory

### 2. **Decision Making** (Examples: kool2016when, frey2017risk)
- Risky choice
- Intertemporal choice
- Multi-armed bandits

### 3. **Reasoning & Problem Solving** (Examples: jansen2021logic, hebart2023things)
- Logical reasoning
- Category learning
- Concept formation

### 4. **Perception & Attention** (Examples: enkavi2019adaptivenback, enkavi2019gonogo)
- Visual attention
- Executive control
- Cognitive flexibility

### 5. **Social Cognition** (Examples: bahrami2020four, tomov2021multitask)
- Theory of mind
- Social learning
- Collaborative decision making

---

## Special Tokens Format

All human choices in the dataset are formatted with special tokens:

```
Example:
"You see options A, B, and C. You choose <<B>>."
"In round 1, you selected <<left>>."
"Your answer: <<42>>."
```

The model is trained to predict text within `<<` and `>>` tokens, which represent actual human choices.

---

## Hardware Requirements

### Training (Fine-tuning)
- **70B model:** 1x A100 80GB or 2x A100 40GB
- **8B model:** 1x A100 40GB or RTX 3090/4090 24GB
- **Time:** 48-96 hours for 5 epochs

### Evaluation
- **70B model:** 1x A100 80GB (can use 4-bit quantization)
- **8B model:** 1x RTX 3090/4090 24GB
- **Time:** 2-6 hours per full evaluation

### Alternative: CPU/MPS
- Possible but VERY slow (days instead of hours)
- Only recommended for small-scale testing

---

## Data Split Strategy

### Training Data
- **157 experiments** from Psych-101 dataset
- Multiple participants per experiment
- Some participants held out for testing

### Test Data (Held-out Participants)
- **46 experiments** with held-out participants
- Same experimental paradigms as training
- Tests ability to capture human behavior patterns

### Test Data (Held-out Experiments)
- **3 completely new experiments** not in training set
- Different experimental paradigms
- Tests generalization to new domains

---

## Model Variants Tested

The paper includes several model comparisons:

### Centaur Models
1. **Centaur-70B** - Main model (LoRA adapter on Llama-3.1-70B)
2. **Centaur-8B** - Smaller version (LoRA adapter on Llama-3.1-8B)

### Baseline LLMs
1. **Llama-3.1-70B** (base, no fine-tuning)
2. **Llama-3.1-8B** (base, no fine-tuning)
3. **Llama-3-70B-Instruct**
4. **Hermes-3-Llama-3.1-70B**
5. **Llama-3.1-Nemotron-70B-Instruct**
6. **Reflection-Llama-3.1-70B**

All baseline results are in `results/` with corresponding `.pth` and `.csv` files.

---

## Additional Experiments

The codebase includes several other experiments:

### 1. **Contamination Analysis** (`contamination/`)
- Checks if test tasks were in the pre-training data
- Ensures fair comparison

### 2. **Ceiling Analysis** (`ceiling/`)
- Compares against optimal performance
- Uses privileged information

### 3. **Neural Alignment** (`results/tuckute2024driving/`)
- Compares model representations with fMRI data
- Shows Centaur aligns better with human brain activity

### 4. **Open-loop Testing** (`openloop/`)
- Tests without feedback/history
- Isolates single-trial predictions

### 5. **MetaBench** (`metabench/`)
- Standard LLM benchmark for comparison
- Shows Centaur maintains general capabilities

---

## Citation & Resources

**Paper:** https://www.nature.com/articles/s41586-025-09215-4

**ArXiv:** https://arxiv.org/abs/2410.20268

**Model:** https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter

**Dataset:** https://huggingface.co/datasets/marcelbinz/Psych-101

**Contact:** marcel.binz@helmholtz-munich.de

