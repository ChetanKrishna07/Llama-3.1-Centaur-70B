# Documentation Index

Welcome! This directory contains comprehensive documentation for understanding, reproducing, and extending the Centaur experiments.

## 📖 Documentation Files

### 1. **QUICK_START.md** ⭐ **START HERE**
**Read this first if you're new to the project**

Contains:
- What is Centaur?
- Quick overview of how experiments were performed
- Simple commands to get started
- 5-minute guide to reproducing results

**Best for:** Getting up to speed quickly

---

### 2. **REPRODUCTION_GUIDE.md**
**Complete guide for reproducing all experiments**

Contains:
- Detailed experimental setup
- Training procedures
- Evaluation pipeline (all 4 scripts)
- Baseline comparisons
- Step-by-step instructions

**Best for:** Reproducing the full paper experiments

---

### 3. **EXPERIMENT_ARCHITECTURE.md**
**System design and architecture explanation**

Contains:
- Visual diagrams of the system
- Evaluation metrics explained
- Task categories in Psych-101
- Key findings from the paper
- File flow diagrams

**Best for:** Understanding how everything fits together

---

### 4. **MODEL_COMPARISON_TABLE.md**
**Reference for all model results and files**

Contains:
- Exact file mappings for all models
- Which results files contain what data
- Task lists (36 + 10 + 3 tasks)
- How to load and compare results
- Code examples for analysis

**Best for:** Working with existing results data

---

### 5. **COT_CENTAUR_GUIDE.md** 🆕
**Complete guide for creating Chain-of-Thought Centaur**

Contains:
- How to create CoT training data
- Two approaches (GPT-4 and rule-based)
- Training scripts for CoT model
- Evaluation and comparison procedures
- Reasoning quality analysis

**Best for:** Creating a CoT version of Centaur

---

## 🚀 Quick Reference

### I want to...

**...understand what Centaur is**
→ Read `QUICK_START.md` (first 2 sections)

**...reproduce the paper's results**
→ Follow `REPRODUCTION_GUIDE.md` step-by-step

**...evaluate a new model**
→ Run: `./evaluate_new_model.sh YOUR_MODEL`

**...compare my model with Centaur**
→ Read `MODEL_COMPARISON_TABLE.md` (sections on comparison)

**...create a CoT version**
→ Follow `COT_CENTAUR_GUIDE.md` completely

**...understand the architecture**
→ Read `EXPERIMENT_ARCHITECTURE.md` (diagrams section)

**...find where results are stored**
→ Check `MODEL_COMPARISON_TABLE.md` (file mappings)

**...see what tasks are evaluated**
→ Check `MODEL_COMPARISON_TABLE.md` (task lists section)

---

## 📁 Key Files in This Directory

### Documentation
- `README.md` - Original project README
- `QUICK_START.md` - Quick start guide ⭐
- `REPRODUCTION_GUIDE.md` - Full reproduction guide
- `EXPERIMENT_ARCHITECTURE.md` - Architecture overview
- `MODEL_COMPARISON_TABLE.md` - Results reference
- `COT_CENTAUR_GUIDE.md` - CoT extension guide 🆕
- `DOCUMENTATION_INDEX.md` - This file

### Training Scripts
- `finetune.py` - Train standard Centaur
- `finetune_cot.py` - Train CoT Centaur 🆕

### Evaluation Scripts
- `test_adapter.py` - Evaluate 36 tasks
- `test_adapter_custom_metrics.py` - Evaluate 10 tasks (custom loss)
- `test_adapter_full_log_likelihoods.py` - Detailed per-trial analysis
- `merge.py` - Combine all results

### Utility Scripts
- `evaluate_new_model.sh` - **Automated evaluation pipeline** ⭐
- `train_centaur_cot.sh` - Train CoT model 🆕
- `create_cot_dataset_rulebased.py` - Generate CoT data 🆕
- `run_minimal.py` - Minimal demo

### Directories
- `results/` - All evaluation results
- `generalization/` - Generalization test scripts
- `plots/` - Visualization scripts
- `scripts/` - SLURM cluster scripts
- `camera_ready/` - Paper figures

---

## 🎯 Common Workflows

### Workflow 1: Quick Evaluation of New Model

```bash
# 1. Read the quick start
cat QUICK_START.md

# 2. Run evaluation
./evaluate_new_model.sh mistralai/Mistral-7B-Instruct-v0.3

# 3. Check results
cat results/all_data_mistralai-Mistral-7B-Instruct-v0.3.csv
```

### Workflow 2: Full Paper Reproduction

```bash
# 1. Read the reproduction guide
cat REPRODUCTION_GUIDE.md

# 2. Train Centaur (if needed)
python finetune.py \
  --model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
  --output_dir "my-centaur"

# 3. Evaluate both Centaur and Llama
./evaluate_new_model.sh ./my-centaur
./evaluate_new_model.sh unsloth/Meta-Llama-3.1-70B-bnb-4bit

# 4. Create visualizations
cd plots/
python fig2_new.py
python tab1_new.py
```

### Workflow 3: Create CoT Centaur

```bash
# 1. Read the CoT guide
cat COT_CENTAUR_GUIDE.md

# 2. Generate CoT dataset
python create_cot_dataset_rulebased.py

# 3. Train CoT model
./train_centaur_cot.sh

# 4. Evaluate and compare
./evaluate_new_model.sh ./centaur-cot-70b
# Compare with standard Centaur (see COT_CENTAUR_GUIDE.md)
```

---

## 📊 Understanding Results

### Result Files Explained

**CSV Files** (`results/all_data_*.csv`)
- Human-readable
- One row per task
- Easy to compare models
- Use with pandas

**PTH Files** (`results/custom_metrics_full_log_likelihoods_*.pth`)
- Per-trial data
- PyTorch tensors
- For statistical analysis
- Use with torch.load()

### Key Metrics

**Negative Log-Likelihood (NLL)**
- Lower is better
- Direct measure of prediction accuracy
- Centaur: ~1.30, Llama: ~1.45

**Pseudo-R²**
- Higher is better (0-1 range)
- R² = 1 - (NLL_model / NLL_random)
- Centaur: ~0.65, Llama: ~0.55

---

## 🆘 Getting Help

### Common Issues

**"Module not found"**
→ Install dependencies: `pip install -r requirements.txt`

**"Out of memory"**
→ Check `REPRODUCTION_GUIDE.md` hardware requirements

**"Results files not found"**
→ Run evaluation first: `./evaluate_new_model.sh MODEL`

**"Model not loading"**
→ Check model name format in `MODEL_COMPARISON_TABLE.md`

### Where to Look

- **Installation issues** → `requirements.txt` and `README.md`
- **Training issues** → `REPRODUCTION_GUIDE.md` (Phase 1)
- **Evaluation issues** → `REPRODUCTION_GUIDE.md` (Phase 2)
- **Results interpretation** → `MODEL_COMPARISON_TABLE.md`
- **CoT-specific issues** → `COT_CENTAUR_GUIDE.md` (Troubleshooting)

---

## 📚 Additional Resources

### Original Paper
- **Nature:** https://www.nature.com/articles/s41586-025-09215-4
- **ArXiv:** https://arxiv.org/abs/2410.20268

### Models
- **Centaur-70B:** https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter
- **Centaur-8B:** https://huggingface.co/marcelbinz/Llama-3.1-Centaur-8B-adapter

### Dataset
- **Psych-101:** https://huggingface.co/datasets/marcelbinz/Psych-101
- **Psych-101 Test:** https://huggingface.co/datasets/marcelbinz/Psych-101-test

### Contact
- **Original authors:** marcel.binz@helmholtz-munich.de
- **Issues:** GitHub issues (if applicable)

---

## 🔄 Updates

### Version History

**v1.0** (Initial)
- Original documentation
- Standard Centaur experiments
- Reproduction guides

**v1.1** (Current) 🆕
- Added CoT Centaur guide
- Added CoT training scripts
- Added comparison utilities
- Improved organization

---

## 📝 Contributing

If you extend this work:

1. Document your changes
2. Update relevant guide files
3. Add examples to this index
4. Share results with community

---

## ⭐ Recommended Reading Order

1. **QUICK_START.md** - Get oriented (15 min)
2. **EXPERIMENT_ARCHITECTURE.md** - Understand system (20 min)
3. **REPRODUCTION_GUIDE.md** - Reproduce experiments (varies)
4. **MODEL_COMPARISON_TABLE.md** - Reference as needed
5. **COT_CENTAUR_GUIDE.md** - If creating CoT version

**Total reading time:** ~1-2 hours for full understanding

---

## 🎓 Citation

```bibtex
@article{binz2024centaur,
  title={Centaur: a foundation model of human cognition},
  author={Binz, Marcel and Akata, Elif and Bethge, Matthias and ...},
  journal={Nature},
  year={2025},
  doi={10.1038/s41586-025-09215-4}
}
```

---

**Last updated:** November 2024

**Questions?** Check the relevant guide or contact the authors.

