# Creating Chain-of-Thought (CoT) Centaur

## Overview

This guide walks you through creating **Centaur-CoT**, a version of Centaur that generates reasoning steps before making predictions. This allows you to:

1. Improve prediction accuracy through explicit reasoning
2. Gain interpretability into the model's decision-making process
3. Compare reasoning-based vs. direct prediction approaches

---

## Architecture Comparison

```
Standard Centaur:
Input: "You see A, B, C. You choose"
Output: "<<B>>"

CoT Centaur:
Input: "You see A, B, C. You choose"
Output: "Let me think: Based on past rewards, B seems promising. <<B>>"
```

---

## Implementation Plan

### Phase 1: Data Preparation (Create CoT Training Dataset)
### Phase 2: Model Training (Fine-tune on CoT Data)
### Phase 3: Evaluation (Compare with Standard Centaur)
### Phase 4: Analysis (Reasoning Quality & Performance)

---

# Phase 1: Data Preparation

## Option A: GPT-4/Claude Generated Reasoning (High Quality)

Create `create_cot_dataset_gpt4.py`:

```python
"""
Generate Chain-of-Thought reasoning for Psych-101 dataset using GPT-4
This creates higher quality reasoning but requires API costs
"""

from datasets import load_dataset
from openai import OpenAI
import json
import re
from tqdm import tqdm
import time

client = OpenAI()  # Set OPENAI_API_KEY environment variable

def extract_choice(text):
    """Extract the choice wrapped in << >> tokens"""
    choice_match = re.search(r'<<(.+?)>>', text)
    if not choice_match:
        return None, text, ""
    
    choice = choice_match.group(1)
    before_choice = text[:choice_match.start()]
    after_choice = text[choice_match.end():]
    return choice, before_choice, after_choice

def generate_reasoning(context, choice, experiment_name, retry=3):
    """Generate plausible reasoning using GPT-4"""
    
    # Determine task type from experiment name
    task_type = "general cognitive task"
    if any(x in experiment_name.lower() for x in ['bandit', 'reward', 'decision']):
        task_type = "reward-based decision task"
    elif any(x in experiment_name.lower() for x in ['memory', 'recall', 'working']):
        task_type = "memory task"
    elif any(x in experiment_name.lower() for x in ['risk', 'gamble', 'lottery']):
        task_type = "risky decision task"
    elif any(x in experiment_name.lower() for x in ['logic', 'reason', 'inference']):
        task_type = "logical reasoning task"
    elif any(x in experiment_name.lower() for x in ['category', 'similar', 'concept']):
        task_type = "categorization task"
    
    prompt = f"""You are simulating human reasoning in a {task_type}.

Context:
{context}

The participant chose: {choice}

Generate a brief, plausible chain-of-thought reasoning (1-2 sentences max) that explains why a human might make this choice. Focus on cognitive processes like:
- Memory retrieval and pattern recognition
- Reward expectation and risk assessment
- Exploration vs exploitation trade-offs
- Logical inference and deduction
- Similarity judgments and categorization

Requirements:
1. Keep it concise (1-2 sentences)
2. Start with "Let me think: "
3. Make it sound human and natural
4. Don't overthink - simple reasoning is fine
5. Be specific to the choice made

Example: "Let me think: Option B has given good rewards recently and seems less risky. "

Your reasoning:"""

    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=80,
                timeout=30
            )
            
            reasoning = response.choices[0].message.content.strip()
            
            # Ensure it starts with "Let me think:"
            if not reasoning.startswith("Let me think:"):
                reasoning = "Let me think: " + reasoning
            
            # Ensure it ends with proper punctuation
            if not reasoning.endswith(('.', '!', '?')):
                reasoning += "."
            
            return reasoning
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Fallback to generic reasoning
                return "Let me think: Considering the context and my experience. "
    
    return "Let me think: Based on the situation. "

def create_cot_dataset(output_file='psych101_cot_train.jsonl', 
                       max_examples=None, 
                       start_from=0):
    """
    Create CoT version of Psych-101 dataset
    
    Args:
        output_file: Where to save the CoT dataset
        max_examples: Limit number of examples (for testing)
        start_from: Skip first N examples (for resuming)
    """
    
    # Load original dataset
    print("Loading Psych-101 dataset...")
    dataset = load_dataset("marcelbinz/Psych-101")
    train_data = dataset['train']
    
    if max_examples:
        train_data = train_data.select(range(start_from, min(start_from + max_examples, len(train_data))))
    else:
        train_data = train_data.select(range(start_from, len(train_data)))
    
    print(f"Processing {len(train_data)} examples...")
    
    cot_data = []
    failed = 0
    
    for i, example in enumerate(tqdm(train_data)):
        try:
            text = example['text']
            experiment = example['experiment']
            
            # Extract choice
            choice, before_choice, after_choice = extract_choice(text)
            
            if choice is None:
                # No choice found, keep original
                cot_data.append(example)
                failed += 1
                continue
            
            # Generate reasoning
            reasoning = generate_reasoning(before_choice, choice, experiment)
            
            # Reconstruct with reasoning
            cot_text = f"{before_choice}{reasoning} You choose <<{choice}>>.{after_choice}"
            
            cot_data.append({
                'text': cot_text,
                'experiment': experiment
            })
            
            # Save incrementally every 100 examples
            if (i + 1) % 100 == 0:
                with open(output_file, 'w') as f:
                    for item in cot_data:
                        f.write(json.dumps(item) + '\n')
                print(f"\nSaved checkpoint at {i + 1} examples")
            
            # Rate limiting
            time.sleep(0.5)  # Avoid hitting API rate limits
            
        except Exception as e:
            print(f"\nError on example {i + start_from}: {e}")
            # Fallback to original
            cot_data.append(example)
            failed += 1
    
    # Final save
    print(f"\nSaving final dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for item in cot_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n{'='*60}")
    print(f"CoT dataset created successfully!")
    print(f"Total examples: {len(cot_data)}")
    print(f"Failed/skipped: {failed}")
    print(f"Success rate: {((len(cot_data) - failed) / len(cot_data)) * 100:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="psych101_cot_train.jsonl")
    parser.add_argument("--max_examples", type=int, default=None, 
                       help="Limit examples (for testing)")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Skip first N examples (for resuming)")
    args = parser.parse_args()
    
    create_cot_dataset(
        output_file=args.output,
        max_examples=args.max_examples,
        start_from=args.start_from
    )
```

**Usage:**

```bash
# Test with 100 examples first
python create_cot_dataset_gpt4.py --max_examples 100 --output test_cot.jsonl

# Run on full dataset (will take hours and cost $$$)
export OPENAI_API_KEY="your-key-here"
python create_cot_dataset_gpt4.py --output psych101_cot_train.jsonl

# Resume if interrupted (example: start from example 5000)
python create_cot_dataset_gpt4.py --start_from 5000 --output psych101_cot_train.jsonl
```

---

## Option B: Rule-Based Reasoning (Fast, Free)

Create `create_cot_dataset_rulebased.py`:

```python
"""
Generate Chain-of-Thought reasoning using rule-based templates
This is fast and free but lower quality than GPT-4
"""

from datasets import load_dataset
import re
import json
from tqdm import tqdm
import random

def extract_choice(text):
    """Extract the choice wrapped in << >> tokens"""
    choice_match = re.search(r'<<(.+?)>>', text)
    if not choice_match:
        return None, text, ""
    
    choice = choice_match.group(1)
    before_choice = text[:choice_match.start()]
    after_choice = text[choice_match.end():]
    return choice, before_choice, after_choice

def generate_reasoning_template(choice, experiment_name, context):
    """Generate reasoning based on task type"""
    
    experiment_lower = experiment_name.lower()
    
    # Task-specific reasoning templates
    if any(x in experiment_lower for x in ['bandit', 'reward', 'slot', 'arm']):
        templates = [
            f"Let me think: Based on past rewards, this option has been promising. ",
            f"Let me think: This choice balances exploration and exploitation well. ",
            f"Let me think: The reward pattern suggests this is a good option. ",
            f"Let me think: I should try this option to maximize my rewards. ",
        ]
    
    elif any(x in experiment_lower for x in ['memory', 'recall', 'recognition', 'working']):
        templates = [
            f"Let me think: Searching my memory for this pattern. ",
            f"Let me think: I recall seeing this sequence before. ",
            f"Let me think: This matches what I remember. ",
            f"Let me think: My working memory suggests this answer. ",
        ]
    
    elif any(x in experiment_lower for x in ['risk', 'gamble', 'lottery', 'bet']):
        templates = [
            f"Let me think: Weighing the risk against potential reward. ",
            f"Let me think: This option balances risk and expected value. ",
            f"Let me think: Considering my risk tolerance, this seems reasonable. ",
            f"Let me think: The expected value makes this attractive. ",
        ]
    
    elif any(x in experiment_lower for x in ['logic', 'reason', 'inference', 'deduc']):
        templates = [
            f"Let me think: Applying logical rules to solve this. ",
            f"Let me think: The logical inference leads to this answer. ",
            f"Let me think: Following the reasoning chain step by step. ",
            f"Let me think: Deductively, this must be correct. ",
        ]
    
    elif any(x in experiment_lower for x in ['category', 'similar', 'concept', 'classif']):
        templates = [
            f"Let me think: Comparing features to find similarities. ",
            f"Let me think: This best matches the category pattern. ",
            f"Let me think: Based on shared features, this fits. ",
            f"Let me think: The concept mapping suggests this answer. ",
        ]
    
    elif any(x in experiment_lower for x in ['attention', 'go', 'nogo', 'response']):
        templates = [
            f"Let me think: Focusing on the relevant stimuli. ",
            f"Let me think: This requires careful attention. ",
            f"Let me think: Inhibiting the wrong response. ",
            f"Let me think: Monitoring and adjusting my response. ",
        ]
    
    elif any(x in experiment_lower for x in ['learn', 'acquire', 'adapt']):
        templates = [
            f"Let me think: Learning from previous trials. ",
            f"Let me think: Adapting based on feedback. ",
            f"Let me think: The pattern I've learned suggests this. ",
            f"Let me think: My experience points to this choice. ",
        ]
    
    elif any(x in experiment_lower for x in ['social', 'trust', 'cooperat']):
        templates = [
            f"Let me think: Considering what others might do. ",
            f"Let me think: Trust and cooperation matter here. ",
            f"Let me think: Taking the social context into account. ",
            f"Let me think: This balances self-interest and fairness. ",
        ]
    
    elif any(x in experiment_lower for x in ['time', 'temporal', 'delay']):
        templates = [
            f"Let me think: Weighing immediate vs delayed rewards. ",
            f"Let me think: Considering the timing of outcomes. ",
            f"Let me think: My time preference suggests this choice. ",
            f"Let me think: The temporal dynamics favor this option. ",
        ]
    
    elif any(x in experiment_lower for x in ['spatial', 'location', 'position']):
        templates = [
            f"Let me think: Considering the spatial layout. ",
            f"Let me think: The position information guides this choice. ",
            f"Let me think: Using spatial memory to decide. ",
            f"Let me think: The location pattern suggests this. ",
        ]
    
    else:
        # Generic templates for unknown task types
        templates = [
            f"Let me think: Based on the context and my experience. ",
            f"Let me think: Considering all available information. ",
            f"Let me think: This seems like the best option. ",
            f"Let me think: Following my intuition and past performance. ",
            f"Let me think: Analyzing the situation carefully. ",
        ]
    
    # Randomly select a template for variety
    return random.choice(templates)

def create_cot_dataset(output_file='psych101_cot_train_rulebased.jsonl'):
    """Create CoT dataset using rule-based reasoning"""
    
    # Load original dataset
    print("Loading Psych-101 dataset...")
    dataset = load_dataset("marcelbinz/Psych-101")
    train_data = dataset['train']
    
    print(f"Processing {len(train_data)} examples...")
    
    cot_data = []
    failed = 0
    
    for example in tqdm(train_data):
        try:
            text = example['text']
            experiment = example['experiment']
            
            # Extract choice
            choice, before_choice, after_choice = extract_choice(text)
            
            if choice is None:
                # No choice found, keep original
                cot_data.append(example)
                failed += 1
                continue
            
            # Generate reasoning
            reasoning = generate_reasoning_template(choice, experiment, before_choice)
            
            # Reconstruct with reasoning
            cot_text = f"{before_choice}{reasoning}You choose <<{choice}>>.{after_choice}"
            
            cot_data.append({
                'text': cot_text,
                'experiment': experiment
            })
            
        except Exception as e:
            print(f"\nError on example: {e}")
            cot_data.append(example)
            failed += 1
    
    # Save
    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for item in cot_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n{'='*60}")
    print(f"CoT dataset created successfully!")
    print(f"Total examples: {len(cot_data)}")
    print(f"Failed/skipped: {failed}")
    print(f"Success rate: {((len(cot_data) - failed) / len(cot_data)) * 100:.1f}%")
    print(f"{'='*60}")
    
    # Show some examples
    print("\nSample CoT examples:")
    for i in range(min(3, len(cot_data))):
        print(f"\n--- Example {i+1} ---")
        print(f"Experiment: {cot_data[i]['experiment']}")
        print(f"Text: {cot_data[i]['text'][:200]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="psych101_cot_train_rulebased.jsonl")
    args = parser.parse_args()
    
    create_cot_dataset(output_file=args.output)
```

**Usage:**

```bash
# Generate rule-based CoT dataset (fast, ~10-20 minutes)
python create_cot_dataset_rulebased.py --output psych101_cot_train.jsonl
```

---

# Phase 2: Model Training

Create `finetune_cot.py`:

```python
"""
Fine-tune Llama 3.1 on CoT-augmented Psych-101 dataset
This creates Centaur-CoT
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, set_seed
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth import FastLanguageModel
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)

@dataclass
class DataTrainingArguments:
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=32768)
    cot_data_path: str = field(
        default="psych101_cot_train.jsonl",
        metadata={"help": "Path to CoT-augmented training data"}
    )

def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load CoT dataset
    print(f"Loading CoT dataset from {data_args.cot_data_path}...")
    train_dataset = load_dataset('json', data_files={'train': [data_args.cot_data_path]})['train'].shuffle()
    print(f"Loaded {len(train_dataset)} training examples")
    
    # Load standard test set (keep evaluation on original format)
    eval_datasets = load_dataset("marcelbinz/Psych-101-test")['test']
    print(f"Loaded {len(eval_datasets)} evaluation examples")

    # Model
    print(f"Loading base model: {model_args.model_name_or_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = data_args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj",
            "up_proj", "down_proj",
        ],
        lora_alpha = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = training_args.seed,
        use_rslora = True,
        loftq_config = None,
    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    # Data collator (still predict << >> tokens)
    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id, 
        instruction_template=r_id, 
        tokenizer=tokenizer
    )

    # Trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_datasets,
        dataset_text_field = data_args.dataset_text_field,
        max_seq_length = data_args.max_seq_length,
        dataset_num_proc = 8,
        data_collator = collator,
        args = UnslothTrainingArguments(
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            gradient_accumulation_steps = training_args.gradient_accumulation_steps,
            warmup_steps = training_args.warmup_steps,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            embedding_learning_rate = training_args.learning_rate / 10,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            log_level = training_args.log_level,
            logging_strategy = training_args.logging_strategy,
            logging_steps = training_args.logging_steps,
            evaluation_strategy = training_args.evaluation_strategy,
            eval_steps = training_args.eval_steps,
            save_strategy = training_args.save_strategy,
            save_steps = training_args.save_steps,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            seed = training_args.seed,
            output_dir = training_args.output_dir,
        ),
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=None)

    # Save final model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {training_args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
```

**Training Script** `train_centaur_cot.sh`:

```bash
#!/bin/bash

# Train Centaur-CoT
python finetune_cot.py \
  --seed 100 \
  --model_name_or_path "unsloth/Meta-Llama-3.1-70B-bnb-4bit" \
  --cot_data_path "psych101_cot_train.jsonl" \
  --max_seq_len 32768 \
  --num_train_epochs 5 \
  --log_level "info" \
  --logging_strategy "steps" \
  --logging_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 999999 \
  --save_strategy "steps" \
  --save_steps 100 \
  --learning_rate 5e-5 \
  --optim "adamw_8bit" \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --output_dir "centaur-cot-70b" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32
```

**Usage:**

```bash
chmod +x train_centaur_cot.sh
./train_centaur_cot.sh
```

---

# Phase 3: Evaluation

Use the existing evaluation pipeline:

```bash
# Evaluate Centaur-CoT
./evaluate_new_model.sh ./centaur-cot-70b

# This will create:
# - results/all_data_centaur-cot-70b.csv
# - results/custom_metrics_full_log_likelihoods_centaur-cot-70b.pth
```

---

# Phase 4: Analysis & Comparison

## Script 1: Quantitative Comparison

Create `compare_cot_vs_standard.py`:

```python
"""
Compare Centaur-CoT vs Standard Centaur performance
"""

import torch
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(model_name):
    """Load per-trial results"""
    try:
        return torch.load(f'results/custom_metrics_full_log_likelihoods_{model_name}.pth')
    except FileNotFoundError:
        print(f"Error: Could not find results for {model_name}")
        return None

def compare_models(standard_name, cot_name):
    """Compare two models statistically"""
    
    print("="*60)
    print("CENTAUR vs CENTAUR-CoT COMPARISON")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    centaur_standard = load_results(standard_name)
    centaur_cot = load_results(cot_name)
    
    if centaur_standard is None or centaur_cot is None:
        print("Error: Could not load results")
        return
    
    # Per-task comparison
    results = []
    all_standard_nll = []
    all_cot_nll = []
    
    print("\nAnalyzing per-task performance...")
    for task_name in centaur_standard.keys():
        if task_name not in centaur_cot:
            print(f"Warning: {task_name} not in CoT results, skipping")
            continue
        
        std_nll = centaur_standard[task_name].numpy()
        cot_nll = centaur_cot[task_name].numpy()
        
        # Ensure same length
        min_len = min(len(std_nll), len(cot_nll))
        std_nll = std_nll[:min_len]
        cot_nll = cot_nll[:min_len]
        
        # Statistical test (paired t-test)
        t_stat, p_value = stats.ttest_rel(std_nll, cot_nll)
        
        # Calculate improvement
        improvement = ((std_nll.mean() - cot_nll.mean()) / std_nll.mean()) * 100
        
        # Effect size (Cohen's d)
        cohens_d = (std_nll.mean() - cot_nll.mean()) / np.sqrt((std_nll.std()**2 + cot_nll.std()**2) / 2)
        
        results.append({
            'task': task_name,
            'standard_nll': std_nll.mean(),
            'cot_nll': cot_nll.mean(),
            'improvement_%': improvement,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'n_trials': len(std_nll)
        })
        
        all_standard_nll.extend(std_nll.tolist())
        all_cot_nll.extend(cot_nll.tolist())
    
    df = pd.DataFrame(results)
    df = df.sort_values('improvement_%', ascending=False)
    
    # Summary statistics
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Total tasks analyzed: {len(df)}")
    print(f"Total trials: {df['n_trials'].sum()}")
    print(f"\nPerformance:")
    print(f"  Standard Centaur mean NLL: {df['standard_nll'].mean():.4f}")
    print(f"  CoT Centaur mean NLL:      {df['cot_nll'].mean():.4f}")
    print(f"  Mean improvement:          {df['improvement_%'].mean():.2f}%")
    
    print(f"\nWin/Loss Record:")
    better = (df['improvement_%'] > 0).sum()
    worse = (df['improvement_%'] < 0).sum()
    tie = (df['improvement_%'] == 0).sum()
    print(f"  CoT better:  {better}/{len(df)} ({better/len(df)*100:.1f}%)")
    print(f"  CoT worse:   {worse}/{len(df)} ({worse/len(df)*100:.1f}%)")
    print(f"  Tied:        {tie}/{len(df)}")
    
    print(f"\nStatistical Significance:")
    sig_better = ((df['improvement_%'] > 0) & (df['significant'])).sum()
    sig_worse = ((df['improvement_%'] < 0) & (df['significant'])).sum()
    print(f"  Significantly better (p<0.05): {sig_better}/{len(df)} ({sig_better/len(df)*100:.1f}%)")
    print(f"  Significantly worse (p<0.05):  {sig_worse}/{len(df)} ({sig_worse/len(df)*100:.1f}%)")
    
    # Overall statistical test
    all_standard_nll = np.array(all_standard_nll)
    all_cot_nll = np.array(all_cot_nll)
    overall_t, overall_p = stats.ttest_rel(all_standard_nll, all_cot_nll)
    print(f"\nOverall t-test:")
    print(f"  t-statistic: {overall_t:.4f}")
    print(f"  p-value: {overall_p:.4e}")
    print(f"  Result: {'CoT significantly better' if overall_p < 0.05 and overall_t > 0 else 'No significant difference' if overall_p >= 0.05 else 'CoT significantly worse'}")
    
    # Top improvements
    print("\n" + "="*60)
    print("TOP 10 IMPROVEMENTS (CoT Better)")
    print("="*60)
    top_improvements = df.nlargest(10, 'improvement_%')
    for idx, row in top_improvements.iterrows():
        sig = "*" if row['significant'] else ""
        print(f"{row['task']:40s} {row['improvement_%']:+6.2f}% (p={row['p_value']:.4f}){sig}")
    
    # Top regressions
    print("\n" + "="*60)
    print("TOP 10 REGRESSIONS (CoT Worse)")
    print("="*60)
    top_regressions = df.nsmallest(10, 'improvement_%')
    for idx, row in top_regressions.iterrows():
        sig = "*" if row['significant'] else ""
        print(f"{row['task']:40s} {row['improvement_%']:+6.2f}% (p={row['p_value']:.4f}){sig}")
    
    # Save detailed results
    output_file = 'results/cot_comparison_detailed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")
    
    # Create visualization
    create_comparison_plots(df, standard_name, cot_name)
    
    return df

def create_comparison_plots(df, standard_name, cot_name):
    """Create visualization of comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Improvement distribution
    ax = axes[0, 0]
    ax.hist(df['improvement_%'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(df['improvement_%'].mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {df["improvement_%"].mean():.2f}%')
    ax.set_xlabel('Improvement (%)')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Distribution of Performance Improvement')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Scatter plot
    ax = axes[0, 1]
    colors = ['red' if x < 0 else 'green' for x in df['improvement_%']]
    ax.scatter(df['standard_nll'], df['cot_nll'], c=colors, alpha=0.6)
    min_val = min(df['standard_nll'].min(), df['cot_nll'].min())
    max_val = max(df['standard_nll'].max(), df['cot_nll'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Equal performance')
    ax.set_xlabel('Standard Centaur NLL')
    ax.set_ylabel('CoT Centaur NLL')
    ax.set_title('Per-Task Performance Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Top/Bottom tasks
    ax = axes[1, 0]
    top_bottom = pd.concat([df.nlargest(10, 'improvement_%'), df.nsmallest(10, 'improvement_%')])
    colors_bar = ['green' if x > 0 else 'red' for x in top_bottom['improvement_%']]
    y_pos = range(len(top_bottom))
    ax.barh(y_pos, top_bottom['improvement_%'], color=colors_bar, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:30] for t in top_bottom['task']], fontsize=8)
    ax.set_xlabel('Improvement (%)')
    ax.set_title('Top 10 Improvements & Regressions')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(alpha=0.3, axis='x')
    
    # Plot 4: P-value vs Improvement
    ax = axes[1, 1]
    colors_sig = ['red' if p < 0.05 else 'gray' for p in df['p_value']]
    ax.scatter(df['improvement_%'], -np.log10(df['p_value']), c=colors_sig, alpha=0.6)
    ax.axhline(-np.log10(0.05), color='blue', linestyle='--', linewidth=2, label='p=0.05')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Improvement (%)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance vs Effect Size')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_plot = 'results/cot_comparison_plots.pdf'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_plot}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", type=str, 
                       default="marcelbinz-Llama-3.1-Centaur-70B-adapter",
                       help="Standard Centaur model name")
    parser.add_argument("--cot", type=str,
                       default="centaur-cot-70b",
                       help="CoT Centaur model name")
    args = parser.parse_args()
    
    df = compare_models(args.standard, args.cot)
```

**Usage:**

```bash
python compare_cot_vs_standard.py \
  --standard marcelbinz-Llama-3.1-Centaur-70B-adapter \
  --cot centaur-cot-70b
```

---

## Script 2: Qualitative Analysis (Inspect Reasoning)

Create `inspect_cot_reasoning.py`:

```python
"""
Inspect the reasoning generated by Centaur-CoT
"""

from unsloth import FastLanguageModel
import transformers
from datasets import load_dataset
import re

def extract_reasoning(text):
    """Extract reasoning from generated text"""
    # Look for "Let me think:" pattern
    match = re.search(r'Let me think:(.+?)You choose <<', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def test_cot_model(model_path, num_examples=20):
    """Test CoT model on sample prompts"""
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 32768,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,  # Allow enough for reasoning + choice
        temperature=0.1,  # Low temperature for consistency
        do_sample=True,
    )
    
    # Load test examples from different domains
    print("Loading test examples...")
    dataset = load_dataset("marcelbinz/Psych-101-test")
    test_data = dataset['test']
    
    # Sample from different experiments
    experiments_seen = set()
    test_examples = []
    
    for example in test_data:
        exp_name = example['experiment'].split('_')[0]  # Get base experiment name
        if exp_name not in experiments_seen and len(test_examples) < num_examples:
            test_examples.append(example)
            experiments_seen.add(exp_name)
    
    print(f"\nTesting on {len(test_examples)} examples from different tasks\n")
    print("="*80)
    
    reasoning_quality = {
        'has_reasoning': 0,
        'no_reasoning': 0,
        'total': 0
    }
    
    for i, example in enumerate(test_examples):
        text = example['text']
        experiment = example['experiment']
        
        # Extract the prompt (before the choice)
        match = re.search(r'(.+?)You choose <<', text, re.DOTALL)
        if not match:
            continue
        
        prompt = match.group(1) + "You choose"
        actual_choice = re.search(r'<<(.+?)>>', text)
        actual_choice = actual_choice.group(1) if actual_choice else "?"
        
        # Generate with CoT
        print(f"\n{'='*80}")
        print(f"Example {i+1} - Task: {experiment}")
        print(f"{'='*80}")
        print(f"\nPrompt:\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}")
        print(f"\nActual human choice: {actual_choice}")
        
        result = pipe(prompt)[0]['generated_text']
        generated = result[len(prompt):]
        
        print(f"\nModel generation:\n{generated}")
        
        # Extract reasoning
        reasoning = extract_reasoning(prompt + generated)
        if reasoning:
            print(f"\nExtracted reasoning:\n>>> {reasoning}")
            reasoning_quality['has_reasoning'] += 1
        else:
            print("\n[No reasoning pattern detected]")
            reasoning_quality['no_reasoning'] += 1
        
        reasoning_quality['total'] += 1
        
        # Extract predicted choice
        pred_match = re.search(r'<<(.+?)>>', generated)
        if pred_match:
            pred_choice = pred_match.group(1)
            match_str = "✓ MATCH" if pred_choice.strip() == actual_choice.strip() else "✗ MISMATCH"
            print(f"\nPredicted choice: {pred_choice} {match_str}")
    
    print("\n" + "="*80)
    print("REASONING QUALITY SUMMARY")
    print("="*80)
    print(f"Examples with reasoning: {reasoning_quality['has_reasoning']}/{reasoning_quality['total']} "
          f"({reasoning_quality['has_reasoning']/reasoning_quality['total']*100:.1f}%)")
    print(f"Examples without reasoning: {reasoning_quality['no_reasoning']}/{reasoning_quality['total']} "
          f"({reasoning_quality['no_reasoning']/reasoning_quality['total']*100:.1f}%)")

def test_specific_prompts(model_path):
    """Test on hand-crafted prompts"""
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 32768,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,
    )
    
    test_prompts = [
        {
            'task': 'Multi-armed bandit',
            'prompt': """You have four slot machines to choose from. Here's your experience so far:
- Machine A: Played 10 times, average reward 7 points
- Machine B: Played 15 times, average reward 5 points  
- Machine C: Played 5 times, average reward 9 points
- Machine D: Never played

Which machine do you play next? You choose"""
        },
        {
            'task': 'Memory recall',
            'prompt': """You saw the following sequence of letters:
A, B, C, D, B, C, E, B

What letter appeared most frequently? You choose"""
        },
        {
            'task': 'Risky choice',
            'prompt': """You must choose between two options:
- Option A: Receive $50 with 100% certainty
- Option B: Receive $150 with 40% chance, or $0 with 60% chance

Which option do you choose? You choose"""
        },
    ]
    
    print("\n" + "="*80)
    print("TESTING ON HAND-CRAFTED PROMPTS")
    print("="*80)
    
    for i, test in enumerate(test_prompts):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: {test['task']}")
        print(f"{'='*80}")
        print(f"\nPrompt:\n{test['prompt']}")
        
        result = pipe(test['prompt'])[0]['generated_text']
        generated = result[len(test['prompt']):]
        
        print(f"\nModel generation:\n{generated}")
        
        reasoning = extract_reasoning(test['prompt'] + generated)
        if reasoning:
            print(f"\nExtracted reasoning:\n>>> {reasoning}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Path to CoT model")
    parser.add_argument("--num_examples", type=int, default=20,
                       help="Number of test examples")
    parser.add_argument("--custom_prompts", action="store_true",
                       help="Test on hand-crafted prompts")
    args = parser.parse_args()
    
    if args.custom_prompts:
        test_specific_prompts(args.model)
    else:
        test_cot_model(args.model, args.num_examples)
```

**Usage:**

```bash
# Test on dataset examples
python inspect_cot_reasoning.py --model ./centaur-cot-70b --num_examples 20

# Test on custom prompts
python inspect_cot_reasoning.py --model ./centaur-cot-70b --custom_prompts
```

---

# Complete Workflow

Create `run_cot_experiment.sh`:

```bash
#!/bin/bash
# Complete workflow for creating and evaluating Centaur-CoT

set -e

echo "=================================================="
echo "CENTAUR-CoT COMPLETE WORKFLOW"
echo "=================================================="

# Step 1: Generate CoT dataset
echo ""
echo "[1/5] Generating CoT dataset..."
echo "Choose: (1) Rule-based (fast, free) or (2) GPT-4 (slow, costs money)"
read -p "Enter choice [1 or 2]: " choice

if [ "$choice" == "1" ]; then
    python create_cot_dataset_rulebased.py --output psych101_cot_train.jsonl
elif [ "$choice" == "2" ]; then
    read -p "Enter your OpenAI API key: " api_key
    export OPENAI_API_KEY="$api_key"
    python create_cot_dataset_gpt4.py --output psych101_cot_train.jsonl
else
    echo "Invalid choice, using rule-based"
    python create_cot_dataset_rulebased.py --output psych101_cot_train.jsonl
fi

# Step 2: Train CoT model
echo ""
echo "[2/5] Training Centaur-CoT (this will take 48-96 hours)..."
read -p "Start training now? [y/n]: " train_now

if [ "$train_now" == "y" ]; then
    ./train_centaur_cot.sh
else
    echo "Skipping training. Run ./train_centaur_cot.sh manually when ready."
    exit 0
fi

# Step 3: Evaluate CoT model
echo ""
echo "[3/5] Evaluating Centaur-CoT..."
./evaluate_new_model.sh ./centaur-cot-70b

# Step 4: Compare with standard Centaur
echo ""
echo "[4/5] Comparing CoT vs Standard Centaur..."
python compare_cot_vs_standard.py \
  --standard marcelbinz-Llama-3.1-Centaur-70B-adapter \
  --cot centaur-cot-70b

# Step 5: Inspect reasoning quality
echo ""
echo "[5/5] Inspecting reasoning quality..."
python inspect_cot_reasoning.py --model ./centaur-cot-70b --num_examples 20

echo ""
echo "=================================================="
echo "EXPERIMENT COMPLETE!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - results/all_data_centaur-cot-70b.csv"
echo "  - results/cot_comparison_detailed.csv"
echo "  - results/cot_comparison_plots.pdf"
```

Make executable:

```bash
chmod +x run_cot_experiment.sh
chmod +x train_centaur_cot.sh
```

---

# Expected Outcomes & Hypotheses

## Hypothesis 1: Task-Specific Performance

**Prediction:** CoT will help most on:
- Complex reasoning tasks (logic, planning)
- Tasks requiring multi-step inference
- Tasks with temporal dependencies

**Prediction:** CoT may not help (or hurt) on:
- Simple perceptual tasks
- Single-step memory recall
- Rapid reaction time tasks

## Hypothesis 2: Overall Performance

**Conservative:** CoT matches standard Centaur (no significant difference)

**Optimistic:** CoT improves by 2-5% on average

**Pessimistic:** CoT slightly worse due to added complexity

## Hypothesis 3: Interpretability

Even if performance is similar, CoT provides:
- Explainable decisions
- Insight into model's "reasoning process"
- Better alignment with human thought processes

---

# Troubleshooting

## Issue: Out of Memory During Training

**Solution:**
```bash
# Reduce batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 64  # Increase this
```

## Issue: CoT Dataset Generation Too Slow (GPT-4)

**Solution:**
1. Use rule-based approach instead
2. Generate subset first (--max_examples 10000)
3. Use faster model (gpt-3.5-turbo)

## Issue: Model Not Generating Reasoning

**Possible causes:**
1. Training data doesn't have consistent format
2. Not enough training epochs
3. Evaluation prompts don't match training format

**Solution:** Check training data format and increase epochs to 7-10

## Issue: Reasoning Quality is Poor

**Solution:**
1. Use GPT-4 for dataset generation instead of rules
2. Add more diverse reasoning templates
3. Fine-tune for more epochs
4. Use larger base model (70B vs 8B)

---

# Next Steps & Extensions

## 1. Multi-Step Reasoning

Extend to multiple reasoning steps:
```
"Let me think: Step 1: Consider past rewards. Step 2: Balance exploration. Step 3: Choose highest expected value. You choose <<B>>."
```

## 2. Self-Consistency

Generate multiple reasoning paths and ensemble:
- Generate 5 different reasonings
- Take majority vote on final choice
- Compare consistency across paths

## 3. Reasoning Verification

Train a separate model to:
- Judge reasoning quality
- Identify logical errors
- Suggest improvements

## 4. Human Evaluation

Recruit human judges to rate:
- Reasoning plausibility
- Alignment with human thinking
- Usefulness for explanation

---

# Citation

If you use this CoT approach in your research, please cite:

```bibtex
@article{binz2024centaur,
  title={Centaur: a foundation model of human cognition},
  author={Binz, Marcel and Akata, Elif and Bethge, Matthias and ...},
  journal={Nature},
  year={2025}
}
```

---

# Contact

For questions or issues:
- Original Centaur: marcel.binz@helmholtz-munich.de
- This CoT extension: [Your contact]

