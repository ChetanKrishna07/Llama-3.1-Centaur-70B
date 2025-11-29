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

