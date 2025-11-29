#!/bin/bash
# Quick script to evaluate a new model on all Centaur benchmarks
# Usage: ./evaluate_new_model.sh <model_name_or_path>
#
# Example:
#   ./evaluate_new_model.sh mistralai/Mistral-7B-Instruct-v0.3
#   ./evaluate_new_model.sh ./my-finetuned-model

set -e  # Exit on error

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_name_or_path>"
    echo ""
    echo "Examples:"
    echo "  $0 mistralai/Mistral-7B-Instruct-v0.3"
    echo "  $0 meta-llama/Llama-2-13b-chat-hf"
    echo "  $0 ./my-finetuned-model"
    exit 1
fi

MODEL_PATH="$1"
# Create a safe filename by replacing / with -
MODEL_NAME=$(echo "$MODEL_PATH" | sed 's/\//-/g' | sed 's/\.\///')

echo "============================================="
echo "Evaluating model: $MODEL_PATH"
echo "Results will be saved with prefix: $MODEL_NAME"
echo "============================================="
echo ""

# Check if model path exists (for local models)
if [[ "$MODEL_PATH" == ./* ]] || [[ "$MODEL_PATH" == /* ]]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Local model directory not found: $MODEL_PATH"
        exit 1
    fi
fi

# Step 1: Standard metrics on held-out participants
echo "[1/6] Running standard evaluation on held-out participants..."
echo "      This evaluates on 36 tasks and may take 2-4 hours on a 80GB GPU"
python test_adapter.py --model "$MODEL_PATH"
echo "✓ Results saved to: results/${MODEL_NAME}.csv"
echo ""

# Step 2: Custom metrics on held-out participants
echo "[2/6] Running custom metrics evaluation on held-out participants..."
echo "      This evaluates on 10 additional tasks"
python test_adapter_custom_metrics.py --model "$MODEL_PATH"
echo "✓ Results saved to: results/custom_metrics_${MODEL_NAME}.csv"
echo ""

# Step 3: Full log-likelihoods for detailed analysis
echo "[3/6] Computing full log-likelihoods for all tasks..."
echo "      This computes per-trial losses for statistical analysis"
python test_adapter_full_log_likelihoods.py --model "$MODEL_PATH"
echo "✓ Results saved to: results/custom_metrics_full_log_likelihoods_${MODEL_NAME}.pth"
echo ""

# Step 4: Generalization to new experiments
echo "[4/6] Testing generalization to new experiments..."
cd generalization/
python generalization.py --model "$MODEL_PATH"
echo "✓ Results saved to: generalization/results/${MODEL_NAME}.csv"
echo ""

# Step 5: Generalization with custom metrics
echo "[5/6] Testing generalization with custom metrics..."
python generalization_custom_metrics.py --model "$MODEL_PATH"
echo "✓ Results saved to: generalization/results/custom_metrics_${MODEL_NAME}.csv"
cd ..
echo ""

# Step 6: Merge all results
echo "[6/6] Merging all results into single file..."
python merge.py --model "$MODEL_NAME"
echo "✓ Final results saved to: results/all_data_${MODEL_NAME}.csv"
echo ""

echo "============================================="
echo "✓ Evaluation complete!"
echo "============================================="
echo ""
echo "Results summary:"
echo "  - Main results: results/all_data_${MODEL_NAME}.csv"
echo "  - Detailed per-trial data: results/custom_metrics_full_log_likelihoods_${MODEL_NAME}.pth"
echo ""
echo "To compare with Centaur and Llama baselines:"
echo "  cd plots/"
echo "  # Edit the plotting scripts to include your model"
echo "  python fig2_new.py"
echo "  python tab1_new.py"
echo ""

