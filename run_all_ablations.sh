#!/bin/bash
#
# Run all ablation experiments sequentially
# Usage: ./run_all_ablations.sh [--no_wandb]
#

set -e  # Exit on error

cd "$(dirname "$0")"

# Parse arguments
WANDB_FLAG=""
if [[ "$1" == "--no_wandb" ]]; then
    WANDB_FLAG="--no_wandb"
    echo "W&B logging disabled"
fi

echo "================================================================================"
echo "RUNNING ALL ABLATION EXPERIMENTS"
echo "================================================================================"
echo ""
echo "This will run 4 ablation experiments sequentially:"
echo "  1. Layer normalization ablation (no RMSNorm)"
echo "  2. Pre-norm vs Post-norm"
echo "  3. Position embeddings ablation (NoPE)"
echo "  4. SwiGLU vs SiLU"
echo ""
echo "Each experiment takes ~2 hours (40K iterations)"
echo "Total time: ~8 hours"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Track results
declare -A RESULTS

# Function to run experiment
run_experiment() {
    local ablation=$1
    local lr=$2
    local name=$3
    
    echo ""
    echo "================================================================================"
    echo "EXPERIMENT: $name"
    echo "Ablation: $ablation"
    echo "Learning rate: $lr"
    echo "================================================================================"
    echo ""
    
    if uv run python experiments/ablations.py \
        --ablation "$ablation" \
        --learning_rate "$lr" \
        --device cuda \
        $WANDB_FLAG; then
        RESULTS["$name"]="✅ SUCCESS"
        echo ""
        echo "✅ $name completed successfully!"
    else
        RESULTS["$name"]="❌ FAILED"
        echo ""
        echo "❌ $name failed!"
        echo "Continuing with next experiment..."
    fi
}

# Run all experiments
run_experiment "layer_norm" "1e-3" "Layer Norm Ablation (lr=1e-3)"
run_experiment "layer_norm" "3e-4" "Layer Norm Ablation (lr=3e-4)"
run_experiment "pre_norm" "1e-3" "Post-Norm (lr=1e-3)"
run_experiment "no_pos_emb" "1e-3" "No Position Embeddings (NoPE)"
run_experiment "swiglu" "1e-3" "SwiGLU vs SiLU"

# Print summary
echo ""
echo "================================================================================"
echo "ABLATION EXPERIMENTS SUMMARY"
echo "================================================================================"
for name in "${!RESULTS[@]}"; do
    echo "$name: ${RESULTS[$name]}"
done
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - checkpoints/ablations/"
echo "  - W&B project: cs336-ablations"
echo ""
echo "🎉 All experiments completed!"

