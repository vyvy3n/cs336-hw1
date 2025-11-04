#!/bin/bash
# Run all learning rate experiments
# Usage: bash experiments/run_all_experiments.sh [device]

set -e  # Exit on error

DEVICE=${1:-cuda}
echo "Running experiments on device: $DEVICE"

# Check if data exists
if [ ! -f "data/tinystories_train_tokens.npy" ]; then
    echo "Error: TinyStories dataset not found!"
    echo "Please run: uv run python scripts/prepare_tinystories.py --vocab_size 10000"
    exit 1
fi

# Create log directory
mkdir -p logs

echo "========================================"
echo "Starting Learning Rate Experiments"
echo "========================================"
echo "Device: $DEVICE"
echo "Start time: $(date)"
echo ""

# Grid sweep
echo "========================================"
echo "Part 1: Grid Sweep"
echo "========================================"
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device $DEVICE \
    2>&1 | tee logs/grid_sweep_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✓ Grid sweep complete!"
echo ""

# Stability sweep
echo "========================================"
echo "Part 2: Stability Sweep"
echo "========================================"
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device $DEVICE \
    2>&1 | tee logs/stability_sweep_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✓ Stability sweep complete!"
echo ""

echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Check your W&B dashboard for results:"
echo "https://wandb.ai"
echo ""
echo "Logs saved to: logs/"
echo "Checkpoints saved to: checkpoints/lr_sweep/"

