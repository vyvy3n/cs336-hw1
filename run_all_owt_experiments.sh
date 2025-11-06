#!/bin/bash
# Run ALL OpenWebText experiments
# - Single OWT training
# - Compare TinyStories vs OWT
# - Learning rate sweep
# - Batch size sweep

set -e  # Exit on error

echo "========================================================================"
echo "🚀 Running ALL OpenWebText Experiments"
echo "========================================================================"
echo ""
echo "This will run:"
echo "  1. Single OWT training (40K iters, eval every 500)"
echo "  2. Compare TinyStories vs OWT (40K iters each, eval every 500)"
echo "  3. Learning rate sweep (4 LRs × 40K iters)"
echo "  4. Batch size sweep (2 batch sizes × 40K iters)"
echo ""
echo "⏱️  Estimated total time: ~50-60 hours on H100"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi
echo ""

# Configuration
MAX_ITERS=40000
EVAL_INTERVAL=500
DEVICE="cuda"
USE_WANDB="--use_wandb"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-wandb)
            USE_WANDB=""
            shift
            ;;
        --max-iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --skip-confirmation)
            # Already confirmed above
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device cuda|cpu] [--no-wandb] [--max-iters N] [--eval-interval N] [--skip-confirmation]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Max iterations: $MAX_ITERS"
echo "  Eval interval: $EVAL_INTERVAL"
echo "  W&B logging: $([ -n "$USE_WANDB" ] && echo "enabled" || echo "disabled")"
echo ""

# Track start time
START_TIME=$(date +%s)

# ============================================================================
# EXPERIMENT 1: Single OWT Training
# ============================================================================
echo ""
echo "========================================================================"
echo "📊 EXPERIMENT 1/4: Single OWT Training"
echo "========================================================================"
echo ""

uv run python experiments/train_owt.py \
    --device $DEVICE \
    --max_iters $MAX_ITERS \
    --eval_interval $EVAL_INTERVAL \
    --learning_rate 0.001 \
    --batch_size 32 \
    $USE_WANDB

echo ""
echo "✅ Experiment 1/4 complete!"
echo ""

# ============================================================================
# EXPERIMENT 2: Compare TinyStories vs OWT
# ============================================================================
echo ""
echo "========================================================================"
echo "📊 EXPERIMENT 2/4: Compare TinyStories vs OWT"
echo "========================================================================"
echo ""

uv run python experiments/compare_datasets.py \
    --device $DEVICE \
    --max_iters $MAX_ITERS \
    --eval_interval $EVAL_INTERVAL \
    $USE_WANDB

echo ""
echo "✅ Experiment 2/4 complete!"
echo ""

# ============================================================================
# EXPERIMENT 3: Learning Rate Sweep
# ============================================================================
echo ""
echo "========================================================================"
echo "📊 EXPERIMENT 3/4: Learning Rate Sweep"
echo "========================================================================"
echo ""

# Learning rates to sweep
LR_VALUES=(1e-5 1e-4 1e-3 5e-3)

echo "Testing learning rates: ${LR_VALUES[@]}"
echo ""

TOTAL_LR=${#LR_VALUES[@]}
CURRENT_LR=0

for LR in "${LR_VALUES[@]}"; do
    CURRENT_LR=$((CURRENT_LR + 1))
    RUN_NAME="owt_lr_${LR}"
    
    echo "--------------------------------------------------------------------"
    echo "[$CURRENT_LR/$TOTAL_LR] Training with LR = $LR"
    echo "--------------------------------------------------------------------"
    
    uv run python experiments/train_owt.py \
        --device $DEVICE \
        --max_iters $MAX_ITERS \
        --eval_interval $EVAL_INTERVAL \
        --learning_rate $LR \
        --batch_size 32 \
        --checkpoint_dir "checkpoints/owt_lr_sweep/lr_${LR}" \
        --wandb_project "cs336-owt-lr-sweep" \
        --wandb_run_name "$RUN_NAME" \
        $USE_WANDB
    
    echo ""
    echo "✓ Completed LR = $LR ($CURRENT_LR/$TOTAL_LR)"
    echo ""
done

echo ""
echo "✅ Experiment 3/4 complete!"
echo ""

# ============================================================================
# EXPERIMENT 4: Batch Size Sweep
# ============================================================================
echo ""
echo "========================================================================"
echo "📊 EXPERIMENT 4/4: Batch Size Sweep"
echo "========================================================================"
echo ""

# Batch sizes to sweep
BATCH_SIZES=(32 128)
LEARNING_RATE=0.001

echo "Testing batch sizes: ${BATCH_SIZES[@]}"
echo "Learning rate: $LEARNING_RATE"
echo ""

TOTAL_BS=${#BATCH_SIZES[@]}
CURRENT_BS=0

for BS in "${BATCH_SIZES[@]}"; do
    CURRENT_BS=$((CURRENT_BS + 1))
    RUN_NAME="owt_bs_${BS}"
    
    echo "--------------------------------------------------------------------"
    echo "[$CURRENT_BS/$TOTAL_BS] Training with Batch Size = $BS"
    echo "--------------------------------------------------------------------"
    
    # Try to run, catch OOM errors
    if uv run python experiments/train_owt.py \
        --device $DEVICE \
        --max_iters $MAX_ITERS \
        --eval_interval $EVAL_INTERVAL \
        --learning_rate $LEARNING_RATE \
        --batch_size $BS \
        --checkpoint_dir "checkpoints/owt_batch_sweep/bs_${BS}" \
        --wandb_project "cs336-owt-batch-sweep" \
        --wandb_run_name "$RUN_NAME" \
        $USE_WANDB; then
        echo ""
        echo "✓ Completed Batch Size = $BS ($CURRENT_BS/$TOTAL_BS)"
        echo ""
    else
        echo ""
        echo "✗ Failed Batch Size = $BS (likely OOM)"
        echo "  Skipping remaining larger batch sizes..."
        echo ""
        break
    fi
done

echo ""
echo "✅ Experiment 4/4 complete!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "⏱️  Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "📁 Results saved to:"
echo "  1. Single OWT:         checkpoints/owt/"
echo "  2. Dataset comparison: checkpoints/tinystories/ & checkpoints/owt/"
echo "  3. LR sweep:           checkpoints/owt_lr_sweep/"
echo "  4. Batch size sweep:   checkpoints/owt_batch_sweep/"
echo ""

if [ -n "$USE_WANDB" ]; then
    echo "📊 W&B Projects:"
    echo "  - cs336-dataset-comparison"
    echo "  - cs336-owt-lr-sweep"
    echo "  - cs336-owt-batch-sweep"
    echo ""
fi

echo "========================================================================"
echo ""
echo "📋 Summary of experiments:"
echo ""
echo "✅ Experiment 1: Single OWT training"
echo "✅ Experiment 2: TinyStories vs OWT comparison"
echo "✅ Experiment 3: Learning rate sweep (${#LR_VALUES[@]} learning rates)"
echo "✅ Experiment 4: Batch size sweep (tested batch sizes)"
echo ""
echo "🎓 You now have all the data needed for comprehensive analysis!"
echo ""
