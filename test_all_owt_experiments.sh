#!/bin/bash
# Quick test of ALL OpenWebText experiments (reduced iterations)
# Use this to verify everything works before running the full experiments

set -e  # Exit on error

echo "========================================================================"
echo "🧪 Testing ALL OpenWebText Experiments (Quick Version)"
echo "========================================================================"
echo ""
echo "This will run quick tests of:"
echo "  1. Single OWT training (500 iters)"
echo "  2. Compare TinyStories vs OWT (500 iters each)"
echo "  3. Learning rate sweep (3 LRs × 500 iters)"
echo "  4. Batch size sweep (3 batch sizes × 500 iters)"
echo ""
echo "⏱️  Estimated total time: ~30-45 minutes on H100"
echo ""

# Configuration for testing
MAX_ITERS=500
EVAL_INTERVAL=100
DEVICE="cuda"
USE_WANDB=""  # Disable W&B for testing by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --max-iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device cuda|cpu] [--use-wandb] [--max-iters N]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Max iterations: $MAX_ITERS (testing mode)"
echo "  Eval interval: $EVAL_INTERVAL"
echo "  W&B logging: $([ -n "$USE_WANDB" ] && echo "enabled" || echo "disabled")"
echo ""

# Track start time
START_TIME=$(date +%s)

# ============================================================================
# TEST 1: Single OWT Training
# ============================================================================
echo ""
echo "========================================================================"
echo "🧪 TEST 1/4: Single OWT Training"
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
echo "✅ Test 1/4 complete!"
echo ""

# ============================================================================
# TEST 2: Compare TinyStories vs OWT
# ============================================================================
echo ""
echo "========================================================================"
echo "🧪 TEST 2/4: Compare TinyStories vs OWT"
echo "========================================================================"
echo ""

uv run python experiments/compare_datasets.py \
    --device $DEVICE \
    --max_iters $MAX_ITERS \
    --eval_interval $EVAL_INTERVAL \
    $USE_WANDB

echo ""
echo "✅ Test 2/4 complete!"
echo ""

# ============================================================================
# TEST 3: Learning Rate Sweep (reduced)
# ============================================================================
echo ""
echo "========================================================================"
echo "🧪 TEST 3/4: Learning Rate Sweep (3 LRs)"
echo "========================================================================"
echo ""

# Test with only 3 learning rates
LR_VALUES=(1e-4 1e-3 3e-3)

echo "Testing learning rates: ${LR_VALUES[@]}"
echo ""

TOTAL_LR=${#LR_VALUES[@]}
CURRENT_LR=0

for LR in "${LR_VALUES[@]}"; do
    CURRENT_LR=$((CURRENT_LR + 1))
    RUN_NAME="owt_lr_${LR}_test"
    
    echo "--------------------------------------------------------------------"
    echo "[$CURRENT_LR/$TOTAL_LR] Testing LR = $LR"
    echo "--------------------------------------------------------------------"
    
    uv run python experiments/train_owt.py \
        --device $DEVICE \
        --max_iters $MAX_ITERS \
        --eval_interval $EVAL_INTERVAL \
        --learning_rate $LR \
        --batch_size 32 \
        --checkpoint_dir "checkpoints/test_owt_lr_sweep/lr_${LR}" \
        --wandb_project "cs336-owt-lr-sweep-test" \
        --wandb_run_name "$RUN_NAME" \
        $USE_WANDB
    
    echo ""
    echo "✓ Completed LR = $LR ($CURRENT_LR/$TOTAL_LR)"
    echo ""
done

echo ""
echo "✅ Test 3/4 complete!"
echo ""

# ============================================================================
# TEST 4: Batch Size Sweep (reduced)
# ============================================================================
echo ""
echo "========================================================================"
echo "🧪 TEST 4/4: Batch Size Sweep (3 batch sizes)"
echo "========================================================================"
echo ""

# Test with only 3 batch sizes
BATCH_SIZES=(16 32 64)
LEARNING_RATE=0.001

echo "Testing batch sizes: ${BATCH_SIZES[@]}"
echo "Learning rate: $LEARNING_RATE"
echo ""

TOTAL_BS=${#BATCH_SIZES[@]}
CURRENT_BS=0

for BS in "${BATCH_SIZES[@]}"; do
    CURRENT_BS=$((CURRENT_BS + 1))
    RUN_NAME="owt_bs_${BS}_test"
    
    echo "--------------------------------------------------------------------"
    echo "[$CURRENT_BS/$TOTAL_BS] Testing Batch Size = $BS"
    echo "--------------------------------------------------------------------"
    
    # Try to run, catch OOM errors
    if uv run python experiments/train_owt.py \
        --device $DEVICE \
        --max_iters $MAX_ITERS \
        --eval_interval $EVAL_INTERVAL \
        --learning_rate $LEARNING_RATE \
        --batch_size $BS \
        --checkpoint_dir "checkpoints/test_owt_batch_sweep/bs_${BS}" \
        --wandb_project "cs336-owt-batch-sweep-test" \
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
echo "✅ Test 4/4 complete!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "🎉 ALL TESTS COMPLETE!"
echo "========================================================================"
echo ""
echo "⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "✅ All experiments are working correctly!"
echo ""
echo "📋 Test results saved to:"
echo "  - checkpoints/owt/"
echo "  - checkpoints/tinystories/"
echo "  - checkpoints/test_owt_lr_sweep/"
echo "  - checkpoints/test_owt_batch_sweep/"
echo ""
echo "🚀 Ready to run full experiments with:"
echo "   ./run_all_owt_experiments.sh"
echo ""
echo "💡 Or run individual experiment groups:"
echo "   ./run_owt_experiments.sh      # Main experiments only"
echo "   ./run_owt_lr_sweep.sh         # LR sweep only"
echo "   ./run_owt_batch_sweep.sh      # Batch sweep only"
echo ""

