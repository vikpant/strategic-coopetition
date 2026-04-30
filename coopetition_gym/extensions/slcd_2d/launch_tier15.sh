#!/bin/bash
# =============================================================================
# launch_tier15.sh
# Target:   Single 8-GPU Vast.ai instance (4090 or 5090). Tier 1.5 scope:
#   Stage A: IPPO + ISAC x 25 (eta, beta) cells x 30 seeds  = 1500 runs
#   Stage B: MADDPG + MASAC x 1 baseline cell x 30 seeds    =   60 runs
#   Stage C: calibration — grid(kappa, xi) x 10 inner seeds = ~200 runs
#   TOTAL: ~1760 runs
# Expected: ~18-22 hr wall clock on 8xRTX 4090 with 32 GPU workers.
#           Cost: ~$40-50 at $2.14/hr.
# =============================================================================
set -euo pipefail

: "${REPO_ROOT:=/workspace/strategic-coopetition}"
: "${OUTPUT_DIR:=/workspace/results_slcd2d_tier15}"
: "${TIMESTEPS:=500000}"
: "${SEEDS:=200-229}"
: "${SWEEP_ALGORITHMS:=IPPO,ISAC}"
: "${VERIFY_ALGORITHMS:=MADDPG,MASAC}"
: "${STAGES:=A,B,C}"
: "${REWARD_TYPES:=integrated}"
: "${MAX_WORKERS:=32}"
: "${NUM_GPUS:=8}"
: "${CAL_INNER_SEEDS:=300-309}"
: "${CAL_GRID_RES:=5}"
: "${CAL_MAX_OUTER:=2}"
: "${REQUIRE_GPU:=1}"

cd "$REPO_ROOT"

echo "=== [1/4] Pre-flight check ==="
if [ "$REQUIRE_GPU" = "1" ]; then
    python -m extensions.slcd_2d.pre_launch_check_tier15 --require-gpu
else
    python -m extensions.slcd_2d.pre_launch_check_tier15
fi

echo "=== [2/4] Dry-run matrix preview ==="
python -m extensions.slcd_2d.campaign_tier15 \
    --stages "$STAGES" \
    --sweep-algorithms "$SWEEP_ALGORITHMS" \
    --verify-algorithms "$VERIFY_ALGORITHMS" \
    --seeds "$SEEDS" \
    --reward-types "$REWARD_TYPES" \
    --timesteps "$TIMESTEPS" \
    --max-workers "$MAX_WORKERS" \
    --num-gpus "$NUM_GPUS" \
    --calibrate-inner-seeds "$CAL_INNER_SEEDS" \
    --calibrate-grid-resolution "$CAL_GRID_RES" \
    --calibrate-max-outer "$CAL_MAX_OUTER" \
    --output "$OUTPUT_DIR" \
    --dry-run

mkdir -p "$OUTPUT_DIR"

echo "=== [3/4] Launching Tier 1.5 campaign in tmux ==="
SESSION=slcd2d_tier15
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
    "cd $REPO_ROOT && \
     python -m extensions.slcd_2d.campaign_tier15 \
        --stages $STAGES \
        --sweep-algorithms $SWEEP_ALGORITHMS \
        --verify-algorithms $VERIFY_ALGORITHMS \
        --seeds $SEEDS \
        --reward-types $REWARD_TYPES \
        --timesteps $TIMESTEPS \
        --max-workers $MAX_WORKERS \
        --num-gpus $NUM_GPUS \
        --calibrate-inner-seeds $CAL_INNER_SEEDS \
        --calibrate-grid-resolution $CAL_GRID_RES \
        --calibrate-max-outer $CAL_MAX_OUTER \
        --output $OUTPUT_DIR \
        2>&1 | tee $OUTPUT_DIR/campaign.log"

echo "=== [4/4] Launched ==="
echo "  tmux session: $SESSION (attach with: tmux attach -t $SESSION)"
echo "  output:       $OUTPUT_DIR"
echo "  manifest:     $OUTPUT_DIR/manifest.json"
echo "  log:          $OUTPUT_DIR/campaign.log"
echo ""
echo "Stage artifacts (after completion):"
echo "  Stage A (sweep):     $OUTPUT_DIR/{reward}/IPPO|ISAC/{cell}/seed_*/result.json"
echo "  Stage B (verify):    $OUTPUT_DIR/{reward}/MADDPG|MASAC/eta0.40_beta0.60/seed_*/result.json"
echo "  Stage C (calibrate): $OUTPUT_DIR/calibration/result.json"
echo "  Summary:             $OUTPUT_DIR/summary.json"
