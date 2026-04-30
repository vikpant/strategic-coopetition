#!/bin/bash
# =============================================================================
# launch_tier1.sh
# Target:   Single 8-GPU Vast.ai instance (4090 or 5090). Tier 1 scope:
#           2 algos (IPPO, ISAC) x 25 (eta, beta) cells x 20 seeds x 1 reward
#           = 1000 runs. Integrated-reward only by default.
# Expected: ~8-10 hr wall clock on 8xRTX 4090 with 32 GPU workers.
#           Each run ~6-10 min on-GPU for 500k timesteps on SLCD (40-step horizon).
# =============================================================================
set -euo pipefail

: "${REPO_ROOT:=/workspace/strategic-coopetition}"
: "${OUTPUT_DIR:=/workspace/results_slcd2d_tier1}"
: "${TIMESTEPS:=500000}"
: "${SEEDS:=200-219}"
: "${ALGORITHMS:=IPPO,ISAC}"
: "${REWARD_TYPES:=integrated}"
: "${MAX_WORKERS:=32}"
: "${NUM_GPUS:=8}"
: "${REQUIRE_GPU:=1}"

cd "$REPO_ROOT"

echo "=== [1/4] Pre-flight check ==="
if [ "$REQUIRE_GPU" = "1" ]; then
    python -m extensions.slcd_2d.pre_launch_check_tier1 --require-gpu
else
    python -m extensions.slcd_2d.pre_launch_check_tier1
fi

echo "=== [2/4] Dry-run matrix preview ==="
python -m extensions.slcd_2d.campaign_tier1 \
    --algorithms "$ALGORITHMS" \
    --seeds "$SEEDS" \
    --reward-types "$REWARD_TYPES" \
    --timesteps "$TIMESTEPS" \
    --max-workers "$MAX_WORKERS" \
    --num-gpus "$NUM_GPUS" \
    --output "$OUTPUT_DIR" \
    --dry-run

echo "=== [3/4] Launching Tier 1 campaign in tmux ==="
SESSION=slcd2d_tier1
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
    "cd $REPO_ROOT && \
     python -m extensions.slcd_2d.campaign_tier1 \
        --algorithms $ALGORITHMS \
        --seeds $SEEDS \
        --reward-types $REWARD_TYPES \
        --timesteps $TIMESTEPS \
        --max-workers $MAX_WORKERS \
        --num-gpus $NUM_GPUS \
        --output $OUTPUT_DIR \
        2>&1 | tee $OUTPUT_DIR/campaign.log"

echo "=== [4/4] Launched ==="
echo "  tmux session: $SESSION  (attach with: tmux attach -t $SESSION)"
echo "  output:       $OUTPUT_DIR"
echo "  manifest:     $OUTPUT_DIR/manifest.json"
echo "  log:          $OUTPUT_DIR/campaign.log"
