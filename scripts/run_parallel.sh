#!/bin/bash
# ============================================================================
# Run reference pipeline in parallel chunks within a salloc session
# ============================================================================
#
# Usage (inside salloc with multiple GPUs):
#   bash scripts/run_parallel.sh data/raw/original_120s.avi
#   bash scripts/run_parallel.sh data/raw/original_120s.avi 4
#   bash scripts/run_parallel.sh data/raw/my_video.avi 2 configs/hpc_reference.yaml
#
# Arguments:
#   $1 - Video path (required)
#   $2 - Number of chunks/GPUs (default: 4)
#   $3 - Config file (default: configs/hpc_reference.yaml)
#   $4 - Extra overrides (optional, space-separated key=value)
#
# What it does:
#   1. Reads total frames from the video automatically
#   2. Divides into N equal chunks
#   3. Launches each chunk on a different GPU in parallel
#   4. Waits for all to finish
#   5. Auto-merges results (video + contacts)
#   6. Prints the final merged video path
#
# ============================================================================

set -euo pipefail

VIDEO="${1:?Usage: bash scripts/run_parallel.sh <video_path> [num_chunks] [config] [overrides]}"
NUM_CHUNKS="${2:-4}"
CONFIG="${3:-configs/hpc_reference.yaml}"
EXTRA_OVERRIDES="${4:-}"

# --- Step 1: Get total frames from video ---
echo "Analyzing video: $VIDEO"
TOTAL_FRAMES=$(python -c "
import cv2, sys
cap = cv2.VideoCapture('$VIDEO')
if not cap.isOpened():
    print('ERROR: cannot open video', file=sys.stderr)
    sys.exit(1)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total / fps if fps > 0 else 0
print(total)
print(f'  Resolution: {w}x{h}', file=sys.stderr)
print(f'  FPS: {fps:.1f}', file=sys.stderr)
print(f'  Total frames: {total}', file=sys.stderr)
print(f'  Duration: {duration:.1f}s ({duration/60:.1f} min)', file=sys.stderr)
cap.release()
")

if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" -eq 0 ]; then
    echo "ERROR: Could not read frame count from video"
    exit 1
fi

# --- Step 2: Calculate frame ranges ---
FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))

echo ""
echo "=== Parallel Execution Plan ==="
echo "  Config:     $CONFIG"
echo "  Video:      $VIDEO"
echo "  Frames:     $TOTAL_FRAMES"
echo "  Chunks:     $NUM_CHUNKS"
echo "  Per chunk:  ~$FRAMES_PER_CHUNK frames"
if [ -n "$EXTRA_OVERRIDES" ]; then
    echo "  Overrides:  $EXTRA_OVERRIDES"
fi
echo ""

mkdir -p outputs/slurm

# --- Step 3: Launch chunks in parallel ---
PIDS=()
LOGS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    START=$((i * FRAMES_PER_CHUNK))
    END=$(( (i + 1) * FRAMES_PER_CHUNK ))
    if [ $END -gt $TOTAL_FRAMES ]; then
        END=$TOTAL_FRAMES
    fi

    LOG="outputs/slurm/chunk_${i}.log"
    LOGS+=("$LOG")
    echo "  Chunk $i: frames $START -> $END (GPU $i) -> $LOG"

    CUDA_VISIBLE_DEVICES=$i python -m src.pipelines.reference.run \
        --config "$CONFIG" \
        --start-frame $START \
        --end-frame $END \
        --chunk-id $i \
        video_path="$VIDEO" contacts.enabled=true $EXTRA_OVERRIDES \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_CHUNKS chunks launched. Waiting..."
echo "  Monitor progress: tail -f outputs/slurm/chunk_0.log"
echo ""

# --- Step 4: Wait and report ---
FAILED=0
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    if wait ${PIDS[$i]}; then
        echo "  Chunk $i: DONE"
    else
        echo "  Chunk $i: FAILED (see outputs/slurm/chunk_${i}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

if [ $FAILED -gt 0 ]; then
    echo "=== $FAILED chunk(s) failed. Check logs in outputs/slurm/ ==="
    exit 1
fi

# --- Step 5: Find run directories from logs and auto-merge ---
echo "=== All chunks completed. Merging... ==="

CHUNK_DIRS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    # Extract "Run directory: /path/to/..." from each chunk's log
    RUN_DIR=$(grep -oP 'Run directory: \K.*' "${LOGS[$i]}" 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        CHUNK_DIRS+=("$RUN_DIR")
        echo "  Chunk $i: $RUN_DIR"
    else
        echo "  Chunk $i: WARNING - could not find run directory in log"
    fi
done

if [ ${#CHUNK_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No run directories found in logs. Merge manually."
    exit 1
fi

echo ""
python scripts/merge_chunks.py "${CHUNK_DIRS[@]}"
