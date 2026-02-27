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
# Output structure:
#   outputs/runs/<BATCH_ID>/
#   ├── chunks/
#   │   ├── chunk_0/  (overlays, contacts, logs per chunk)
#   │   ├── chunk_1/
#   │   ├── chunk_2/
#   │   └── chunk_3/
#   ├── overlays/
#   │   └── <pipeline>_<model>_<date>_merged.avi   ← FINAL VIDEO
#   ├── contacts/
#   │   ├── contacts_per_frame.csv
#   │   ├── contact_bouts.csv
#   │   └── session_summary.json
#   └── logs/
#       └── run.log
#
# ============================================================================

set -euo pipefail

VIDEO="${1:?Usage: bash scripts/run_parallel.sh <video_path> [num_chunks] [config] [overrides]}"
NUM_CHUNKS="${2:-4}"
CONFIG="${3:-configs/hpc_reference.yaml}"
EXTRA_OVERRIDES="${4:-}"

# --- Step 1: Create batch directory ---
BATCH_ID="$(date +%Y-%m-%d_%H%M%S)_reference_batch"
BATCH_DIR="outputs/runs/$BATCH_ID"
CHUNKS_DIR="$BATCH_DIR/chunks"
LOGS_DIR="$BATCH_DIR/logs"
mkdir -p "$CHUNKS_DIR" "$LOGS_DIR"

# --- Step 2: Get total frames from video ---
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

# --- Step 3: Calculate frame ranges ---
FRAMES_PER_CHUNK=$(( (TOTAL_FRAMES + NUM_CHUNKS - 1) / NUM_CHUNKS ))

echo ""
echo "=== Parallel Execution Plan ==="
echo "  Batch ID:   $BATCH_ID"
echo "  Batch dir:  $BATCH_DIR"
echo "  Config:     $CONFIG"
echo "  Video:      $VIDEO"
echo "  Frames:     $TOTAL_FRAMES"
echo "  Chunks:     $NUM_CHUNKS"
echo "  Per chunk:  ~$FRAMES_PER_CHUNK frames"
if [ -n "$EXTRA_OVERRIDES" ]; then
    echo "  Overrides:  $EXTRA_OVERRIDES"
fi
echo ""

BATCH_START=$(date +%s)

# --- Step 4: Launch chunks in parallel ---
PIDS=()
CHUNK_STARTS=()
LOGS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    START=$((i * FRAMES_PER_CHUNK))
    END=$(( (i + 1) * FRAMES_PER_CHUNK ))
    if [ $END -gt $TOTAL_FRAMES ]; then
        END=$TOTAL_FRAMES
    fi

    LOG="$LOGS_DIR/chunk_${i}.log"
    LOGS+=("$LOG")
    CHUNK_STARTS+=("$(date +%s)")
    echo "  Chunk $i: frames $START -> $END (GPU $i)"

    CUDA_VISIBLE_DEVICES=$i python -m src.pipelines.reference.run \
        --config "$CONFIG" \
        --start-frame $START \
        --end-frame $END \
        --chunk-id $i \
        output_dir="$CHUNKS_DIR" \
        video_path="$VIDEO" contacts.enabled=true $EXTRA_OVERRIDES \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NUM_CHUNKS chunks launched. Waiting..."
echo "  Monitor: tail -f $LOGS_DIR/chunk_0.log"
echo ""

# --- Step 5: Wait and report with timing ---
FAILED=0
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    if wait ${PIDS[$i]}; then
        ELAPSED=$(( $(date +%s) - ${CHUNK_STARTS[$i]} ))
        MINS=$((ELAPSED / 60))
        SECS=$((ELAPSED % 60))
        echo "  Chunk $i: DONE (${MINS}m ${SECS}s)"
    else
        ELAPSED=$(( $(date +%s) - ${CHUNK_STARTS[$i]} ))
        MINS=$((ELAPSED / 60))
        SECS=$((ELAPSED % 60))
        echo "  Chunk $i: FAILED after ${MINS}m ${SECS}s (see $LOGS_DIR/chunk_${i}.log)"
        FAILED=$((FAILED + 1))
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - BATCH_START ))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "  Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "=== $FAILED chunk(s) failed. Check logs in $LOGS_DIR/ ==="
    exit 1
fi

# --- Step 6: Find chunk directories and merge ---
echo ""
echo "=== All chunks completed. Merging... ==="

CHUNK_DIRS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    RUN_DIR=$(grep -oP 'Run directory: \K.*' "${LOGS[$i]}" 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        CHUNK_DIRS+=("$RUN_DIR")
        echo "  Chunk $i: $RUN_DIR"
    else
        echo "  Chunk $i: WARNING - could not find run directory in log"
    fi
done

if [ ${#CHUNK_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No run directories found in logs."
    exit 1
fi

echo ""
python scripts/merge_chunks.py "${CHUNK_DIRS[@]}" -o "$BATCH_DIR"

# --- Step 7: Compress contacts into tar.gz ---
CONTACTS_TAR=""
if [ -d "$BATCH_DIR/contacts" ]; then
    CONTACTS_TAR="$BATCH_DIR/contacts_${BATCH_ID}.tar.gz"
    tar -czf "$CONTACTS_TAR" -C "$BATCH_DIR" contacts/
    echo "Contacts compressed: $CONTACTS_TAR"
fi

# --- Step 8: Print download commands ---
REMOTE_USER="${USER:-s4948012}"
REMOTE_HOST="bunya.rcc.uq.edu.au"
ABS_BATCH_DIR="$(cd "$BATCH_DIR" && pwd)"

# Find the merged video
MERGED_VIDEO=$(find "$BATCH_DIR/overlays" -name "*_merged.*" 2>/dev/null | head -1)
ABS_MERGED=""
if [ -n "$MERGED_VIDEO" ]; then
    ABS_MERGED="$(cd "$(dirname "$MERGED_VIDEO")" && pwd)/$(basename "$MERGED_VIDEO")"
fi

ABS_CONTACTS=""
if [ -n "$CONTACTS_TAR" ]; then
    ABS_CONTACTS="$(cd "$(dirname "$CONTACTS_TAR")" && pwd)/$(basename "$CONTACTS_TAR")"
fi

LOCAL_DIR="C:\\Users\\CatherineVaras\\Downloads\\yolo-sam2-lab-tracking\\outputs"

echo ""
echo "============================================================"
echo "  BATCH COMPLETE: $BATCH_ID"
echo "  TOTAL TIME:     ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  BATCH DIR:      $ABS_BATCH_DIR"
if [ -n "$ABS_MERGED" ]; then
    echo "  MERGED VIDEO:   $ABS_MERGED"
fi
if [ -n "$ABS_CONTACTS" ]; then
    echo "  CONTACTS:       $ABS_CONTACTS"
fi
echo "============================================================"
echo ""
echo "# Download results (PowerShell):"
if [ -n "$ABS_MERGED" ]; then
    echo "scp ${REMOTE_USER}@${REMOTE_HOST}:${ABS_MERGED} \\"
    echo "  \"${LOCAL_DIR}\\\""
fi
if [ -n "$ABS_CONTACTS" ]; then
    echo "scp ${REMOTE_USER}@${REMOTE_HOST}:${ABS_CONTACTS} \\"
    echo "  \"${LOCAL_DIR}\\\""
fi
echo ""
