#!/bin/bash
#SBATCH --job-name=reference-infer
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/reference_%j.out
#SBATCH --error=outputs/slurm/reference_%j.err

# ============================================================================
# Reference Pipeline on Bunya HPC
# ============================================================================
#
# Uses SAM2ImagePredictor + IdentityMatcher (ported from reference pipeline).
# YOLO detect-only (no BoT-SORT). Centroid fallback for occlusion recovery.
#
# Usage:
#   sbatch slurm/run_reference.sbatch
#
#   # Override config and parameters:
#   sbatch --export=ALL,CONFIG=configs/hpc_reference.yaml,OVERRIDES="video_path=data/raw/other.mp4 contacts.enabled=true" slurm/run_reference.sbatch
#
# ============================================================================

set -euo pipefail

# Default config (can be overridden via --export=CONFIG=...)
CONFIG="${CONFIG:-configs/hpc_reference.yaml}"

# Optional config overrides (space-separated key=value pairs)
OVERRIDES="${OVERRIDES:-}"

echo "=== Reference Pipeline ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Config:    $CONFIG"
echo "Overrides: $OVERRIDES"
echo "=========================="

# Navigate to project root
cd ~/Balbi/yolo-sam2-lab-tracking

# Load modules and activate virtual environment
module load python/3.10.4-gcccore-11.3.0 2>/dev/null || true
source .venv/bin/activate

# Create slurm output directory
mkdir -p outputs/slurm

# Run pipeline
if [ -n "$OVERRIDES" ]; then
    python -m src.pipelines.reference.run --config "$CONFIG" $OVERRIDES
else
    python -m src.pipelines.reference.run --config "$CONFIG"
fi

echo "=== Job complete ==="
