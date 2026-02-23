# Running on UQ Bunya HPC

## Cluster Details

- **Slurm GPU partition:** `gpu_cuda`
- **Python venv:** `~/yolo-sam2-lab-tracking/.venv`
- **SAM2:** installed from official repo via editable install
- **CUDA:** loaded via module system
- **Verified:** SAM 2.1 tiny loads and runs with PyTorch + CUDA

## Initial Setup (one-time)

```bash
# =========================
# 0) Login (from your laptop)
# ssh s4948012@bunya.rcc.uq.edu.au
# =========================

# =========================
# 1) Reserve a GPU node (run on Bunya)
# =========================
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --time=02:00:00 --mem=16G srun --pty bash -

# =========================
# 2) Load modules (do this BEFORE activating venv)
# =========================
module load python/3.10.4
module load cuda
python --version
nvidia-smi   # optional sanity check that you are on a GPU node

# =========================
# 3) Go to project (clone first time only)
# =========================
mkdir -p ~/Balbi
cd ~/Balbi

# First time only:
# git clone https://github.com/catherinevaraspadilla-commits/yolo-sam2-lab-tracking.git
cd yolo-sam2-lab-tracking

# =========================
# 4) Create / recreate venv (recommended if you saw libpython errors)
# =========================
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# =========================
# 5) Install PyTorch (CUDA 11.8 wheels)
# =========================
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Quick check (should say CUDA available: True)
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.version.cuda)"

# =========================
# 6) Install project dependencies
# =========================
mkdir -p models/sam2
git clone https://github.com/facebookresearch/segment-anything-2.git models/sam2/segment-anything-2
python -m pip install -r requirements.txt
python -m pip install ultralytics opencv-python roboflow

# =========================
# 7) Download SAM2 checkpoints
# =========================
cd models/sam2/segment-anything-2/checkpoints
bash download_ckpts.sh

# Back to project root
cd ~/Balbi/yolo-sam2-lab-tracking
```

## Running Inference

### Via Slurm (recommended for long videos)

```bash
# Default: uses configs/hpc_full.yaml
sbatch slurm/run_infer.sbatch

# Custom config:
sbatch --export=CONFIG=configs/hpc_full.yaml slurm/run_infer.sbatch

# Check job status
squeue -u $USER

# View output
tail -f outputs/slurm/infer_<jobid>.out
```

### Frame Extraction

```bash
sbatch slurm/run_extract_frames.sbatch

# Or interactive:
srun --partition=gpu_cuda --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
source .venv/bin/activate
python scripts/extract_frames.py all --config configs/hpc_full.yaml
```

## Config Differences (local vs HPC)

| Parameter | local_quick.yaml | hpc_full.yaml |
|-----------|-----------------|---------------|
| video_path | `data/clips/output-6s.mp4` | `data/raw/original.mp4` |
| SAM2 model | tiny (~39 MB) | large (~225 MB) |
| device | auto | cuda |
| max_frames | 150 | null (all) |
| sampling target | 50 frames | 200 frames |

## Output Location

All outputs go to `outputs/runs/<timestamp>/` — same structure as local runs. Copy results back to your machine:

```bash
# From your local machine
scp -r bunya:~/yolo-sam2-lab-tracking/outputs/runs/<run-dir> ./outputs/runs/
```

## YOLO Training

```bash
sbatch slurm/run_train_yolo.sbatch
```

Edit `slurm/run_train_yolo.sbatch` to configure training parameters (dataset path, epochs, batch size, etc.).
