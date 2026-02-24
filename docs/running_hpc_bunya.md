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

scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/overlay.avi `
 "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs"

 scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-23_191503_sam2_yolo/overlays/overlay.avi ` "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs"
```

2026-02-23_195107_sam2_yolo


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

## GPU vs CPU — Cómo elegir device

El parámetro `models.device` en el YAML controla dónde corren los modelos:

| Valor | Qué hace |
|-------|----------|
| `"auto"` | Usa GPU si hay CUDA disponible, si no cae a CPU |
| `"cuda"` | Fuerza GPU (falla si no hay CUDA) |
| `"cpu"` | Fuerza CPU (funciona siempre, pero SAM2 es muy lento) |

### Si te salió CPU en Bunya

Si usaste `local_quick.yaml` y viste `Using device: cpu`, es porque:
1. Estás en el **login node** (no tiene GPU) — necesitas pedir un nodo GPU con `salloc`
2. No cargaste el módulo CUDA — corre `module load cuda` antes de activar el venv
3. PyTorch no tiene CUDA — reinstalar con `pip install torch --index-url https://download.pytorch.org/whl/cu118`

Diagnóstico rápido:
```bash
nvidia-smi                    # Debe mostrar la GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # Debe decir True
```

### Correr interactivamente con GPU (forma rápida)

```bash
# 1) Pedir un nodo GPU
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --time=02:00:00 --mem=16G
srun --pty bash

# 2) Setup del ambiente
module load python/3.10.4
module load cuda
cd ~/Balbi/yolo-sam2-lab-tracking
source .venv/bin/activate

# 3) Correr con GPU — usa cualquier config + override del device
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml models.device=cuda
```

Esto usa SAM2 tiny + max_frames=150 pero en GPU. Ideal para experimentar rápido.

### Overrides útiles desde la línea de comandos

Cualquier parámetro del YAML se puede sobreescribir con `clave.subclave=valor`:

```bash
# GPU + tiny + solo 300 frames
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    models.device=cuda scan.max_frames=300

# GPU + modelo large + video completo
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    models.device=cuda \
    models.sam2_checkpoint=models/sam2/segment-anything-2/checkpoints/sam2.1_hiera_large.pt \
    models.sam2_config=configs/sam2.1/sam2.1_hiera_l.yaml \
    scan.max_frames=null

# Cambiar confianza de YOLO
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    models.device=cuda detection.confidence=0.4

# Solo mostrar keypoints con alta confianza
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    models.device=cuda detection.keypoint_min_conf=0.5

# Usar otro video
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    models.device=cuda video_path=data/raw/otro_video.mp4
```

### Resumen de overrides

| Override | Efecto |
|----------|--------|
| `models.device=cuda` | Forzar GPU |
| `models.device=cpu` | Forzar CPU |
| `scan.max_frames=300` | Procesar solo 300 frames |
| `scan.max_frames=null` | Procesar todo el video |
| `detection.confidence=0.4` | Subir umbral de detección YOLO |
| `detection.keypoint_min_conf=0.5` | Solo keypoints con alta confianza |
| `video_path=data/raw/video.mp4` | Usar otro video de entrada |

## Config Differences (local vs HPC)

| Parameter | local_quick.yaml | hpc_full.yaml |
|-----------|-----------------|---------------|
| video_path | `data/clips/output-6s.mp4` | `data/raw/original.mp4` |
| SAM2 model | tiny (~39 MB) | large (~225 MB) |
| device | auto (cae a CPU si no hay GPU) | cuda (requiere GPU) |
| max_frames | 150 | null (todo) |
| sampling target | 50 frames | 200 frames |

Puedes usar cualquiera de los dos en Bunya, combinando con overrides según necesites.

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
