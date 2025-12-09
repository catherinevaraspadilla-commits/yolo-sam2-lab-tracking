# To activate your environment with all the required dependencies

# 1) Crear venv
python -m venv .venv

# 2) Activar venv
.\.venv\Scripts\Activate.ps1

# 3) Instalar PyTorch con CUDA (ajusta según tu GPU si hace falta)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4) Instalar el resto
pip install -r requirements.txt


