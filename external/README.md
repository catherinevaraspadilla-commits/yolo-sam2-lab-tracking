First we need to clone the repository of sam 2 here 

git clone https://github.com/facebookresearch/segment-anything-2.git external/segment-anything-2

and then we install the libraries

cd external/segment-anything-2
pip install -e .

en git bash ejecuta lo siguiente (no powershell para este)
cd external/segment-anything-2/checkpoints
bash download_ckpts.sh
