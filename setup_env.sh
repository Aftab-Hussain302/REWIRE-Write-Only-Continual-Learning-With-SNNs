#!/usr/bin/env bash
# One-time environment on a fresh GPU box (Ubuntu, NVIDIA driver present). Creates a venv so system packages are untouched.
set -e
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126 || pip install -q torch torchvision
pip install -q -r requirements.txt
python3 -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "activate later with:  . .venv/bin/activate"
