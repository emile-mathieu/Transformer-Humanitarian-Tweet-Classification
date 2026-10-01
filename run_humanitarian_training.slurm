#!/bin/bash
#SBATCH --job-name=humanitarian_transformer
#SBATCH --output=logs/humanitarian_transformer_%j.out
#SBATCH --error=logs/humanitarian_transformer_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -e

source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate humaid

cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -u main.py
