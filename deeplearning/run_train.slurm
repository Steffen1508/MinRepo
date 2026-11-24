#!/bin/bash
#SBATCH --job-name=skatetrick
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# Job-script til AAU AI-LAB. Alle argumenter sendes videre til train.py:
#   sbatch run_train.slurm --model r2plus1d --freeze partial --crop-mode skater --name r21d_partial_skater
#
# Engangsopsætning (venv oven på PyTorch-containeren, kun OpenCV mangler):
#   srun singularity exec $CONTAINER python -m venv --system-site-packages ~/skate-venv
#   srun singularity exec $CONTAINER ~/skate-venv/bin/pip install "opencv-python-headless<4.12"

CONTAINER=${CONTAINER:-/ceph/container/pytorch/pytorch_26.10.sif}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

srun singularity exec --nv "$CONTAINER" \
    ~/skate-venv/bin/python train.py --data-root Dataset_Kaggle --workers 8 \
    --external-root SkateboardML-1.0/Tricks --external-list SkateboardML-1.0/testlist02.txt "$@"
