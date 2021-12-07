#!/bin/bash
#SBATCH -J cnnfull
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --constraint=intel
#SBATCH -o indivcnn.out # STDOUT
#SBATCH -e indivcnn.err # STDERR

module purge

module load CUDA/10.1.105
module load cuDNN/7.6.4.38-CUDA-10.1.105

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mf0


python voas/main.py --model voas_cnn --name cnn_full_degraded --data-splits ./data/data_splits_hpc.json --patch-len 128 --epochs 100 --batch_size 32 --steps_epoch 2048 --val_steps 440