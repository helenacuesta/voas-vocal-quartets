#!/bin/bash
#SBATCH -J revclstm
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=288:00:00
#SBATCH --mem=72GB
#SBATCH -o revclstm.out # STDOUT
#SBATCH -e revclstm.err # STDERR

module purge

module load CUDA/10.1.105
module load cuDNN/7.6.4.38-CUDA-10.1.105

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mf0


python voas/main.py --model voas_clstm_reverse --name voas_clstm_reverse --data-splits ./data/data_splits_hpc.json --patch-len 128 --epochs 65 --batch_size 20 --steps_epoch 2048 --val_steps 440