#!/bin/bash
#
#SBATCH --job-name=feats
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --time=96:00:00
#SBATCH --output=feats.out
#SBATCH --error=feats.err


module purge

module load rubberband/1.9.0-foss-2017a
module load FFmpeg/4.1-foss-2017a
module load SoX/14.4.2-foss-2017a

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mf0

python voas/data_prep.py
