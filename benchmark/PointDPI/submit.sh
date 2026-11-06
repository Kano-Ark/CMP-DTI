#!/bin/bash
#SBATCH -J src
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515063029/code-yjx/PointDPI/log_coldpair.txt
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate envyjx
cd /share/home/u2515063029/code-yjx/PointDPI
python /share/home/u2515063029/code-yjx/PointDPI/PointDPI_train.py