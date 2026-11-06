#!/bin/bash
#SBATCH -J sgcca
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515063029/code-yjx/SGcCA/log_cold_drug.txt
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate envyjx
cd /share/home/u2515063029/code-yjx/SGcCA
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python /share/home/u2515063029/code-yjx/SGcCA/code/main_cold_source_style.py \
  --dataset drugbank \
  --split cold_drug \
  --train_file dataset/drugbank/cold_drug/train_drug_cold.csv \
  --test_file dataset/drugbank/cold_drug/test_drug_cold.csv \