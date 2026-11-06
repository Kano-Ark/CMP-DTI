#!/bin/bash
#SBATCH -J PointDPI_CV5
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515063029/code-yjx/PointDPI/log_pointdpi_cv5_%j.out

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate envyjx

# DTI datasets and generated split CSVs are stored in DCGCN.
DCGCN_DIR="/share/home/u2515063029/code-yjx/DCGCN"
POINTDPI_DIR="/share/home/u2515063029/code-yjx/PointDPI"

# Change to drugbank if that is the dataset folder you want to evaluate.
TASK="drugbank"

# Each split is run sequentially; one sbatch submission, one GPU.
SPLITS=(drug_scaffold protein_similarity40)

mkdir -p "${POINTDPI_DIR}/logs"
cd "${DCGCN_DIR}"

for split in "${SPLITS[@]}"; do
    for fold in 0 1 2 3 4; do
        echo "============================================================"
        echo "PointDPI: task=${TASK}, split=${split}, fold=${fold}"
        echo "Train: ${DCGCN_DIR}/dataset/${TASK}/result/${split}/train_cv${fold}.csv"
        echo "Test : ${DCGCN_DIR}/dataset/${TASK}/result/${split}/test_cv${fold}.csv"
        echo "============================================================"

        python -u "${POINTDPI_DIR}/PointDPI_train_cv5.py" \
            --task "${TASK}" \
            --split "${split}" \
            --fold "${fold}" \
            2>&1 | tee "${POINTDPI_DIR}/logs/${TASK}_${split}_cv${fold}.log"
    done
done

python -u "${POINTDPI_DIR}/summarize_pointdpi_cv.py" \
    --task "${TASK}" \
    --splits "${SPLITS[@]}" \
    --folds 5

echo "All PointDPI CV5 runs and summary are complete."