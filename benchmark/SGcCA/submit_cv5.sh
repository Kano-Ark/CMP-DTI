#!/bin/bash
#SBATCH -J SGcCA_CV5
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515063029/code-yjx/SGcCA/log_cv5_%j.txt

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate envyjx

SGCCA_DIR="/share/home/u2515063029/code-yjx/SGcCA"
DCGCN_DIR="/share/home/u2515063029/code-yjx/DCGCN"
TASK="drugbank"

# One Slurm job, one GPU, sequentially execute all folds.
SPLITS=(drug_scaffold protein_similarity40)
FOLDS=(0 1 2 3 4)

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$SGCCA_DIR"
mkdir -p logs

DATA_ROOT="${DCGCN_DIR}/dataset/${TASK}"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: DCGCN dataset directory not found: $DATA_ROOT" >&2
    exit 1
fi

for split in "${SPLITS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        TRAIN_CSV="${DATA_ROOT}/result/${split}/train_cv${fold}.csv"
        TEST_CSV="${DATA_ROOT}/result/${split}/test_cv${fold}.csv"

        if [[ ! -f "$TRAIN_CSV" || ! -f "$TEST_CSV" ]]; then
            echo "ERROR: missing split file: $TRAIN_CSV or $TEST_CSV" >&2
            exit 1
        fi

        echo "============================================================"
        echo "Start: model=SGcCA task=${TASK} split=${split} fold=${fold} time=$(date)"
        echo "Train: ${TRAIN_CSV}"
        echo "Test : ${TEST_CSV}"
        echo "============================================================"

        python -u main_cv5.py \
            --task "$TASK" \
            --split "$split" \
            --fold "$fold" \
            --dti-root "$DATA_ROOT" \
            2>&1 | tee "logs/${TASK}_${split}_cv${fold}.log"

        echo "Finished: task=${TASK} split=${split} fold=${fold} time=$(date)"
    done
done

python -u summarize_sgcca_cv.py \
    --task "$TASK" \
    --splits "${SPLITS[@]}" \
    --folds 5 \
    --result-root "./result/${TASK}"

echo "All SGcCA CV5 runs and mean±SD summary completed at $(date)"
