import argparse
import os
import warnings
from datetime import datetime
from time import time
from pathlib import Path
from copy import copy

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import SGcADTI
from trainer import Trainer
from utils import set_seed, graph_collate_func, mkdir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 路径设置
# =====================================================
# 当前文件在 SGcCA/code/main.py
# 项目根目录是 SGcCA/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(description="SGcCA cold-start training")

parser.add_argument(
    "--cfg",
    default=str(PROJECT_ROOT / "configs" / "SGcCA.yaml"),
    help="path to config file",
    type=str,
)

parser.add_argument(
    "--dataset",
    default="drugbank",
    help="dataset name, e.g. drugbank or bindingdb",
    type=str,
)

parser.add_argument(
    "--split",
    default="cold_drug",
    choices=["cold_drug", "cold_protein", "cold_target"],
    help="cold split type",
    type=str,
)

parser.add_argument(
    "--train_file",
    default=None,
    help="predefined train csv file",
    type=str,
)

parser.add_argument(
    "--test_file",
    default=None,
    help="predefined test csv file",
    type=str,
)

parser.add_argument(
    "--runs",
    default=3,
    help="number of repeated runs",
    type=int,
)

parser.add_argument(
    "--val_frac",
    default=0.1,
    help="validation ratio split from train file",
    type=float,
)

args = parser.parse_args()


def read_dti_csv(file_path):
    """
    读取 DTI 数据文件。

    支持两种格式：
    1. 有表头：
       SMILES,Protein,Y

    2. 无表头：
       第一列 SMILES，第二列 Protein，第三列 Y
    """
    file_path = str(file_path)

    # 先按有表头 CSV 读取
    df = pd.read_csv(file_path)

    cols_lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    has_header = (
        "smiles" in cols_lower
        and "protein" in cols_lower
        and (
            "y" in cols_lower
            or "label" in cols_lower
            or "interaction" in cols_lower
        )
    )

    if has_header:
        smiles_col = cols_lower["smiles"]
        protein_col = cols_lower["protein"]

        if "y" in cols_lower:
            y_col = cols_lower["y"]
        elif "label" in cols_lower:
            y_col = cols_lower["label"]
        else:
            y_col = cols_lower["interaction"]

        df = df[[smiles_col, protein_col, y_col]].copy()
        df.columns = ["SMILES", "Protein", "Y"]

    else:
        # 如果没有标准表头，就按无表头重新读取
        df = pd.read_csv(file_path, header=None)
        df = df.iloc[:, :3].copy()
        df.columns = ["SMILES", "Protein", "Y"]

    df = df.dropna(subset=["SMILES", "Protein", "Y"]).reset_index(drop=True)
    df["Y"] = df["Y"].astype(int)

    return df


def get_default_cold_files(dataset, split):
    """
    根据你的目录结构自动找 cold 文件。
    """
    dataset_dir = PROJECT_ROOT / "dataset" / dataset

    if split == "cold_drug":
        split_dir = dataset_dir / "cold_drug"
        train_file = split_dir / "train_drug_cold.csv"
        test_file = split_dir / "test_drug_cold.csv"

    else:
        # 兼容 cold_protein 和 cold_target 两种目录名
        if (dataset_dir / "cold_protein").exists():
            split_dir = dataset_dir / "cold_protein"
        else:
            split_dir = dataset_dir / "cold_target"

        train_file = split_dir / "train_protein_cold.csv"
        test_file = split_dir / "test_protein_cold.csv"

    return train_file, test_file


def split_train_val(train_full, val_frac, seed):
    """
    从 train 文件里划一部分做验证集。
    注意：不从 test 里划验证集，避免 test 泄漏。
    """
    train_full = shuffle(train_full, random_state=seed).reset_index(drop=True)

    val_size = int(len(train_full) * val_frac)
    val_size = max(1, val_size)

    train_set = train_full.iloc[:-val_size].copy()
    val_set = train_full.iloc[-val_size:].copy()

    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    return train_set, val_set


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    # =====================================================
    # cold 文件路径
    # =====================================================
    if args.train_file is None or args.test_file is None:
        train_file, test_file = get_default_cold_files(
            args.dataset,
            args.split,
        )
    else:
        train_file = Path(args.train_file)
        test_file = Path(args.test_file)

    print("Train file:", train_file)
    print("Test file :", test_file)

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    if not Path(test_file).exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_full = read_dti_csv(train_file)
    test_set_original = read_dti_csv(test_file)

    # =====================================================
    # 输出目录
    # =====================================================
    cfg.RESULT.OUTPUT_DIR = str(
        PROJECT_ROOT
        / "result"
        / args.dataset
        / f"{args.split}_predefined"
    )

    mkdir(cfg.RESULT.OUTPUT_DIR)

    print(cfg.RESULT.OUTPUT_DIR)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    metrics = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
    }

    metrics = pd.DataFrame(metrics)

    # =====================================================
    # 多次运行
    # =====================================================
    for seed, fold in enumerate(range(1, args.runs + 1)):
        torch.cuda.empty_cache()

        run_seed = cfg.SOLVER.SEED + seed

        cv_out_path = os.path.join(
            cfg.RESULT.OUTPUT_DIR,
            f"random_{fold}",
        )

        os.makedirs(cv_out_path, exist_ok=True)

        print("*" * 60 + str(fold) + "-random" + "*" * 60)

        set_seed(run_seed)

        train_set, val_set = split_train_val(
            train_full,
            args.val_frac,
            run_seed,
        )

        test_set = copy(test_set_original)
        test_set.reset_index(drop=True, inplace=True)

        print("data preprocess end !!!")
        print(f"train:{len(train_set)}")
        print(f"dev:{len(val_set)}")
        print(f"test:{len(test_set)}")

        # 保存实际使用的数据划分，方便检查
        train_set.to_csv(
            os.path.join(cv_out_path, "train_split_used.csv"),
            index=False,
        )

        val_set.to_csv(
            os.path.join(cv_out_path, "val_split_used.csv"),
            index=False,
        )

        test_set.to_csv(
            os.path.join(cv_out_path, "test_split_used.csv"),
            index=False,
        )

        train_dataset = DTIDataset(train_set.index.values, train_set)
        val_dataset = DTIDataset(val_set.index.values, val_set)
        test_dataset = DTIDataset(test_set.index.values, test_set)

        model = SGcADTI(**cfg).to(device)
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.LR,
        )

        torch.backends.cudnn.benchmark = True

        params = {
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "shuffle": True,
            "num_workers": cfg.SOLVER.NUM_WORKERS,
            "drop_last": True,
            "collate_fn": graph_collate_func,
        }

        train_generator = DataLoader(train_dataset, **params)

        params["shuffle"] = False
        params["drop_last"] = False

        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)

        trainer = Trainer(
            model,
            opt,
            device,
            train_generator,
            val_generator,
            test_generator,
            **cfg,
        )

        result = trainer.train()
        result = pd.DataFrame(result, index=[0])

        metrics = pd.concat((metrics, result), axis=0)

        trainer.save_result(cv_out_path)

        metrics.to_csv(
            os.path.join(cfg.RESULT.OUTPUT_DIR, "result.csv"),
            index=False,
        )

        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    summary = metrics.describe().loc[["mean", "std"]]
    summary.to_csv(
        os.path.join(cfg.RESULT.OUTPUT_DIR, "result_summary.csv")
    )

    print("\nResult summary:")
    print(summary)

    return metrics


if __name__ == "__main__":
    print(f"start time: {datetime.now()}")
    s = time()

    result = main()
    print(result)

    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2) / 3600}h")