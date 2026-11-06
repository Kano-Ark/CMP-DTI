import argparse
import os
import warnings
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import SGcADTI
from trainer_cold import Trainer
from utils import graph_collate_func, mkdir, set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="SGcCA training with predefined cold-start splits")
    parser.add_argument("--cfg", default="./configs/SGcCA.yaml", type=str, help="path to config file")
    parser.add_argument("--dataset", default="drugbank", type=str, help="dataset name, e.g., drugbank")
    parser.add_argument("--data_root", default="./dataset", type=str, help="root folder containing dataset splits")
    parser.add_argument(
        "--split",
        default="cold_drug",
        choices=["cold_drug", "cold_target", "cold_protein"],
        help="cold_drug or cold_target/cold_protein",
    )
    parser.add_argument("--train_file", default=None, type=str, help="optional path to predefined train csv")
    parser.add_argument("--test_file", default=None, type=str, help="optional path to predefined test csv")
    parser.add_argument("--val_file", default=None, type=str, help="optional path to predefined validation csv")
    parser.add_argument("--val_frac", default=0.1, type=float, help="validation ratio split from train_file when --val_file is not provided")
    parser.add_argument("--runs", default=3, type=int, help="number of repeated runs")
    parser.add_argument("--seed", default=42, type=int, help="base random seed")
    parser.add_argument("--output_root", default="./result", type=str, help="result root folder")
    return parser.parse_args()


def normalize_split_name(split):
    if split == "cold_protein":
        return "cold_target"
    return split


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find split file. Tried:\n" + "\n".join(paths))


def auto_split_paths(args):
    split = normalize_split_name(args.split)
    base = os.path.join(args.data_root, args.dataset)

    if split == "cold_drug":
        train_candidates = [
            os.path.join(base, "cold_drug", "train_drug_cold.csv"),
            os.path.join(base, "cold_drug", "train_cold_drug.csv"),
            os.path.join(base, "drug_cold", "train_drug_cold.csv"),
        ]
        test_candidates = [
            os.path.join(base, "cold_drug", "test_drug_cold.csv"),
            os.path.join(base, "cold_drug", "test_cold_drug.csv"),
            os.path.join(base, "drug_cold", "test_drug_cold.csv"),
        ]
    elif split == "cold_target":
        train_candidates = [
            os.path.join(base, "cold_target", "train_protein_cold.csv"),
            os.path.join(base, "cold_target", "train_target_cold.csv"),
            os.path.join(base, "cold_target", "train_cold_target.csv"),
            os.path.join(base, "cold_protein", "train_protein_cold.csv"),
            os.path.join(base, "protein_cold", "train_protein_cold.csv"),
        ]
        test_candidates = [
            os.path.join(base, "cold_target", "test_protein_cold.csv"),
            os.path.join(base, "cold_target", "test_target_cold.csv"),
            os.path.join(base, "cold_target", "test_cold_target.csv"),
            os.path.join(base, "cold_protein", "test_protein_cold.csv"),
            os.path.join(base, "protein_cold", "test_protein_cold.csv"),
        ]
    else:
        raise ValueError(f"Unsupported split: {args.split}")

    train_path = args.train_file if args.train_file is not None else first_existing(train_candidates)
    test_path = args.test_file if args.test_file is not None else first_existing(test_candidates)
    return train_path, test_path


def _standardize_columns(df, path):
    # Common names used by DTI datasets. The dataloader requires SMILES, Protein, Y.
    smiles_candidates = ["SMILES", "smiles", "Smiles", "drug", "Drug", "compound_iso_smiles", "compound", "Compound"]
    protein_candidates = ["Protein", "protein", "Target", "target", "sequence", "Sequence", "target_sequence", "protein_sequence"]
    label_candidates = ["Y", "y", "Label", "label", "interaction", "Interaction", "affinity", "Affinity"]

    cols = list(df.columns)
    smiles_col = next((c for c in smiles_candidates if c in cols), None)
    protein_col = next((c for c in protein_candidates if c in cols), None)
    label_col = next((c for c in label_candidates if c in cols), None)

    if smiles_col is None or protein_col is None or label_col is None:
        if df.shape[1] < 3:
            raise ValueError(f"{path} must contain at least three columns: SMILES, Protein, Y")
        df = df.iloc[:, :3].copy()
        df.columns = ["SMILES", "Protein", "Y"]
    else:
        df = df[[smiles_col, protein_col, label_col]].copy()
        df.columns = ["SMILES", "Protein", "Y"]

    df = df.dropna(subset=["SMILES", "Protein", "Y"]).copy()
    df["SMILES"] = df["SMILES"].astype(str)
    df["Protein"] = df["Protein"].astype(str)
    df["Y"] = df["Y"].astype(int)
    df = df.drop_duplicates(subset=["SMILES", "Protein", "Y"]).reset_index(drop=True)
    return df


def read_interaction_file(path):
    # First try normal CSV with header. If it is whitespace txt or headerless, fall back.
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # If the first row was actually data but pandas interpreted it as header, retry as headerless.
    required = {"SMILES", "Protein", "Y"}
    lower_cols = {str(c).lower() for c in df.columns}
    looks_named = any(c.lower() in lower_cols for c in ["smiles", "protein", "target", "label", "y"])
    if not looks_named:
        try:
            df_no_header = pd.read_csv(path, header=None)
            if df_no_header.shape[1] == 1:
                df_no_header = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
            if df_no_header.shape[1] >= 3 and len(df_no_header) >= len(df):
                df = df_no_header
        except Exception:
            pass

    return _standardize_columns(df, path)


def check_binary_labels(name, df):
    labels = sorted(df["Y"].unique().tolist())
    if len(labels) < 2:
        raise ValueError(f"{name} split has only one class {labels}. Please provide a valid split or change val_frac/seed.")


def make_val_from_train(train_df, val_frac, seed):
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError("--val_frac must be between 0 and 1 when --val_file is not provided.")

    stratify = train_df["Y"] if train_df["Y"].value_counts().min() >= 2 else None
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def split_report(train, val, test, split):
    split = normalize_split_name(split)
    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    train_drugs = set(train["SMILES"].unique())
    val_drugs = set(val["SMILES"].unique())
    trainval_drugs = set(trainval["SMILES"].unique())
    test_drugs = set(test["SMILES"].unique())

    train_targets = set(train["Protein"].unique())
    val_targets = set(val["Protein"].unique())
    trainval_targets = set(trainval["Protein"].unique())
    test_targets = set(test["Protein"].unique())

    report = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_pos": int(train["Y"].sum()),
        "val_pos": int(val["Y"].sum()),
        "test_pos": int(test["Y"].sum()),
        "train_drugs": len(train_drugs),
        "val_drugs": len(val_drugs),
        "test_drugs": len(test_drugs),
        "train_targets": len(train_targets),
        "val_targets": len(val_targets),
        "test_targets": len(test_targets),
        "overlap_train_test_drugs": len(train_drugs & test_drugs),
        "overlap_trainval_test_drugs": len(trainval_drugs & test_drugs),
        "overlap_train_test_targets": len(train_targets & test_targets),
        "overlap_trainval_test_targets": len(trainval_targets & test_targets),
    }

    if split == "cold_drug":
        if report["overlap_trainval_test_drugs"] != 0:
            raise ValueError(
                "This is not a valid cold-drug split: train/val and test share drugs. "
                f"overlap_trainval_test_drugs={report['overlap_trainval_test_drugs']}"
            )
    elif split == "cold_target":
        if report["overlap_trainval_test_targets"] != 0:
            raise ValueError(
                "This is not a valid cold-target split: train/val and test share targets. "
                f"overlap_trainval_test_targets={report['overlap_trainval_test_targets']}"
            )
    return report


def load_predefined_splits(args, run_seed):
    train_path, test_path = auto_split_paths(args)
    raw_train = read_interaction_file(train_path)
    test = read_interaction_file(test_path)

    if args.val_file is not None:
        train = raw_train
        val = read_interaction_file(args.val_file)
    else:
        train, val = make_val_from_train(raw_train, args.val_frac, run_seed)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    check_binary_labels("train", train)
    check_binary_labels("val", val)
    check_binary_labels("test", test)

    return train, val, test, train_path, test_path


def run_one(args, cfg, run_idx):
    run_seed = args.seed + run_idx
    set_seed(run_seed)
    torch.cuda.empty_cache()

    train_set, val_set, test_set, train_path, test_path = load_predefined_splits(args, run_seed)
    report = split_report(train_set, val_set, test_set, args.split)

    print("=" * 80)
    print(f"Run {run_idx + 1}/{args.runs} | split={args.split} | seed={run_seed}")
    print(f"train_file: {train_path}")
    print(f"test_file:  {test_path}")
    for k, v in report.items():
        print(f"{k}: {v}")

    train_dataset = DTIDataset(train_set.index.values, train_set)
    val_dataset = DTIDataset(val_set.index.values, val_set)
    test_dataset = DTIDataset(test_set.index.values, test_set)

    batch_size = cfg.SOLVER.BATCH_SIZE
    train_drop_last = len(train_dataset) >= batch_size

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": train_drop_last,
        "collate_fn": graph_collate_func,
    }
    eval_params = dict(train_params)
    eval_params["shuffle"] = False
    eval_params["drop_last"] = False

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **eval_params)
    test_loader = DataLoader(test_dataset, **eval_params)

    model = SGcADTI(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    trainer = Trainer(model, opt, device, train_loader, val_loader, test_loader, **cfg)
    result = trainer.train()

    run_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, f"seed_{run_seed}")
    os.makedirs(run_dir, exist_ok=True)
    trainer.save_result(run_dir)
    pd.DataFrame([report]).to_csv(os.path.join(run_dir, "split_report.csv"), index=False)
    train_set.to_csv(os.path.join(run_dir, "train_split_used.csv"), index=False)
    val_set.to_csv(os.path.join(run_dir, "val_split_used.csv"), index=False)
    test_set.to_csv(os.path.join(run_dir, "test_split_used.csv"), index=False)

    result.update(report)
    result["seed"] = run_seed
    result["split"] = normalize_split_name(args.split)
    result["dataset"] = args.dataset
    return result


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    torch.cuda.empty_cache()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    split_name = normalize_split_name(args.split)
    cfg.RESULT.OUTPUT_DIR = os.path.join(args.output_root, args.dataset, split_name + "_predefined")
    mkdir(cfg.RESULT.OUTPUT_DIR)

    print(f"Config yaml: {args.cfg}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {split_name}")
    print(f"Output: {cfg.RESULT.OUTPUT_DIR}")
    print(f"Running on: {device}\n")

    all_results = []
    for run_idx in range(args.runs):
        result = run_one(args, cfg, run_idx)
        all_results.append(result)
        pd.DataFrame(all_results).to_csv(os.path.join(cfg.RESULT.OUTPUT_DIR, "result_runs.csv"), index=False)

    result_df = pd.DataFrame(all_results)
    metric_cols = ["auroc", "auprc", "accuracy", "precision", "recall", "f1", "mcc"]
    summary = result_df[metric_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(cfg.RESULT.OUTPUT_DIR, "result_summary.csv"))

    print("\nFinal results over runs:")
    print(result_df[["seed"] + metric_cols])
    print("\nMean ± std:")
    for col in metric_cols:
        std = result_df[col].std(ddof=1) if len(result_df) > 1 else 0.0
        print(f"{col}: {result_df[col].mean():.4f} ± {std:.4f}")


if __name__ == "__main__":
    print(f"start time: {datetime.now()}")
    s = time()
    main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2) / 3600:.4f} h")
