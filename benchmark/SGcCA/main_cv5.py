import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from time import time
from pathlib import Path
from copy import copy

# The original SGcCA entry point lives under ./code/. This CV runner is in the
# repository root, so explicitly prioritize ./code when importing configs,
# dataloader, models, trainer, and utils. Keep YAML paths relative to repo root.
_CODE_DIR = Path(__file__).resolve().parent / "code"
if _CODE_DIR.is_dir():
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import SGcADTI
from trainer import Trainer
from utils import set_seed, graph_collate_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGcCA DrugBank CV runner using split CSVs generated in DCGCN"
    )
    parser.add_argument("--cfg", default="./configs/SGcCA.yaml", type=str,
                        help="SGcCA config YAML")
    parser.add_argument("--task", default="drugbank", type=str,
                        help="dataset name in DCGCN/dataset, default: drugbank")
    parser.add_argument("--split", required=True, choices=["drug_scaffold", "protein_similarity40"],
                        help="split directory under DCGCN/dataset/<task>/result/")
    parser.add_argument("--fold", required=True, type=int, choices=range(5),
                        help="CV fold index, 0..4")
    parser.add_argument("--dti-root", default="../DCGCN/dataset/drugbank", type=str,
                        help="dataset root in DCGCN containing drug_SMILES.csv, protein sequence CSV, and result/")
    parser.add_argument("--drug-smiles-file", default=None, type=str,
                        help="optional explicit path to the headerless drug ID,SMILES mapping CSV")
    parser.add_argument("--protein-seq-file", default=None, type=str,
                        help="optional explicit path to the headerless protein ID,sequence mapping CSV")
    parser.add_argument("--val-total-fraction", default=0.10, type=float,
                        help="fraction of train+test pairs reserved as internal validation; default 10%%")
    return parser.parse_args()


def _read_two_column_mapping(path, kind):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} mapping file not found: {path.resolve()}")
    frame = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if frame.shape[1] < 2:
        raise ValueError(f"{kind} mapping must have at least 2 columns (ID, value): {path}")
    ids = frame.iloc[:, 0].astype(str).str.strip()
    values = frame.iloc[:, 1].astype(str).str.strip()
    mapping = {}
    for identifier, value in zip(ids, values):
        if identifier and value:
            mapping[identifier] = value
    if not mapping:
        raise ValueError(f"No valid {kind} mappings found in {path}")
    return mapping


def _find_mapping(root, explicit_path, candidates, kind):
    if explicit_path:
        return _read_two_column_mapping(explicit_path, kind)
    root = Path(root)
    for name in candidates:
        path = root / name
        if path.is_file():
            print(f"[data] {kind} mapping: {path.resolve()}")
            return _read_two_column_mapping(path, kind)
    tried = ", ".join(str(root / name) for name in candidates)
    raise FileNotFoundError(
        f"Could not find {kind} mapping under {root.resolve()}. Tried: {tried}. "
        "Use --drug-smiles-file or --protein-seq-file to specify the file explicitly."
    )


def _read_pair_file(path, drug_smiles, protein_sequences):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DTI split file not found: {path.resolve()}")
    pairs = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if pairs.shape[1] < 3:
        raise ValueError(f"Expected headerless 3-column DTI file (drug_id, protein_id, dti): {path}")

    records = []
    missing_drugs = set()
    missing_proteins = set()
    invalid_labels = []
    for row_idx, row in pairs.iloc[:, :3].iterrows():
        drug_id = str(row.iloc[0]).strip()
        protein_id = str(row.iloc[1]).strip()
        label_text = str(row.iloc[2]).strip()
        if drug_id not in drug_smiles:
            missing_drugs.add(drug_id)
            continue
        if protein_id not in protein_sequences:
            missing_proteins.add(protein_id)
            continue
        try:
            label = int(float(label_text))
        except ValueError:
            invalid_labels.append((row_idx, label_text))
            continue
        if label not in (0, 1):
            invalid_labels.append((row_idx, label_text))
            continue
        records.append({
            "SMILES": drug_smiles[drug_id],
            "Protein": protein_sequences[protein_id],
            "Y": label,
            "drug_id": drug_id,
            "protein_id": protein_id,
        })

    if missing_drugs or missing_proteins or invalid_labels:
        examples = []
        if missing_drugs:
            examples.append(f"missing drug IDs {len(missing_drugs)} (examples: {list(sorted(missing_drugs))[:5]})")
        if missing_proteins:
            examples.append(f"missing protein IDs {len(missing_proteins)} (examples: {list(sorted(missing_proteins))[:5]})")
        if invalid_labels:
            examples.append(f"invalid labels {len(invalid_labels)} (examples: {invalid_labels[:5]})")
        raise ValueError(
            f"Could not map all rows in {path}: {'; '.join(examples)}. "
            "Refusing to silently drop DTI pairs."
        )
    if not records:
        raise ValueError(f"No usable DTI pairs in {path}")
    frame = pd.DataFrame(records, columns=["SMILES", "Protein", "Y", "drug_id", "protein_id"])
    frame["Y"] = frame["Y"].astype(int)
    return frame


def _make_train_val(train_frame, test_frame, val_total_fraction, seed):
    if not 0.0 < val_total_fraction < 1.0:
        raise ValueError("--val-total-fraction must be between 0 and 1")
    # Keep the external DCGCN test set untouched. Reserve the requested fraction
    # of train+test pairs from the training CSV for SGcCA's internal validation.
    desired_val_n = int(round(val_total_fraction * (len(train_frame) + len(test_frame))))
    desired_val_n = max(1, min(desired_val_n, len(train_frame) - 1))
    labels = train_frame["Y"]
    stratify = None
    counts = labels.value_counts()
    if desired_val_n >= 2 and (len(train_frame) - desired_val_n) >= 2 and len(counts) == 2 and counts.min() >= 2:
        stratify = labels
    train_set, val_set = train_test_split(
        train_frame,
        test_size=desired_val_n,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    return (
        train_set.reset_index(drop=True),
        val_set.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def _result_to_dict(result):
    if isinstance(result, pd.DataFrame):
        return result.iloc[0].to_dict() if len(result) else {}
    if isinstance(result, pd.Series):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)) and len(result) == 1 and isinstance(result[0], dict):
        return result[0]
    raise TypeError(
        f"Unexpected Trainer.train() result type: {type(result).__name__}; "
        "expected dict, Series, DataFrame, or a single-item list of dicts."
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    args = parse_args()
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    task = args.task.lower()
    split = args.split
    fold = args.fold
    dti_root = Path(args.dti_root).expanduser().resolve()
    split_root = dti_root / "result" / split
    train_csv = split_root / f"train_cv{fold}.csv"
    test_csv = split_root / f"test_cv{fold}.csv"

    drug_smiles = _find_mapping(
        dti_root, args.drug_smiles_file,
        ["drug_SMILES.csv", "drug_smiles.csv"], "drug-SMILES"
    )
    protein_sequences = _find_mapping(
        dti_root, args.protein_seq_file,
        ["protein_seq.csv", "prorein_seq.csv", "protein_sequence.csv", "protein_sequences.csv"],
        "protein-sequence"
    )

    train_pairs = _read_pair_file(train_csv, drug_smiles, protein_sequences)
    test_pairs = _read_pair_file(test_csv, drug_smiles, protein_sequences)
    train_set, val_set, test_set = _make_train_val(
        train_pairs, test_pairs, args.val_total_fraction, seed=42 + fold
    )

    run_dir = Path("./result") / task / split / f"cv{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.RESULT.OUTPUT_DIR = str(run_dir)

    print("=" * 72)
    print(f"SGcCA | task={task} | split={split} | fold={fold}")
    print(f"Config YAML: {args.cfg}")
    print(f"DCGCN data root: {dti_root}")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV : {test_csv}")
    print(f"Output dir: {run_dir.resolve()}")
    print(f"Running on: {device}")
    print(f"Pairs after mapping: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print(f"Train label counts: {train_set['Y'].value_counts().to_dict()}")
    print(f"Val label counts: {val_set['Y'].value_counts().to_dict()}")
    print(f"Test label counts: {test_set['Y'].value_counts().to_dict()}")
    print(f"Hyperparameters: {dict(cfg)}")
    print("=" * 72)

    # Retain the original SGcCA dataset interface: SMILES, Protein, Y.
    train_dataset = DTIDataset(train_set.index.values, train_set)
    val_dataset = DTIDataset(val_set.index.values, val_set)
    test_dataset = DTIDataset(test_set.index.values, test_set)

    set_seed(cfg.SOLVER.SEED)
    model = SGcADTI(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    torch.backends.cudnn.benchmark = True

    common_params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "collate_fn": graph_collate_func,
    }
    train_generator = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **common_params
    )
    val_generator = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common_params
    )
    test_generator = DataLoader(
        test_dataset, shuffle=False, drop_last=False, **common_params
    )

    trainer = Trainer(
        model, optimizer, device,
        train_generator, val_generator, test_generator, **cfg
    )
    result = trainer.train()
    result_dict = _result_to_dict(result)

    # Keep the original Trainer's result/plot/checkpoint behavior.
    trainer.save_result(str(run_dir))

    result_row = {
        "task": task,
        "split": split,
        "fold": fold,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        **result_dict,
    }
    pd.DataFrame([result_row]).to_csv(run_dir / "metrics.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result_row), f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {run_dir / 'metrics.json'}")
    print(f"[DONE] task={task} split={split} fold={fold}")
    return result_row


if __name__ == "__main__":
    started = time()
    print(f"Start time: {datetime.now()}")
    output = main()
    print(output)
    print(f"End time: {datetime.now()}")
    print(f"Runtime: {(time() - started) / 3600:.4f} h")