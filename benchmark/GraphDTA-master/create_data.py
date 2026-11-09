"""Prepare DrugBank drug-target interaction (DTI) data for GraphDTA.

This replaces the original Davis/KIBA DTA preprocessing.  It reads

    data/drugbank/drug_SMILES.csv                 drug_id,SMILES
    data/drugbank/protein_seq.csv                 protein_id,sequence
    data/drugbank/result/<split>/train_cv{i}.csv  drug_id,protein_id,label
    data/drugbank/result/<split>/test_cv{i}.csv   drug_id,protein_id,label

and writes PyTorch-Geometric datasets to

    data/processed/drugbank_<split>_{train,test}_cv{i}.pt

Labels are binary: 1 = interaction, 0 = no interaction.

Example
-------
    python create_data.py --split drug_scaffold
    python create_data.py --split protein_similarity40 --folds 0 1 2
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem

from utils import (
    MAX_SEQ_LEN,
    TestbedDataset,
    load_drug_smiles,
    load_dti_pairs,
    load_protein_sequences,
    seq_cat,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "drugbank")
DEFAULT_ROOT = os.path.join(BASE_DIR, "data")


def _implicit_valence(atom):
    """Return implicit valence without triggering RDKit deprecation warnings."""
    try:
        return atom.GetValence(Chem.ValenceType.IMPLICIT)
    except (AttributeError, TypeError):  # older RDKit
        return atom.GetImplicitValence()


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
                "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
                "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge",
                "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg",
                "Pb", "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            _implicit_valence(atom), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Map inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """Convert a SMILES string to (num_atoms, atom_features, directed_edges)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES: {}".format(smile))

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / feature.sum())
    features = np.asarray(features, dtype=np.float32)

    edges = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append([begin, end])
        edges.append([end, begin])

    return c_size, features, edges


def discover_folds(split_dir):
    """Return the sorted fold indices that have both train and test files."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError("Split directory not found: {}".format(split_dir))

    folds = []
    for name in sorted(os.listdir(split_dir)):
        match = re.fullmatch(r"train_cv(\d+)\.csv", name)
        if match is None:
            continue
        fold = int(match.group(1))
        test_path = os.path.join(split_dir, "test_cv{}.csv".format(fold))
        if os.path.isfile(test_path):
            folds.append(fold)
        else:
            print("WARNING: {} has no matching {}".format(
                os.path.join(split_dir, name), os.path.basename(test_path)))
    return folds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create DrugBank DTI datasets in PyTorch-Geometric format."
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing drug_SMILES.csv, protein_seq.csv "
                             "and result/ (default: %(default)s)")
    parser.add_argument("--split", default="drug_scaffold",
                        help="Sub-directory of result/ holding the CV split files "
                             "(default: %(default)s)")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Fold indices to process (default: all folds found)")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="PyG root directory; processed files go to "
                             "<root>/processed (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-create processed files that already exist")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    root = os.path.abspath(args.root)
    processed_dir = os.path.join(root, "processed")
    split_dir = os.path.join(data_dir, "result", args.split)

    drug_path = os.path.join(data_dir, "drug_SMILES.csv")
    protein_path = os.path.join(data_dir, "protein_seq.csv")

    print("Data directory : {}".format(data_dir))
    print("Split          : {}".format(args.split))
    print("Processed dir  : {}".format(processed_dir))

    drug_smiles = load_drug_smiles(drug_path)
    protein_seq = load_protein_sequences(protein_path)
    print("Drugs          : {}".format(len(drug_smiles)))
    print("Proteins       : {}".format(len(protein_seq)))

    available_folds = discover_folds(split_dir)
    if not available_folds:
        raise ValueError("No train_cv*.csv/test_cv*.csv pairs found in {}".format(split_dir))
    folds = available_folds if args.folds is None else sorted(set(args.folds))
    missing = [f for f in folds if f not in available_folds]
    if missing:
        raise ValueError(
            "Fold(s) {} not found under {}. Available folds: {}".format(
                missing, split_dir, available_folds))
    print("Folds          : {}".format(folds))

    # Load all requested splits and validate that every ID can be mapped.
    fold_data = {}
    for fold in folds:
        train_df = load_dti_pairs(os.path.join(split_dir, "train_cv{}.csv".format(fold)))
        test_df = load_dti_pairs(os.path.join(split_dir, "test_cv{}.csv".format(fold)))
        for split_name, df in (("train", train_df), ("test", test_df)):
            missing_drugs = sorted(set(df["drug_id"]) - set(drug_smiles))
            missing_prots = sorted(set(df["protein_id"]) - set(protein_seq))
            if missing_drugs:
                raise ValueError(
                    "Fold {} {} has {} drug ID(s) absent from drug_SMILES.csv: {}".format(
                        fold, split_name, len(missing_drugs), missing_drugs[:10]))
            if missing_prots:
                raise ValueError(
                    "Fold {} {} has {} protein ID(s) absent from protein_seq.csv: {}".format(
                        fold, split_name, len(missing_prots), missing_prots[:10]))
        fold_data[fold] = {"train": train_df, "test": test_df}
        print("Fold {}: train={} (pos={}, neg={}), test={} (pos={}, neg={})".format(
            fold,
            len(train_df), int((train_df["label"] == 1).sum()), int((train_df["label"] == 0).sum()),
            len(test_df), int((test_df["label"] == 1).sum()), int((test_df["label"] == 0).sum()),
        ))

    # Decide which datasets still need to be built.
    pending = []
    for fold in folds:
        for split_name in ("train", "test"):
            df = fold_data[fold][split_name]
            dataset_name = "drugbank_{}_{}_cv{}".format(args.split, split_name, fold)
            processed_path = os.path.join(processed_dir, dataset_name + ".pt")
            if os.path.isfile(processed_path) and not args.force:
                print("[skip] {} already exists".format(processed_path))
            else:
                pending.append((fold, split_name, dataset_name, processed_path, df))

    # Metadata covers all requested folds, even the ones that were skipped.
    all_smiles = {
        drug_smiles[drug_id]
        for split_data in fold_data.values()
        for df in split_data.values()
        for drug_id in df["drug_id"]
    }
    all_sequences = {
        protein_seq[protein_id]
        for split_data in fold_data.values()
        for df in split_data.values()
        for protein_id in df["protein_id"]
    }
    truncated = sum(1 for seq in all_sequences if len(seq) > MAX_SEQ_LEN)

    os.makedirs(processed_dir, exist_ok=True)

    if pending:
        # Build each unique molecular graph once and reuse it for every dataset.
        needed_smiles = sorted({
            drug_smiles[drug_id]
            for (_, _, _, _, df) in pending
            for drug_id in df["drug_id"]
        })
        print("\nBuilding molecular graphs for {} unique SMILES ...".format(
            len(needed_smiles)))
        smile_graph = {}
        for i, smiles in enumerate(needed_smiles, 1):
            smile_graph[smiles] = smile_to_graph(smiles)
            if i % 500 == 0 or i == len(needed_smiles):
                print("  graphs: {}/{}".format(i, len(needed_smiles)))

        # Encode each unique protein sequence once.
        needed_sequences = sorted({
            protein_seq[protein_id]
            for (_, _, _, _, df) in pending
            for protein_id in df["protein_id"]
        })
        print("Encoding {} unique protein sequences (max length {}) ...".format(
            len(needed_sequences), MAX_SEQ_LEN))
        encoded_seq = {seq: seq_cat(seq) for seq in needed_sequences}

        for fold, split_name, dataset_name, processed_path, df in pending:
            if os.path.isfile(processed_path):
                os.remove(processed_path)
            xd = [drug_smiles[drug_id] for drug_id in df["drug_id"]]
            xt = [encoded_seq[protein_seq[pid]] for pid in df["protein_id"]]
            y = df["label"].astype(float).tolist()
            print("\nPreparing {} ({} pairs)".format(dataset_name, len(df)))
            TestbedDataset(
                root=root,
                dataset=dataset_name,
                xd=xd,
                xt=xt,
                y=y,
                smile_graph=smile_graph,
                log=True,
            )
    else:
        print("\nAll requested processed datasets already exist; nothing to build.")

    summary_rows = []
    for fold in folds:
        for split_name in ("train", "test"):
            df = fold_data[fold][split_name]
            dataset_name = "drugbank_{}_{}_cv{}".format(args.split, split_name, fold)
            summary_rows.append({
                "split": args.split,
                "fold": fold,
                "subset": split_name,
                "dataset": dataset_name,
                "pairs": len(df),
                "positives": int((df["label"] == 1).sum()),
                "negatives": int((df["label"] == 0).sum()),
            })

    summary_path = os.path.join(
        processed_dir, "drugbank_{}_summary.csv".format(args.split))
    new_summary = pd.DataFrame(summary_rows)
    if os.path.isfile(summary_path):
        old_summary = pd.read_csv(summary_path)
        summary = pd.concat([old_summary, new_summary], ignore_index=True)
        summary = summary.drop_duplicates(["split", "fold", "subset"], keep="last")
    else:
        summary = new_summary
    summary = summary.sort_values(["split", "fold", "subset"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    print("\nDataset summary written to {}".format(summary_path))

    metadata = {
        "data_dir": data_dir,
        "split": args.split,
        "folds": folds,
        "max_seq_len": MAX_SEQ_LEN,
        "drugs": len(drug_smiles),
        "proteins": len(protein_seq),
        "unique_smiles_in_folds": len(all_smiles),
        "truncated_proteins": truncated,
    }
    metadata_path = os.path.join(
        processed_dir, "drugbank_{}_preprocess.json".format(args.split))
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Preprocessing metadata written to {}".format(metadata_path))


if __name__ == "__main__":
    main()
