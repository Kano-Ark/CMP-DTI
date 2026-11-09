"""Shared utilities for the DrugBank DTI version of GraphDTA.

The original GraphDTA pipeline predicts drug-target affinity (DTA) as a
regression value.  This version predicts drug-target interaction (DTI) as a
binary label:

    1 -> the drug and the protein interact
    0 -> they do not interact

Expected input files (under ``data/drugbank`` by default)::

    drug_SMILES.csv                     drug_id,SMILES
    protein_seq.csv                     protein_id,amino_acid_sequence
    result/<split>/train_cv{i}.csv      drug_id,protein_id,label
    result/<split>/test_cv{i}.csv       drug_id,protein_id,label

The CSV files may or may not contain a header row; both are handled.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data, InMemoryDataset

try:  # PyTorch Geometric >= 2.0
    from torch_geometric.loader import DataLoader
except ImportError:  # pragma: no cover - older PyG releases
    from torch_geometric.data import DataLoader

# Protein sequences are encoded with the standard 20 amino-acid alphabet used
# by the original GraphDTA code.  Index 0 is reserved for padding (and for any
# non-standard residue, which is mapped to padding).
SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}

# GraphDTA's protein branch uses a Conv1d with in_channels=1000, i.e. the
# sequence length is part of the architecture.  Keep this fixed unless the
# model code is changed as well.
MAX_SEQ_LEN = 1000


def seq_cat(prot, max_seq_len=MAX_SEQ_LEN):
    """Encode an amino-acid sequence as a fixed-length integer vector."""
    x = np.zeros(max_seq_len, dtype=np.int64)
    for i, ch in enumerate(str(prot)[:max_seq_len]):
        x[i] = SEQ_DICT.get(ch.upper(), 0)
    return x


def _read_id_table(path):
    """Read a two-column ``id,value`` CSV, with or without a header."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Input file not found: {}".format(path))
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                     skipinitialspace=True)
    if df.empty or df.shape[1] < 2:
        raise ValueError("File {} does not contain at least two columns".format(path))
    first_cell = str(df.iloc[0, 0]).strip().lower()
    if first_cell in {"drug_id", "protein_id", "compound_id", "entity_id", "id"}:
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :2].copy()
    df.columns = ["entity_id", "value"]
    df["entity_id"] = df["entity_id"].str.strip()
    df["value"] = df["value"].str.strip()
    df = df[(df["entity_id"] != "") & (df["value"] != "")].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows found in {}".format(path))
    return df


def _table_to_dict(df, path):
    """Convert an id/value table to a dict, rejecting conflicting duplicates."""
    nunique = df.groupby("entity_id")["value"].nunique()
    conflicts = nunique[nunique > 1]
    if len(conflicts) > 0:
        examples = ", ".join(conflicts.index[:5].tolist())
        raise ValueError(
            "{} contains IDs mapped to different values (e.g. {}).".format(path, examples)
        )
    df = df.drop_duplicates("entity_id")
    return dict(zip(df["entity_id"], df["value"]))


def load_drug_smiles(path):
    """Load ``drug_id -> SMILES`` from ``drug_SMILES.csv``."""
    return _table_to_dict(_read_id_table(path), path)


def load_protein_sequences(path):
    """Load ``protein_id -> sequence`` from ``protein_seq.csv``."""
    return _table_to_dict(_read_id_table(path), path)


def load_dti_pairs(path):
    """Load a DTI split file as a DataFrame with drug_id/protein_id/label.

    Labels are validated to be binary (0/1).  Duplicate (drug, protein) pairs
    are rejected because silently keeping one of them could change the label
    distribution of a split.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Split file not found: {}".format(path))
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                     skipinitialspace=True)
    if df.empty or df.shape[1] < 3:
        raise ValueError("File {} does not contain three columns".format(path))
    first_cell = str(df.iloc[0, 0]).strip().lower()
    if first_cell in {"drug_id", "compound_id", "id"}:
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :3].copy()
    df.columns = ["drug_id", "protein_id", "label"]
    for col in ("drug_id", "protein_id", "label"):
        df[col] = df[col].str.strip()

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any():
        bad = df.loc[labels.isna(), "label"].head(5).tolist()
        raise ValueError("Non-numeric labels in {}: {}".format(path, bad))
    df["label"] = labels.astype(int)
    if not df["label"].isin([0, 1]).all():
        raise ValueError("DTI labels must be 0 or 1 in {}".format(path))

    df = df[(df["drug_id"] != "") & (df["protein_id"] != "")].reset_index(drop=True)
    dup = df.duplicated(["drug_id", "protein_id"], keep=False)
    if dup.any():
        examples = df.loc[dup, ["drug_id", "protein_id"]].head(5).values.tolist()
        raise ValueError("Duplicate drug-protein pairs in {}: {}".format(path, examples))
    return df


class TestbedDataset(InMemoryDataset):
    """PyTorch-Geometric dataset for drug-target interaction pairs.

    Each sample stores:

    * ``x``          atom feature matrix of the drug graph
    * ``edge_index`` directed bond list of the drug graph
    * ``target``     encoded protein sequence, shape ``(1, MAX_SEQ_LEN)``
    * ``y``          binary interaction label, shape ``(1,)``
    * ``c_size``     number of atoms in the drug graph, shape ``(1,)``

    ``target`` is stored as a 2-D tensor so that PyG concatenates samples along
    the batch dimension (giving ``(batch_size, MAX_SEQ_LEN)``).
    """

    def __init__(self, root="data", dataset="drugbank_drug_scaffold_train_cv0",
                 xd=None, xt=None, y=None, transform=None, pre_transform=None,
                 smile_graph=None, log=True):
        self.dataset = dataset
        super(TestbedDataset, self).__init__(
            root=root, transform=transform, pre_transform=pre_transform, log=log
        )
        if os.path.isfile(self.processed_paths[0]):
            data, slices = self._torch_load(self.processed_paths[0])
            self.data, self.slices = data, slices
        else:
            self.process(xd, xt, y, smile_graph)
            data, slices = self._torch_load(self.processed_paths[0])
            self.data, self.slices = data, slices

    @staticmethod
    def _torch_load(path):
        """Load a processed file on both old and new PyTorch versions."""
        try:
            out = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 1.13
            out = torch.load(path, map_location="cpu")
        if isinstance(out, tuple) and len(out) == 3:  # PyG >= 2.4 save format
            data, slices, _ = out
        else:
            data, slices = out
        return data, slices

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def _download(self):
        # Inputs are read by create_data.py; nothing to download.
        pass

    def _process(self):
        # The processed data is built explicitly in __init__ (it needs the
        # xd/xt/y/smile_graph arguments, which PyG's automatic process() call
        # cannot provide).
        os.makedirs(self.processed_dir, exist_ok=True)

    def process(self, xd, xt, y, smile_graph):
        if xd is None or xt is None or y is None or smile_graph is None:
            raise ValueError(
                "No processed file for '{}' and no raw arrays were supplied. "
                "Run create_data.py first.".format(self.dataset)
            )
        assert len(xd) == len(xt) == len(y), "xd, xt and y must have the same length"

        data_list = []
        data_len = len(xd)
        print_every = max(1, data_len // 10)
        for i in range(data_len):
            if i == 0 or (i + 1) % print_every == 0 or i + 1 == data_len:
                print("  converting graphs: {}/{}".format(i + 1, data_len))
            smiles = xd[i]
            c_size, features, edge_index = smile_graph[smiles]

            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            GCNData = Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor([float(y[i])], dtype=torch.float),
            )
            # Shape (1, MAX_SEQ_LEN): see the class docstring.
            GCNData.target = torch.tensor(
                np.asarray(xt[i], dtype=np.int64).reshape(1, -1), dtype=torch.long
            )
            GCNData.c_size = torch.tensor([int(c_size)], dtype=torch.long)
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Graph construction done. Saving to {} ...".format(self.processed_paths[0]))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def classification_metrics(y_true, y_prob, threshold=0.5):
    """Compute binary DTI metrics from labels and predicted probabilities."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size != y_prob.size:
        raise ValueError("y_true and y_prob must have the same length")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    has_both_classes = len(np.unique(y_true)) == 2
    auc = roc_auc_score(y_true, y_prob) if has_both_classes else float("nan")
    auprc = average_precision_score(y_true, y_prob) if has_both_classes else float("nan")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "n": int(y_true.size),
        "positives": int(tp + fn),
        "negatives": int(tn + fp),
        "auc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
