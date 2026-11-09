"""Training helpers shared by training.py and training_validation.py."""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import (
    DataLoader,
    TestbedDataset,
    classification_metrics,
    load_dti_pairs,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "drugbank")
DEFAULT_ROOT = os.path.join(BASE_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "results")

MODEL_REGISTRY = {
    "ginconv": GINConvNet,
    "gat": GATNet,
    "gat_gcn": GAT_GCN,
    "gcn": GCNNet,
}

MODEL_NAMES = {
    "ginconv": "GINConvNet",
    "gat": "GATNet",
    "gat_gcn": "GAT_GCN",
    "gcn": "GCNNet",
}


def add_common_args(parser):
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing drug_SMILES.csv, protein_seq.csv "
                             "and result/ (default: %(default)s)")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="PyG root directory containing processed/ "
                             "(default: %(default)s)")
    parser.add_argument("--split", default="drug_scaffold",
                        help="Split name under result/ (default: %(default)s)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0],
                        help="Fold indices to train (default: 0; use 0 1 2 3 4 for "
                             "the full 5-fold CV)")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="ginconv",
                        help="GraphDTA backbone (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stopping patience in epochs; 0 disables it "
                             "(default: %(default)s)")
    parser.add_argument("--min_delta", type=float, default=1e-5,
                        help="Minimum metric improvement to reset patience "
                             "(default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size (default: %(default)s)")
    parser.add_argument("--test_batch_size", type=int, default=512,
                        help="Evaluation batch size (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Adam learning rate (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Adam weight decay (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold used to compute accuracy/F1/etc. "
                             "(default: %(default)s)")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', 'cuda', or 'cuda:N' (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes (default: %(default)s)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for models, metrics and predictions "
                             "(default: %(default)s)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print progress every N epochs, and whenever the best "
                             "score improves (default: %(default)s)")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU instead.")
        return torch.device("cpu")
    return torch.device(name)


def build_model(model_name, device):
    model = MODEL_REGISTRY[model_name](n_output=1)
    return model.to(device)


def make_loader(dataset, batch_size, shuffle, seed=0, num_workers=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator if shuffle else None,
    )


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data).view(-1)
        loss = loss_fn(logits, data.y.view(-1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / max(1, total_samples)


@torch.no_grad()
def predict(model, loader, device):
    """Return labels and interaction probabilities in loader order."""
    model.eval()
    all_labels = []
    all_probs = []
    for data in loader:
        data = data.to(device)
        logits = model(data).view(-1)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(data.y.view(-1).cpu())
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.cat(all_probs).numpy()
    return labels, probs


def evaluate(model, loader, device, threshold=0.5):
    labels, probs = predict(model, loader, device)
    metrics = classification_metrics(labels, probs, threshold=threshold)
    return metrics, labels, probs


def test_pairs(data_dir, split, fold):
    """Load the test-pair table in the same order used by create_data.py."""
    path = os.path.join(data_dir, "result", split, "test_cv{}.csv".format(fold))
    return load_dti_pairs(path)


def result_row(model_name, split, fold, epoch, metrics):
    row = {
        "model": MODEL_NAMES[model_name],
        "split": split,
        "fold": int(fold),
        "best_epoch": int(epoch),
    }
    row.update(metrics)
    return row


def write_result_csv(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def update_summary_csv(path, rows, key=("model", "split", "fold")):
    """Merge rows into an existing summary CSV, keeping the newest value."""
    new = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(list(key), keep="last")
    else:
        combined = new
    combined = combined.sort_values(list(key)).reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def write_predictions_csv(path, pairs, labels, probs, threshold=0.5):
    """Write per-pair predictions; ``pairs`` must be in dataset order."""
    if len(pairs) != len(labels):
        raise ValueError(
            "Prediction/label length mismatch: {} pairs vs {} predictions".format(
                len(pairs), len(labels)))
    expected = pairs["label"].to_numpy().astype(int)
    if not np.array_equal(expected, np.asarray(labels).astype(int)):
        raise ValueError(
            "Dataset labels do not align with the pair table. The processed "
            "dataset may be stale; re-run create_data.py.")
    out = pairs[["drug_id", "protein_id"]].copy()
    out["label"] = expected
    out["probability"] = np.asarray(probs, dtype=float)
    out["prediction"] = (out["probability"] >= threshold).astype(int)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return path


def processed_path(root, dataset_name):
    return os.path.join(os.path.abspath(root), "processed", dataset_name + ".pt")


def require_processed(root, dataset_names, split, folds):
    missing = [name for name in dataset_names if not os.path.isfile(processed_path(root, name))]
    if missing:
        raise SystemExit(
            "Missing processed dataset(s):\n  {}\n"
            "Run: python create_data.py --split {} --folds {}".format(
                "\n  ".join(missing), split, " ".join(str(f) for f in folds)))


def print_metric_summary(rows, metric_names=("auc", "auprc", "accuracy", "f1", "mcc")):
    if not rows:
        return
    print("\n===== Summary over {} fold(s) =====".format(len(rows)))
    for metric in metric_names:
        values = np.array([row[metric] for row in rows], dtype=float)
        print("  {:18s} {:.4f} +/- {:.4f}".format(metric, values.mean(), values.std()))
