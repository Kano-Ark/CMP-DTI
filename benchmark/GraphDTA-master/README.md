# GraphDTA for DrugBank DTI

This repository is the original [GraphDTA](https://github.com/thinng/GraphDTA)
codebase converted from **drug-target affinity (DTA) regression** to
**drug-target interaction (DTI) binary classification** on DrugBank.

Each sample is a `(drug, protein)` pair and the model predicts the probability
that the pair interacts:

* `1` — interaction
* `0` — no interaction

The drug is encoded as a molecular graph and the protein as an amino-acid
sequence; the two branches are fused before a single interaction logit is
produced.  Training uses `BCEWithLogitsLoss`; applying a sigmoid to the logit
gives the interaction probability.

## Input data

All inputs live under `data/drugbank/`:

```
data/drugbank/
├── drug_SMILES.csv                 # drug_id,SMILES
├── protein_seq.csv                 # protein_id,amino_acid_sequence
└── result/
    ├── drug_scaffold/
    │   ├── train_cv0.csv ... train_cv4.csv
    │   └── test_cv0.csv  ... test_cv4.csv
    └── protein_similarity40/
        ├── train_cv0.csv ... train_cv4.csv
        └── test_cv0.csv  ... test_cv4.csv
```

`drug_SMILES.csv` and `protein_seq.csv` have two columns (`id,value`).
Each `train_cv{i}.csv` / `test_cv{i}.csv` has three columns
`drug_id,protein_id,label` with binary labels.  Header rows are optional; the
loader accepts files with or without a header.

The DrugBank data shipped here contains 6,876 drugs, 4,628 proteins, and five
cross-validation folds of about 30k training pairs and 7.5k test pairs each.
Preprocessing all five folds writes roughly 2.5 GB of `.pt` files; use
`--folds` to process a subset.

## Installation

```bash
conda create -n graphdta python=3.11
conda activate graphdta
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.6, PyTorch Geometric 2.8.1, RDKit 2025.03.3,
and scikit-learn 1.6.  The four GraphDTA backbones used here only need the pure
Python parts of PyTorch Geometric, so no `torch-scatter` / `torch-sparse`
compilation is required.  PyG may print a warning that `torch-scatter` is
missing; this is only a speed hint (optional to install).

## 1. Preprocess the data

```bash
# All folds of the drug-scaffold split
python create_data.py --split drug_scaffold

# Selected folds of the protein-similarity split
python create_data.py --split protein_similarity40 --folds 0 1
```

This reads the two CSV tables and the requested fold files, builds one
molecular graph per unique SMILES, encodes the protein sequences, and writes:

```
data/processed/drugbank_<split>_train_cv{i}.pt
data/processed/drugbank_<split>_test_cv{i}.pt
data/processed/drugbank_<split>_summary.csv
data/processed/drugbank_<split>_preprocess.json
```

Use `--force` to rebuild processed files that already exist.

## 2. Train a model

```bash
# One fold
python training.py --split drug_scaffold --folds 0 --model ginconv

# Full 5-fold cross-validation
python training.py --split drug_scaffold --folds 0 1 2 3 4 --model ginconv
```

Available backbones: `ginconv` (GINConvNet, default), `gat` (GATNet),
`gat_gcn` (GAT_GCN), and `gcn` (GCNNet).

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--folds` | `0` | Fold indices, e.g. `0 1 2 3 4` |
| `--model` | `ginconv` | Backbone name |
| `--epochs` | `200` | Maximum epochs |
| `--patience` | `30` | Early-stopping patience (`0` disables) |
| `--batch_size` | `512` | Training batch size |
| `--lr` | `5e-4` | Adam learning rate |
| `--threshold` | `0.5` | Probability threshold for accuracy/F1/MCC |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `--output_dir` | `results/` | Where models, metrics and predictions are written |

Note that `training.py` keeps the original GraphDTA model-selection protocol:
the checkpoint with the best **test** AUC is saved.  For an unbiased test
estimate, use `training_validation.py`.

## 3. Train with a validation split

```bash
python training_validation.py --split drug_scaffold --folds 0 --model ginconv
```

This holds out 20% of the training split (`--valid_ratio`), selects the
checkpoint with the best validation AUC, and evaluates it once on the test
split.

## Outputs

For each fold, `results/` contains:

```
model_<Model>_drugbank_<split>_cv<fold>.model
result_<Model>_drugbank_<split>_cv<fold>.csv
predictions_<Model>_drugbank_<split>_cv<fold>.csv
summary_<Model>_drugbank_<split>.csv            # training.py
summary_<Model>_drugbank_<split>_validation.csv # training_validation.py
```

`result_*.csv` reports:

* `auc` — area under the ROC curve
* `auprc` — area under the precision-recall curve (average precision)
* `accuracy`, `precision`, `recall` (sensitivity), `specificity`
* `f1`, `mcc`, `balanced_accuracy`
* `n`, `positives`, `negatives`, `threshold`

`best_epoch` is the epoch whose checkpoint was kept: the best test-AUC epoch in
`training.py`, or the best validation-AUC epoch in `training_validation.py`.

`predictions_*.csv` contains `drug_id, protein_id, label, probability,
prediction` for every test pair, in the same order as the input test CSV.

## Protein pLDDT / tier statistics

`protein_plddt_stats.py` merges `data/train.csv` and `data/test.csv`, keeps the
unique proteins that occur, and matches them against
`data/protein_structure_summary.csv`:

```bash
python protein_plddt_stats.py
```

By default it counts `overall_tier` values as **low / mid / high** (`medium` is
reported as `mid`) and averages `mean_plddt_all` over the unique used proteins.
Other summary columns can be selected explicitly:

```bash
python protein_plddt_stats.py --tier-column confidence_tier
python protein_plddt_stats.py --plddt-column mean_plddt_aligned
python protein_plddt_stats.py --output data/used_protein_plddt.csv
```

The last command also writes a per-protein `protein_id, tier, plddt` table.

## Notes

* Protein sequences are truncated/padded to **1,000 residues**, matching the
  original GraphDTA architecture (`Conv1d(in_channels=1000)` on the protein
  branch).  354 proteins in this DrugBank release are longer than 1,000
  residues.  Changing `MAX_SEQ_LEN` also requires changing the model input
  dimension.
* The legacy Davis/KIBA DTA files under `data/davis` and `data/kiba` are kept
  for reference but are no longer used by the pipeline.
* GraphDTA reference: Nguyen et al., *GraphDTA: Predicting drug-target binding
  affinity with graph neural networks*, Bioinformatics (2021).
