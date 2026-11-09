#!/usr/bin/env python
"""Summarise pLDDT / confidence tiers for proteins used by DTI train+test.

The script:

1. reads ``data/train.csv`` and ``data/test.csv`` (three columns:
   ``drug_id, protein_id, label``);
2. merges them and keeps the unique protein IDs that actually occur;
3. matches those proteins against ``data/protein_structure_summary.csv``;
4. reports how many used proteins are ``low`` / ``mid`` / ``high`` according to
   the summary's ``overall_tier`` column (``medium`` is reported as ``mid``);
5. reports the mean pLDDT of the used proteins (default column:
   ``mean_plddt_all``).

Examples
--------
    python protein_plddt_stats.py
    python protein_plddt_stats.py --tier-column confidence_tier
    python protein_plddt_stats.py --plddt-column mean_plddt_aligned
    python protein_plddt_stats.py --output data/used_protein_plddt.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN = os.path.join(BASE_DIR, "data", "train.csv")
DEFAULT_TEST = os.path.join(BASE_DIR, "data", "test.csv")
DEFAULT_SUMMARY = os.path.join(BASE_DIR, "data", "protein_structure_summary.csv")

# Both ``overall_tier`` (low/medium/high) and ``confidence_tier``
# (very_low/low/confident/very_high) are mapped onto low/mid/high.
TIER_ALIASES = {
    "very_low": "low",
    "low": "low",
    "confident": "mid",
    "medium": "mid",
    "mid": "mid",
    "very_high": "high",
    "high": "high",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count low/mid/high proteins used by train+test and compute "
                    "their mean pLDDT."
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN,
                        help="Training pair CSV (default: %(default)s)")
    parser.add_argument("--test", default=DEFAULT_TEST,
                        help="Test pair CSV (default: %(default)s)")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY,
                        help="Protein structure summary CSV (default: %(default)s)")
    parser.add_argument("--tier-column", default="overall_tier",
                        help="Tier column in the summary (default: %(default)s)")
    parser.add_argument("--plddt-column", default="mean_plddt_all",
                        help="pLDDT column in the summary (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="Optional CSV with one row per used protein")
    return parser.parse_args()


def read_pair_file(path):
    """Read a DTI pair file with columns drug_id, protein_id, label."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Pair file not found: {}".format(path))

    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if df.shape[1] < 3:
        raise ValueError(
            "{} has {} column(s); expected at least 3 "
            "(drug_id, protein_id, label)".format(path, df.shape[1]))

    df = df.iloc[:, :3].copy()
    df.columns = ["drug_id", "protein_id", "label"]
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()

    if df.empty:
        raise ValueError("{} contains no rows".format(path))

    # The provided files have no header; drop one if it is present.
    first_label = pd.to_numeric(pd.Series([df.iloc[0]["label"]]), errors="coerce").iloc[0]
    if pd.isna(first_label) or first_label not in (0, 1):
        print("[info] {}: treating the first row as a header".format(path))
        df = df.iloc[1:].reset_index(drop=True)

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any():
        examples = df.loc[labels.isna(), "label"].head(5).tolist()
        raise ValueError(
            "{} contains non-numeric labels: {}".format(path, examples))
    if not labels.isin([0, 1]).all():
        raise ValueError("{} contains labels other than 0/1".format(path))
    df["label"] = labels.astype(int)

    if (df["drug_id"] == "").any() or (df["protein_id"] == "").any():
        raise ValueError("{} contains empty drug_id/protein_id values".format(path))
    return df


def normalise_tier(value):
    text = str(value).strip().lower()
    return TIER_ALIASES.get(text, text)


def main():
    args = parse_args()

    train = read_pair_file(args.train)
    test = read_pair_file(args.test)
    merged = pd.concat([train, test], ignore_index=True)

    used_ids = pd.Index(merged["protein_id"].drop_duplicates())
    train_ids = set(train["protein_id"])
    test_ids = set(test["protein_id"])

    print("\n===== DTI pair set =====")
    print("Train pairs      : {:,}".format(len(train)))
    print("Test pairs       : {:,}".format(len(test)))
    print("Merged pairs     : {:,}".format(len(merged)))
    print("Unique proteins  : {:,} (train {:,}, test {:,}, overlap {:,})".format(
        len(used_ids), len(train_ids), len(test_ids), len(train_ids & test_ids)))

    summary = pd.read_csv(args.summary, dtype={"protein_id": str},
                          keep_default_na=False)
    for column in ("protein_id", args.tier_column, args.plddt_column):
        if column not in summary.columns:
            raise SystemExit(
                "Column '{}' not found in {}. Available columns: {}".format(
                    column, args.summary, ", ".join(summary.columns)))
    summary["protein_id"] = summary["protein_id"].astype(str).str.strip()

    duplicated = summary["protein_id"].duplicated(keep=False)
    if duplicated.any():
        examples = summary.loc[duplicated, "protein_id"].drop_duplicates().head(5).tolist()
        print("[warning] duplicate protein_id values in summary (e.g. {}); "
              "keeping the first row per protein".format(examples))
        summary = summary.drop_duplicates("protein_id", keep="first")

    used = summary[summary["protein_id"].isin(set(used_ids))].copy()
    missing = sorted(set(used_ids) - set(used["protein_id"]))

    used["_tier"] = used[args.tier_column].map(normalise_tier)
    used["_plddt"] = pd.to_numeric(used[args.plddt_column], errors="coerce")

    tier_counts = used["_tier"].value_counts()
    total_used = len(used_ids)
    total_tiered = int(tier_counts.sum())

    print("\n===== Protein confidence tiers ({}) =====".format(args.tier_column))
    for tier in ("low", "mid", "high"):
        print("  {:4s}: {:,}".format(tier, int(tier_counts.get(tier, 0))))
    unknown = tier_counts.drop(labels=["low", "mid", "high"], errors="ignore")
    for tier, count in unknown.items():
        print("  {:4s}: {:,}".format(tier, int(count)))
    if total_tiered != total_used:
        print("  (tier counts cover {:,} of {:,} used proteins)".format(
            total_tiered, total_used))

    if missing:
        print("\n[warning] {:,} used protein(s) not found in {}:".format(
            len(missing), args.summary))
        print("  " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))

    valid = used.dropna(subset=["_plddt"])
    print("\n===== pLDDT of used proteins ({}) =====".format(args.plddt_column))
    print("Proteins with pLDDT: {:,} / {:,}".format(len(valid), total_used))
    if len(valid):
        values = valid["_plddt"].to_numpy(dtype=float)
        print("Mean pLDDT         : {:.2f}".format(values.mean()))
        print("Median pLDDT       : {:.2f}".format(float(np.median(values))))
        print("Std / min / max    : {:.2f} / {:.2f} / {:.2f}".format(
            values.std(ddof=1), values.min(), values.max()))

        print("\nPer-tier pLDDT:")
        per_tier = (valid.groupby("_tier")["_plddt"]
                    .agg(count="count", mean="mean", std="std")
                    .reset_index()
                    .rename(columns={"_tier": "tier", "count": "proteins",
                                     "mean": "mean_plddt", "std": "std_plddt"}))
        print(per_tier.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    else:
        print("No pLDDT values available for the used proteins.")

    if args.output:
        output_columns = ["protein_id", args.tier_column, args.plddt_column]
        out = used[output_columns].copy()
        out.columns = ["protein_id", "tier", "plddt"]
        out["tier"] = out["tier"].map(normalise_tier)
        out.to_csv(args.output, index=False)
        print("\nPer-protein table written to {}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
