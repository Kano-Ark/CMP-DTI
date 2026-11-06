#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate SGcCA DrugBank scaffold/protein-similarity CV fold metrics."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = {
    "AUROC": ("auroc", "AUROC", "auc", "AUC", "roc_auc"),
    "AUPRC": ("auprc", "AUPRC", "aupr", "AUPR", "average_precision"),
    "Accuracy": ("accuracy", "Accuracy", "acc", "ACC"),
    "Precision": ("precision", "Precision"),
    "Recall": ("recall", "Recall", "sensitivity", "SEN"),
    "F1": ("f1", "F1", "f1_score"),
    "MCC": ("mcc", "MCC"),
}


def pick_metric(record, aliases):
    for key in aliases:
        if key in record and record[key] is not None and record[key] != "":
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SGcCA CV results as mean ± sample SD."
    )
    parser.add_argument("--task", default="drugbank")
    parser.add_argument("--splits", nargs="+",
                        default=["drug_scaffold", "protein_similarity40"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--result-root", default=None,
                        help="Default: ./result/<task>")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--decimals", type=int, default=4)
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    result_root = Path(args.result_root) if args.result_root else Path("result") / args.task
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "cv_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []
    missing = []

    for split in args.splits:
        rows = []
        for fold in range(args.folds):
            metrics_path = result_root / split / f"cv{fold}" / "metrics.json"
            if not metrics_path.is_file():
                missing.append(str(metrics_path))
                continue
            with metrics_path.open("r", encoding="utf-8") as f:
                record = json.load(f)

            row = {
                "task": record.get("task", args.task),
                "split": record.get("split", split),
                "fold": record.get("fold", fold),
                "n_train": record.get("n_train", ""),
                "n_val": record.get("n_val", ""),
                "n_test": record.get("n_test", ""),
            }
            for metric, aliases in METRICS.items():
                row[metric] = pick_metric(record, aliases)
            rows.append(row)
            detail_rows.append(row)

        if len(rows) != args.folds:
            continue

        summary = {"task": args.task, "split": split, "n_folds": len(rows)}
        for metric in METRICS:
            vals = [row[metric] for row in rows if row[metric] is not None]
            mean = statistics.mean(vals) if vals else float("nan")
            sd = statistics.stdev(vals) if len(vals) >= 2 else float("nan")
            summary[f"{metric}_n"] = len(vals)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_SD"] = sd
            if math.isfinite(mean) and math.isfinite(sd):
                summary[f"{metric}_mean±SD"] = (
                    f"{mean:.{args.decimals}f} ± {sd:.{args.decimals}f}"
                )
            else:
                summary[f"{metric}_mean±SD"] = "NaN ± NaN"
        summary_rows.append(summary)

    if missing:
        print("ERROR: metrics.json is missing for these folds; summary not written:")
        for path in missing:
            print("  " + path)
        print("Complete the requested runs, then run this script again.")
        return 2

    detail_path = output_dir / "fold_metrics.csv"
    detail_fields = ["task", "split", "fold", "n_train", "n_val", "n_test", *METRICS.keys()]
    with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_path = output_dir / "summary_mean_sd.csv"
    summary_fields = ["task", "split", "n_folds"]
    for metric in METRICS:
        summary_fields.extend([
            f"{metric}_n", f"{metric}_mean", f"{metric}_SD", f"{metric}_mean±SD"
        ])
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-fold metrics: {detail_path}")
    print(f"Saved mean±SD summary:  {summary_path}")
    for row in summary_rows:
        print(f"\n[{row['task']} / {row['split']}]")
        for metric in METRICS:
            print(f"  {metric}: {row[f'{metric}_mean±SD']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
