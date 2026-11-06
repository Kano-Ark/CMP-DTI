#!/usr/bin/env python3
"""Summarize PointDPI CV metrics saved by PointDPI_train_cv5.py."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

METRICS = ("AUC", "AUPR", "ACC", "SEN", "SPE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bindingdb")
    parser.add_argument(
        "--splits", nargs="+",
        default=["drug_scaffold", "protein_similarity40"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_root = script_dir / "runs_cv" / args.task
    summary_dir = Path(args.output_dir) if args.output_dir else run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    summary_rows = []
    missing = []

    for split in args.splits:
        split_rows = []
        for fold in range(args.folds):
            metric_path = run_root / split / f"cv{fold}" / "metrics.json"
            if not metric_path.exists():
                missing.append(str(metric_path))
                continue
            with metric_path.open() as f:
                row = json.load(f)
            row.setdefault("task", args.task)
            row.setdefault("split", split)
            row.setdefault("fold", fold)
            split_rows.append(row)
            fold_rows.append({
                "task": row.get("task", args.task),
                "split": row.get("split", split),
                "fold": row.get("fold", fold),
                "seed": row.get("seed", ""),
                "best_epoch": row.get("best_epoch", ""),
                "n_train": row.get("n_train", ""),
                "n_test": row.get("n_test", ""),
                **{m: row.get(m, "") for m in METRICS},
                "model_path": row.get("model_path", ""),
            })

        if len(split_rows) != args.folds:
            continue

        summary = {
            "task": args.task,
            "split": split,
            "n_folds": len(split_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in split_rows]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) >= 2 else float("nan")
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_SD"] = sd
            summary[f"{metric}_mean±SD"] = f"{mean:.4f} ± {sd:.4f}"
        summary_rows.append(summary)

    if missing:
        print("ERROR: missing fold metrics files; summary was not considered complete:")
        for path in missing:
            print(f"  {path}")
        print("Run the missing folds first, then rerun this script.")
        return 2

    fold_fields = [
        "task", "split", "fold", "seed", "best_epoch", "n_train", "n_test",
        *METRICS, "model_path"
    ]
    fold_csv = summary_dir / "fold_metrics.csv"
    with fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(fold_rows)

    summary_fields = ["task", "split", "n_folds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_SD", f"{metric}_mean±SD"])
    summary_csv = summary_dir / "summary_mean_sd.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-fold metrics: {fold_csv}")
    print(f"Saved mean±SD summary:  {summary_csv}")
    for row in summary_rows:
        print(f"\n[{row['task']} / {row['split']}]")
        for metric in METRICS:
            print(f"  {metric}: {row[f'{metric}_mean±SD']}")


if __name__ == "__main__":
    raise SystemExit(main())
