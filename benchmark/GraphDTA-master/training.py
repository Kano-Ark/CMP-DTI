"""Train a GraphDTA model on DrugBank DTI (binary interaction) data.

Model selection follows the original GraphDTA ``training.py`` protocol: after
every epoch the model is evaluated on the test split and the checkpoint with the
best test AUC is kept.  If you need an unbiased test estimate, use
``training_validation.py`` instead, which selects the model on a held-out part
of the training split.

Examples
--------
    python training.py --split drug_scaffold --folds 0 --model ginconv
    python training.py --split drug_scaffold --folds 0 1 2 3 4 --model gcn
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from train_utils import (
    MODEL_NAMES,
    add_common_args,
    build_model,
    evaluate,
    make_loader,
    print_metric_summary,
    require_processed,
    resolve_device,
    result_row,
    set_seed,
    test_pairs,
    train_epoch,
    update_summary_csv,
    write_predictions_csv,
    write_result_csv,
)
from utils import TestbedDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GraphDTA for DrugBank drug-target interaction prediction."
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    prefix = "drugbank_{}".format(args.split)
    model_st = MODEL_NAMES[args.model]
    train_names = ["{}_train_cv{}".format(prefix, fold) for fold in args.folds]
    test_names = ["{}_test_cv{}".format(prefix, fold) for fold in args.folds]
    require_processed(args.root, train_names + test_names, args.split, args.folds)

    os.makedirs(args.output_dir, exist_ok=True)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Device      : {}".format(device))
    print("Model       : {}".format(model_st))
    print("Split       : {}".format(args.split))
    print("Folds       : {}".format(args.folds))
    print("Epochs      : {} (patience {})".format(args.epochs, args.patience))
    print("Batch size  : {}".format(args.batch_size))
    print("Learning rate: {}".format(args.lr))
    print("Output dir  : {}".format(os.path.abspath(args.output_dir)))

    all_rows = []
    for fold_number, fold in enumerate(args.folds, 1):
        print("\n" + "=" * 78)
        print("Fold {} ({}/{})".format(fold, fold_number, len(args.folds)))
        print("=" * 78)
        set_seed(args.seed + fold)

        train_data = TestbedDataset(
            root=args.root, dataset="{}_train_cv{}".format(prefix, fold), log=False)
        test_data = TestbedDataset(
            root=args.root, dataset="{}_test_cv{}".format(prefix, fold), log=False)
        print("Train samples: {} | Test samples: {}".format(len(train_data), len(test_data)))

        train_loader = make_loader(
            train_data, args.batch_size, shuffle=True,
            seed=args.seed + fold, num_workers=args.num_workers)
        test_loader = make_loader(
            test_data, args.test_batch_size, shuffle=False,
            num_workers=args.num_workers)
        pairs = test_pairs(args.data_dir, args.split, fold)
        if len(pairs) != len(test_data):
            raise RuntimeError(
                "Test split has {} pairs but the processed dataset has {} samples. "
                "Re-run create_data.py.".format(len(pairs), len(test_data)))

        model = build_model(args.model, device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model_file = os.path.join(
            args.output_dir, "model_{}_{}_cv{}.model".format(model_st, prefix, fold))
        result_file = os.path.join(
            args.output_dir, "result_{}_{}_cv{}.csv".format(model_st, prefix, fold))
        prediction_file = os.path.join(
            args.output_dir, "predictions_{}_{}_cv{}.csv".format(model_st, prefix, fold))

        best_score = -float("inf")
        best_metrics = None
        best_epoch = 0
        best_labels = None
        best_probs = None
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            metrics, labels, probs = evaluate(
                model, test_loader, device, threshold=args.threshold)
            score = metrics["auc"]
            if np.isnan(score):
                score = metrics["balanced_accuracy"]

            improved = score > best_score + args.min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(metrics)
                best_labels = labels
                best_probs = probs
                bad_epochs = 0
                torch.save(model.state_dict(), model_file)
            else:
                bad_epochs += 1

            if improved or epoch % args.log_interval == 0 or epoch == args.epochs:
                print("Epoch {:03d} | train_loss {:.4f} | test_auc {:.4f} | "
                      "test_ap {:.4f} | test_acc {:.4f} | best_auc {:.4f} @ epoch {:03d} {}".format(
                          epoch, train_loss, metrics["auc"], metrics["auprc"],
                          metrics["accuracy"], best_score, best_epoch,
                          "*" if improved else ""))

            if args.patience > 0 and bad_epochs >= args.patience:
                print("Early stopping after {} epochs without improvement.".format(bad_epochs))
                break

        if best_metrics is None:
            raise RuntimeError("Training finished without producing a checkpoint")
        row = result_row(args.model, args.split, fold, best_epoch, best_metrics)
        write_result_csv(result_file, row)
        write_predictions_csv(
            prediction_file, pairs, best_labels, best_probs, threshold=args.threshold)
        all_rows.append(row)

        print("\nBest epoch {} | AUC {:.4f} | AUPRC {:.4f} | accuracy {:.4f} | "
              "F1 {:.4f} | MCC {:.4f}".format(
                  best_epoch, row["auc"], row["auprc"], row["accuracy"],
                  row["f1"], row["mcc"]))
        print("Model       : {}".format(model_file))
        print("Metrics     : {}".format(result_file))
        print("Predictions : {}".format(prediction_file))

    summary_file = os.path.join(
        args.output_dir, "summary_{}_{}.csv".format(model_st, args.split))
    update_summary_csv(summary_file, all_rows)
    print("\nCross-fold summary: {}".format(summary_file))
    print_metric_summary(all_rows)


if __name__ == "__main__":
    main()
