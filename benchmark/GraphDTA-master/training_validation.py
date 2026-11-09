"""Train GraphDTA on DrugBank DTI data with a validation-based checkpoint.

A random fraction of the training split is held out for model selection.  The
best checkpoint according to validation AUC is then evaluated once on the test
split, which gives a cleaner test estimate than ``training.py`` (the latter
follows the original GraphDTA protocol of selecting on the test set).

Examples
--------
    python training_validation.py --split drug_scaffold --folds 0 --model ginconv
    python training_validation.py --split protein_similarity40 --folds 0 1 2 3 4
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
        description="Train GraphDTA for DTI with a held-out validation split."
    )
    add_common_args(parser)
    parser.add_argument("--valid_ratio", type=float, default=0.2,
                        help="Fraction of the training split used for validation "
                             "(default: %(default)s)")
    args = parser.parse_args()
    if not 0.0 < args.valid_ratio < 1.0:
        parser.error("--valid_ratio must be between 0 and 1")
    return args


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

    print("Device       : {}".format(device))
    print("Model        : {}".format(model_st))
    print("Split        : {}".format(args.split))
    print("Folds        : {}".format(args.folds))
    print("Epochs       : {} (patience {})".format(args.epochs, args.patience))
    print("Batch size   : {}".format(args.batch_size))
    print("Learning rate: {}".format(args.lr))
    print("Valid ratio  : {}".format(args.valid_ratio))
    print("Output dir   : {}".format(os.path.abspath(args.output_dir)))

    all_rows = []
    for fold_number, fold in enumerate(args.folds, 1):
        print("\n" + "=" * 78)
        print("Fold {} ({}/{})".format(fold, fold_number, len(args.folds)))
        print("=" * 78)
        set_seed(args.seed + fold)

        full_train = TestbedDataset(
            root=args.root, dataset="{}_train_cv{}".format(prefix, fold), log=False)
        test_data = TestbedDataset(
            root=args.root, dataset="{}_test_cv{}".format(prefix, fold), log=False)
        pairs = test_pairs(args.data_dir, args.split, fold)
        if len(pairs) != len(test_data):
            raise RuntimeError(
                "Test split has {} pairs but the processed dataset has {} samples. "
                "Re-run create_data.py.".format(len(pairs), len(test_data)))

        train_size = int((1.0 - args.valid_ratio) * len(full_train))
        train_size = max(1, min(train_size, len(full_train) - 1))
        valid_size = len(full_train) - train_size
        split_generator = torch.Generator()
        split_generator.manual_seed(args.seed + fold)
        train_data, valid_data = torch.utils.data.random_split(
            full_train, [train_size, valid_size], generator=split_generator)
        print("Train samples: {} | Valid samples: {} | Test samples: {}".format(
            len(train_data), len(valid_data), len(test_data)))

        train_loader = make_loader(
            train_data, args.batch_size, shuffle=True,
            seed=args.seed + fold, num_workers=args.num_workers)
        valid_loader = make_loader(
            valid_data, args.test_batch_size, shuffle=False,
            num_workers=args.num_workers)
        test_loader = make_loader(
            test_data, args.test_batch_size, shuffle=False,
            num_workers=args.num_workers)

        model = build_model(args.model, device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model_file = os.path.join(
            args.output_dir, "model_{}_{}_cv{}.model".format(model_st, prefix, fold))
        result_file = os.path.join(
            args.output_dir, "result_{}_{}_cv{}.csv".format(model_st, prefix, fold))
        prediction_file = os.path.join(
            args.output_dir, "predictions_{}_{}_cv{}.csv".format(model_st, prefix, fold))

        best_valid_score = -float("inf")
        best_valid_epoch = 0
        best_test_metrics = None
        best_labels = None
        best_probs = None
        bad_epochs = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            valid_metrics, _, _ = evaluate(
                model, valid_loader, device, threshold=args.threshold)
            valid_score = valid_metrics["auc"]
            if np.isnan(valid_score):
                valid_score = valid_metrics["balanced_accuracy"]

            improved = valid_score > best_valid_score + args.min_delta
            if improved:
                best_valid_score = valid_score
                best_valid_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), model_file)
                test_metrics, labels, probs = evaluate(
                    model, test_loader, device, threshold=args.threshold)
                best_test_metrics = dict(test_metrics)
                best_labels = labels
                best_probs = probs
            else:
                bad_epochs += 1

            if improved or epoch % args.log_interval == 0 or epoch == args.epochs:
                message = ("Epoch {:03d} | train_loss {:.4f} | valid_auc {:.4f} | "
                           "valid_ap {:.4f} | best_valid_auc {:.4f} @ epoch {:03d}".format(
                               epoch, train_loss, valid_metrics["auc"],
                               valid_metrics["auprc"], best_valid_score,
                               best_valid_epoch))
                if improved:
                    message += " | test_auc {:.4f} | test_ap {:.4f} *".format(
                        best_test_metrics["auc"], best_test_metrics["auprc"])
                print(message)

            if args.patience > 0 and bad_epochs >= args.patience:
                print("Early stopping after {} epochs without improvement.".format(bad_epochs))
                break

        if best_test_metrics is None:
            raise RuntimeError("Training finished without producing a checkpoint")
        row = result_row(args.model, args.split, fold, best_valid_epoch, best_test_metrics)
        row["best_valid_auc"] = float(best_valid_score)
        write_result_csv(result_file, row)
        write_predictions_csv(
            prediction_file, pairs, best_labels, best_probs, threshold=args.threshold)
        all_rows.append(row)

        print("\nBest valid epoch {} (valid AUC {:.4f}) | test AUC {:.4f} | "
              "test AUPRC {:.4f} | test accuracy {:.4f} | test F1 {:.4f}".format(
                  best_valid_epoch, best_valid_score, row["auc"], row["auprc"],
                  row["accuracy"], row["f1"]))
        print("Model       : {}".format(model_file))
        print("Metrics     : {}".format(result_file))
        print("Predictions : {}".format(prediction_file))

    summary_file = os.path.join(
        args.output_dir, "summary_{}_{}_validation.csv".format(model_st, args.split))
    update_summary_csv(summary_file, all_rows)
    print("\nCross-fold summary: {}".format(summary_file))
    print_metric_summary(all_rows)


if __name__ == "__main__":
    main()
