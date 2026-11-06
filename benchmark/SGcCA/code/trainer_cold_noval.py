import os

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from models import binary_cross_entropy, cross_entropy_logits


class Trainer(object):
    """
    No-validation trainer for predefined cold-start splits.

    It trains on the whole train split and evaluates the final epoch model on test.
    No model selection or threshold tuning is performed on validation/test data.
    For threshold-dependent metrics, a fixed threshold of 0.5 is used.
    """

    def __init__(self, model, optim, device, train_dataloader, test_dataloader, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["DECODER"]["BINARY"]

        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.lr_decay = config["SOLVER"]["LR_DECAY"]
        self.decay_interval = config["SOLVER"]["DECAY_INTERVAL"]
        self.use_ld = config["SOLVER"]["USE_LD"]

        self.train_loss_epoch = []
        self.test_metrics = {}
        self.config = config
        self.y_label = None
        self.y_score = None
        self.threshold = 0.5

    def train(self):
        for _ in range(self.epochs):
            self.current_epoch += 1
            if self.use_ld and self.current_epoch % self.decay_interval == 0:
                self.optim.param_groups[0]["lr"] *= self.lr_decay

            train_loss = self.train_epoch()
            self.train_loss_epoch.append(train_loss)

        auroc, auprc, accuracy, precision, recall, f1, mcc, test_loss = self.test()

        print(
            "Test at Final Epoch "
            + str(self.current_epoch)
            + " with test loss "
            + str(test_loss)
            + " AUROC "
            + str(auroc)
            + " AUPRC "
            + str(auprc)
            + " F1 "
            + str(f1)
            + " Precision "
            + str(precision)
            + " Recall "
            + str(recall)
            + " Accuracy "
            + str(accuracy)
            + " MCC "
            + str(mcc)
            + " Threshold "
            + str(self.threshold)
        )

        self.test_metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "final_epoch": self.current_epoch,
            "threshold": self.threshold,
        }
        return self.test_metrics

    def save_result(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        pd.DataFrame(self.test_metrics, index=[0]).to_csv(os.path.join(out_path, "test.csv"), index=False)

        pd.DataFrame({"epoch": np.arange(1, len(self.train_loss_epoch) + 1), "train_loss": self.train_loss_epoch}).to_csv(
            os.path.join(out_path, "train_loss.csv"), index=False
        )

        if self.y_label is None:
            return

        df_pred = pd.DataFrame({"y_label": self.y_label, "y_score": self.y_score})
        df_pred.to_csv(os.path.join(out_path, "y_pred.csv"), index=False)

        fpr, tpr, _ = roc_curve(self.y_label, self.y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(out_path, "roc.png"), dpi=300)
        plt.close()

        precision, recall, _ = precision_recall_curve(self.y_label, self.y_score)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f"AUPRC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(os.path.join(out_path, "pr.png"), dpi=300)
        plt.close()

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0.0
        num_batches = len(self.train_dataloader)
        for _, (v_d, v_s, v_p, labels, drug_name, protein_name) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            if not isinstance(v_d, dgl.DGLGraph):
                raise TypeError(f"Expected DGLGraph, got {type(v_d)}")
            if not (isinstance(v_s, torch.Tensor) and isinstance(v_p, torch.Tensor)):
                raise TypeError(f"Expected torch.Tensor, got {type(v_s)} and {type(v_p)}")
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(labels)}")

            v_d = v_d.to(self.device)
            v_s = v_s.to(self.device)
            v_p = v_p.to(self.device)
            labels = labels.float().to(self.device)

            self.optim.zero_grad()
            v_d, v_p, f, score, vs_pooled, vd_pooled, vp_pooled, fusion_ds_pooled = self.model(v_d, v_s, v_p)
            if self.n_class == 1:
                _, loss = binary_cross_entropy(score, labels)
            else:
                _, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / max(num_batches, 1)
        print("Training at Epoch " + str(self.current_epoch) + " with training loss " + str(loss_epoch))
        return loss_epoch

    def test(self):
        test_loss = 0.0
        y_label, y_score = [], []
        num_batches = len(self.test_dataloader)

        with torch.no_grad():
            self.model.eval()
            for _, (v_d, v_s, v_p, labels, drug_name, protein_name) in enumerate(self.test_dataloader):
                v_d = v_d.to(self.device)
                v_s = v_s.to(self.device)
                v_p = v_p.to(self.device)
                labels = labels.float().to(self.device)

                v_d, v_p, f, score, vs_pooled, vd_pooled, vp_pooled, fusion_ds_pooled = self.model(v_d, v_s, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)

                test_loss += loss.item()
                y_label.extend(labels.detach().cpu().numpy().tolist())
                y_score.extend(n.detach().cpu().numpy().tolist())

        y_label = np.asarray(y_label)
        y_score = np.asarray(y_score)
        auroc = roc_auc_score(y_label, y_score)
        auprc = average_precision_score(y_label, y_score)
        test_loss = test_loss / max(num_batches, 1)

        y_pred = (y_score >= self.threshold).astype(int)
        accuracy = accuracy_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred, zero_division=0)
        recall = recall_score(y_label, y_pred, zero_division=0)
        f1 = f1_score(y_label, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_label, y_pred)

        self.y_label = y_label
        self.y_score = y_score
        return auroc, auprc, accuracy, precision, recall, f1, mcc, test_loss
