import dgl
import torch
import copy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score, precision_recall_curve, precision_score, recall_score, matthews_corrcoef
from models import binary_cross_entropy, cross_entropy_logits
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def copy_model(model):
    model_path = r'../output/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        torch.save(model, model_path+'model.pt')
    new_model = torch.load(model_path + 'model.pt')
    return new_model

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["DECODER"]["BINARY"]

        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.lr_decay = config["SOLVER"]["LR_DECAY"]
        self.decay_interval = config["SOLVER"]["DECAY_INTERVAL"]
        self.use_ld = config['SOLVER']["USE_LD"]

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_auprc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}

        self.config = config
        self.y_label = None
        self.y_score = None


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if self.use_ld:
                if self.current_epoch % self.decay_interval == 0:
                    self.optim.param_groups[0]['lr'] *= self.lr_decay

            train_loss = self.train_epoch()

            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")

            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                # self.best_model = copy_model(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
        auroc, auprc, f1, precision, recall, accuracy, mcc, test_loss, thred_optim = self.test(dataloader="test")

        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " F1 " + str(f1) + " Precision " + str(precision) + " Recall " +
              str(recall) + " Accuracy " + str(accuracy) + " MCC " + str(mcc) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["precision"] = precision
        self.test_metrics["recall"] = recall
        self.test_metrics["f1"] = f1
        self.test_metrics["mcc"] = mcc


        return self.test_metrics

    def save_result(self, cv_out_path):
        test_metrics = pd.DataFrame(self.test_metrics, index=[0])
        test_metrics.to_csv(os.path.join(cv_out_path, "test.csv"), index=False)

        if self.y_label is None:
            return

        # 保存预测
        df_pred = pd.DataFrame({
            "y_label": self.y_label,
            "y_score": self.y_score
        })
        df_pred.to_csv(os.path.join(cv_out_path, "y_pred.csv"), index=False)

        # ===== ROC =====
        fpr, tpr, _ = roc_curve(self.y_label, self.y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(cv_out_path, "roc.png"), dpi=300)
        plt.close()

        # ===== PR =====
        precision, recall, _ = precision_recall_curve(self.y_label, self.y_score)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, label=f"AUPRC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(os.path.join(cv_out_path, "pr.png"), dpi=300)
        plt.close()

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_s, v_p, labels, drug_name, protein_name) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            if not isinstance(v_d, dgl.DGLGraph):
                raise TypeError(f"Expected DGLGraph, got {type(v_d)}")
                # 确保v_s和v_p是张量
            if not (isinstance(v_s, torch.Tensor) and isinstance(v_p, torch.Tensor)):
                raise TypeError(f"Expected torch.Tensor, got {type(v_s)} and {type(v_p)}")
                # 确保labels也是张量
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(labels)}")
            v_d, v_s, v_p, labels = v_d.to(self.device), v_s.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score, vs_pooled, vd_pooled, vp_pooled, fusion_ds_pooled = self.model(v_d, v_s, v_p)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()


        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred, fusion, v_s1, v_d1, v_p1, fusion_ds1, drug_names, protein_names = [], [], [], [], [], [], [], [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)

        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_s, v_p, labels, drug_name, protein_name) in enumerate(data_loader):
                v_d, v_s, v_p, labels = v_d.to(self.device), v_s.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score, vs_pooled, vd_pooled, vp_pooled, fusion_ds_pooled = self.model(v_d, v_s, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score, vs_pooled, vd_pooled, vp_pooled, fusion_ds_pooled = self.best_model(v_d, v_s, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
                drug_names += list(drug_name)
                protein_names += list(protein_name)

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches



        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            try:
                precision = tpr / (tpr + fpr)
            except RuntimeError:
                raise ('RuntimeError: the divide==0')
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            accuracy = accuracy_score(y_label, y_pred_s)
            recall = recall_score(y_label, y_pred_s)
            mcc = matthews_corrcoef(y_label, y_pred_s)
            precision = precision_score(y_label, y_pred_s)
            self.y_label = np.array(y_label)
            self.y_score = np.array(y_pred)

            return auroc, auprc, accuracy, precision, recall, np.max(f1[5:]), mcc, test_loss, thred_optim
        else:
            return auroc, auprc, test_loss