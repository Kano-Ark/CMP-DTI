import argparse
import os
import warnings
from datetime import datetime
from time import time
from utils import cv2, cv3, cv4
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import SGcADTI
from trainer import Trainer
from utils import set_seed, graph_collate_func, mkdir
from sklearn.utils import shuffle
from copy import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', default=r'./configs/SGcCA.yaml', help="path to config file", type=str)
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.RESULT.OUTPUT_DIR = f"./result/{data_name}/{cv}"
    print(cfg.RESULT.OUTPUT_DIR)

    mkdir(cfg.RESULT.OUTPUT_DIR)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./dataset/'

    datasets_path = os.path.join(dataFolder, data_name + ".txt")
    datasets = pd.read_csv(datasets_path, header=None, sep=' ')
    datasets.columns = ['SMILES', 'Protein', 'Y']


    # datasets_path = os.path.join(dataFolder, 'drugbank.csv')
    # datasets = pd.read_csv(datasets_path)

    metrics = {"auroc": [], "auprc": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []}
    metrics = pd.DataFrame(metrics)

    for seed, fold in enumerate(range(1, 4)):
        torch.cuda.empty_cache()

        cv_out_path = f"./result/{data_name}/{cv}/random_{fold}"
        os.makedirs(cv_out_path, exist_ok=True)

        print("*" * 60 + str(fold) + "-random" + "*" * 60)
        data = copy(datasets)
        data = shuffle(data, random_state=seed + 42)


        set_seed(cfg.SOLVER.SEED)
        block_size = len(data) // 10
        dataset_train, dataset_test = data[:block_size * 7], data[block_size * 7:]
        size = len(dataset_test)
        if cv == "cv1":
            train_set, val_set, test_set = dataset_train, dataset_test[: int(size * 1 / 3)], dataset_test[
                                                                                             int(size * 1 / 3):]
        elif cv == "cv2":
            train_set, val_set, test_set = cv2(data)
        elif cv == "cv3":
            train_set, val_set, test_set = cv3(data)
        elif cv == "cv4":
            train_set, val_set, test_set = cv4(data)

        train_set.reset_index(drop=True, inplace=True)
        val_set.reset_index(drop=True, inplace=True)
        test_set.reset_index(drop=True, inplace=True)

        print("data preprocess end !!!")

        print(f"train:{len(train_set)}")
        print(f"dev:{len(val_set)}")
        print(f"test:{len(test_set)}")

        train_dataset = DTIDataset(train_set.index.values, train_set)
        val_dataset = DTIDataset(val_set.index.values, val_set)
        test_dataset = DTIDataset(test_set.index.values, test_set)

        model = SGcADTI(**cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        torch.backends.cudnn.benchmark = True

        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': graph_collate_func}


        train_generator = DataLoader(train_dataset, **params)

        params['shuffle'] = False
        params['drop_last'] = False

        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)


        trainer = Trainer(model, opt, device, train_generator, val_generator, test_generator, **cfg)


        result = trainer.train()
        result = pd.DataFrame(result, index=[0])

        metrics = pd.concat((metrics, result), axis=0)
        trainer.save_result(cv_out_path)

        metrics.to_csv(os.path.join(cfg.RESULT.OUTPUT_DIR, f"result.csv"), index=False)
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return metrics


if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    DATASETS = ["kiba"]
    for data_name in DATASETS:
        for cv_id in [1]:
            cv = f"cv{cv_id}"
            print(data_name, cv)
            result = main()
            print(result)

    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)/3600}h")