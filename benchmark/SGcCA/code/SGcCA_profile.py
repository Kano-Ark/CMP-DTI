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
from models import SGcADTI, binary_cross_entropy, cross_entropy_logits
from trainer import Trainer
from utils import set_seed, graph_collate_func, mkdir
from sklearn.utils import shuffle
from copy import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Script lives in <project>/code/. Resolve config and dataset paths from project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CFG = os.path.join(PROJECT_ROOT, "configs", "SGcCA.yaml")
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', default=DEFAULT_CFG, help="path to config file", type=str)
parser.add_argument('--profile-only', action='store_true',
                    help='Profile parameters/GPU memory with one real batch and exit.')
parser.add_argument('--data-name', default='kiba', type=str,
                    help='Dataset basename under ./dataset (default: kiba).')
parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR, type=str,
                    help='Dataset directory; defaults to <SGcCA project>/dataset.')
parser.add_argument('--cv', default='cv1', choices=['cv1', 'cv2', 'cv3', 'cv4'],
                    help='Data split protocol used to obtain the profiling batch.')
parser.add_argument('--profile-batch-size', type=int, default=None,
                    help='Override config batch size only for profiling.')
args, _unknown_args = parser.parse_known_args()

def _profile_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_mb(num_bytes):
    return num_bytes / (1024.0 ** 2)


def _profile_parameter_report(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    print("\n========== MODEL PARAMETERS ==========")
    print(f"Total parameters:          {total:,}")
    print(f"Trainable parameters:      {trainable:,}")
    print(f"Non-trainable parameters:  {frozen:,}")
    print(f"Parameters (M):            {total / 1e6:.6f}")
    print(f"Parameter tensors:         {_profile_mb(param_bytes):.2f} MB")
    print(f"Buffers:                   {_profile_mb(buffer_bytes):.2f} MB")
    print(f"Model state (params+buf):  {_profile_mb(param_bytes + buffer_bytes):.2f} MB")
    return total, trainable, frozen


def _profile_gpu_start(model, device):
    model = model.to(device)
    if not torch.cuda.is_available():
        print("\nCUDA is unavailable; GPU memory metrics cannot be measured.")
        return model, None

    torch.cuda.empty_cache()
    _profile_sync()
    # Report PyTorch allocator usage after the model is on the GPU.
    print("\n========== GPU MODEL MEMORY ==========")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Model/current allocated:  {_profile_mb(torch.cuda.memory_allocated()):.2f} MB")
    print(f"Current reserved:         {_profile_mb(torch.cuda.memory_reserved()):.2f} MB")
    print(f"Device total memory:      {_profile_mb(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory):.2f} MB")
    return model, True


def _profile_forward_peak(run_forward):
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        run_forward()
    _profile_sync()
    print("\n========== FORWARD / INFERENCE MEMORY ==========")
    print(f"Peak allocated:           {_profile_mb(torch.cuda.max_memory_allocated()):.2f} MB")
    print(f"Peak reserved:            {_profile_mb(torch.cuda.max_memory_reserved()):.2f} MB")


def _profile_training_peak(model, optimizer, run_loss):
    if not torch.cuda.is_available():
        return
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    loss = run_loss()
    loss.backward()
    optimizer.step()
    _profile_sync()
    print("\n========== TRAINING-STYLE MEMORY (1 BATCH) ==========")
    print(f"Profiling loss:            {loss.detach().item():.6f}")
    print(f"Peak allocated:           {_profile_mb(torch.cuda.max_memory_allocated()):.2f} MB")
    print(f"Peak reserved:            {_profile_mb(torch.cuda.max_memory_reserved()):.2f} MB")
    print(f"Current allocated:        {_profile_mb(torch.cuda.memory_allocated()):.2f} MB")
    print("Optimizer steps:          1")


def profile_sgcca(model, optimizer, train_generator, cfg, device):
    """Profile SGcCA using the same six-field batch and loss as Trainer.train_epoch."""
    try:
        batch = next(iter(train_generator))
    except StopIteration:
        raise RuntimeError("The SGcCA training DataLoader is empty.")

    if len(batch) != 6:
        raise RuntimeError(
            "Expected SGcCA batch (v_d, v_s, v_p, labels, drug_name, protein_name); "
            f"received {len(batch)} fields."
        )

    v_d, v_s, v_p, labels, drug_name, protein_name = batch
    model, _ = _profile_gpu_start(model, device)

    v_d = v_d.to(device)
    v_s = v_s.to(device)
    v_p = v_p.to(device)
    labels = labels.float().to(device)

    _profile_parameter_report(model)
    print("\n========== PROFILE INPUT ==========")
    print(f"Batch size:                {len(labels)}")
    print(f"Drug graph batch:          {type(v_d).__name__}")
    print(f"Drug feature shape:        {tuple(v_s.shape)}")
    print(f"Protein feature shape:     {tuple(v_p.shape)}")

    n_class = cfg["DECODER"]["BINARY"]

    def get_model_outputs():
        return model(v_d, v_s, v_p)

    model.eval()

    def forward_only():
        outputs = get_model_outputs()
        return outputs[3]

    _profile_forward_peak(forward_only)

    def training_loss():
        outputs = get_model_outputs()
        score = outputs[3]
        if n_class == 1:
            _n, loss = binary_cross_entropy(score, labels)
        else:
            _n, loss = cross_entropy_logits(score, labels)
        return loss

    _profile_training_peak(model, optimizer, training_loss)
    print("\n========== PROFILE FINISHED ==========")
    print("Full training started: NO")
    print("Epochs executed: 0")
    print("Dataset batches used: 1")
    print("Optimizer steps used for memory measurement: 1")
    print("Checkpoint saved: NO")
    print("Exiting now.")


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        project_candidate = os.path.join(PROJECT_ROOT, cfg_path)
        cfg_path = project_candidate if os.path.isfile(project_candidate) else os.path.abspath(cfg_path)
    cfg.merge_from_file(cfg_path)
    cfg.RESULT.OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", data_name, cv)
    print(cfg.RESULT.OUTPUT_DIR)

    if not args.profile_only:
        mkdir(cfg.RESULT.OUTPUT_DIR)
    if args.profile_only and args.profile_batch_size is not None:
        cfg.SOLVER.BATCH_SIZE = args.profile_batch_size
    print(f"Config yaml: {cfg_path}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = args.dataset_dir
    if not os.path.isabs(dataFolder):
        project_candidate = os.path.join(PROJECT_ROOT, dataFolder)
        dataFolder = project_candidate if os.path.isdir(project_candidate) else os.path.abspath(dataFolder)

    datasets_path = os.path.join(dataFolder, data_name + ".txt")
    datasets = pd.read_csv(datasets_path, header=None, sep=' ')
    datasets.columns = ['SMILES', 'Protein', 'Y']


    # datasets_path = os.path.join(dataFolder, 'drugbank.csv')
    # datasets = pd.read_csv(datasets_path)

    metrics = {"auroc": [], "auprc": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []}
    metrics = pd.DataFrame(metrics)

    fold_range = range(1, 2) if args.profile_only else range(1, 4)
    for seed, fold in enumerate(fold_range):
        torch.cuda.empty_cache()

        cv_out_path = f"./result/{data_name}/{cv}/random_{fold}"
        if not args.profile_only:
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

        model = SGcADTI(**cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        torch.backends.cudnn.benchmark = True

        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': not args.profile_only, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': not args.profile_only, 'collate_fn': graph_collate_func}


        train_generator = DataLoader(train_dataset, **params)

        params['shuffle'] = False
        params['drop_last'] = False

        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)

        if args.profile_only:
            profile_sgcca(model, opt, train_generator, cfg, device)
            return metrics

        model = model.to(device)
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
    DATASETS = [args.data_name] if args.profile_only else ["kiba"]
    for data_name in DATASETS:
        cv_ids = [int(args.cv.replace("cv", ""))] if args.profile_only else [1]
        for cv_id in cv_ids:
            cv = f"cv{cv_id}"
            print(data_name, cv)
            result = main()
            print(result)

    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)/3600}h")