# CMP-DTI Training Details

---

## 1. Task Definition

CMP-DTI formulates drug--target interaction prediction as a two-class classification task.

For each drug--target pair, the model integrates the following representations:

### Drug side
- SMILES sequence representation
- Three-dimensional drug structural representation
- Atomic connectivity information
- Pairwise atomic distance information

### Protein side
- ESM-2 protein sequence representation
- AlphaFold-derived residue-level structural representation
- Sequence-adjacency protein graph representation

The final classifier outputs two logits corresponding to:
- class `0`: non-interaction
- class `1`: interaction

The model is optimized using cross-entropy loss.

---

## 2. Software Environment

The implementation is based on PyTorch.

| Component | Version |
|---|---|
| Python | `3.9` |
| PyTorch | `2.4.0` |
| CUDA | `11.8` |
| cuDNN | `9.3.0` |
| RDKit | `2023.09.02` |
| NumPy | `1.23.5` |
| pandas | `2.0.3` |
| scikit-learn | `1.3.2` |
| fair-esm / ESM | `2.0.0` |

Install dependencies with:

```bash
pip install -r requirements.txt
```

The versions reported here should match the environment used to generate the manuscript results.

---

## 3. Hardware Environment

The reported experiments were conducted using NVIDIA Tesla A30 GPUs.

| Item | Configuration |
|---|---|
| GPU | NVIDIA Tesla A30 |
| Number of GPUs | `4` |
| GPU memory | `24GB` |
| CPU | `Xeon(R) Silver 4314` |
| RAM | `512GB` |
| Operating system | `CentOS` |

---

## 4. ESM-2 Protein Sequence Features

Protein sequence representations are extracted using:

```text
esm2_t33_650M_UR50D
```

Configuration:

| Setting | Value |
|---|---|
| Selected layer | 33 |
| Residue embedding dimension | 1280 |
| Maximum preprocessing length | 1024 residues |
| Special tokens | Removed before pooling |
| Pooling | Mean pooling |
| Fine-tuning | Frozen |
| CMP-DTI output dimension | 256 |

The beginning and end special tokens are excluded, and residue representations are mean-pooled to obtain a fixed-length protein representation.

---

## 5. Main Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Shared feature dimension | 256 |
| Batch size | 128 |
| Learning rate | \(1\times10^{-4}\) |
| Training epochs | 100 |
| Optimizer | Adam |
| Drug SMILES embedding dimension | 128 |
| Drug SMILES Transformer layers | 2 |
| Drug SMILES attention heads | 8 |
| Drug structural embedding size | 512 |
| Drug structural Transformer layers | 2 |
| Drug structural attention heads | 4 |
| Drug CNN channels | 32 -> 128 |
| Protein sequence input dimension | 1280 |
| Protein sequence output dimension | 256 |
| Protein structural embedding size | 512 |
| Protein global tokens | 16 |
| Protein attention heads | 8 |
| Protein CNN channels | 32 -> 128 |
| Protein GCN layers | 2 |
| Protein GCN hidden dimension | 256 |
| Classifier hidden dimensions | 1024 -> 256 |
| First classifier activation | LeakyReLU (negative slope = 0.2) |
| Second classifier activation | Tanh |
| Classifier output | 2 logits |
| Loss function | Cross-entropy |

## 6. Classifier and Loss

The fused drug--target representation is passed through a three-layer fully connected classifier:

```text
input
  |
  v
Linear -> 1024
  |
LeakyReLU(0.2)
  |
  v
Linear -> 256
  |
Tanh
  |
  v
Linear -> 2 logits
```

The model is trained using two-class cross-entropy loss. The final linear layer outputs two unnormalized logits; softmax is not applied before `CrossEntropyLoss` during training.

---
