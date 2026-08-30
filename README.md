# CMP-DTI

Here is the code for paper **CMP-DTI: Cross-Modal Molecular Point Cloud Fusion for Drug--Target Interaction Prediction**.

# System Requirements

The source code developed in Python 3.9 using PyTorch 2.4.0 The required python dependencies are given below. 

numpy==1.23.5

torch==2.4.0+cu124

scikit_learn==1.3.2

rdkit==2023.9.2

## 1. Overview

CMP-DTI represents drugs and proteins using multiple complementary modalities.

### Drug representations

- SMILES sequence
- Atomic three-dimensional coordinates
- Atomic connectivity matrix
- Pairwise atomic distance matrix

### Protein representations

- Protein sequence features extracted by ESM-2
- Residue-level C-alpha structural representation derived from AlphaFold structures
- Sequence-adjacency protein graph

The modality-specific representations are encoded separately and subsequently integrated for binary drug–target interaction prediction.

**Important:** CMP-DTI does not use a docked drug–protein complex as model input and does not explicitly construct atom–residue contact maps or binding-pocket representations.

---

## 2. Repository Structure

The recommended repository structure is:

```text
CMP-DTI/
│
├── README.md
├── LICENSE
├── requirements.txt
├── dataset/
│   ├── download.py
│   ├── drug3d.py
│   ├── esm2.py
│   ├── seq2graph.py
│   └── 3dnormal.py
├── main.py
├── utils.py
