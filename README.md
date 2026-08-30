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
```

## 3. Train
Useing
```bash
python main.py
```
The default dataset is Drugbank. You can change the dataset by modifying the area around line 630 in the *main.py* code.
There are also lines 647 and 717 that perform a cold start.
```python
#warm
with open('dataset/' + task + '/result/CV5/train_cv0.csv') as f1:  #647
with open('dataset/' + task + '/result/CV5/test_cv0.csv') as f1:  #717
#cold drug
with open('dataset/' + task + '/result/cold_start_drug/train_drug_cold.csv') as f1:  #647
with open('dataset/' + task + '/result/cold_start_drug/test_drug_cold.csv') as f1:  #717
#cold protein
with open('dataset/' + task + '/result/cold_start_protein/train_protein_cold.csv') as f1:  #647
with open('dataset/' + task + '/result/cold_start_protein/test_proten_cold.csv') as f1:  #717
#cold pair
with open('dataset/' + task + '/result/cold_pair/train_cold_pair.csv') as f1:  #647
with open('dataset/' + task + '/result/cold_pair/test_cold_pair.csv') as f1:  #717
```

