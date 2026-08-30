import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


# ===================== 配置区 =====================
IN_CSV = "drug_SMILES.csv"
ATOM_DICT_PATH = "atom_dict.txt"
OUT_DIR = "Drug_Point_Graph"
FAIL_LOG = os.path.join(OUT_DIR, "missing_drugs.txt")

SEED = 2026

# 3D 构象生成与优化
USE_ETKDG_V3 = True
OPTIMIZE = True          # True: MMFF/UFF 优化；False: 不优化

# 距离图权重：A_ij = exp(-d^2 / (2*sigma^2))
# 若 SIGMA=None，则用分子内距离的中位数自适应
SIGMA = None
SIGMA_SCALE = 1.0        # 想让 L_dist 的非对角更接近某个范围，可调这个倍数，如 0.8/1.2

TXT_FMT = "%.6f"         # 输出格式：与示例一致的 6 位小数
# =================================================


def load_atom_dict(path: str):
    atoms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                atoms.append(t)
    atom2id = {t: i for i, t in enumerate(atoms)}  # C=0, N=1, O=2, H=3, ...
    return atoms, atom2id


def atom_to_token(atom: Chem.Atom) -> str:
    """
    把 RDKit Atom 转成你的 atom_dict token:
      - 元素大写：Cl -> CL, Br -> BR, Na -> NA, Si -> SI
      - 带电：N +1 -> N1+，O -1 -> O1-，RU +2 -> RU2+ ...
    """
    sym_up = atom.GetSymbol().upper()
    chg = int(atom.GetFormalCharge())

    if chg == 0:
        return sym_up

    sign = "+" if chg > 0 else "-"
    mag = abs(chg)
    return f"{sym_up}{mag}{sign}"


def normalized_laplacian_from_adj(A: np.ndarray) -> np.ndarray:
    """
    L = I - D^{-1/2} A D^{-1/2}
    A: nonnegative adjacency, shape (n,n)
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(deg)

    An = (A * inv_sqrt[None, :]) * inv_sqrt[:, None]
    L = np.eye(n, dtype=np.float32) - An.astype(np.float32)
    return L.astype(np.float32)


def coords_from_mol(mol: Chem.Mol) -> np.ndarray:
    """
    读取 conformer 坐标，返回 (n,3)，并中心化
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i] = [p.x, p.y, p.z]
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def estimate_sigma(coords: np.ndarray) -> float:
    """
    用分子内 pairwise 距离中位数估 sigma（自适应）
    """
    X = coords.astype(np.float32)
    n = X.shape[0]
    xx = np.sum(X * X, axis=1, keepdims=True)
    dist2 = xx + xx.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)
    d = np.sqrt(dist2 + 1e-12)
    tri = d[np.triu_indices(n, k=1)]
    return float(np.median(tri) + 1e-6)


def build_distance_laplacian(coords: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    L_dist：用稠密高斯核构 A，再算 normalized Laplacian
    """
    X = coords.astype(np.float32)
    n = X.shape[0]

    xx = np.sum(X * X, axis=1, keepdims=True)
    dist2 = xx + xx.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)

    if sigma is None:
        sigma = estimate_sigma(X)
    sigma *= float(SIGMA_SCALE)

    A = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    np.fill_diagonal(A, 0.0)

    return normalized_laplacian_from_adj(A)


def build_bond_laplacian(mol: Chem.Mol) -> np.ndarray:
    """
    L_bond：用键连接构 A（无权 0/1），再算 normalized Laplacian
    这块会产生 -0.288675/-0.408248/-0.5 等典型值，并且大量 0（稀疏）
    """
    n = mol.GetNumAtoms()
    A = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        A[i, j] = 1.0
        A[j, i] = 1.0
    return normalized_laplacian_from_adj(A)


def atomic_sequence_column(mol: Chem.Mol, atom2id: dict) -> np.ndarray:
    """
    s：最后一列 atomic sequence，用 atom_dict.txt 的行号做 ID
    输出为 float32 只是为了和你看到的 1.00000/2.00000 形式一致
    """
    ids = []
    for a in mol.GetAtoms():
        tok = atom_to_token(a)
        if tok in atom2id:
            ids.append(atom2id[tok])
        else:
            # 兜底：退回到元素 token（比如某些电荷形态没列在字典）
            tok0 = a.GetSymbol().upper()
            if tok0 in atom2id:
                ids.append(atom2id[tok0])
            else:
                raise ValueError(f"Atom token not in dict: {tok} (fallback {tok0} also missing)")
    return np.array(ids, dtype=np.float32).reshape(-1, 1)


def smiles_to_3d_mol_keepH(smiles: str) -> Chem.Mol:
    """
    SMILES -> RDKit Mol -> 添加显式氢 -> 生成 3D 构象 -> (可选) 优化
    注意：不 RemoveHs，保证 H 在图里（匹配你的 atom_dict）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("MolFromSmiles failed")

    mol = Chem.AddHs(mol)

    if USE_ETKDG_V3:
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        code = AllChem.EmbedMolecule(mol, params)
    else:
        code = AllChem.EmbedMolecule(mol, randomSeed=SEED)

    if code != 0:
        raise ValueError(f"EmbedMolecule failed code={code}")

    if OPTIMIZE:
        # 优先 MMFF，不行再 UFF
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol)

    return mol


def build_dcgcN_drug_matrix(smiles: str, atom2id: dict) -> np.ndarray:
    mol = smiles_to_3d_mol_keepH(smiles)
    coords = coords_from_mol(mol)

    L_dist = build_distance_laplacian(coords, sigma=SIGMA)   # (n,n)
    L_bond = build_bond_laplacian(mol)                       # (n,n)
    s = atomic_sequence_column(mol, atom2id)                 # (n,1)

    X = np.concatenate([L_dist, L_bond, s], axis=1).astype(np.float32)  # (n,2n+1)
    return X


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    _, atom2id = load_atom_dict(ATOM_DICT_PATH)

    df = pd.read_csv(IN_CSV, dtype=str)
    if not {"id", "smiles"}.issubset(df.columns):
        raise ValueError("drug_SMILES.csv 必须包含列：id, smiles")

    missing = []
    total = len(df)

    for k, row in df.iterrows():
        did = str(row["id"])
        smi = str(row["smiles"])

        out_path = os.path.join(OUT_DIR, f"{did}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        try:
            X = build_dcgcN_drug_matrix(smi, atom2id)
            np.savetxt(out_path, X, fmt=TXT_FMT, delimiter=" ")
            if (k + 1) % 50 == 0 or (k + 1) == total:
                print(f"[{k+1}/{total}] OK  {did}  shape={X.shape}")
        except Exception as e:
            print(f"[{k+1}/{total}] FAIL {did}  reason={e}")
            missing.append(did)

    if missing:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"Missing saved: {FAIL_LOG}  count={len(missing)}")

    print("Done.")


if __name__ == "__main__":
    main()
