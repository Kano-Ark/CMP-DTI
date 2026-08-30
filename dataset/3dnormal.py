import os
import sys
import numpy as np
import pandas as pd
import requests

# 需要：pip install gemmi pandas requests numpy
import gemmi


# ===================== 配置区 =====================
PROTEIN_CSV = "protein_seq.csv"
OUT_DIR = "protein_image_3D_normal"
CACHE_DIR = "afdb_cache"

# 超长蛋白处理：None 表示不截断（不推荐，矩阵会巨大）
MAX_LEN = 1024

# 保存格式：True -> 保存为 n*n 矩阵（推荐）
# 如果你想存成一行（flatten），改成 True
FLATTEN_TO_ONE_LINE = False
TXT_FMT = "%.6f"
# ===============================================


def read_protein_ids(csv_path: str):
    df = pd.read_csv(csv_path)
    # 兼容列名：uniprot_id / id / 第一列
    if "uniprot_id" in df.columns:
        ids = df["uniprot_id"].astype(str).tolist()
    elif "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        ids = df.iloc[:, 0].astype(str).tolist()
    # 去重且保持顺序
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


import os
import gzip
import requests

UA = {"User-Agent": "Mozilla/5.0 (DTI-preprocess; contact: your_email@example.com)"}

def _pick_model_url_from_api_items(items):
    """
    items: list[dict] from AFDB API
    兼容旧/新字段（2025 有 breaking changes，仍在双支持期） :contentReference[oaicite:2]{index=2}
    优先选 mmCIF，其次 PDB；若有多个，选“序列最长”的那个（通常覆盖最大）
    """
    candidates = []
    for it in items:
        # 常见旧字段
        for key, fmt in [
            ("cifUrl", "cif"), ("mmcifUrl", "cif"), ("bcifUrl", "cif"),
            ("pdbUrl", "pdb"),
            ("cif_url", "cif"), ("mmcif_url", "cif"), ("pdb_url", "pdb"),
        ]:
            url = it.get(key)
            if url:
                seq = it.get("sequence") or it.get("uniprotSequence") or ""
                candidates.append((fmt, len(seq), url))

        # 兼容可能的新结构字段（不同版本可能叫 structures/models/files 等）
        for container_key in ["structures", "models", "files"]:
            arr = it.get(container_key)
            if isinstance(arr, list):
                for obj in arr:
                    if not isinstance(obj, dict):
                        continue
                    url = obj.get("url") or obj.get("fileUrl") or obj.get("downloadUrl")
                    f = (obj.get("format") or obj.get("fileFormat") or "").lower()
                    if not url:
                        continue
                    if "cif" in f:
                        fmt = "cif"
                    elif "pdb" in f:
                        fmt = "pdb"
                    else:
                        continue
                    seq = it.get("sequence") or it.get("uniprotSequence") or ""
                    candidates.append((fmt, len(seq), url))

    if not candidates:
        return None, None

    # 优先 cif，再按序列长度降序
    candidates.sort(key=lambda x: (0 if x[0] == "cif" else 1, -x[1]))
    fmt, _, url = candidates[0]
    return url, fmt


def _download_url(url, out_path):
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    data = r.content
    # 处理 .gz
    if url.endswith(".gz"):
        data = gzip.decompress(data)
        # 去掉 .gz 后缀
        if out_path.endswith(".gz"):
            out_path = out_path[:-3]
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def try_download_afdb_structure(uniprot_id: str, cache_dir: str):
    """
    先走 AFDB API（最稳），拿到真实下载链接；
    再 fallback 你原来的 /files/AF-... 规则。
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 1) AFDB API
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        r = requests.get(api, headers=UA, timeout=60)
        if r.status_code == 200:
            items = r.json()
            if isinstance(items, dict):
                items = [items]
            url, fmt = _pick_model_url_from_api_items(items)
            if url:
                # 根据 url 给缓存文件名（保留后缀）
                fname = os.path.basename(url)
                out_path = os.path.join(cache_dir, fname)
                if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                    out_path = _download_url(url, out_path)
                # gemmi 读入需要知道 fmt（cif/pdb）
                if out_path.endswith(".cif"):
                    return out_path, "cif"
                if out_path.endswith(".pdb"):
                    return out_path, "pdb"
                return out_path, fmt
    except Exception:
        pass

    # 2) fallback：尝试常见 files 命名（含 .gz）
    base = f"AF-{uniprot_id}-F1-model_v4"
    candidates = [
        (f"{base}.cif", "cif"),
        (f"{base}.pdb", "pdb"),
        (f"{base}.cif.gz", "cif"),
        (f"{base}.pdb.gz", "pdb"),
    ]
    for fname, fmt in candidates:
        out_path = os.path.join(cache_dir, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # 若是 gz，读之前可解压；这里简单起见你也可以直接返回 gz 路径并在读取前解压
            if out_path.endswith(".gz"):
                out_path = _download_url(f"https://alphafold.ebi.ac.uk/files/{fname}", out_path)
            return out_path, fmt

        url = f"https://alphafold.ebi.ac.uk/files/{fname}"
        try:
            out_path = _download_url(url, out_path)
            return out_path, fmt
        except Exception:
            continue

    raise RuntimeError(f"AFDB structure not found for {uniprot_id} (API + files fallback all failed)")

def load_ca_coords(struct_path: str, fmt: str):
    """
    抽取 Cα 坐标作为点云，返回 coords: (n,3) float32
    """
    coords = []

    if fmt == "cif":
        st = gemmi.read_structure(struct_path)
        model = st[0]
        for chain in model:
            for res in chain:
                ca = res.find_atom("CA", altloc="*")
                if ca is None:
                    continue
                p = ca.pos
                coords.append([p.x, p.y, p.z])

    elif fmt == "pdb":
        st = gemmi.read_structure(struct_path)  # gemmi 也能读 pdb
        model = st[0]
        for chain in model:
            for res in chain:
                ca = res.find_atom("CA", altloc="*")
                if ca is None:
                    continue
                p = ca.pos
                coords.append([p.x, p.y, p.z])
    else:
        raise ValueError(f"Unknown format: {fmt}")

    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] == 0:
        raise RuntimeError(f"No CA atoms parsed from {struct_path}")

    # 中心化（只做平移，不做旋转/尺度归一）
    coords = coords - coords.mean(axis=0, keepdims=True)
    return coords


def uniform_sample(coords: np.ndarray, max_len: int):
    n = coords.shape[0]
    if max_len is None or n <= max_len:
        return coords
    idx = np.linspace(0, n - 1, max_len).round().astype(np.int64)
    return coords[idx]


def estimate_sigma(coords: np.ndarray):
    """
    用子采样估一个 sigma（距离中位数），更稳且便宜
    """
    n = coords.shape[0]
    m = min(512, n)
    idx = np.linspace(0, n - 1, m).round().astype(np.int64)
    X = coords[idx].astype(np.float32)

    # dist2 = a + b - 2X X^T
    xx = np.sum(X * X, axis=1, keepdims=True)  # (m,1)
    dist2 = xx + xx.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)
    d = np.sqrt(dist2 + 1e-12)

    tri = d[np.triu_indices(m, k=1)]
    sigma = float(np.median(tri) + 1e-6)
    return sigma


def build_normalized_laplacian(coords: np.ndarray):
    """
    A_ij = exp(-d_ij^2/(2*sigma^2)), A_ii=0
    L = I - D^{-1/2} A D^{-1/2}
    返回 L: (n,n) float32
    """
    X = coords.astype(np.float32)
    n = X.shape[0]

    sigma = estimate_sigma(X)
    sigma2 = 2.0 * sigma * sigma

    xx = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    dist2 = xx + xx.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)

    A = np.exp(-dist2 / sigma2).astype(np.float32)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(deg)

    An = (A * inv_sqrt[None, :]) * inv_sqrt[:, None]
    L = np.eye(n, dtype=np.float32) - An
    return L


def save_txt_matrix(out_path: str, mat: np.ndarray, flatten: bool):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if flatten:
        mat = mat.reshape(1, -1)
    np.savetxt(out_path, mat, fmt=TXT_FMT)


def main():
    ids = read_protein_ids(PROTEIN_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    missing = []
    for i, uid in enumerate(ids, 1):
        out_path = os.path.join(OUT_DIR, f"{uid}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(ids)}] skip (exists): {uid}")
            continue

        try:
            struct_path, fmt = try_download_afdb_structure(uid, CACHE_DIR)
            coords = load_ca_coords(struct_path, fmt)
            coords = uniform_sample(coords, MAX_LEN)

            L = build_normalized_laplacian(coords)
            save_txt_matrix(out_path, L, FLATTEN_TO_ONE_LINE)
            print(f"[{i}/{len(ids)}] OK: {uid}  n={L.shape[0]} -> {out_path}")

        except Exception as e:
            print(f"[{i}/{len(ids)}] FAIL: {uid}  reason={e}")
            missing.append(uid)

    if missing:
        miss_path = os.path.join(OUT_DIR, "missing_ids.txt")
        with open(miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"Missing saved: {miss_path}  count={len(missing)}")


if __name__ == "__main__":
    main()
