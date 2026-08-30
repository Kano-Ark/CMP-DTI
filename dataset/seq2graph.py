"""
build_protein_graph_txt.py
----------------------------------
功能：
  从蛋白质序列 CSV 文件生成每个蛋白的图输入矩阵 xg，
  并将结果保存为 .txt 文件 (N, N+1)

输入文件示例：
  (有header)
    protein_id,sequence
    P001,MKTFFVLVLLLSVTVQAT
  (无header)
    P001,MKTFFVLVLLLSVTVQAT
"""

import torch
import numpy as np
import pandas as pd
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import os

# ==============================
# 1️⃣ 基础设置
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = "./dtinet/protein_seq.csv"

# ------------------------------
# 若CSV有首行标签（header行）：
# df = pd.read_csv(csv_path)
#
# 若CSV没有首行标签（无header行）：
df = pd.read_csv(csv_path, header=None, names=["protein_id", "sequence"])
# ------------------------------
#df = pd.read_csv(csv_path)  # ✅ 默认假设有header

print(f"✅ 读取到 {len(df)} 条蛋白序列")

# ==============================
# 2️⃣ 加载 ESM 模型
# ==============================
print("🔹 加载 ESM 模型中 ...")
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
model.eval()

# ==============================
# 3️⃣ 定义辅助函数
# ==============================

def sequence_to_features(seq):
    """
    将蛋白质序列转为节点特征矩阵 X
    输入：sequence (str)
    输出：X (torch.FloatTensor), shape = (N, d)
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    X = outputs.last_hidden_state.squeeze(0)  # (N, d)
    return X.cpu()

def build_adj_matrix(N):
    """
    构建基于序列相邻的邻接矩阵 A (N, N)
    """
    A = torch.zeros((N, N))
    A[range(N - 1), range(1, N)] = 1
    A[range(1, N), range(N - 1)] = 1
    return A

def build_xg(X, A):
    """
    合并A和节点类型，生成 xg (N, d+N)
    """
    # node_type = torch.argmax(X, dim=1, keepdim=True).float()  # 简化版节点类型
    xg = torch.cat([X, A], dim=1)
    return xg

# ==============================
# 4️⃣ 处理所有蛋白序列并保存为TXT
# ==============================
save_dir = "./dataset/protein_graphs"
os.makedirs(save_dir, exist_ok=True)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building protein graphs"):
    pid, seq = row[0], str(row[1]).strip().upper()

    # Step1: 特征矩阵 X
    X = sequence_to_features(seq)
    N = X.shape[0]

    # Step2: 邻接矩阵
    A = build_adj_matrix(N)

    # Step3: 合并
    xg = build_xg(X, A)  # (N, N+1)

    # Step4: 保存为 txt 文件
    txt_path = os.path.join(save_dir, f"{pid}.txt")
    np.savetxt(txt_path, xg.numpy(), fmt="%.6f")

print(f"✅ 所有蛋白图已保存到: {save_dir}/")
