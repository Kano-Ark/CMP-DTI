import random
import os
import torch.nn as nn
from torch.utils.data import dataloader, TensorDataset, DataLoader, Dataset
from utils import *
import csv
import torch.nn.functional as F

#################基础设置################
LR = 0.0001
SEED = 3142
# VAL_SIZE = 7486
BATCH_SIZE = 128
EPOCH = 100
Feature_Size = 256   # [256, 512]
Alpha_d = 1
Alpha_p = 1

###############药物编码模块###############
# SMILES_Coding
DSC_Kernel_Num = 32
DSC_Kernel_Size = 8
Drug_SMILES_Input_Size = 128      # [128, 256]

# Image_Coding
Drug_Point_Hidden_Size = 512   # [128, 256, 512]
DPC_Kernel_Num = 32
DPC_Kernel_Size = 8   # [8, 16]

###############蛋白编码模块###############
# Bert_Coding
Protein_Bert_Hidden_Size = 1280

#Point_Coding
Protein_Point_Hidden_Size = 512   # [128, 256, 512]
PPC_Kernel_Num = 32
PPC_Kernel_Size = 8   # [8, 16]

###############数据处理设置################
Drug_Max_Lengtgh = 100
Protein_Max_Lengtgh = 1024
AA_Dict = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z']
Protein_Dic_Length = 23
Atom_Point_Dict_Length = 79
atom_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35, "0": 3,
            "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41,
            "@": 8, "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46,
            "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, "b": 21, "e": 57,
            "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62,
            "r": 27, "u": 63, "t": 28, "y": 64}

Atom_Dic_Length = 64

metrics_file = "val_metrics_drug.csv"
if not os.path.exists(metrics_file):
    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "AUC", "AUPR", "ACC", "SEN", "SPE"])  # 表头

class GraphConvolution(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, a):
        support = torch.mm(x, self.weight)  # X*W
        r = torch.mm(a, support)    # A*X*W
        return r


class DrugSMILESCoding(nn.Module):
    def __init__(self, hid_dim=Drug_SMILES_Input_Size, out_dim=Feature_Size,
                 vocab_size=Atom_Dic_Length, max_len=256, nhead=8, nlayers=2, dropout=0.1):
        super(DrugSMILESCoding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=hid_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, dim_feedforward=hid_dim*4, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.norm = nn.LayerNorm(hid_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x shape: [B, L]  (B=batch size, L=sequence length)
        pos_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)  # [B, L]
        x = self.embedding(x) + self.pos_embedding(pos_ids)  # [B, L, H]
        x = self.transformer_encoder(x)  # [B, L, H]
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [B, H, L] for pooling
        x = self.pool(x).squeeze(-1)  # [B, H]
        x = self.fc(x)  # [B, out_dim]
        return x


class DrugPointCoding(nn.Module):
    def __init__(self, point_hid_dim=128, point_output_dim=256,
                 channel=32, kernel_size=3,
                 n_layers=2, n_heads=4,
                 dropout=0.1):
        """
        xd: [B, N, 2N+1]
        """
        super().__init__()

        # ========== 基础节点融合：Cross-Attention ==========
        self.base_cross_attn = nn.MultiheadAttention(
            embed_dim=point_hid_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.base_attn_norm = nn.LayerNorm(point_hid_dim)

        # ========== 共享输入处理 ==========
        self.embedding = nn.Embedding(Atom_Point_Dict_Length, point_hid_dim)
        self.proj = nn.Linear(point_hid_dim, point_hid_dim)

        self.dist_encoder = nn.Linear(1, point_hid_dim)
        self.ctx_encoder = nn.Linear(1, point_hid_dim)

        # ========== 全局分支：Transformer ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=point_hid_dim,
            nhead=n_heads,
            dim_feedforward=point_hid_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.global_fc = nn.Linear(point_hid_dim, point_hid_dim)

        # ========== 局部分支：CNN ==========
        self.conv_local = nn.Sequential(
            nn.Conv1d(point_hid_dim, channel, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channel),
            nn.Conv1d(channel, channel * 2, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channel * 2),
            nn.Conv1d(channel * 2, channel * 4, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channel * 4),
        )
        self.local_pool = nn.AdaptiveMaxPool1d(1)
        self.local_fc = nn.Linear(channel * 4, point_hid_dim)

        # ========== 分支 attention 融合 ==========
        # 用一个可学习 fusion token 读取 [global, local] 两个token
        self.fusion_token = nn.Parameter(torch.randn(1, 1, point_hid_dim))
        self.branch_fuse_attn = nn.MultiheadAttention(
            embed_dim=point_hid_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.branch_fuse_norm = nn.LayerNorm(point_hid_dim)

        # ========== FCN 输出 ==========
        self.fcn = nn.Sequential(
            nn.Linear(point_hid_dim, point_hid_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(point_hid_dim * 2, point_output_dim),
        )
        self.output_norm = nn.LayerNorm(point_output_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, xd):
        """
        xd: (B, N, 2N+1)
        - 前 N: d_d (距离矩阵)
        - 中 N: d_c (上下文矩阵)
        - 最后一列: d_a (原子类别id)
        """
        B, N, _ = xd.shape

        d_d = xd[:, :, :N]         # (B, N, N)
        d_c = xd[:, :, N:2*N]      # (B, N, N)
        d_a = xd[:, :, -1].long()  # (B, N)

        # ========= 共享特征 =========
        h = self.embedding(d_a)     # (B, N, D)
        h = self.proj(h)            # (B, N, D)

        # 距离/上下文 -> (B, N, D)
        dist_feat = self.dist_encoder(d_d.unsqueeze(-1)).mean(dim=2)
        ctx_feat  = self.ctx_encoder(d_c.unsqueeze(-1)).mean(dim=2)
        struct_feat = dist_feat + ctx_feat

        # ========= 基础节点 Cross-Attn 融合 =========
        attn_out, _ = self.base_cross_attn(
            query=h, key=struct_feat, value=struct_feat,
            need_weights=False
        )
        h_base = self.base_attn_norm(h + attn_out)  # (B, N, D)

        # ========= 并行分支 =========
        # 1) Global: Transformer
        h_global_seq = self.transformer_encoder(h_base)  # (B, N, D)
        h_global = self.global_pool(h_global_seq.permute(0, 2, 1)).squeeze(-1)  # (B, D)
        h_global = self.global_fc(h_global)  # (B, D)
        h_global = F.normalize(h_global, p=2, dim=1)

        # 2) Local: CNN
        h_local = self.conv_local(h_base.permute(0, 2, 1))           # (B, C*4, N)
        h_local = self.local_pool(h_local).squeeze(-1)               # (B, C*4)
        h_local = self.local_fc(h_local)                             # (B, D)
        h_local = F.normalize(h_local, p=2, dim=1)

        # ========= 仅 attention 融合 (global/local 两token) =========
        branch_tokens = torch.stack([h_global, h_local], dim=1)  # (B, 2, D)
        q = self.fusion_token.expand(B, -1, -1)                  # (B, 1, D)

        fused, _ = self.branch_fuse_attn(
            query=q, key=branch_tokens, value=branch_tokens,
            need_weights=False
        )
        fused_vec = self.branch_fuse_norm(q + fused).squeeze(1)  # (B, D)

        # ========= FCN 输出 =========
        out = self.fcn(fused_vec)         # (B, out_dim)
        out = self.output_norm(out)
        out = self.activation(out)
        return out



class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    def __init__(self, d_model, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        """
        x: (B, 2, D) - 两个特征的拼接
        """
        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm(x + ffn_output)
        
        return x


class DrugCoding(nn.Module):
    def __init__(self):
        super(DrugCoding, self).__init__()
        self.coding1 = DrugSMILESCoding()
        self.coding2 = DrugPointCoding()

    def forward(self, x_smiles, x_image):
        e_graph = self.coding1(x_smiles)
        e_image = self.coding2(x_image)

        return e_graph, e_image


class ProteinBertCoding(nn.Module):
    def __init__(self, bert_hid_dim=Protein_Bert_Hidden_Size, bert_output_dim=Feature_Size):
        super(ProteinBertCoding, self).__init__()
        self.seq_coding = nn.Sequential(
            nn.Linear(1280, bert_hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(bert_hid_dim, bert_output_dim),
            nn.Sigmoid(),
        )

    def forward(self, xp):
        ep = self.seq_coding(xp)
        return ep


class ProteinPointCoding(nn.Module):
    """
    xp: [B, N, N+1]
        xp[:, :, :N]   -> p_t  (拓扑/邻接类矩阵) [B, N, N]
        xp[:, :, N:]   -> p_a  (离散残基类型id)  [B, N, 1]
    输出:
        out: [B, point_output_dim]
    """

    def __init__(
        self,
        point_hid_dim=Protein_Point_Hidden_Size,   # 512
        point_output_dim=Feature_Size,             # 256
        channel=PPC_Kernel_Num,                    # 32
        kernel_size=PPC_Kernel_Size,               # 8
        # global summary
        global_token_num=16,
        n_heads=8,
        dropout=0.1,
        max_len=Protein_Max_Lengtgh               # 1024
    ):
        super().__init__()
        assert point_hid_dim % n_heads == 0, "point_hid_dim 必须能被 n_heads 整除"

        # ========== pa embedding ==========
        self.aa_embed = nn.Embedding(len(AA_Dict) + 1, point_hid_dim, padding_idx=0)
        self.aa_proj  = nn.Linear(point_hid_dim, point_hid_dim)

        # ========== positional embedding（给并行分支用） ==========
        self.pos_embedding = nn.Embedding(max_len, point_hid_dim)

        # ========== pa-pt 注意力融合（pa 读 pt-struct token） ==========
        self.pa_pt_attn = nn.MultiheadAttention(
            embed_dim=point_hid_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pa_pt_norm = nn.LayerNorm(point_hid_dim)
        self.pa_pt_drop = nn.Dropout(dropout)

        # ========== CNN 分支（局部） ==========
        self.conv_local = nn.Sequential(
            nn.Conv1d(point_hid_dim, channel, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channel),

            nn.Conv1d(channel, channel * 2, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channel * 2),

            nn.Conv1d(channel * 2, channel * 4, kernel_size, padding=kernel_size - 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channel * 4),
        )
        self.local_pool = nn.AdaptiveMaxPool1d(1)
        self.local_fc   = nn.Linear(channel * 4, point_hid_dim)

        # ========== 另一并行模块：global tokens 读序列（全局压缩） ==========
        self.global_token_num = global_token_num
        self.global_tokens = nn.Parameter(torch.randn(1, global_token_num, point_hid_dim))

        self.readout_attn = nn.MultiheadAttention(
            embed_dim=point_hid_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.readout_norm = nn.LayerNorm(point_hid_dim)
        self.global_fc    = nn.Linear(point_hid_dim, point_hid_dim)

        # ========== 两分支输出 cross-attn 融合 ==========
        self.cnn_to_global = nn.MultiheadAttention(
            embed_dim=point_hid_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.global_to_cnn = nn.MultiheadAttention(
            embed_dim=point_hid_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cnn_norm    = nn.LayerNorm(point_hid_dim)
        self.global_norm = nn.LayerNorm(point_hid_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        # ========== 最终 FC 输出 ==========
        self.out_head = nn.Sequential(
            nn.Linear(point_hid_dim * 2, point_hid_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(point_hid_dim * 2, point_output_dim),
        )
        self.out_norm = nn.LayerNorm(point_output_dim)
        self.out_act  = nn.LeakyReLU(0.2)

    def forward(self, xp):
        # ----- split -----
        N = xp.shape[1]
        p_t = xp[:, :, :N]                       # [B, N, N]
        p_a = xp[:, :, N:].squeeze(-1).long()    # [B, N]
        B, N = p_a.shape

        # padding mask（id==0 为 PAD）
        pad_mask = (p_a == 0)                    # [B, N]

        # ========== 1) pa token ==========
        h_a = self.aa_proj(self.aa_embed(p_a))   # [B, N, D]

        # ========== 2) pt -> struct token（用邻接聚合得到结构表示） ==========
        # (可选) 做一个简单度归一化，避免不同节点度导致尺度漂移
        deg = p_t.sum(dim=-1, keepdim=True).clamp_min(1e-6)     # [B, N, 1]
        h_t = torch.bmm(p_t, h_a) / deg                          # [B, N, D]

        # ========== 3) pa-pt 注意力融合（pa 读结构 token） ==========
        # key_padding_mask 作用于 key/value：pad 位置不被读入
        attn_out, _ = self.pa_pt_attn(
            query=h_a,
            key=h_t,
            value=h_t,
            key_padding_mask=pad_mask,
            need_weights=False
        )
        h = self.pa_pt_norm(h_a + self.pa_pt_drop(attn_out))     # [B, N, D]

        # 加位置编码（给并行分支用）
        pos_ids = torch.arange(N, device=xp.device).unsqueeze(0).expand(B, N)
        h = h + self.pos_embedding(pos_ids)

        # （可选）把 padding 位置清零，减少 CNN 污染
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # ========== 4) 并行分支 A：CNN ==========
        x_local = self.conv_local(h.permute(0, 2, 1))            # [B, C*4, ~N]
        v_cnn = self.local_pool(x_local).squeeze(-1)             # [B, C*4]
        v_cnn = self.local_fc(v_cnn)                             # [B, D]
        v_cnn = F.normalize(v_cnn, p=2, dim=1)

        # ========== 5) 并行分支 B：global tokens 读序列 ==========
        g0 = self.global_tokens.expand(B, -1, -1)                # [B, M, D]
        g_attn, _ = self.readout_attn(
            query=g0,
            key=h,
            value=h,
            key_padding_mask=pad_mask,
            need_weights=False
        )
        g = self.readout_norm(g0 + g_attn)                       # [B, M, D]
        v_global = g.max(dim=1).values                           # [B, D]
        v_global = self.global_fc(v_global)                      # [B, D]
        v_global = F.normalize(v_global, p=2, dim=1)

        # ========== 6) 两分支输出 cross-attn 融合 ==========
        t_cnn = v_cnn.unsqueeze(1)                               # [B, 1, D]
        t_g   = v_global.unsqueeze(1)                            # [B, 1, D]

        # cnn -> global
        c_ctx, _ = self.cnn_to_global(
            query=t_cnn, key=t_g, value=t_g, need_weights=False
        )
        t_cnn = self.cnn_norm(t_cnn + self.fuse_drop(c_ctx))

        # global -> cnn（用更新后的 t_cnn）
        g_ctx, _ = self.global_to_cnn(
            query=t_g, key=t_cnn, value=t_cnn, need_weights=False
        )
        t_g = self.global_norm(t_g + self.fuse_drop(g_ctx))

        # ========== 7) FC 输出 ==========
        fused = torch.cat([t_cnn.squeeze(1), t_g.squeeze(1)], dim=1)   # [B, 2D]
        out = self.out_head(fused)                                     # [B, out_dim]
        out = self.out_norm(out)
        out = self.out_act(out)
        return out

    
class ProteinGraphEncoder(nn.Module):
    """
    输入: xg  (B, max_n, d + max_n)
      - 前 d 列: 节点特征 X (padding 后为 0)
      - 后 max_n 列: 邻接矩阵 A 的 padding 版本
    输出: (B, output_size)
    """
    def __init__(self, in_node_dim=320, hidden_dim=256, output_size=256, dropout=0.1):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.lin_in = nn.Linear(in_node_dim, hidden_dim)
        self.lin_gcn1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_size),
            nn.LeakyReLU(0.2)
        )

    def _norm_adj(self, A, mask):
        """
        A: (B, N, N) dense adjacency (0/1)
        mask: (B, N) True for valid nodes
        返回: 归一化后的 A_hat = D^{-1/2} (A+I) D^{-1/2}
        """
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)

        # 只在有效节点范围内加 self-loop，padding 节点不加
        mask_f = mask.float()
        M = mask_f.unsqueeze(2) * mask_f.unsqueeze(1)  # (B,N,N) valid-valid
        A = A * M
        A_tilde = (A + I) * M

        deg = A_tilde.sum(dim=-1)  # (B,N)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (B,N,N)

        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        return A_hat

    def forward(self, xg):
        """
        xg: (B, max_n, d+max_n)
        """
        B, N, Dtot = xg.shape
        d = self.in_node_dim
        assert Dtot >= d + N, f"xg last dim should be >= d+N, got {Dtot}, need {d+N}"

        X = xg[:, :, :d]          # (B,N,d)
        A = xg[:, :, d:d+N]       # (B,N,N)

        # mask: 哪些节点是有效的（X 不全为 0 视为有效）
        mask = (X.abs().sum(dim=-1) > 0)  # (B,N) bool

        # 节点初始投影
        h = self.lin_in(X)  # (B,N,hidden)
        h = F.leaky_relu(h, 0.2)
        h = self.dropout(h)

        # 归一化邻接
        A_hat = self._norm_adj(A, mask)  # (B,N,N)

        # GCN layer 1
        h1 = A_hat @ self.lin_gcn1(h)
        h1 = self.norm1(h1)
        h1 = F.leaky_relu(h1, 0.2)
        h1 = self.dropout(h1)

        # GCN layer 2
        h2 = A_hat @ self.lin_gcn2(h1)
        h2 = self.norm2(h2)
        h2 = F.leaky_relu(h2, 0.2)

        # masked global max pool
        # padding 节点置为 -inf，确保 maxpool 不选到
        neg_inf = torch.finfo(h2.dtype).min
        h2_masked = h2.masked_fill(~mask.unsqueeze(-1), neg_inf)  # (B,N,H)
        g = h2_masked.max(dim=1).values  # (B,H)

        out = self.readout(g)  # (B, output_size)
        return out


class ProteinCoding(nn.Module):
    def __init__(self):
        super(ProteinCoding, self).__init__()
        self.coding1 = ProteinBertCoding()
        self.coding2 = ProteinPointCoding()
        self.coding3 = ProteinGraphEncoder(in_node_dim=320, output_size=Feature_Size)  # 直接嵌入图模态

    def forward(self, x_bert, x_point, x_graph):
        e_bert = self.coding1(x_bert)
        e_point = self.coding2(x_point)
        e_graph = self.coding3(x_graph)

        return e_bert, e_point, e_graph


class PreNetMLP(nn.Module):
    def __init__(self, smiles_output_dim=Feature_Size, bert_output_dim=Feature_Size):
        super(PreNetMLP, self).__init__()
        self.d_c = DrugCoding()
        self.p_c = ProteinCoding()
        self.fc1 = nn.Linear((smiles_output_dim + bert_output_dim)*2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
        self.act1 = nn.LeakyReLU(0.2)
        self.act2 = nn.Tanh()

    def forward(self, d_s, d_i, p_b, p_p, p_g):
        eds, edi = self.d_c(d_s, d_i)
        epb, epp, epg = self.p_c(p_b, p_p, p_g)

        self.alpha = nn.Parameter(torch.tensor(0.5))  # epg权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # epp权重
        e_protein = self.alpha * epp + self.beta * epg
        #e=torch.cat((eds, epb,epg), dim=1)
        #e=torch.cat((edi, epp,epg), dim=1)
        e = torch.cat((eds, edi, epb, e_protein), dim=1)
        s0 = self.fc1(e)
        a0 = self.act1(s0)
        s1 = self.fc2(a0)
        a1 = self.act2(s1)
        s2 = self.fc3(a1)
        return eds, edi, epb, epp, s2


def seed_torch(seed):
    random.seed()
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(SEED)
    task = 'drugbank'   # select from 'drugbank', 'bindingdb' and 'dtinet'
    #task = 'kiba'  # select from 'drugbank', 'bindingdb' and 'dtinet'
    #task = 'bindingdb'  # select from 'drugbank', 'bindingdb' and 'dtinet'
    #confi = 'confi60'
    # parser = argparse.ArgumentParser(description='train and test set')
    # parser.add_argument('--task', type=str, default='drugbank', help='task name')
    # parser.add_argument('--path', type=str, default='', help='file name')
    # args = parser.parse_args()
    # task = args.task
    # path = args.path
    # 将所有的药物和蛋白所需数据读入内存中方便后续快速取用，确保针对每种药物/蛋白只处理一次
    fold = 0
    print('FOLD:', fold)
    drug_smiles_data, drug_points_data = data_preparation_drug_all(task)
    protein_bert_data, protein_point_data = data_preparation_protein_all(task)
    protein_graph_data = data_preparation_protein_graph_all(task)

    with open('dataset/' + task + '/result/CV5/train_cv0.csv') as f1:
        train_data = f1.readlines()
    num_sample = len(train_data)
    drug_idx_train, protein_idx_train, label_train = data_preparation(train_data, task)
    print(label_train.size())

    dataset = TensorDataset(drug_idx_train, protein_idx_train, label_train)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    
    model = PreNetMLP()
    # print(model)
    loss_func1 = nn.MSELoss()
    loss_func2 = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    if torch.cuda.is_available():
        model = model.cuda()
    print('Start training...')
    max_auc = 0
    max_aupr = 0
    loss_list = []
    for epoch in range(EPOCH):
        for step, data in enumerate(dataloader):
            batch_d_idx, batch_p_idx, batch_y = data
            batch_xd1, batch_xd2 = data_preparation_drug(drug_smiles_data, drug_points_data, batch_d_idx)  # xd1存序列，xd2存分子图片
            batch_xp1, batch_xp2 = data_preparation_protein(protein_bert_data, protein_point_data, batch_p_idx)  # xp1存序列特征，xp2存3d点云
            batch_xg = [protein_graph_data[i] for i in batch_p_idx.tolist()]
            # padding 成 batch
            max_n = max(xg.shape[0] for xg in batch_xg)
            batch_size = len(batch_xg)
            # 计算 d：d = xg.shape[1] - n
            # 假设 d 在所有样本里一致
            n0 = batch_xg[0].shape[0]
            d = batch_xg[0].shape[1] - n0
            batch_xg_tensor = torch.zeros(batch_size, max_n,d + max_n)
            for i, xg in enumerate(batch_xg):
                n = xg.shape[0]
                di = xg.shape[1] - n
                assert di == d, f"inconsistent node feature dim: {di} vs {d}"

                # xg: (n, d+n) = [X | A]
                # 填 X
                batch_xg_tensor[i, :n, :d] = xg[:, :d]
                # 填 A 到前 n 列
                batch_xg_tensor[i, :n, d:d+n] = xg[:, d:d+n]

            if torch.cuda.is_available():
                    batch_xd1 = batch_xd1.cuda()
                    batch_xd2 = batch_xd2.cuda()
                    batch_xp1 = batch_xp1.cuda()
                    batch_xp2 = batch_xp2.cuda()
                    batch_xg_tensor = batch_xg_tensor.cuda()
                    batch_y = batch_y.cuda()
            # print(epoch, step)
            batch_ed1, batch_ed2, batch_ep1, batch_ep2, batch_pre = model(batch_xd1, batch_xd2, batch_xp1, batch_xp2, batch_xg_tensor)
            # loss1 = loss_func1(batch_ed1, batch_ed2)
            # loss2 = loss_func1(batch_ep1, batch_ep2)
            loss3 = loss_func2(batch_pre, batch_y)
            loss = loss3
            loss_list.append(loss3.cpu().item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 10 == 0:
                if torch.cuda.is_available():
                    print('Epoch: ', epoch, ' | Step: ', step, '/', int(num_sample / BATCH_SIZE) + 1,
                          '| loss: %.20f' % loss.cpu().item())
                else:
                    print('Epoch: ', epoch, '| loss: %.20f' % loss.item())

        with open('dataset/' + task + '/result/CV5/test_cv0.csv') as f1:
            val_data = f1.readlines()
        # num_sample = len(train_data)
        drug_idx_val, protein_idx_val, label_val = data_preparation(val_data, task)

        dataset_val = TensorDataset(drug_idx_val, protein_idx_val, label_val)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False)
        pre_val, y_val = [], []
        for step_val, data_val in enumerate(dataloader_val):
            batch_d_idx_val, batch_p_idx_val, batch_y_val = data_val
            batch_xd1_val, batch_xd2_val = data_preparation_drug(drug_smiles_data, drug_points_data, batch_d_idx_val)   # xd1存序列，xd2存分子图片
            batch_xp1_val, batch_xp2_val = data_preparation_protein(protein_bert_data, protein_point_data, batch_p_idx_val)  # xp1存序列特征，xp2存3d点云
            batch_graph = [protein_graph_data[i] for i in batch_p_idx_val.cpu().numpy()]
            max_n = max(xg.shape[0] for xg in batch_graph)
            batch_size = len(batch_graph)
            n1 = batch_graph[0].shape[0]
            d1 = batch_graph[0].shape[1] - n1
            batch_xg_val = torch.zeros(batch_size, max_n, d1 + max_n)
            for i, xg in enumerate(batch_graph):
                n = xg.shape[0]
                di = xg.shape[1] - n
                assert di == d, f"inconsistent node feature dim: {di} vs {d}"

                # xg: (n, d+n) = [X | A]
                # 填 X
                batch_xg_val[i, :n, :d] = xg[:, :d]
                # 填 A 到前 n 列
                batch_xg_val[i, :n, d:d+n] = xg[:, d:d+n]

            if torch.cuda.is_available():
                batch_xd1_val = batch_xd1_val.cuda()
                batch_xd2_val = batch_xd2_val.cuda()
                batch_xp1_val = batch_xp1_val.cuda()
                batch_xp2_val = batch_xp2_val.cuda()
                batch_xg_val = batch_xg_val.cuda()
                batch_y_val = batch_y_val.cuda()
            # print(epoch, step)
            batch_ed1_val, batch_ed2_val, batch_ep1_val, batch_ep2_val, batch_pre_val = model(batch_xd1_val, batch_xd2_val, batch_xp1_val, batch_xp2_val, batch_xg_val)
            pre_val += batch_pre_val.detach().cpu().numpy()[:, 1].tolist()
            y_val += batch_y_val.detach().cpu().numpy().tolist()

        val_metrics = val_evalute(pre_val, y_val)   # auc, aupr, acc, sen, spe
        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3], val_metrics[4]])
        print('AUC = ', str(val_metrics[0]), ' | AUPR = ', str(val_metrics[1]))
        if val_metrics[0] > max_auc and val_metrics[1] > max_aupr:
            max_auc = val_metrics[0]
            max_aupr = val_metrics[1]
            max_acc = val_metrics[2]
            max_sen = val_metrics[3]
            max_spe = val_metrics[4]
            best_model = model
            print('Get a better performance! Max_AUC = ' + str(max_auc) + ' and Max_AUPR = ' + str(val_metrics[1]) + '\nACC SEN SPE = ' + str(max_acc) +' '+ str(max_sen) +' '+ str(max_spe))
    torch.save(best_model.state_dict(), 'models/esm_' + task + '3.pth')
    # np.savetxt('only_structure/loss_' + task + '.csv', loss_list)
    print('kiba：max_auc = '+str(max_auc)+' max_acc = '+ str(max_acc))
    print(fold)