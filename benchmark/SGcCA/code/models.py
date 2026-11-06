import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import reduce
from dgllife.model.gnn import GCN
from dgllife.model.gnn import GAT
from ScConv import ScConv
from CEAA import CrossEfficientAdditiveAttnetion


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class SGcADTI(nn.Module):
    def __init__(self, **config):
        super(SGcADTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        # kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]

        self.mapd = nn.Linear(256, 128)
        self.mapp = nn.Linear(1200, 290)

        self.drug_extractor_GCN = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                               padding=drug_padding,
                                               hidden_feats=drug_hidden_feats)

        self.drug_extractor_SMILES = MolecularScConv(protein_emb_dim, num_filters, protein_padding)
        # self.drug_extractor_SMILES = MolecularCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.protein_extractor = ProteinScConv(protein_emb_dim, num_filters, protein_padding)
        # self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.ceaa = CrossEfficientAdditiveAttnetion(embed_dim=128)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, smiles, v_p, mode="train"):
        v_d = self.drug_extractor_GCN(bg_d)  # v_d.shape(64, 290, 128)
        v_s = self.drug_extractor_SMILES(smiles)  # v_s.shape(64, 290, 128)
        v_p = self.protein_extractor(v_p)  # v_p.shape:(64, 1200, 128)

        # ceaa
        fusion_ds = self.ceaa(v_d, v_s)
        # fusion_ds = torch.cat([v_d, v_s], dim=-1)
        # print(fusion_ds.shape)
        fusion_ds = self.mapd(fusion_ds)
        # print(fusion_ds.shape)
        v_p = v_p.transpose(1, 2)
        # print("v_p ", v_p .shape)
        v_p = self.mapp(v_p)
        # print("v_p ", v_p.shape)
        v_p = v_p.transpose(1, 2)

        f = self.ceaa(fusion_ds, v_p)
        f = reduce(f, "B H W-> B W", "max")
        v_s1 = reduce(v_s, "B H W-> B W", "max")
        v_d1 = reduce(v_d, "B H W-> B W", "max")
        v_p1 = reduce(v_p, "B H W-> B W", "max")
        fusion_ds1 = reduce(fusion_ds, "B H W-> B W", "max")
        vs_pooled = F.max_pool1d(v_s1, kernel_size=2)
        vd_pooled = F.max_pool1d(v_d1, kernel_size=2)
        vp_pooled = F.max_pool1d(v_p1, kernel_size=2)
        fusion_ds_pooled = F.max_pool1d(fusion_ds1, kernel_size=2)
        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score, vs_pooled, vd_pooled, vp_pooled, fusion_ds_pooled
        elif mode == "eval":
            return v_d, v_p, score, None


import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# class MolecularGAT(nn.Module):
#     def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None):
#         super(MolecularGAT, self).__init__()
#         self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
#         if padding:
#             with torch.no_grad():
#                 self.init_transform.weight[-1].fill_(0)
#         self.gat = GAT(in_feats=dim_embedding, hidden_feats=hidden_feats)
#         self.output_feats = hidden_feats[-1]
#
#     def forward(self, batch_graph):
#         node_feats = batch_graph.ndata.pop('h')
#         node_feats = self.init_transform(node_feats)
#         node_feats = self.gat(batch_graph, node_feats)
#         batch_size = batch_graph.batch_size
#         node_feats = node_feats.view(batch_size, -1, self.output_feats)
#         return node_feats


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


# 消融实验
# class ProteinCNN(nn.Module):
#     def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
#         super(ProteinCNN, self).__init__()
#         if padding:
#             self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
#         else:
#             self.embedding = nn.Embedding(26, embedding_dim)
#         in_ch = [embedding_dim] + num_filters
#         self.in_ch = in_ch[-1]
#         kernels = kernel_size
#         self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
#         self.bn1 = nn.BatchNorm1d(in_ch[1])
#         self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
#         self.bn2 = nn.BatchNorm1d(in_ch[2])
#         self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
#         self.bn3 = nn.BatchNorm1d(in_ch[3])
# 
#     def forward(self, v):
#         v = self.embedding(v.long())
#         v = v.transpose(2, 1)
#         v = self.bn1(F.relu(self.conv1(v)))
#         v = self.bn2(F.relu(self.conv2(v)))
#         v = self.bn3(F.relu(self.conv3(v)))
#         v = v.view(v.size(0), v.size(2), -1)
#         return v
# 
# class MolecularCNN(nn.Module):
#     def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
#         super(MolecularCNN, self).__init__()
#         if padding:
#             self.embedding = nn.Embedding(65, embedding_dim, padding_idx=0)
#         else:
#             self.embedding = nn.Embedding(65, embedding_dim)
#         in_ch = [embedding_dim] + num_filters
#         self.in_ch = in_ch[-1]
#         kernels = kernel_size
#         self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding='same')
#         self.bn1 = nn.BatchNorm1d(in_ch[1])
#         self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding='same')
#         self.bn2 = nn.BatchNorm1d(in_ch[2])
#         self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding='same')
#         self.bn3 = nn.BatchNorm1d(in_ch[3])
# 
#     def forward(self, v):
#         v = self.embedding(v.long())
#         v = v.transpose(2, 1)
#         v = self.bn1(F.relu(self.conv1(v)))
#         v = self.bn2(F.relu(self.conv2(v)))
#         v = self.bn3(F.relu(self.conv3(v)))
#         v = v.view(v.size(0), v.size(2), -1)
#         return v

class MolecularScConv(nn.Module):
    def __init__(self, embedding_dim, num_filters, padding=True):
        super(MolecularScConv, self).__init__()
        if padding:
            self.embedding = nn.Embedding(65, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(65, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.conv1 = ScConv(op_channel=128)

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = v.unsqueeze(-2)
        v = self.conv1(v).squeeze(-2)
        v = v.squeeze(-2)
        v = v.reshape(v.size(0), v.size(2), -1)
        return v


class ProteinScConv(nn.Module):
    def __init__(self, embedding_dim, num_filters, padding=True):
        super(ProteinScConv, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        # kernels = kernel_size
        self.conv1 = ScConv(op_channel=128)

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = v.unsqueeze(-2)
        v = self.conv1(v)
        v = v.squeeze(-2)
        v = v.reshape(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x