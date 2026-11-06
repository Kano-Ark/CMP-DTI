import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein, integer_label_smiles


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)

        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]

        # ====== 关键补丁：超长分子先裁剪到 max_drug_nodes ======
        if num_actual_nodes > self.max_drug_nodes:
            keep = torch.arange(self.max_drug_nodes, dtype=torch.int32, device=v_d.device)

            # 1) 裁剪节点特征
            actual_node_feats = actual_node_feats[:self.max_drug_nodes]

            # 2) 裁剪图的节点/边（DGL写法）
            #    node_subgraph 会保留这些节点诱导子图，并自动重编号
            v_d = v_d.subgraph(keep)
            if '_ID' in v_d.ndata:
                v_d.ndata.pop('_ID')
            if '_ID' in v_d.edata:
                v_d.edata.pop('_ID')
            num_actual_nodes = self.max_drug_nodes  # 更新

        # ====== 继续原逻辑：给真实节点加“是否虚拟节点”bit ======
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats

        # ====== padding 虚拟节点（这里 num_virtual_nodes 一定 >= 0）======
        virtual_node_feat = torch.cat(
            (torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1
        )
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        # ====== 下面保持你原来的 ======
        v_s = self.df.iloc[index]['SMILES']
        drug_name = self.df.iloc[index]["SMILES"]
        v_s = integer_label_smiles(v_s)

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["Y"]
        protein_name = self.df.iloc[index]["Protein"]
        return v_d, v_s, v_p, y, drug_name, protein_name