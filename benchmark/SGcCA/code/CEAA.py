import torch
import torch.nn as nn
import einops
import torch.nn.functional as F


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=128, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x, y):
        query = self.to_query(x)
        key = self.to_key(y)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query  #BxNxD

        out = self.final(out) # BxNxD

        return out



class CrossEfficientAdditiveAttnetion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.map = nn.Linear(256, 128)
        self.eaa = EfficientAdditiveAttnetion()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, y):
        # Norm and Attention for x Branch
        x_norm1 = self.norm1(x)
        x_ = self.map(self.eaa(x_norm1, y))
        x_attn = x_ + x
        x_norm2 = self.norm2(x_attn)
        x = self.mlp(x_norm2)

        # Norm and Attention for y Branch
        y_norm1 = self.norm1(y)
        y_ = self.map(self.eaa(y_norm1, x))
        y_attn = y_ + y
        y_norm2 = self.norm2(y_attn)
        y = self.mlp(y_norm2)

        # x_pooled = F.max_pool1d(x, kernel_size=2)
        # y_pooled = F.max_pool1d(y, kernel_size=2)
        output = torch.cat((x, y), dim=-1)



        return output