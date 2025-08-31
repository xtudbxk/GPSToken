import torch
import torch.nn as nn

class AttnBlock(torch.nn.Module):
    """ an conditional Transformer block """
    def __init__(self, n_embed, mlp_ratio=4, roic=9):
        super().__init__()
        self.roimlp = nn.Linear(roic, 1)

        self.ln1 = nn.LayerNorm(n_embed)
        self.attn1 = torch.nn.MultiheadAttention(embed_dim=n_embed, num_heads=8, batch_first=True)

        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, mlp_ratio*n_embed),
            nn.GELU(),  # nice
            nn.Linear(mlp_ratio*n_embed, n_embed),
            nn.Dropout(0.0),
        )

    def forward(self, x, conds):
        _b, _n, _c = x.shape

        # roi mlp
        cond_embed = self.roimlp(conds) # [b*n, c, roic] -> [b*n, c, 1]
        x = x + cond_embed.view(_b, _n, _c)

        # self-atten
        x_norm = self.ln1(x) # [b, n, c]
        attn, _ = self.attn1(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn

        # mlp
        x = x + self.mlp(self.ln2(x))

        return x
