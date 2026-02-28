import torch
import torch.nn as nn
import math


def build_range_mask(num_patches, grid_size, R, device):
    """
    num_patches: N (without CLS)
    grid_size: sqrt(N)
    R: range radius
    """
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append((i, j))

    mask = torch.zeros((num_patches + 1, num_patches + 1), device=device)

    # CLS attends to all
    mask[0, :] = 1
    mask[:, 0] = 1

    for i in range(num_patches):
        xi, yi = coords[i]
        for j in range(num_patches):
            xj, yj = coords[j]
            if abs(xi - xj) <= R and abs(yi - yj) <= R:
                mask[i + 1, j + 1] = 1

    return mask


class PatchRangeAttention(nn.Module):
    def __init__(self, dim, num_heads, grid_size, R):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.R = R

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask = build_range_mask(
            num_patches=(self.grid_size ** 2),
            grid_size=self.grid_size,
            R=self.R,
            device=x.device
        )

        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class PRTBlock(nn.Module):
    def __init__(self, dim, num_heads, grid_size, R):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PatchRangeAttention(dim, num_heads, grid_size, R)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchRangeTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=6,
        R=2
    ):
        super().__init__()

        self.grid_size = img_size // patch_size
        num_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            PRTBlock(embed_dim, num_heads, self.grid_size, R)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])
