"""Perceiver-style attention blocks used for multimodal fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import config

class Attention(nn.Module):
    """Multi-head self-attention used inside latent processing blocks."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    """Cross-attention from latent queries to modality context features."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        b, n, _, h = *x.shape, self.heads
        
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            # Simplified mask handling
            pass

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PerceiverIO(nn.Module):
    """Lightweight Perceiver-style fusion module.

    Latent tokens repeatedly cross-attend to input features and then refine
    themselves using latent self-attention.
    """
    def __init__(self, depth, dim, queries_dim, logits_dim=None, num_latents=128, latent_dim=1024, cross_heads=1, latent_heads=8, cross_dim_head=64, latent_dim_head=64, weight_tie_layers=False, decoder_ff=False):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        
        self.cross_attend_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                CrossAttention(latent_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=0.),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Linear(latent_dim * 4, latent_dim)
            )
            for i in range(depth)
        ])

        self.latent_self_attend_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=0.),
                 nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Linear(latent_dim * 4, latent_dim)
            )
            for i in range(depth)
        ])
        
    def forward(self, data):
        b, *_ = data.shape
        x = repeat(self.latents, 'n d -> b n d', b=b)

        for cross_attend, self_attend in zip(self.cross_attend_blocks, self.latent_self_attend_blocks):
            x = cross_attend[0](x) # Norm
            x = cross_attend[1](x, context=data) + x # Cross-attention
            x = cross_attend[2](x) # Norm
            ff_out = cross_attend[5](cross_attend[4](cross_attend[3](x))) # Feedforward
            x = ff_out + x
            
            x = self_attend[0](x) # Norm
            x = self_attend[1](x) + x # Self-attention
            x = self_attend[2](x) # Norm
            ff_out = self_attend[5](self_attend[4](self_attend[3](x))) # Feedforward
            x = ff_out + x
            
        return x
