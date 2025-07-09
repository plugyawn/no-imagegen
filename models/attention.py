"""Attention mechanisms for W-NO model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for W-NO."""
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention for conditional generation."""
    
    def __init__(self, dim: int, context_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context):
        b, c, h, w = x.shape
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)
        
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class AttentionBlock(nn.Module):
    """Combined attention block with self and cross attention."""
    
    def __init__(self, dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.self_attn = MultiHeadSelfAttention(dim, heads, dim_head)
        
        self.use_cross_attn = context_dim is not None
        if self.use_cross_attn:
            self.norm2 = nn.GroupNorm(8, dim)
            self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head)
            
    def forward(self, x, context=None):
        x = x + self.self_attn(self.norm1(x))
        
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
            
        return x