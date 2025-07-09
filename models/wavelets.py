"""Advanced wavelet transforms for W-NO."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from typing import Optional


class AdvancedWaveletConv2d(nn.Module):
    """Wavelet convolution with support for multiple wavelet types."""
    
    def __init__(self, in_channels: int, out_channels: int, wave: str = 'db4', level: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wave = wave
        self.level = level
        
        # Wavelet transforms
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # Learnable convolutions for each frequency band
        self.conv_ll = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_high = nn.ModuleList([
            nn.Conv2d(in_channels * 3, out_channels * 3, 1) for _ in range(level)
        ])
        
        # Adaptive mixing weights
        self.mix_weights = nn.Parameter(torch.ones(level + 1))
        
    def forward(self, x):
        # Perform multi-level DWT
        yl, yh = self.dwt(x)
        
        # Process low-frequency component
        yl_out = self.conv_ll(yl)
        
        # Process high-frequency components at each level
        yh_out = []
        for i, (yh_level, conv) in enumerate(zip(yh, self.conv_high)):
            # yh_level has shape (B, C, 3, H, W) - combine the 3 orientations
            b, c, _, h, w = yh_level.shape
            yh_level = yh_level.view(b, c * 3, h, w)
            yh_level_out = conv(yh_level)
            yh_level_out = yh_level_out.view(b, self.out_channels, 3, h, w)
            yh_out.append(yh_level_out)
        
        # Reconstruct with adaptive mixing
        yl_out = yl_out * self.mix_weights[0]
        for i, yh_level in enumerate(yh_out):
            yh_level = yh_level * self.mix_weights[i + 1]
            yh_out[i] = yh_level
            
        # Inverse DWT
        out = self.idwt((yl_out, yh_out))
        return out


class WaveletAttention(nn.Module):
    """Attention mechanism in wavelet domain."""
    
    def __init__(self, channels: int, wave: str = 'db4'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # Attention for different frequency bands
        self.attn_ll = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        self.attn_high = nn.Sequential(
            nn.Conv2d(channels * 3, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels * 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Wavelet transform
        yl, yh = self.dwt(x)
        
        # Apply attention to low frequencies
        yl_attn = self.attn_ll(yl)
        yl = yl * yl_attn
        
        # Apply attention to high frequencies
        b, c, _, h, w = yh[0].shape
        yh_combined = yh[0].view(b, c * 3, h, w)
        yh_attn = self.attn_high(yh_combined)
        yh_attn = yh_attn.view(b, c, 3, h, w)
        yh = [yh[0] * yh_attn]
        
        # Reconstruct
        out = self.idwt((yl, yh))
        return out