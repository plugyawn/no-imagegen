"""Configuration for progressive image generation."""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ModelConfig:
    """Configuration for the W-NO model."""
    width: int = 192
    n_layers: int = 6
    modes: int = 16
    upscale_factor: float = 2.0
    min_resolution: int = 4
    max_resolution: int = 256  # Increased from 64
    use_attention: bool = True  # New: attention layers
    attention_heads: int = 8
    wavelet_type: str = 'db4'  # More sophisticated wavelets
    
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32  # Reduced for higher resolutions
    learning_rate: float = 3e-4
    discriminator_lr: float = 1e-4
    epochs: int = 100
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    
    # Progressive growing
    progressive_growing: bool = True
    resolution_schedule: List[Tuple[int, int]] = None  # (resolution, epochs)
    
    # Loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.1
    gradient_weight: float = 0.05
    adversarial_weight: float = 0.1
    fid_frequency: int = 5  # Compute FID every N epochs
    
    def __post_init__(self):
        if self.resolution_schedule is None:
            self.resolution_schedule = [
                (64, 25),    # 64x64 for 25 epochs
                (128, 25),   # 128x128 for 25 epochs
                (256, 50),   # 256x256 for 50 epochs
            ]
            

@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = 'celeba'  # Options: cifar10, celeba, ffhq
    data_path: str = './data'
    num_workers: int = 4
    image_size: int = 256
    center_crop: Optional[int] = None
    conditional: bool = True  # Enable conditional generation
    num_classes: Optional[int] = None  # For class-conditional generation