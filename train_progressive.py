"""Progressive training for high-resolution image generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, Tuple

from config import ModelConfig, TrainingConfig, DataConfig
from models.attention import AttentionBlock
from models.wavelets import AdvancedWaveletConv2d, WaveletAttention
from models.metrics import FIDCalculator, InceptionScore
from main import FNO_Double, PatchDiscriminator, downsample, make_pyramid
import lpips
from pytorch_msssim import ssim


class ProgressiveWNO(nn.Module):
    """Progressive W-NO with attention and advanced wavelets."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.current_resolution = config.min_resolution
        
        # Resolution-specific parameters
        self.resolution_params = nn.ModuleDict()
        
        # Base model (shared across resolutions)
        self.base_model = self._build_base_model()
        
        # Resolution-specific heads
        for res in [64, 128, 256]:
            self.resolution_params[str(res)] = nn.ModuleDict({
                'input_proj': nn.Conv2d(7, config.width, 1),  # RGB + noise + coords + skip
                'output_proj': nn.Conv2d(config.width, 3, 1),
                'attention': AttentionBlock(config.width, heads=config.attention_heads) if config.use_attention else nn.Identity()
            })
            
    def _build_base_model(self):
        """Build the base W-NO model."""
        layers = []
        
        for i in range(self.config.n_layers):
            # Wavelet convolution
            layers.append(AdvancedWaveletConv2d(
                self.config.width, 
                self.config.width,
                wave=self.config.wavelet_type
            ))
            
            # Dense layer
            layers.append(nn.Sequential(
                nn.GroupNorm(8, self.config.width),
                nn.Conv2d(self.config.width, self.config.width * 4, 1),
                nn.GELU(),
                nn.Conv2d(self.config.width * 4, self.config.width, 1)
            ))
            
            # Local convolutions
            layers.append(nn.Sequential(
                nn.GroupNorm(8, self.config.width),
                nn.Conv2d(self.config.width, self.config.width, 3, padding=1),
                nn.GELU()
            ))
            
            # Add attention every 2 layers
            if self.config.use_attention and i % 2 == 1:
                layers.append(WaveletAttention(self.config.width, self.config.wavelet_type))
                
        return nn.ModuleList(layers)
        
    def forward(self, x_up: torch.Tensor, noise: torch.Tensor, coords: torch.Tensor, 
                x_skip: torch.Tensor, resolution: int) -> torch.Tensor:
        """Forward pass at specific resolution."""
        # Get resolution-specific parameters
        res_params = self.resolution_params[str(resolution)]
        
        # Concatenate inputs
        x = torch.cat([x_up, noise, coords, x_skip], dim=1)
        x = res_params['input_proj'](x)
        
        # Apply base model
        for layer in self.base_model:
            x = x + layer(x) if isinstance(layer, (AdvancedWaveletConv2d, WaveletAttention)) else layer(x)
            
        # Apply resolution-specific attention
        x = res_params['attention'](x)
        
        # Output projection
        out = res_params['output_proj'](x)
        
        # Residual connection and clipping
        out = x_up + out
        out = torch.clamp(out, 0, 1)
        
        return out
        
    def set_resolution(self, resolution: int):
        """Set the current training resolution."""
        self.current_resolution = resolution


class ProgressiveDiscriminator(nn.Module):
    """Progressive discriminator that grows with generator."""
    
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleDict({
            '64': PatchDiscriminator(ndf=64),
            '128': PatchDiscriminator(ndf=128),
            '256': PatchDiscriminator(ndf=256)
        })
        
    def forward(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        return self.discriminators[str(resolution)](x)


def create_coordinate_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create normalized coordinate grid."""
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
    return coords


def generate_noise_schedule(resolution: int, base_std: float = 0.1) -> float:
    """Adaptive noise schedule based on resolution."""
    # Less noise at higher resolutions
    scale = 64 / resolution
    return base_std * np.sqrt(scale)


def train_progressive(config: TrainingConfig, model_config: ModelConfig, data_config: DataConfig):
    """Progressive training with multiple resolutions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    model = ProgressiveWNO(model_config).to(device)
    discriminator = ProgressiveDiscriminator().to(device)
    
    # Optimizers
    opt_g = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999))
    
    # Mixed precision training
    scaler_g = GradScaler() if config.mixed_precision else None
    scaler_d = GradScaler() if config.mixed_precision else None
    
    # Loss functions
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    fid_calculator = FIDCalculator(device)
    is_calculator = InceptionScore(device)
    
    # Progressive training loop
    for resolution, epochs in config.resolution_schedule:
        print(f"\n=== Training at {resolution}x{resolution} resolution ===")
        model.set_resolution(resolution)
        
        # Create dataset for current resolution
        dataset = create_dataset(data_config, resolution)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                              shuffle=True, num_workers=data_config.num_workers)
        
        for epoch in range(epochs):
            # Training
            model.train()
            discriminator.train()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for i, (images, labels) in enumerate(pbar):
                images = images.to(device)
                
                # Create image pyramid
                pyramid = make_pyramid(images, model_config.min_resolution)
                
                # Generate samples progressively
                with autocast(enabled=config.mixed_precision):
                    generated = generate_progressive(
                        model, pyramid[0], resolution, 
                        model_config.upscale_factor, device
                    )
                    
                    # Discriminator loss
                    real_pred = discriminator(images, resolution)
                    fake_pred = discriminator(generated.detach(), resolution)
                    
                    d_loss = F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean()
                    
                # Update discriminator
                opt_d.zero_grad()
                if scaler_d:
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()
                else:
                    d_loss.backward()
                    opt_d.step()
                    
                # Generator loss
                if i % 2 == 0:  # Update generator every other step
                    with autocast(enabled=config.mixed_precision):
                        # Reconstruction loss
                        l1_loss = F.l1_loss(generated, images)
                        
                        # Perceptual loss
                        perceptual_loss = lpips_fn(generated, images).mean()
                        
                        # SSIM loss
                        ssim_loss = 1 - ssim(generated, images)
                        
                        # Adversarial loss
                        fake_pred = discriminator(generated, resolution)
                        adv_loss = -fake_pred.mean()
                        
                        # Total generator loss
                        g_loss = (config.l1_weight * l1_loss +
                                config.perceptual_weight * perceptual_loss +
                                config.ssim_weight * ssim_loss +
                                config.adversarial_weight * adv_loss)
                                
                    # Update generator
                    opt_g.zero_grad()
                    if scaler_g:
                        scaler_g.scale(g_loss).backward()
                        scaler_g.step(opt_g)
                        scaler_g.update()
                    else:
                        g_loss.backward()
                        opt_g.step()
                        
                # Update metrics
                if i % 100 == 0:
                    fid_calculator.update(images, generated.detach())
                    
                pbar.set_postfix({
                    'G': f"{g_loss.item():.4f}",
                    'D': f"{d_loss.item():.4f}"
                })
                
            # Compute FID every few epochs
            if (epoch + 1) % config.fid_frequency == 0:
                fid = fid_calculator.compute()
                print(f"FID at epoch {epoch+1}: {fid:.2f}")
                
        # Save checkpoint for this resolution
        torch.save({
            'model': model.state_dict(),
            'discriminator': discriminator.state_dict(),
            'resolution': resolution,
            'config': config
        }, f'checkpoint_{resolution}.pt')


def generate_progressive(model: ProgressiveWNO, x_start: torch.Tensor, 
                        target_resolution: int, upscale_factor: float,
                        device: torch.device) -> torch.Tensor:
    """Generate images progressively from coarse to fine."""
    x = x_start
    current_res = x.shape[-1]
    
    while current_res < target_resolution:
        # Upsample
        x_up = F.interpolate(x, scale_factor=upscale_factor, mode='bilinear', align_corners=False)
        current_res = x_up.shape[-1]
        
        # Generate noise
        noise_std = generate_noise_schedule(current_res)
        noise = torch.randn_like(x_up) * noise_std
        
        # Create coordinate grid
        coords = create_coordinate_grid(current_res, current_res, device)
        coords = coords.expand(x_up.shape[0], -1, -1, -1)
        
        # Apply model
        x = model(x_up, noise, coords, x_up, min(current_res, target_resolution))
        
    return x


def create_dataset(config: DataConfig, resolution: int):
    """Create dataset for specific resolution."""
    # This is a placeholder - implement actual dataset loading
    # For now, return CIFAR-10 resized to target resolution
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ])
    
    if config.dataset == 'cifar10':
        return datasets.CIFAR10(config.data_path, train=True, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")


if __name__ == "__main__":
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Start progressive training
    train_progressive(training_config, model_config, data_config)