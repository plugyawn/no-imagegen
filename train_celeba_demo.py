"""Demo training script for CelebA with reduced requirements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime

from config import ModelConfig, TrainingConfig, DataConfig
from train_progressive import ProgressiveWNO, ProgressiveDiscriminator, create_coordinate_grid
from models.metrics import FIDCalculator
from main import make_pyramid
from pytorch_msssim import ssim
import lpips


def create_celeba_demo_dataset(data_path, resolution=128, num_samples=1000):
    """Create a demo CelebA dataset with limited samples."""
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # CelebA faces are 178x218, crop to square
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ])
    
    # Try to load CelebA
    try:
        dataset = datasets.CelebA(
            root=data_path,
            split='train',
            transform=transform,
            download=False  # Don't auto-download due to size
        )
        # Use subset for demo
        indices = list(range(min(num_samples, len(dataset))))
        dataset = Subset(dataset, indices)
        print(f"Loaded {len(dataset)} CelebA images")
    except:
        print("CelebA not found. Using CIFAR-10 as fallback...")
        # Fallback to CIFAR-10 for demo
        dataset = datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
            ]),
            download=True
        )
        indices = list(range(min(num_samples, len(dataset))))
        dataset = Subset(dataset, indices)
        
    return dataset


def visualize_generation(model, device, save_path, resolution=128):
    """Generate and save sample images."""
    model.eval()
    
    with torch.no_grad():
        # Generate 16 samples
        batch_size = 16
        
        # Start from noise at lowest resolution
        x = torch.randn(batch_size, 3, 4, 4, device=device) * 0.1
        
        # Progressive generation
        while x.shape[-1] < resolution:
            # Upsample
            x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            h, w = x_up.shape[-2:]
            
            # Generate noise (less noise at higher resolutions)
            noise_std = 0.1 * np.sqrt(64 / h)
            noise = torch.randn_like(x_up) * noise_std
            
            # Coordinate grid
            coords = create_coordinate_grid(h, w, device)
            coords = coords.expand(batch_size, -1, -1, -1)
            
            # Apply model
            x = model(x_up, noise, coords, x_up, min(h, resolution))
        
        # Save grid
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < batch_size:
                img = x[i].cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved samples to {save_path}")


def train_demo():
    """Demo training with reduced requirements."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Demo configuration
    resolution = 128  # Lower resolution for demo
    batch_size = 16 if torch.cuda.is_available() else 4
    epochs = 10  # Quick demo
    
    # Model configuration
    model_config = ModelConfig(
        max_resolution=resolution,
        width=128,  # Smaller model for demo
        n_layers=4,
        use_attention=True
    )
    
    # Create models
    generator = ProgressiveWNO(model_config).to(device)
    discriminator = ProgressiveDiscriminator().to(device)
    generator.set_resolution(resolution)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Optimizers
    opt_g = torch.optim.AdamW(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Loss functions
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # Create dataset
    dataset = create_celeba_demo_dataset('./data', resolution, num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Training loop
    os.makedirs('outputs', exist_ok=True)
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            # Handle different dataset formats
            if isinstance(batch, list) and len(batch) == 2:
                images, _ = batch
            else:
                images = batch[0] if isinstance(batch, tuple) else batch
                
            images = images.to(device)
            batch_size_actual = images.shape[0]
            
            # Create image pyramid
            pyramid = make_pyramid(images, model_config.min_resolution)
            
            # Progressive generation
            x = pyramid[0]
            
            while x.shape[-1] < resolution:
                x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                h, w = x_up.shape[-2:]
                
                noise_std = 0.1 * np.sqrt(64 / h)
                noise = torch.randn_like(x_up) * noise_std
                
                coords = create_coordinate_grid(h, w, device)
                coords = coords.expand(batch_size_actual, -1, -1, -1)
                
                x = generator(x_up, noise, coords, x_up, min(h, resolution))
            
            generated = x
            
            # Discriminator update
            real_pred = discriminator(images, resolution)
            fake_pred = discriminator(generated.detach(), resolution)
            d_loss = F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean()
            
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            
            # Generator update (every other step)
            if i % 2 == 0:
                fake_pred = discriminator(generated, resolution)
                adv_loss = -fake_pred.mean()
                
                # Reconstruction losses
                l1_loss = F.l1_loss(generated, images)
                perceptual_loss = lpips_fn(generated, images).mean()
                ssim_loss = 1 - ssim(generated, images, data_range=1.0)
                
                # Total loss
                g_loss = l1_loss + 0.1 * perceptual_loss + 0.1 * ssim_loss + 0.1 * adv_loss
                
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                
                total_g_loss += g_loss.item()
            
            total_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G': f"{g_loss.item() if i % 2 == 0 else 0:.4f}",
                'D': f"{d_loss.item():.4f}"
            })
        
        # Generate samples after each epoch
        visualize_generation(generator, device, f'outputs/samples_epoch_{epoch+1}.png', resolution)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'model_config': model_config,
            }, f'outputs/checkpoint_epoch_{epoch+1}.pt')
    
    print("\nDemo training complete!")
    print(f"Generated samples saved in outputs/")
    
    # Final visualization with more samples
    visualize_generation(generator, device, 'outputs/final_samples.png', resolution)


if __name__ == "__main__":
    # First try to ensure we have required packages
    try:
        import lpips
        import pytorch_msssim
    except ImportError:
        print("Installing required packages...")
        os.system("pip install lpips pytorch-msssim")
    
    train_demo()