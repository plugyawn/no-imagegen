"""Main training script for photorealistic image generation with W-NO."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from config import ModelConfig, TrainingConfig, DataConfig
from train_progressive import ProgressiveWNO, ProgressiveDiscriminator
from models.conditional import ConditionalWNO, ConditionalDataset
from models.perceptual_loss import CombinedPerceptualLoss
from models.metrics import FIDCalculator, InceptionScore
from datasets.natural_images import get_natural_dataset
from main import make_pyramid
from pytorch_msssim import ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Train photorealistic W-NO')
    parser.add_argument('--dataset', type=str, default='celeba', 
                       choices=['cifar10', 'celeba', 'ffhq', 'imagenet', 'lsun_bedroom'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    return parser.parse_args()


def create_models(args, config: ModelConfig):
    """Create generator and discriminator models."""
    # Base generator
    generator = ProgressiveWNO(config)
    
    # Add conditioning if requested
    if args.conditional:
        if args.dataset == 'imagenet':
            generator = ConditionalWNO(generator, cond_type='class', num_classes=1000)
        elif args.dataset in ['celeba', 'ffhq']:
            generator = ConditionalWNO(generator, cond_type='text')  # For attribute conditioning
            
    # Discriminator
    discriminator = ProgressiveDiscriminator()
    
    return generator, discriminator


def create_dataloaders(args, resolution: int):
    """Create data loaders for training and validation."""
    # Get dataset
    if args.dataset == 'cifar10':
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(args.data_path, train=True, transform=transform, download=True)
        val_dataset = datasets.CIFAR10(args.data_path, train=False, transform=transform, download=True)
    else:
        train_dataset = get_natural_dataset(args.dataset, args.data_path, resolution, split='train')
        val_dataset = get_natural_dataset(args.dataset, args.data_path, resolution, split='val')
        
    # Wrap with conditional dataset if needed
    if args.conditional:
        train_dataset = ConditionalDataset(train_dataset)
        val_dataset = ConditionalDataset(val_dataset)
        
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
                          
    return train_loader, val_loader


def train_epoch(generator, discriminator, train_loader, opt_g, opt_d, 
                perceptual_loss, device, args, scaler_g=None, scaler_d=None):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar):
        if args.conditional:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
        else:
            images, _ = batch
            images = images.to(device)
            labels = None
            
        batch_size = images.shape[0]
        resolution = images.shape[-1]
        
        # Create image pyramid
        pyramid = make_pyramid(images, generator.config.min_resolution)
        
        # Generate samples
        with autocast(enabled=scaler_g is not None):
            # Start from coarsest level
            x = pyramid[0]
            
            # Progressive generation
            while x.shape[-1] < resolution:
                # Upsample
                x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                h, w = x_up.shape[-2:]
                
                # Generate noise
                noise_std = 0.1 * np.sqrt(64 / h)  # Adaptive noise
                noise = torch.randn_like(x_up) * noise_std
                
                # Coordinate grid
                coords = create_coordinate_grid(h, w, device)
                coords = coords.expand(batch_size, -1, -1, -1)
                
                # Apply generator
                if args.conditional:
                    x = generator(x_up, noise, coords, x_up, h, labels)
                else:
                    x = generator(x_up, noise, coords, x_up, h)
                    
            generated = x
            
            # Discriminator predictions
            real_pred = discriminator(images, resolution)
            fake_pred = discriminator(generated.detach(), resolution)
            
            # Discriminator loss (Hinge loss)
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
            
        # Generator update every other step
        if i % 2 == 0:
            with autocast(enabled=scaler_g is not None):
                # Adversarial loss
                fake_pred = discriminator(generated, resolution)
                adv_loss = -fake_pred.mean()
                
                # Reconstruction losses
                l1_loss = F.l1_loss(generated, images)
                
                # Perceptual loss
                p_loss = perceptual_loss(generated, images)
                
                # SSIM loss
                ssim_loss = 1 - ssim(generated, images, data_range=1.0)
                
                # Edge loss (gradient matching)
                def gradient_loss(pred, target):
                    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
                    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
                    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
                    
                grad_loss = gradient_loss(generated, images)
                
                # Total generator loss
                g_loss = (1.0 * l1_loss + 
                         0.1 * p_loss + 
                         0.1 * ssim_loss + 
                         0.05 * grad_loss + 
                         0.1 * adv_loss)
                         
            # Update generator
            opt_g.zero_grad()
            if scaler_g:
                scaler_g.scale(g_loss).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                g_loss.backward()
                opt_g.step()
                
            total_g_loss += g_loss.item()
            
        total_d_loss += d_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'G': f"{g_loss.item() if i % 2 == 0 else 0:.4f}",
            'D': f"{d_loss.item():.4f}"
        })
        
    return total_g_loss / (len(train_loader) // 2), total_d_loss / len(train_loader)


def create_coordinate_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create normalized coordinate grid."""
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([xx, yy], dim=0).unsqueeze(0)


def evaluate(generator, val_loader, device, args):
    """Evaluate model performance."""
    generator.eval()
    fid_calc = FIDCalculator(device)
    is_calc = InceptionScore(device)
    
    all_generated = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if args.conditional:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            else:
                images, _ = batch
                images = images.to(device)
                labels = None
                
            # Generate from noise
            batch_size = images.shape[0]
            resolution = images.shape[-1]
            
            # Start from noise
            x = torch.randn(batch_size, 3, 4, 4, device=device) * 0.1
            
            # Progressive generation
            while x.shape[-1] < resolution:
                x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                h, w = x_up.shape[-2:]
                
                noise_std = 0.1 * np.sqrt(64 / h)
                noise = torch.randn_like(x_up) * noise_std
                
                coords = create_coordinate_grid(h, w, device)
                coords = coords.expand(batch_size, -1, -1, -1)
                
                if args.conditional:
                    x = generator(x_up, noise, coords, x_up, h, labels)
                else:
                    x = generator(x_up, noise, coords, x_up, h)
                    
            # Update metrics
            fid_calc.update(images, x)
            all_generated.append(x)
            
    # Compute metrics
    fid = fid_calc.compute()
    
    all_generated = torch.cat(all_generated, dim=0)[:5000]  # Use first 5k samples
    is_mean, is_std = is_calc.compute(all_generated)
    
    return fid, is_mean, is_std


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configurations
    model_config = ModelConfig(max_resolution=args.resolution)
    
    # Create models
    generator, discriminator = create_models(args, model_config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        print(f"Loaded checkpoint from {args.checkpoint}")
        
    # Optimizers
    opt_g = torch.optim.AdamW(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Mixed precision
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    
    # Loss functions
    perceptual_loss = CombinedPerceptualLoss(use_lpips=True, use_clip=True, use_dino=False).to(device)
    
    # Data loaders
    train_loader, val_loader = create_dataloaders(args, args.resolution)
    
    # Training loop
    best_fid = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Set resolution for progressive training
        generator.set_resolution(args.resolution)
        
        # Train
        g_loss, d_loss = train_epoch(
            generator, discriminator, train_loader, 
            opt_g, opt_d, perceptual_loss, device, args,
            scaler_g, scaler_d
        )
        
        print(f"Avg losses - G: {g_loss:.4f}, D: {d_loss:.4f}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            fid, is_mean, is_std = evaluate(generator, val_loader, device, args)
            print(f"FID: {fid:.2f}, IS: {is_mean:.2f} ± {is_std:.2f}")
            
            # Save best model
            if fid < best_fid:
                best_fid = fid
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'epoch': epoch,
                    'fid': fid,
                    'args': args
                }, os.path.join(args.output_dir, 'best_model.pt'))
                
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'epoch': epoch,
                'args': args
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
    print(f"\nTraining complete! Best FID: {best_fid:.2f}")


if __name__ == "__main__":
    main()