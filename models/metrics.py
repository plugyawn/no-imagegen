"""Advanced metrics for image generation quality evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from typing import Tuple, Optional


class InceptionV3Features(nn.Module):
    """InceptionV3 feature extractor for FID computation."""
    
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        
        # Extract layers up to the final average pooling
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # Resize to 299x299 if needed
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        # Extract features
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x


def calculate_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """Calculate Fréchet Inception Distance between real and fake features."""
    
    # Convert to numpy
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()
    
    # Calculate statistics
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


class FIDCalculator:
    """Helper class for computing FID during training."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = InceptionV3Features().to(device).eval()
        self.real_features = []
        self.fake_features = []
        
    @torch.no_grad()
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        """Update feature banks with new images."""
        real_feat = self.inception(real_images)
        fake_feat = self.inception(fake_images)
        
        self.real_features.append(real_feat)
        self.fake_features.append(fake_feat)
        
    def compute(self) -> float:
        """Compute FID from accumulated features."""
        if not self.real_features:
            return float('inf')
            
        real_features = torch.cat(self.real_features, dim=0)
        fake_features = torch.cat(self.fake_features, dim=0)
        
        fid = calculate_fid(real_features, fake_features)
        
        # Clear features
        self.real_features = []
        self.fake_features = []
        
        return fid


class InceptionScore:
    """Inception Score calculator."""
    
    def __init__(self, device='cuda', splits=10):
        self.device = device
        self.splits = splits
        self.inception = models.inception_v3(pretrained=True).to(device).eval()
        
    @torch.no_grad()
    def compute(self, images: torch.Tensor) -> Tuple[float, float]:
        """Compute Inception Score."""
        # Resize to 299x299
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            
        # Get predictions
        preds = []
        for i in range(0, len(images), 32):  # Process in batches
            batch = images[i:i+32]
            pred = F.softmax(self.inception(batch), dim=1)
            preds.append(pred)
            
        preds = torch.cat(preds, dim=0)
        
        # Calculate IS
        split_scores = []
        for k in range(self.splits):
            part = preds[k * len(preds) // self.splits:(k + 1) * len(preds) // self.splits]
            py = part.mean(dim=0)
            scores = part * (torch.log(part) - torch.log(py.unsqueeze(0)))
            scores = scores.sum(dim=1).mean().exp()
            split_scores.append(scores.item())
            
        return np.mean(split_scores), np.std(split_scores)