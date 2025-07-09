"""Modern perceptual losses using CLIP and DINO."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import clip


class CLIPPerceptualLoss(nn.Module):
    """Perceptual loss using CLIP image encoder."""
    
    def __init__(self, model_name: str = 'ViT-B/32', layers: Optional[List[int]] = None):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name)
        self.model.eval()
        
        # Freeze CLIP
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Layers to extract features from
        self.layers = layers or [3, 6, 9, 11]  # Different transformer layers
        
        # Register hooks
        self.features = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to extract features."""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
            
        # For Vision Transformer
        for i, layer in enumerate(self.model.visual.transformer.resblocks):
            if i in self.layers:
                layer.register_forward_hook(hook_fn(f'layer_{i}'))
                
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CLIP perceptual loss."""
        # Normalize inputs to CLIP's expected range
        pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # Extract features
        self.features = {}
        _ = self.model.visual(pred)
        pred_features = {k: v.clone() for k, v in self.features.items()}
        
        self.features = {}
        _ = self.model.visual(target)
        target_features = self.features
        
        # Compute loss
        loss = 0
        for layer_name in pred_features:
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            loss += F.l1_loss(pred_feat, target_feat)
            
        return loss / len(pred_features)


class DINOPerceptualLoss(nn.Module):
    """Perceptual loss using DINO self-supervised features."""
    
    def __init__(self, model_name: str = 'dino_vits16', layers: Optional[List[int]] = None):
        super().__init__()
        
        # Load DINO model
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.model.eval()
        
        # Freeze DINO
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Layers to extract features from
        self.layers = layers or [3, 6, 9, 11]
        
        # Feature storage
        self.features = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks."""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
            
        # Register hooks on transformer blocks
        for i, block in enumerate(self.model.blocks):
            if i in self.layers:
                block.register_forward_hook(hook_fn(f'block_{i}'))
                
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute DINO perceptual loss."""
        # Resize to DINO's expected input
        pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # DINO normalization (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # Extract features
        self.features = {}
        _ = self.model(pred)
        pred_features = {k: v.clone() for k, v in self.features.items()}
        
        self.features = {}
        _ = self.model(target)
        target_features = self.features
        
        # Compute loss
        loss = 0
        for layer_name in pred_features:
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            
            # Reshape transformer outputs if needed
            if pred_feat.dim() == 3:  # (B, N, D)
                B, N, D = pred_feat.shape
                H = W = int((N - 1) ** 0.5)  # Remove CLS token
                pred_feat = pred_feat[:, 1:].reshape(B, H, W, D).permute(0, 3, 1, 2)
                target_feat = target_feat[:, 1:].reshape(B, H, W, D).permute(0, 3, 1, 2)
                
            loss += F.l1_loss(pred_feat, target_feat)
            
        return loss / len(pred_features)


class CombinedPerceptualLoss(nn.Module):
    """Combined perceptual loss using multiple models."""
    
    def __init__(self, use_lpips: bool = True, use_clip: bool = True, 
                 use_dino: bool = False, weights: Optional[dict] = None):
        super().__init__()
        self.losses = nn.ModuleDict()
        
        # Default weights
        self.weights = weights or {
            'lpips': 1.0,
            'clip': 0.5,
            'dino': 0.5
        }
        
        # Initialize losses
        if use_lpips:
            import lpips
            self.losses['lpips'] = lpips.LPIPS(net='alex')
            
        if use_clip:
            self.losses['clip'] = CLIPPerceptualLoss()
            
        if use_dino:
            self.losses['dino'] = DINOPerceptualLoss()
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined perceptual loss."""
        total_loss = 0
        
        for name, loss_fn in self.losses.items():
            if name == 'lpips':
                loss = loss_fn(pred, target).mean()
            else:
                loss = loss_fn(pred, target)
                
            total_loss += self.weights.get(name, 1.0) * loss
            
        return total_loss


class StyleLoss(nn.Module):
    """Style loss using Gram matrices."""
    
    def __init__(self, feature_extractor: str = 'vgg19'):
        super().__init__()
        
        if feature_extractor == 'vgg19':
            vgg = models.vgg19(pretrained=True).features
            self.layers = [4, 9, 18, 27, 36]  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
            self.model = vgg
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
            
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix."""
        B, C, H, W = x.shape
        features = x.view(B, C, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute style loss."""
        # Extract features
        pred_features = []
        target_features = []
        
        x_pred = pred
        x_target = target
        
        for i, layer in enumerate(self.model):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.layers:
                pred_features.append(x_pred)
                target_features.append(x_target)
                
        # Compute style loss
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += F.l1_loss(pred_gram, target_gram)
            
        return loss / len(pred_features)