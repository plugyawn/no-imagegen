"""Conditional generation modules for W-NO."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import clip


class ClassEmbedding(nn.Module):
    """Class embedding for conditional generation."""
    
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Learned class embeddings
        self.embeddings = nn.Embedding(num_classes, embed_dim)
        
        # Project to model dimension
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Get class embeddings."""
        embed = self.embeddings(labels)
        return self.proj(embed)


class TextEncoder(nn.Module):
    """Text encoder using CLIP for text-to-image generation."""
    
    def __init__(self, embed_dim: int, clip_model: str = 'ViT-B/32'):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model)
        self.clip_model.eval()
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Project CLIP embeddings to model dimension
        clip_dim = self.clip_model.text_projection.shape[1]
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    @torch.no_grad()
    def encode_text(self, text: Union[str, list]) -> torch.Tensor:
        """Encode text using CLIP."""
        if isinstance(text, str):
            text = [text]
            
        tokens = clip.tokenize(text).to(next(self.proj.parameters()).device)
        text_features = self.clip_model.encode_text(tokens)
        return text_features
        
    def forward(self, text: Union[str, list, torch.Tensor]) -> torch.Tensor:
        """Forward pass with text encoding and projection."""
        if isinstance(text, (str, list)):
            text_features = self.encode_text(text)
        else:
            text_features = text
            
        return self.proj(text_features.float())


class ConditionalModulation(nn.Module):
    """FiLM-style modulation for conditional generation."""
    
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        
        # Generate scale and shift parameters
        self.film = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2)
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply conditional modulation."""
        # Get scale and shift
        film_params = self.film(cond)
        scale, shift = film_params.chunk(2, dim=-1)
        
        # Reshape for broadcasting
        scale = scale.view(-1, self.channels, 1, 1)
        shift = shift.view(-1, self.channels, 1, 1)
        
        # Apply modulation
        return x * (1 + scale) + shift


class ConditionalWNO(nn.Module):
    """Conditional Wavelet Neural Operator."""
    
    def __init__(self, base_model: nn.Module, cond_type: str = 'class', 
                 num_classes: Optional[int] = None, embed_dim: int = 512):
        super().__init__()
        self.base_model = base_model
        self.cond_type = cond_type
        self.embed_dim = embed_dim
        
        # Conditioning modules
        if cond_type == 'class':
            assert num_classes is not None
            self.cond_encoder = ClassEmbedding(num_classes, embed_dim)
        elif cond_type == 'text':
            self.cond_encoder = TextEncoder(embed_dim)
        else:
            raise ValueError(f"Unknown conditioning type: {cond_type}")
            
        # Add modulation layers to base model
        self._add_modulation_layers()
        
    def _add_modulation_layers(self):
        """Add conditional modulation to base model layers."""
        # Find all convolution layers and add modulation after them
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == self.base_model.config.width:
                # Add modulation after this layer
                setattr(self.base_model, f"{name}_mod", 
                       ConditionalModulation(module.out_channels, self.embed_dim))
                       
    def forward(self, x_up: torch.Tensor, noise: torch.Tensor, coords: torch.Tensor,
                x_skip: torch.Tensor, resolution: int, 
                condition: Union[torch.Tensor, str, list]) -> torch.Tensor:
        """Forward pass with conditioning."""
        # Encode condition
        if self.cond_type == 'class':
            cond_embed = self.cond_encoder(condition)
        else:  # text
            cond_embed = self.cond_encoder(condition)
            
        # Store conditioning in base model for access during forward
        self.base_model._current_condition = cond_embed
        
        # Run base model
        output = self.base_model(x_up, noise, coords, x_skip, resolution)
        
        # Clear conditioning
        delattr(self.base_model, '_current_condition')
        
        return output


class ConditionalDataset(torch.utils.data.Dataset):
    """Dataset wrapper for conditional generation."""
    
    def __init__(self, base_dataset, num_classes: Optional[int] = None):
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        # Get base item
        if hasattr(self.base_dataset, 'targets'):
            image = self.base_dataset.data[idx]
            label = self.base_dataset.targets[idx]
        else:
            image, label = self.base_dataset[idx]
            
        # Process image if needed
        if hasattr(self.base_dataset, 'transform') and self.base_dataset.transform:
            image = self.base_dataset.transform(image)
            
        return image, label