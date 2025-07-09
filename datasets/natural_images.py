"""Dataset loaders for natural images (CelebA, FFHQ, ImageNet)."""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import Optional, Callable, Tuple
import json


class CelebADataset(Dataset):
    """CelebA dataset for face generation."""
    
    def __init__(self, root: str, split: str = 'train', 
                 resolution: int = 256, center_crop: Optional[int] = None):
        self.root = root
        self.split = split
        self.resolution = resolution
        
        # Setup transforms
        transform_list = []
        if center_crop:
            transform_list.append(transforms.CenterCrop(center_crop))
        transform_list.extend([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose(transform_list)
        
        # Load image paths
        self.image_dir = os.path.join(root, 'img_align_celeba')
        self.images = self._load_split()
        
    def _load_split(self):
        """Load train/val/test split."""
        split_file = os.path.join(self.root, 'list_eval_partition.txt')
        images = []
        
        with open(split_file, 'r') as f:
            for line in f:
                img_name, partition = line.strip().split()
                # 0: train, 1: val, 2: test
                if (self.split == 'train' and partition == '0') or \
                   (self.split == 'val' and partition == '1') or \
                   (self.split == 'test' and partition == '2'):
                    images.append(img_name)
                    
        return images
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Return with dummy label for compatibility
        return img, 0


class FFHQDataset(Dataset):
    """FFHQ dataset for high-quality face generation."""
    
    def __init__(self, root: str, resolution: int = 256):
        self.root = root
        self.resolution = resolution
        
        # Find all images
        self.images = []
        for res_dir in os.listdir(root):
            if res_dir.startswith('images'):
                res_path = os.path.join(root, res_dir)
                if os.path.isdir(res_path):
                    for img in os.listdir(res_path):
                        if img.endswith('.png'):
                            self.images.append(os.path.join(res_path, img))
                            
        self.images.sort()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0


class ImageNetDataset(Dataset):
    """ImageNet dataset for diverse natural image generation."""
    
    def __init__(self, root: str, split: str = 'train', 
                 resolution: int = 256, num_classes: Optional[int] = None):
        self.root = root
        self.split = split
        self.resolution = resolution
        
        # Load class mapping
        self.class_to_idx = self._load_class_mapping()
        if num_classes:
            # Limit to first N classes
            self.class_to_idx = {k: v for k, v in self.class_to_idx.items() 
                               if v < num_classes}
                               
        # Load images
        self.samples = self._load_samples()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(int(resolution * 1.15)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
        
    def _load_class_mapping(self):
        """Load ImageNet class to index mapping."""
        class_file = os.path.join(self.root, 'imagenet_class_index.json')
        with open(class_file, 'r') as f:
            class_idx = json.load(f)
            
        return {v[0]: int(k) for k, v in class_idx.items()}
        
    def _load_samples(self):
        """Load image paths and labels."""
        samples = []
        split_dir = os.path.join(self.root, self.split)
        
        for class_dir in os.listdir(split_dir):
            if class_dir in self.class_to_idx:
                class_path = os.path.join(split_dir, class_dir)
                class_idx = self.class_to_idx[class_dir]
                
                for img in os.listdir(class_path):
                    if img.endswith(('.JPEG', '.jpg', '.png')):
                        samples.append((os.path.join(class_path, img), class_idx))
                        
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class LSUNDataset(Dataset):
    """LSUN dataset for scene generation."""
    
    def __init__(self, root: str, category: str = 'bedroom', 
                 split: str = 'train', resolution: int = 256):
        self.root = root
        self.category = category
        self.split = split
        self.resolution = resolution
        
        # Import lmdb
        try:
            import lmdb
            self.env = lmdb.open(os.path.join(root, f'{category}_{split}_lmdb'),
                               readonly=True, lock=False)
        except ImportError:
            raise ImportError("Please install lmdb: pip install lmdb")
            
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
            
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            img_bytes = txn.get(str(idx).encode())
            
        # Load image from bytes
        import io
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0


def get_natural_dataset(name: str, root: str, resolution: int = 256, 
                       split: str = 'train', **kwargs) -> Dataset:
    """Factory function to get natural image datasets."""
    
    if name.lower() == 'celeba':
        return CelebADataset(root, split, resolution, **kwargs)
    elif name.lower() == 'ffhq':
        return FFHQDataset(root, resolution)
    elif name.lower() == 'imagenet':
        return ImageNetDataset(root, split, resolution, **kwargs)
    elif name.lower().startswith('lsun'):
        category = name.split('_')[1] if '_' in name else 'bedroom'
        return LSUNDataset(root, category, split, resolution)
    else:
        raise ValueError(f"Unknown dataset: {name}")