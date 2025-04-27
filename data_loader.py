import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from collections import defaultdict
from torchvision import transforms

class data_loader(Dataset):
    """
    Dataset class for generating image pairs for Siamese network training.

    Args:
        data_dir (str): Root directory of LFW images (class folders).
        pairs_file (str): Path to the .txt file defining image pairs.
        augment (bool): Whether to apply augmentation.
        split (bool): If True, split the pairs into train/val.
        mode (str): 'train' or 'val' — used if split=True.
        val_ratio (float): Ratio of val pairs to total (default 0.2).
        seed (int): Random seed for consistent splits.
    """
    def __init__(self, data_dir, pairs_file, augment=False, split=False, val_ratio=0.2, mode='train', seed=42):
        self.data_dir = data_dir
        self.augment = augment
        self.mode = mode

        # Load and (optionally) split the pairs
        all_pairs = self._load_pairs_file(pairs_file)
        if split:
            random.seed(seed)
            random.shuffle(all_pairs)
            val_size = int(len(all_pairs) * val_ratio)
            if mode == 'train':
                self.pairs = all_pairs[val_size:]
            elif mode == 'val':
                self.pairs = all_pairs[:val_size]
            else:
                raise ValueError("mode must be 'train' or 'val'")
        else:
            self.pairs = all_pairs

        # Base transform: resize + convert to tensor (normalizes to [0,1])
        self.base_transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor()  # Converts to float32 and scales [0, 255] → [0, 1]
        ])

        # Optional augmentation (only applied when augment=True)
        self.augment_transform = transforms.Compose([
            transforms.RandomApply([ # From paper
                transforms.RandomAffine(
                    degrees=10,                # ±10°
                    translate=(0.03, 0.03),    # ±3 %
                    scale=(0.9, 1.1))          # ±10 % zoom
            ], p=0.9),
            transforms.RandomHorizontalFlip(p=0.5),  # extra for faces
        ])

    def _load_pairs_file(self, file_path):
        pairs = []
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    name, idx1, idx2 = parts
                    img1 = f"{name}/{name}_{int(idx1):04d}.jpg"
                    img2 = f"{name}/{name}_{int(idx2):04d}.jpg"
                    label = 1
                elif len(parts) == 4:
                    name1, idx1, name2, idx2 = parts
                    img1 = f"{name1}/{name1}_{int(idx1):04d}.jpg"
                    img2 = f"{name2}/{name2}_{int(idx2):04d}.jpg"
                    label = 0
                else:
                    continue
                pairs.append((img1, img2, label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def load_image(self, img_path):
        """
        Loads an image from disk, applies optional augmentation and transforms.
        """
        full_path = os.path.join(self.data_dir, img_path)
        image = Image.open(full_path).convert('L')
        if self.augment and self.mode == 'train':
            image = self.augment_transform(image)
        return self.base_transform(image)

    def __getitem__(self, idx):
        img_path1, img_path2, label = self.pairs[idx]
        image1 = self.load_image(img_path1)
        image2 = self.load_image(img_path2)
        label = torch.tensor(float(label), dtype=torch.float32)
        img_name1 = os.path.basename(img_path1)
        img_name2 = os.path.basename(img_path2)
        return image1, image2, label, img_name1, img_name2