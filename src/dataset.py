from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from torchvision import transforms

class myDataSet(Dataset):
    def __init__(self, dataset, should_augment=False):
        self.dataset = dataset
        self.should_augment = should_augment

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset["path"].iloc[idx]

        # Load PNG as a color image
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Image could not be loaded: {path}")

        # OpenCV loads color images in BGR order; convert to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Change axis order: (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # 1. Convert image to a float tensor
        img = torch.tensor(img, dtype=torch.float32)

        # 2. Scale pixel values to [0, 1]
        img = img / 255.0

        if self.should_augment:
            img = self.augmentation(img)

        target = int(self.dataset["label"].iloc[idx])
        
        name = self.dataset["name"].iloc[idx]

        return {
            "xray": img,
            "name": name,
            "target": torch.tensor(target, dtype=torch.long),
        }