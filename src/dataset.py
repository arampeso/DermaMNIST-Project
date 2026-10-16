from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class myDataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset["path"].iloc[idx]

        # Load PNG as a grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Image could not be loaded: {path}")

        # Convert grayscale to 3 channels: (H, W) -> (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Change axis order: (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # 1. Convert image to a float tensor
        img = torch.tensor(img, dtype=torch.float32)

        # 2. Scale pixel values to [0, 1]
        img = img / 255.0

        target = int(self.dataset["label"].iloc[idx])
        
        name = self.dataset["name"].iloc[idx]

        return {
            "xray": img,
            "name": name,
            "target": torch.tensor(target, dtype=torch.long),
        }