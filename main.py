import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from src.prepare_data import SEED

from src.args import get_args
from src.dataset import myDataSet
from src.model import MyModel
from src.trainer import train_and_validate
from src.utils import get_device, plot_losses, summarize_results

device = get_device()
args = get_args()

os.makedirs(args.output_dir, exist_ok=True)

split_results = []
split_progress = tqdm(range(5), leave=True, position=0)

# --------------- Train/Val Loop ---------------
for split in split_progress:

    split_progress.set_description(f"Split {split + 1}/5")

    # --------------- 1. Load specific fold data ---------------
    train_set = pd.read_csv(
        os.path.join(args.split_dir, f"split_{split}_train.csv"))
    
    val_set = pd.read_csv(
        os.path.join(args.split_dir, f"split_{split}_val.csv"))


    # --------------- 2. From dataframe to Dataset ---------------
    train_dataset = myDataSet(train_set, should_augment=True)
    val_dataset = myDataSet(val_set, should_augment=False)

    # --------------- 3. Create dataloaders ---------------
    labels = train_set["label"].to_numpy()
    _, label_indices, class_counts = np.unique(
        labels,
        return_inverse=True,
        return_counts=True,
    )
    sample_weights = torch.as_tensor(
        1.0 / (class_counts[label_indices]),
        dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED + split))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False)


    # --------------- 4. Create model, loss function, and optimizer ---------------

    model = MyModel(backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # --------------- 5. Train and validate model ---------------
    result = train_and_validate(
        model       = model,
        train_loader= train_loader,
        val_loader  = val_loader,
        criterion   = criterion,
        optimizer   = optimizer,
        epochs      = args.epochs,
        split       = split,
        device      = device,
        out_dir     = args.output_dir
    )

    split_results.append(result)

# --------------- # Plot loss for all folds ---------------
plot_losses(split_results, args.output_dir)

# --------------- # Summarize results of all folds ---------------
summarize_results(split_results, args.output_dir)