import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args import get_args
from src.dataset import myDataSet
from src.model import MyModel
from src.trainer import train_and_validate
from src.utils import get_device

device = get_device()
args = get_args()

os.makedirs(args.output_dir, exist_ok=True)

split_results = []
split_progress = tqdm(range(5), leave=True, position=0)

# --------------- 1. Train/Val Loop ---------------
for split in split_progress:

    split_progress.set_description(f"Split {split + 1}/5")

    # --------------- 1. Load specific fold data ---------------
    train_set = pd.read_csv(
        os.path.join(args.split_dir, f"split_{split}_train.csv"))
    
    val_set = pd.read_csv(
        os.path.join(args.split_dir, f"split_{split}_val.csv"))


    # --------------- 2. From dataframe to Dataset ---------------
    train_dataset = myDataSet(train_set)
    val_dataset = myDataSet(val_set)

    # --------------- 3. Create dataloaders ---------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    
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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        split=split,
        device=device,
        out_dir=args.output_dir
    )

    split_results.append(result)

# --------------- # Save results of all folds ---------------
results_df = pd.DataFrame(split_results)
results_path = os.path.join(
    args.output_dir,
    "cross_validation_results.csv"
)
summary_df = results_df[["split", "best_epoch", "best_val_loss"]]
summary_df.to_csv(results_path, index=False)

# --------------- # Print summary ---------------
mean_val_loss = summary_df["best_val_loss"].mean()
std_val_loss = summary_df["best_val_loss"].std(ddof=1)

print("\n========== Cross-Validation Results ==========")
print(summary_df.to_string(index=False))
print(f"\nMean best validation loss: {mean_val_loss:.4f}")
print(f"Standard deviation: {std_val_loss:.4f}")
print(f"Results table saved: {results_path}\n")