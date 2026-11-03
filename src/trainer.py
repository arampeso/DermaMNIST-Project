import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.args import get_args
from src.utils import compute_metrics

args = get_args()

def validate_model(model, val_loader, criterion, device):
    """ 
    Validates the model on the validation set and returns:
    Loss, balanced accuracy, macro F1, ROC-AUC and average precision.
    """
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["xray"].to(device)
            labels = batch["target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

    val_loss = running_loss / len(val_loader)
    bal_acc, f1, auc, ap = compute_metrics(
        np.concatenate(all_labels), np.concatenate(all_probs)
    )

    return val_loss, bal_acc, f1, auc, ap


def train_and_validate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=args.epochs,
        split=0,
        device="cpu",
        out_dir="sessions"
):
    """ 
    Trains the model, validates after every epoch, saves the best model
    and returns its loss history.
    """
    os.makedirs(out_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
    }
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0

    epoch_progress = tqdm(
        range(epochs),
        desc=f"Split {split} epochs",
        leave=False,
        position=1
    )

    for epoch in epoch_progress:

        model.train()
        running_loss = 0.0
        all_labels, all_probs = [], []

        for batch_index, batch in enumerate(train_loader):
            inputs = batch["xray"].to(device)
            labels = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
            epoch_progress.set_postfix(loss=f"{running_loss / (batch_index + 1):.4f}")

        train_loss = running_loss / len(train_loader)

        val_loss, _, val_f1, _, _ = validate_model(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_progress.set_postfix(
            loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(out_dir, f"best_model_split_{split}.pth"))

    return {
        "split": split,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "history": history,
    }