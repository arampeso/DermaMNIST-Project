import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.args import get_args
from src.utils import compute_metrics, plot_metrics

args = get_args()

def validate_model(model, val_loader, criterion, device):
    """ 
    Validates the model on the validation set and returns:
    Loss, balanced accuracy, ROC-AUC and average precision.
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
    bal_acc, auc, ap = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))

    return val_loss, bal_acc, auc, ap


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
    and saves the plots (one for training and one for validation).
    Plots:
    - Loss over epoch
    - Balanced Accuracy over epoch
    - ROC-AUC over epoch
    - Average Precision over epoch
    """
    os.makedirs(out_dir, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_bal_acc": [], "val_bal_acc": [],
        "train_auc": [], "val_auc": [],
        "train_ap": [], "val_ap": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_labels, all_probs = [], []

        for batch in train_loader:
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

        train_loss = running_loss / len(train_loader)
        train_bal_acc, train_auc, train_ap = compute_metrics(
            np.concatenate(all_labels), np.concatenate(all_probs)
        )

        val_loss, val_bal_acc, val_auc, val_ap = validate_model(model, val_loader, criterion, device)

        for key, value in zip(
                history.keys(),
                [train_loss, val_loss, train_bal_acc, val_bal_acc,
                 train_auc, val_auc, train_ap, val_ap]
        ):
            history[key].append(value)

        print("\n")
        print(f"------- Epoch {epoch + 1}/{epochs} -------")
        print("Training   - Loss: {:.4f} | BalAcc: {:.4f} | ROC-AUC: {:.4f} | AP: {:.4f}".format(
            train_loss, train_bal_acc, train_auc, train_ap))
        print("Validation - Loss: {:.4f} | BalAcc: {:.4f} | ROC-AUC: {:.4f} | AP: {:.4f}".format(
            val_loss, val_bal_acc, val_auc, val_ap))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(out_dir, f"best_model_split_{split}.pth"))

    plot_metrics(history, split, out_dir)

    return {
        "split": split,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history
    }