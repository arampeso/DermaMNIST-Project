import os
import numpy as np
import torch
from tqdm import tqdm
from src.args import get_args
from src.utils import compute_metrics


def validate_model(model, val_loader, criterion, device):
    """Evaluate the model on the validation set and calculate its metrics.

    Return: Validation loss, balanced accuracy, macro F1, ROC-AUC, and average precision.
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
        epochs,
        split,
        device,
        out_dir,
):
    """Train and validate the model each epoch, saving the checkpoint with the best macro F1.

    Return: A dictionary containing the best validation metrics, epoch, split, and loss history.
    """
    os.makedirs(out_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
    }
    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_epoch = 0
    latest_val_loss = None
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer = optimizer,
        step_size = 5,
        gamma     = 0.1,
    )
    epoch_progress = tqdm(
        range(epochs),
        desc=f"Split {split} epochs",
        leave=False,
        position=1
    )

    for epoch in epoch_progress:

        model.train()
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            inputs = batch["xray"].to(device)
            labels = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            epoch_progress.set_postfix(
                train_loss=f"{running_loss / (batch_index + 1):.4f}",
                val_loss=(
                    f"{latest_val_loss:.4f}"
                    if latest_val_loss is not None
                    else "pending"
                ),
            )

        train_loss = running_loss / len(train_loader)

        val_loss, _, val_f1, _, _ = validate_model(
            model, val_loader, criterion, device
        )
        latest_val_loss = val_loss
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)

        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}"
        )

        if val_f1 > best_val_f1:
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