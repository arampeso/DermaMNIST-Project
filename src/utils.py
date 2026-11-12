import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_device():
    """Select the available compute device, preferring MPS and then CUDA.

    Return: The selected PyTorch device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        
    else:
        device = torch.device("cpu")

    print(f"Device used: {device}")
    return device

def plot_losses(split_results, out_dir):
    """Plot and save training loss, validation loss, and macro-F1 curves for each split.

    Return: None.
    """
    nrows = 5
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows), squeeze=False)
    column_titles = ("Training loss", "Validation loss", "Validation macro F1")

    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row, result in enumerate(sorted(split_results, key=lambda item: item["split"])):
        history = result["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[row, 0].plot(epochs, history["train_loss"])
        axes[row, 1].plot(epochs, history["val_loss"])
        axes[row, 2].plot(epochs, history["val_macro_f1"])

        for column, ylabel in enumerate(("Loss", "Loss", "Macro F1")):
            axes[row, column].set_ylabel(f"Split {result['split']}\n{ylabel}")
            axes[row, column].set_xlabel("Epoch")

        axes[row, 2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_all_splits.png"))
    plt.close(fig)


def summarize_results(split_results, out_dir):
    """Save the per-split cross-validation results and print aggregate metrics.

    Return: None.
    """
    summary_df = pd.DataFrame(split_results)[
        ["split", "best_epoch", "best_val_loss", "best_val_f1"]
    ]
    results_path = os.path.join(out_dir, "cross_validation_results.csv")
    summary_df.to_csv(results_path, index=False)

    mean_val_loss = summary_df["best_val_loss"].mean()
    std_val_loss = summary_df["best_val_loss"].std(ddof=1)
    mean_val_f1 = summary_df["best_val_f1"].mean()
    std_val_f1 = summary_df["best_val_f1"].std(ddof=1)

    print("\n========== Cross-Validation Results ==========")
    print(summary_df.to_string(index=False))
    print(f"\nMean best validation loss: {mean_val_loss:.4f}")
    print(f"Standard deviation: {std_val_loss:.4f}")
    print(f"\nMean validation macro F1: {mean_val_f1:.4f}")
    print(f"Standard deviation: {std_val_f1:.4f}")
    print(f"\nResults table saved: {results_path}\n")


def compute_metrics(labels, probs):
    """Calculate balanced accuracy, macro F1, ROC-AUC, and average precision.

    Return: A tuple containing balanced accuracy, macro F1, ROC-AUC, and average precision.
    """
    preds = probs.argmax(axis=1)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(
        labels,
        preds,
        labels=np.arange(probs.shape[1]),
        average="macro",
        zero_division=0,
    )

    onehot = np.eye(probs.shape[1])[labels]
    auc = roc_auc_score(labels, probs, multi_class="ovr")
    ap = average_precision_score(onehot, probs, average="macro")

    return bal_acc, f1, auc, ap


def save_confusion_matrix(labels, probs, class_names, path):
    """Create and save a row-normalized confusion matrix plot.

    Return: None.
    """
    preds = probs.argmax(axis=1)
    class_ids = np.arange(len(class_names))
    cm = confusion_matrix(
        labels,
        preds,
        labels=class_ids,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=".1%",
    )
    ax.set_title("Ensemble confusion matrix (row-normalized)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {path}")