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

def get_device():
    """ Returns the appropriate device (CPU, CUDA, or MPS) based on availability. """
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        
    else:
        device = torch.device("cpu")

    print(f"Device used: {device}")
    return device

def plot_metrics(history, split, out_dir):
    """ 
    Saves one combined plot per split (one subplot per metric,
    training vs. validation over epochs).
    """
    metrics = {
        "loss": "Loss",
        "bal_acc": "Balanced Accuracy",
        "f1": "Macro F1",
        "auc": "ROC-AUC",
        "ap": "Average Precision",
    }
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, (key, name) in zip(axes.flatten(), metrics.items()):
        ax.plot(epochs, history[f"train_{key}"], label="Training")
        ax.plot(epochs, history[f"val_{key}"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()

    axes.flatten()[-1].set_visible(False)
    fig.suptitle(f"Metrics over Epochs (Split {split})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"metrics_split_{split}.png"))
    plt.close(fig)


def compute_metrics(labels, probs):
    """
    Computes balanced accuracy, macro F1, ROC-AUC and average precision.
    labels: (N,) int array, probs: (N, C) softmax probabilities.
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


def save_confusion_matrix(labels, probs, path):
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Ensemble confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {path}")