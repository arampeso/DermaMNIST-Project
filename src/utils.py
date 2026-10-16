import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
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
        "auc": "ROC-AUC",
        "ap": "Average Precision",
    }
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (key, name) in zip(axes.flatten(), metrics.items()):
        ax.plot(epochs, history[f"train_{key}"], label="Training")
        ax.plot(epochs, history[f"val_{key}"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()

    fig.suptitle(f"Metrics over Epochs (Split {split})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"metrics_split_{split}.png"))
    plt.close(fig)


def compute_metrics(labels, probs):
    """
    Computes balanced accuracy, ROC-AUC and average precision.
    labels: (N,) int array, probs: (N, C) softmax probabilities.
    """
    preds = probs.argmax(axis=1)
    bal_acc = balanced_accuracy_score(labels, preds)

    preds = probs.argmax(axis=1)
    bal_acc = balanced_accuracy_score(labels, preds)

    onehot = np.eye(probs.shape[1])[labels]
    auc = roc_auc_score(labels, probs, multi_class="ovr")
    ap = average_precision_score(onehot, probs, average="macro")

    return bal_acc, auc, ap