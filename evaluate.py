import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.args import get_args
from src.dataset import myDataSet
from src.model import MyModel
from src.utils import get_device, compute_metrics
from src.utils import save_confusion_matrix

args = get_args()
device = get_device()


def evaluate_ensemble(models, test_loader, device):
    """
    Evaluates the ensemble (mean of the softmax probabilities of all models)
    on the test set and returns the metrics together with labels and probabilities.
    """
    for model in models:
        model.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["xray"].to(device)
            labels = batch["target"].to(device)

            batch_probs = []
            for model in models:
                prob = torch.softmax(model(inputs), dim=1)
                batch_probs.append(prob)

            probs = torch.stack(batch_probs).mean(dim=0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    bal_acc, f1, auc, ap = compute_metrics(labels, probs)

    metrics = {
        "bal_acc": bal_acc,
        "f1": f1,
        "auc": auc,
        "ap": ap,
    }
    return metrics, labels, probs


def main():

    model_folder = Path("sessions/")
    model_files = sorted(model_folder.glob("*.pth"))

    # --------------- 1. Load test data ---------------
    test_set = pd.read_csv(os.path.join(args.split_dir, "test.csv"))

    # --------------- 2. From dataframe to Dataset ---------------
    test_dataset = myDataSet(test_set)

    # --------------- 3. Create dataloader ---------------
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False)

    # --------------- 4. Load all models once ---------------
    models = []
    for model_file in model_files:
        model = MyModel(backbone=args.backbone).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        models.append(model)

    # --------------- 5. Evaluate the ensemble ---------------
    results, labels, probs = evaluate_ensemble(models, test_loader, device)
    results["model"] = f"ensemble_{len(models)}"

    # --------------- 6. Save results and confusion matrix ---------------
    results_df = pd.DataFrame([results]).set_index("model")
    results_df.to_csv(model_folder / "results.csv")
    print(f"Ensemble evaluation results saved to {model_folder / 'results.csv'}")

    save_confusion_matrix(labels, probs, model_folder / "confusion_matrix.png")
    print()
    

if __name__ == "__main__":
    main()