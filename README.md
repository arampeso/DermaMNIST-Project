# DermaMNIST-Project

A PyTorch project for classifying skin lesions using the
[DermaMNIST](https://medmnist.com/) dataset. It trains an image classification
model for seven classes, evaluates the models on a separate test set, and saves
metrics and plots.

## Project structure

```text
.
├── main.py                 # Cross-validation training
├── evaluate.py             # Test evaluation of model ensembles
├── requirements.txt        # Python dependencies
├── src/
│   ├── args.py             # Command-line arguments
│   ├── dataset.py          # Image loading and augmentation
│   ├── model.py            # ResNet and CNN models
│   ├── prepare_data.py     # Dataset download and splitting
│   ├── trainer.py          # Training and validation logic
│   └── utils.py            # Metrics and plots
├── example_images/         # Example images
└── sessions/               # Saved models and outputs
```

## Requirements

- Python 3.11 and `pip`
- An internet connection for the initial data preparation. Pretrained ResNet
  weights may also be downloaded during the first training run.

## Installation

Run all commands in this section from the project directory:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow

### 1. Prepare the data

```bash
python -m src.prepare_data
```

The script downloads DermaMNIST, saves the images to `data/images/`, and creates
the CSV files in `data/splits/`. It reserves 20% of the data as an independent
test set; the remaining 80% are divided into five stratified training and
validation folds for cross-validation.

**Warning:** Each time it runs, the script deletes existing PNG files in
`data/images/` and CSV files in `data/splits/` before generating new ones.

### 2. Train the model

```bash
python main.py
```

By default, five models are trained using `resnet18`. The program validates the
model on each fold and saves the best checkpoint for that fold. At the end, it
generates the cross-validation results and loss curves. During training, the
`WeightedRandomSampler` balances class exposure by sampling with replacement
using inverse square-root class frequencies. Each training epoch retains the
original number of samples; validation and test data are unchanged. Images are
normalized with the ImageNet mean and standard deviation expected by the
pretrained ResNet backbones, after any training augmentations.

### 3. Evaluate the models on the test set

```bash
python evaluate.py
```

The evaluation script loads all `.pth` models from the output directory and
averages their predicted probabilities to form an ensemble. It then evaluates
the ensemble on `data/splits/test.csv`.

## Options

`main.py` and `evaluate.py` accept the same arguments. Examples:

```bash
python main.py -backbone resnet34 -epochs 30 -batch_size 16
python evaluate.py -output_dir sessions -backbone resnet18
```

Argument names use a **single** hyphen, as shown in the examples. Training and
evaluation must use the same backbone setting and the same `-output_dir`.

## Outputs

The following files are saved in the default output directory, `sessions/`:

- `best_model_split_0.pth` through `best_model_split_4.pth`: best model weights
  for the five folds
- `cross_validation_results.csv`: validation results for each fold
- `loss_all_splits.png`: training and validation curves
- `results.csv`: aggregated test results for the ensemble
- `class_metrics.csv`: per-class metrics
- `confusion_matrix.png`: normalized confusion matrix
#

> **Note:** This project is intended for educational and research purposes.
> Its predictions are not a medical diagnosis and must not be used as a
> substitute for an examination by a medical professional.