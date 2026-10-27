# Fashion-MNIST Optuna + Plotly experiment

The companion notebook is designed as a tuning-first Fashion-MNIST CNN experiment using PyTorch, Optuna, and Plotly.

## Notebook workflow

- Validates the Fashion-MNIST CSV schema and explores the ten classes.
- Creates stratified train/validation splits from the training CSV only.
- Tunes nine CNN/training hyperparameters with Optuna TPE and Hyperband pruning.
- Optimizes validation macro-F1 and records accuracy/loss diagnostics.
- Uses file-backed Optuna JournalStorage so the study resumes without SQLAlchemy/SQLite compiled-extension dependencies.
- Retrains the selected configuration and evaluates on the official test split only once.
- Saves interactive Plotly HTML charts in `images/`; model checkpoints, trial tables, predictions, and metrics go in `artifacts/`.
- Notebook output cells are cleared; generated output paths are not printed.

## Data files

Place these files beside the notebook or under a `data/` directory:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

The repository already contains a test CSV. The training CSV is approximately 127 MB and exceeds GitHub's standard 100 MB per-file limit, so it is intentionally not committed as an ordinary Git blob. Keep it locally or use Git LFS / a dataset host if you need versioned distribution.

## Environment

Install the notebook dependencies in your virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scikit-learn plotly optuna kaleido joblib
```

Install the PyTorch build that matches your operating system and accelerator from the official selector: https://pytorch.org/get-started/locally/

## Output folders

```text
images/       # interactive Plotly HTML and optional PNG exports
artifacts/    # Optuna journal, trial exports, model checkpoints, metrics
```

Do not commit transient model checkpoints, database/journal files, or generated predictions unless they are intentionally part of a release.
