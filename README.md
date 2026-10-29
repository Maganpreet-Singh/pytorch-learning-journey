<!--
  Repository: PyTorch Learning Journey
  Maintainer: Maganpreet Singh
  Purpose: A verified, notebook-first map of the repository's current learning work.
-->

<div align="center">

  <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-flame.svg" width="108" alt="PyTorch logo" />

  <h1>⚡ PyTorch Learning Journey</h1>

  <h3>From tensor fundamentals to neural networks, computer vision, transfer learning, and optimization.</h3>

  <p>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" /></a>
    <a href="https://github.com/Maganpreet-Singh/pytorch-learning-journey"><img src="https://img.shields.io/badge/Status-Active%20Learning-2EA44F?style=for-the-badge&logo=github" alt="Active learning" /></a>
  </p>

  <p>
    <a href="#-quick-navigation">Navigate</a> ·
    <a href="#-repository-tour">Repository tour</a> ·
    <a href="#-learning-roadmap">Learning roadmap</a> ·
    <a href="#-fashion-mnist-lab">Fashion-MNIST lab</a> ·
    <a href="#-run-the-notebooks">Run notebooks</a> ·
    <a href="#-contributing">Contribute</a>
  </p>

  <p><em>Understand the math. Write the code. Inspect the evidence. Improve the model.</em></p>

  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=3&section=header" width="100%" alt="" />

</div>

---

## 🌌 Welcome to the lab

This repository is a hands-on record of learning **PyTorch and practical deep learning**. It brings together notebooks, experiments, visualizations, evaluation reports, model checkpoints, and optimization outputs.

The goal is bigger than making a neural network produce a prediction. It is to understand the full path from numerical data to a trained model:

- How tensors represent data and how shape, data type, and device affect computation.
- How autograd records operations and calculates gradients.
- How `torch.nn.Module` organizes layers and trainable parameters.
- How datasets and data loaders turn examples into batches.
- How training loops combine forward passes, loss functions, backpropagation, and optimizer updates.
- How to evaluate classification models beyond a single accuracy number.
- How convolutional networks learn from image structure.
- How transfer learning reuses pretrained representations.
- How hyperparameter optimization can help compare configurations.
- How experiment artifacts make model development easier to inspect and reproduce.

This is a **living learning repository**, not a claim that every topic is mastered or every experiment is production-ready. The notebooks are the main learning material. The figures and artifacts are supporting evidence from individual runs.

> [!IMPORTANT]
> This repository is for education and experimentation. The tabular breast-cancer benchmark experiment is **not a medical device** and must never be used for diagnosis, screening, treatment, or clinical decision-making.

## ✨ What is inside?

<table>
  <tr>
    <td align="center" width="25%">
      <h2>11</h2>
      <sub>Jupyter notebooks</sub>
    </td>
    <td align="center" width="25%">
      <h2>179</h2>
      <sub>Image and visualization files</sub>
    </td>
    <td align="center" width="25%">
      <h2>86</h2>
      <sub>Experiment artifacts</sub>
    </td>
    <td align="center" width="25%">
      <h2>2</h2>
      <sub>Core experiment areas</sub>
    </td>
  </tr>
</table>

<details>
<summary><strong>What do these counts mean?</strong></summary>

They reflect the tracked repository tree checked while this README was updated: 11 files ending in `.ipynb`, 179 files under `images/`, and 86 files under `artifacts/`. They describe the current committed tree, not a promise that every file is required by every notebook. The two experiment areas refers to the broad tabular-classification and Fashion-MNIST work highlighted here.

</details>

---

## 🧭 Quick navigation

Choose a path based on what you want to learn.

| I want to… | Start here |
|---|---|
| Learn tensors, shapes, indexing, and operations | [Tensor fundamentals](./tensors_in_pytorch.ipynb) |
| Understand computation graphs and gradients | [Autograd](./pytorch_autograd.ipynb) |
| Build models using PyTorch modules | [Neural-network modules](./pytorch_nn_module.ipynb) |
| Prepare datasets and mini-batches | [Dataset and DataLoader](./dataset_and_dataloader.ipynb) |
| Understand a structured training workflow | [Training pipeline](./pytorch_training_pipeline.ipynb) |
| Explore Fashion-MNIST with an ANN | [Fashion-MNIST ANN](./Fashion_MNIST_ANN_PyTorch.ipynb) |
| Explore CNN-based image classification | [Fashion-MNIST CNN](./cnn_fashion_mnist_pytorch.ipynb) |
| Study transfer learning with ResNet-18 | [Fashion-MNIST transfer learning](./Fashion_MNIST_Transfer_Learning.ipynb) |
| Run a hyperparameter-search experiment | [Fashion-MNIST Optuna](./Fashion_MNIST_Optuna.ipynb) |
| Explore another CNN + Optuna workflow | [Fashion-MNIST CNN Optuna](./fashion_mnist_cnn_optuna.ipynb) |

**Recommended route:** tensors → autograd → modules → data loading → training pipeline → ANN → CNN → transfer learning → optimization.

---

## 🗂️ Repository tour

The following map reflects the current top-level files and directories.

```text
pytorch-learning-journey/
│
├── README.md
├── .gitignore
├── .gitattributes
│
├── tensors_in_pytorch.ipynb
├── pytorch_autograd.ipynb
├── pytorch_nn_module.ipynb
├── dataset_and_dataloader.ipynb
├── pytorch_training_pipeline.ipynb
│
├── fashion_mnist.ipynb
├── Fashion_MNIST_ANN_PyTorch.ipynb
├── cnn_fashion_mnist_pytorch.ipynb
├── Fashion_MNIST_Transfer_Learning.ipynb
├── Fashion_MNIST_Optuna.ipynb
├── fashion_mnist_cnn_optuna.ipynb
│
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
├── fmnist_small.csv
│
├── images/
│   ├── plots/
│   ├── sample galleries
│   ├── training curves
│   ├── confusion matrices
│   └── interactive HTML visualizations
│
└── artifacts/
    ├── model checkpoints
    ├── metrics and classification reports
    ├── confusion matrices
    ├── prediction exports
    ├── Optuna study outputs
    └── experiment metadata
```

### 📓 Notebook directory

<details open>
<summary><strong>01 · PyTorch foundations</strong></summary>

| Notebook | Focus | Suggested level |
|---|---|---|
| [`tensors_in_pytorch.ipynb`](./tensors_in_pytorch.ipynb) | Tensor creation, dimensions, indexing, reshaping, arithmetic, and device concepts | Beginner |
| [`pytorch_autograd.ipynb`](./pytorch_autograd.ipynb) | Automatic differentiation, computation graphs, and gradients | Beginner → Intermediate |
| [`pytorch_nn_module.ipynb`](./pytorch_nn_module.ipynb) | Layers, parameters, model structure, and the `nn.Module` API | Beginner → Intermediate |
| [`dataset_and_dataloader.ipynb`](./dataset_and_dataloader.ipynb) | Dataset interfaces, batching, shuffling, and data delivery | Beginner → Intermediate |
| [`pytorch_training_pipeline.ipynb`](./pytorch_training_pipeline.ipynb) | A structured tabular classification workflow and evaluation | Intermediate |

</details>

<details open>
<summary><strong>02 · Fashion-MNIST and computer vision</strong></summary>

| Notebook | Focus | Suggested level |
|---|---|---|
| [`fashion_mnist.ipynb`](./fashion_mnist.ipynb) | Introductory Fashion-MNIST exploration and classification | Beginner → Intermediate |
| [`Fashion_MNIST_ANN_PyTorch.ipynb`](./Fashion_MNIST_ANN_PyTorch.ipynb) | ANN-based image classification and detailed evaluation | Intermediate |
| [`cnn_fashion_mnist_pytorch.ipynb`](./cnn_fashion_mnist_pytorch.ipynb) | Convolutional neural network workflow | Intermediate |
| [`Fashion_MNIST_Transfer_Learning.ipynb`](./Fashion_MNIST_Transfer_Learning.ipynb) | Transfer learning with a ResNet-18-based workflow | Intermediate → Advanced |
| [`Fashion_MNIST_Optuna.ipynb`](./Fashion_MNIST_Optuna.ipynb) | Hyperparameter optimization with Optuna | Intermediate |
| [`fashion_mnist_cnn_optuna.ipynb`](./fashion_mnist_cnn_optuna.ipynb) | CNN experiment combined with hyperparameter optimization | Intermediate → Advanced |

</details>

### 📁 Supporting directories

| Directory | What it contains | How to use it |
|---|---|---|
| [`images/`](./images/) | Static figures, SVGs, and interactive HTML visualizations | Browse experiment outputs and visual analysis |
| [`images/plots/`](./images/plots/) | Interactive Plotly-style experiment plots | Open individual HTML files in a browser |
| [`artifacts/`](./artifacts/) | Saved model weights, metrics, reports, CSV/JSON outputs, and study files | Inspect results or load a compatible checkpoint |

### 🧾 Dataset files

| File | Role |
|---|---|
| [`fashion-mnist_train.csv`](./fashion-mnist_train.csv) | Fashion-MNIST training CSV asset tracked in this repository |
| [`fashion-mnist_test.csv`](./fashion-mnist_test.csv) | Fashion-MNIST test CSV asset |
| [`fmnist_small.csv`](./fmnist_small.csv) | Smaller Fashion-MNIST-related CSV asset used for experimentation |

> [!NOTE]
> Always inspect the loading and preprocessing cells in the notebook you plan to run. CSV schemas, label columns, and expected working directories can differ between experiments.

---

## 🛤️ Learning roadmap

The route below is dependency-aware: first understand the data structure, then differentiation, then model building, then training and applied experiments.

### Phase 1 — Tensor fluency

**Notebook:** [`tensors_in_pytorch.ipynb`](./tensors_in_pytorch.ipynb)

A tensor can represent a scalar, vector, matrix, image, batch, or higher-dimensional structure. Most deep-learning debugging becomes easier once tensor shapes and operations feel natural.

Topics to master:

- Creating tensors from values, ranges, constants, and random values.
- Scalar, vector, matrix, and higher-rank tensors.
- Shape, rank, number of elements, dtype, and device.
- Indexing and slicing.
- Reshaping, flattening, stacking, and concatenation.
- Element-wise arithmetic versus matrix multiplication.
- Reduction operations such as `sum`, `mean`, `min`, and `max`.
- Broadcasting and dimension compatibility.
- CPU and accelerator placement.
- NumPy interoperability.
- Random seeds and reproducibility.

**Mini challenge:** build a tensor of shape `(4, 3, 2)`. Predict the output shape of several indexing and reshape operations before running them. Add assertions to verify your predictions.

### Phase 2 — Autograd and the chain rule

**Notebook:** [`pytorch_autograd.ipynb`](./pytorch_autograd.ipynb)

Neural networks learn by adjusting parameters using gradients. Autograd records the operations needed to calculate derivatives.

Core ideas:

- Derivatives and the chain rule.
- `requires_grad=True`.
- Computation graphs.
- Calling `backward()`.
- Gradient accumulation.
- Leaf and non-leaf tensors.
- Detaching tensors.
- Gradient tracking contexts.
- Debugging missing gradients.
- Understanding why gradients are cleared during ordinary training iterations.

Try this small example:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

y.backward()

print("Function value:", y.item())
print("Gradient:", x.grad.item())
```

For (y=x^2+2x+1), the derivative is (2x+2). At (x=3), the expected gradient is (8).

**Mini challenge:** derive the gradient manually, calculate it with autograd, and compare the results for several input values.

### Phase 3 — Build models with `nn.Module`

**Notebook:** [`pytorch_nn_module.ipynb`](./pytorch_nn_module.ipynb)

A PyTorch model is more than a list of layers. It is a structured module that owns parameters and defines a forward computation.

Study:

- `nn.Module` and model initialization.
- Linear layers, weights, and biases.
- Activation functions.
- The `forward()` method.
- Sequential model composition.
- Parameter inspection and counting.
- Training mode and evaluation mode.
- How output dimensions relate to the task.
- How the model definition connects to the loss function.

A compact illustrative model:

```python
import torch
from torch import nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

model = SimpleClassifier(
    input_features=10,
    hidden_features=32,
    classes=2,
)

print(model)
```

The dimensions are illustrative. Match them to the data and task in a real experiment.

**Mini challenge:** pass a batch through the model and write down the shape after each layer. Explain why the final output has its particular number of columns.

### Phase 4 — Data pipelines

**Notebook:** [`dataset_and_dataloader.ipynb`](./dataset_and_dataloader.ipynb)

The `Dataset` and `DataLoader` abstractions separate sample access from batch delivery.

Understand:

- `Dataset`, `__len__()`, and `__getitem__()`.
- `DataLoader` and mini-batches.
- Shuffling and deterministic behavior.
- Batch size and incomplete final batches.
- Feature and target shapes.
- Tensor conversion.
- Train, validation, and test splitting.
- Data leakage and preprocessing discipline.
- Inspecting a batch before training.

Illustrative batch example:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

features = torch.randn(100, 4)
labels = torch.randint(0, 2, (100,))

dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_features, batch_labels in loader:
    print("Features:", batch_features.shape)
    print("Labels:", batch_labels.shape)
    break
```

**Mini challenge:** implement a custom dataset, inspect one sample, then inspect a batch. Identify which dimension is the batch dimension.

### Phase 5 — Training and evaluation

**Notebook:** [`pytorch_training_pipeline.ipynb`](./pytorch_training_pipeline.ipynb)

This notebook connects the building blocks into a tabular classification workflow. Its documented experiment uses a public breast-cancer benchmark dataset and compares baseline and neural-network approaches.

A complete workflow should make the following steps explicit:

1. Set up the environment and random-state controls.
2. Load and validate the data.
3. Inspect target distribution and feature quality.
4. Split training, validation, and test data.
5. Fit preprocessing using training data only.
6. Construct tensors and data loaders.
7. Define the model, loss, and optimizer.
8. Train and monitor validation behavior.
9. Select checkpoints and thresholds using validation data.
10. Evaluate once on held-out test data.
11. Export metrics, figures, configuration, and model artifacts.
12. Explain what the evaluation does and does not prove.

**Important:** the breast-cancer dataset is a public benchmark for education. The workflow is not clinically validated. Never use its predictions for personal health decisions.

### Phase 6 — ANN image classification

**Notebook:** [`Fashion_MNIST_ANN_PyTorch.ipynb`](./Fashion_MNIST_ANN_PyTorch.ipynb)

Fashion-MNIST is a useful first image-classification benchmark. It contains grayscale clothing images across ten categories.

Topics to investigate:

- Image loading and label mapping.
- Pixel values and normalization.
- Flattening images for a fully connected network.
- Hidden layers and nonlinear activations.
- Multi-class loss functions.
- Training and validation behavior.
- Confusion matrices and per-class metrics.
- Prediction confidence and misclassified examples.
- Model checkpointing and exported reports.

**Mini challenge:** inspect the five most confusing categories in the confusion matrix. Explain why visually similar categories may be difficult to distinguish.

### Phase 7 — Convolutional neural networks

**Notebooks:** [`cnn_fashion_mnist_pytorch.ipynb`](./cnn_fashion_mnist_pytorch.ipynb) · [`fashion_mnist_cnn_optuna.ipynb`](./fashion_mnist_cnn_optuna.ipynb)

CNNs process images while preserving local spatial structure. Study:

- Convolution filters and feature maps.
- Channels, kernels, stride, and padding.
- Pooling and downsampling.
- Receptive fields.
- Flattening and classification heads.
- Parameter counts.
- Training stability and overfitting.
- Inspecting learned filters and intermediate activations.
- Comparing CNNs with fully connected baselines.

**Mini challenge:** visualize intermediate feature maps for one input image. Describe what changes as the image passes through the network.

### Phase 8 — Transfer learning

**Notebook:** [`Fashion_MNIST_Transfer_Learning.ipynb`](./Fashion_MNIST_Transfer_Learning.ipynb)

Transfer learning adapts a model trained on one dataset or task to another. The repository includes a ResNet-18 Fashion-MNIST workflow.

Questions to investigate:

- Why pretrained weights can be useful.
- Which layers are frozen and which are trainable.
- How input channels and image sizes are adapted.
- How the classifier head is replaced.
- Feature extraction versus fine-tuning.
- How to compare a pretrained model with a baseline fairly.
- How pretrained-weight licensing and provenance should be documented.

**Mini challenge:** compare frozen-backbone feature extraction with fine-tuning. Keep the validation and test protocol consistent.

### Phase 9 — Hyperparameter optimization

**Notebooks:** [`Fashion_MNIST_Optuna.ipynb`](./Fashion_MNIST_Optuna.ipynb) · [`fashion_mnist_cnn_optuna.ipynb`](./fashion_mnist_cnn_optuna.ipynb)

Optuna helps organize experiments that search over model and training configurations.

A disciplined search should:

- Define a clear objective.
- Search only a justified set of parameters.
- Keep training and validation data separate.
- Use validation metrics for trial selection.
- Save trial history and the best configuration.
- Compare the tuned configuration against a baseline.
- Reserve test data for final evaluation.
- Record the search budget and experiment conditions.

**Mini challenge:** run a small search, inspect the trial history, and compare the selected model against a baseline using the same evaluation protocol.

---

## 👕 Fashion-MNIST lab

Fashion-MNIST is the largest applied theme in this repository. The work moves from data exploration and dense networks toward CNNs, transfer learning, and hyperparameter search.

### The ten classes

| Label | Category |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Visual exploration

The repository contains generated figures and interactive visualizations under [`images/`](./images/). Examples include class distributions, sample galleries, training diagnostics, confusion matrices, confidence plots, and model-analysis visualizations.

Some useful starting points:

- [Class distribution figure](./images/01_class_distribution_matplotlib.png)
- [Sample examples by class](./images/01_examples_by_class.png)
- [Class mean images](./images/class_mean_images.png)
- [Confusion matrix](./images/confusion_matrix.png)
- [Training dashboard](./images/training_dashboard.png)
- [Misclassification gallery](./images/misclassification_gallery.png)

<details>
<summary><strong>Open an interactive visualization</strong></summary>

GitHub does not execute notebook or HTML JavaScript inside the README page. For an interactive HTML figure, open the file from the repository or download it and open it in a browser.

The current `images/` directory contains a mixture of PNG, SVG, and HTML files. The HTML files are interactive assets; the static images are directly previewable in many GitHub contexts.

</details>

### A recorded model result

The repository currently contains [`artifacts/final_metrics.json`](./artifacts/final_metrics.json), which records a Fashion-MNIST test result:

| Metric | Recorded value |
|---|---:|
| Test accuracy | **89.65%** |
| Weighted precision | 89.66% |
| Weighted recall | 89.65% |
| Weighted F1-score | 89.51% |
| Macro one-vs-rest ROC-AUC | 0.9929 |
| Test samples | 10,000 |
| Trainable parameters | 396,202 |
| Epochs completed | 12 |
| Best epoch (by validation loss) | 7 |

These values are copied from the tracked JSON artifact, not estimated. They describe that specific saved experiment and evaluation run; they are **not a universal benchmark result for every notebook or model in this repository**. Check the notebook and associated metadata before comparing it with another run.

Related result files:

- [Final metrics JSON](./artifacts/final_metrics.json)
- [Final metrics CSV](./artifacts/final_metrics.csv)
- [Experiment summary](./artifacts/experiment_summary.json)
- [Model metadata](./artifacts/model_metadata.json)
- [Classification report](./artifacts/classification_report.csv)
- [Confusion matrix](./artifacts/confusion_matrix.csv)
- [Per-class metrics](./artifacts/ann_per_class_metrics.csv)

---

## 🔬 How to read an experiment

A model score is only useful when its context is clear. When reviewing any notebook, ask the following questions.

<details>
<summary><strong>1. What exactly is the task?</strong></summary>

Identify the input, target, prediction type, label mapping, and intended use. Classification, regression, ranking, and detection require different evaluation approaches.

</details>

<details>
<summary><strong>2. What data was used?</strong></summary>

Record the dataset source, version, license, sample count, feature or image dimensions, target distribution, and data-quality checks.

</details>

<details>
<summary><strong>3. How was the data split?</strong></summary>

Find the training, validation, and test strategy. Check for duplicates, leakage, and whether preprocessing was fitted on the training partition only.

</details>

<details>
<summary><strong>4. What did the model learn?</strong></summary>

Inspect the architecture, parameter count, loss function, optimizer, learning rate, batch size, epochs, regularization, and checkpoint-selection rule.

</details>

<details>
<summary><strong>5. How was performance measured?</strong></summary>

Read the metric definitions. For classification, inspect precision, recall, F1-score, confusion matrices, per-class performance, ROC-AUC where appropriate, and confidence behavior.

</details>

<details>
<summary><strong>6. Where does the model fail?</strong></summary>

Look at misclassified examples, class-specific errors, confidence distributions, and examples where the model is confidently wrong. Error analysis often teaches more than the headline metric.

</details>

<details>
<summary><strong>7. Can the experiment be repeated?</strong></summary>

Look for saved configuration, random seeds, package versions, preprocessing details, checkpoints, and exported evaluation outputs.

</details>

### Experiment log template

Use this structure when adding a new experiment.

| Field | Record |
|---|---|
| Experiment name | A short, searchable title |
| Question | What are you trying to learn? |
| Dataset | Source, version, license, and task |
| Split | Train / validation / test strategy |
| Preprocessing | Transformations and where they are fitted |
| Model | Architecture and parameter count |
| Loss | Optimization objective |
| Optimizer | Algorithm and settings |
| Hyperparameters | Learning rate, batch size, epochs, etc. |
| Randomness | Seeds and deterministic settings |
| Compute | CPU/GPU and software versions |
| Metrics | Validation and final test results |
| Error analysis | What fails and where? |
| Limitations | What the run does not establish |
| Next experiment | One controlled change to test |

Never fill metric fields with guessed numbers. Use actual outputs from a completed run.

---

## ⚙️ Run the notebooks

The repository is notebook-first. You can work in VS Code, Jupyter Notebook, or JupyterLab.

### 1 · Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/pytorch-learning-journey.git
cd pytorch-learning-journey
```

### 2 · Create an isolated environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Windows — Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3 · Install PyTorch

PyTorch installation commands depend on your operating system, package manager, Python version, and compute platform.

Use the official selector:

**[→ Install PyTorch — Start Locally](https://pytorch.org/get-started/locally/)**

Then install the common notebook and data-science packages:

```bash
python -m pip install jupyter numpy pandas matplotlib scikit-learn
```

For notebooks that use interactive charts or hyperparameter search:

```bash
python -m pip install plotly optuna
```

Install only the additional dependencies required by the notebook you intend to run. Do not assume a CPU-only installation and a CUDA-enabled installation use the same command.

### 4 · Verify PyTorch

Run this in a notebook cell:

```python
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Tensor shape:", tuple(x.shape))
print("Tensor dtype:", x.dtype)
print("Tensor device:", x.device)
```

A CUDA result of `False` does not by itself mean PyTorch is broken. It can mean a CPU build is installed or a compatible accelerator is not available in the environment.

### 5 · Launch Jupyter

```bash
jupyter notebook
```

Or open the project folder in VS Code, select the `.venv` interpreter, open a notebook, and select the matching kernel.

### 6 · Run a notebook from top to bottom

- Start with a fresh kernel.
- Read Markdown cells before executing code.
- Run cells in order.
- Inspect data paths and expected columns.
- Check tensor shapes before model training.
- Confirm that model and input tensors use compatible devices.
- Avoid overwriting valuable artifacts unintentionally.
- Restart and run all cells before treating a notebook as reproducible.

> [!TIP]
> No GPU is needed for tensor basics or many small experiments. Start with the CPU if that is what your environment supports. Learn the workflow before optimizing the hardware.

---

## 🧪 A better notebook workflow

The most effective learning loop is:

`PREDICT → RUN → INSPECT → EXPLAIN → MODIFY → RECORD`

1. **Predict:** write down what you expect a code cell to produce.
2. **Run:** execute a small example.
3. **Inspect:** examine values, shapes, gradients, metrics, or visualizations.
4. **Explain:** describe why the result occurred in your own words.
5. **Modify:** change one input or parameter and repeat.
6. **Record:** document the lesson, the mistake, and the fix.

### Debug shapes early

```python
print("Input shape:", inputs.shape)
print("Input dtype:", inputs.dtype)
print("Input device:", inputs.device)

print("Target shape:", targets.shape)
print("Target dtype:", targets.dtype)
print("Target device:", targets.device)
```

The variable names above are examples; use the names in your own notebook.

### Keep comparisons fair

When comparing models or configurations:

- Keep data splits consistent.
- Fit preprocessing only on training data.
- Use the same validation protocol.
- Do not select hyperparameters using the test set.
- Record the model configuration and random seed.
- Compare more than one metric where appropriate.
- Explain trade-offs and failure cases.

### Keep notebooks reviewable

- Use clear Markdown headings.
- Explain why a code block exists.
- Use meaningful variable names.
- Add assertions for critical assumptions.
- Label figures and explain their takeaway.
- Avoid huge accidental outputs.
- Restart the kernel and run all cells before committing.
- Keep credentials, private data, and local secrets out of notebooks.

---

## 📦 Artifacts and reproducibility

The [`artifacts/`](./artifacts/) directory currently contains model checkpoints, CSV and JSON metrics, classification reports, prediction exports, experiment metadata, and Optuna-related files.

### Artifact families

| Artifact type | Examples | Why it matters |
|---|---|---|
| Model checkpoints | `.pt`, `.pth` | Save model parameters or serialized model state |
| Metrics | `final_metrics.json`, `final_metrics.csv` | Preserve measured evaluation results |
| Classification reports | `classification_report.csv`, per-class metric files | Show performance by class |
| Confusion matrices | `confusion_matrix.csv`, normalized variants | Reveal where predictions are confused |
| Predictions | `test_predictions.csv`, probability exports | Support prediction inspection and error analysis |
| Configuration | `model_config.json`, hyperparameter JSON files | Document experiment settings |
| Training history | `final_training_history.csv` | Track loss and metric behavior over epochs |
| Optimization outputs | Optuna CSV and database files | Preserve trial-level search information |

### Reproducibility checklist

- [ ] Record Python and package versions.
- [ ] Record dataset source, version, and license.
- [ ] Record split strategy and random seed.
- [ ] Save model architecture and configuration.
- [ ] Save preprocessing objects or exact preprocessing steps.
- [ ] Record loss function and optimizer settings.
- [ ] Record batch size, learning rate, and number of epochs.
- [ ] Save validation and test metrics separately.
- [ ] Document the checkpoint-selection rule.
- [ ] Document compute hardware and relevant settings.
- [ ] Keep a record of limitations and known failure modes.

A random seed helps control randomness, but exact bit-for-bit reproducibility can still depend on hardware, libraries, and nondeterministic operations.

### Checkpoint loading reminder

A state dictionary is not always a complete, ready-to-run model. Reconstruct the same architecture before loading its parameters, and follow the save/load convention used by the notebook that produced the checkpoint.

Never load an unknown serialized file from an untrusted source. PyTorch serialization can involve Python object deserialization; treat downloaded model files with care.

---

## 🧰 Technology stack

| Tool | Role in this repository |
|---|---|
| [Python](https://www.python.org/) | Main programming language |
| [PyTorch](https://pytorch.org/) | Tensor computation, autograd, neural networks |
| [Jupyter](https://jupyter.org/) | Interactive notebooks |
| [NumPy](https://numpy.org/) | Numerical arrays and interoperability |
| [Pandas](https://pandas.pydata.org/) | Tabular data inspection and manipulation |
| [scikit-learn](https://scikit-learn.org/) | Preprocessing, dataset utilities, and evaluation |
| [Matplotlib](https://matplotlib.org/) | Static plots |
| [Plotly](https://plotly.com/python/) | Interactive visualizations |
| [Optuna](https://optuna.org/) | Hyperparameter optimization |
| [Git](https://git-scm.com/) | Version control |
| [GitHub](https://github.com/) | Collaboration and repository hosting |

The exact dependency set can differ across notebooks. This table describes the technologies used or referenced by the repository; it is not a pinned environment specification.

---

## 🧯 Troubleshooting

| Symptom | What to check |
|---|---|
| Tensor shape mismatch | Print the shape before each operation and verify dimensions |
| Dtype error | Inspect `.dtype` and convert targets or inputs only when appropriate |
| CPU/GPU mismatch | Move inputs and model parameters to the same device |
| Missing gradients | Check `requires_grad`, graph tracking, and detach operations |
| Gradients unexpectedly grow | Check whether gradients are cleared between optimization steps |
| Notebook variable undefined | Run cells in order or restart and run all |
| Dataset path not found | Inspect the notebook's expected working directory and file names |
| Plot not displayed | Check plotting backend and notebook output |
| Plot not saved | Confirm the output directory exists and inspect the save path |
| Results differ between runs | Review random seeds, shuffling, hardware, and nondeterministic operations |
| CUDA unavailable | Verify the PyTorch build and accelerator setup using the official selector |
| Out-of-memory error | Reduce batch size, image resolution, or model size |

**Debugging rule:** reduce the failing notebook cell to the smallest example that reproduces the error. A small reproducible failure is easier to understand than a large stack trace surrounded by unrelated code.

---

## 🧠 Progress board

This is a learning checklist, not an externally verified certification. Mark a topic complete when you can explain and implement it independently.

<details>
<summary><strong>PyTorch fundamentals</strong></summary>

- [ ] Create tensors from common sources.
- [ ] Explain shape, rank, dtype, and device.
- [ ] Index and slice tensors.
- [ ] Reshape, stack, and concatenate tensors.
- [ ] Explain broadcasting.
- [ ] Distinguish element-wise operations from matrix multiplication.
- [ ] Explain autograd and computation graphs.
- [ ] Calculate gradients using `backward()`.
- [ ] Explain gradient accumulation and clearing.
- [ ] Build a model with `nn.Module`.
- [ ] Inspect parameters and count trainable parameters.
- [ ] Choose a suitable loss function.

</details>

<details>
<summary><strong>Data and training</strong></summary>

- [ ] Implement a custom `Dataset`.
- [ ] Configure a `DataLoader`.
- [ ] Inspect samples and batches.
- [ ] Split data without leakage.
- [ ] Build a complete training loop.
- [ ] Separate training, validation, and test responsibilities.
- [ ] Save and load model checkpoints.
- [ ] Track learning curves.
- [ ] Evaluate with task-appropriate metrics.
- [ ] Perform error analysis.

</details>

<details>
<summary><strong>Computer vision</strong></summary>

- [ ] Load and inspect image data.
- [ ] Understand channels, height, and width.
- [ ] Explain convolution, stride, and padding.
- [ ] Explain pooling and feature maps.
- [ ] Build and train a CNN.
- [ ] Inspect intermediate feature maps.
- [ ] Compare an ANN and CNN fairly.
- [ ] Experiment with transfer learning.
- [ ] Explain feature extraction versus fine-tuning.

</details>

<details>
<summary><strong>Experiment quality</strong></summary>

- [ ] Record configurations and package versions.
- [ ] Establish a reproducible data split.
- [ ] Choose validation metrics before tuning.
- [ ] Keep the test set untouched during model selection.
- [ ] Save metrics and prediction outputs.
- [ ] Compare a model against a baseline.
- [ ] Document limitations and failure cases.
- [ ] Make a notebook easy for another person to run.

</details>

---

## 🚀 Future directions

As the fundamentals become stronger, useful next experiments include:

| Direction | What it would add |
|---|---|
| Optimizers in depth | Compare SGD, momentum, Adam, and weight decay |
| Learning-rate scheduling | Study step, cosine, and adaptive schedules |
| Regularization | Explore dropout, augmentation, and generalization |
| CNN architectures | Compare compact CNN designs and residual connections |
| Explainability | Explore Grad-CAM and feature attribution carefully |
| Model calibration | Study confidence quality, not only classification accuracy |
| Transfer learning | Compare feature extraction and fine-tuning |
| Experiment tracking | Record configurations, metrics, and run history |
| Inference packaging | Separate training, loading, preprocessing, and prediction |
| Deployment | Build a small, documented inference application |

These are proposed future directions. They should not be read as claims that each item already has a completed notebook.

---

## 📚 Learning resources

Official documentation is the best source for current API behavior and installation instructions.

### PyTorch core

- [PyTorch website](https://pytorch.org/)
- [Install PyTorch — Start Locally](https://pytorch.org/get-started/locally/)
- [Official PyTorch tutorials](https://pytorch.org/tutorials/)
- [PyTorch API documentation](https://pytorch.org/docs/stable/index.html)
- [Tensor documentation](https://pytorch.org/docs/stable/tensors.html)
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [Neural-network package](https://pytorch.org/docs/stable/nn.html)
- [Dataset and DataLoader](https://pytorch.org/docs/stable/data.html)
- [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

### Supporting tools

- [Python documentation](https://docs.python.org/3/)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Plotly Python documentation](https://plotly.com/python/)
- [Optuna documentation](https://optuna.readthedocs.io/)

### Practical companion

- [Learn PyTorch for Deep Learning — GitHub](https://github.com/mrdbourke/pytorch-deep-learning)

Use resources to understand ideas, then implement them yourself. The real test is being able to explain the code and adapt it to a new problem.

---

## 🤝 Contributing

This repository is primarily a personal learning journal, but constructive educational feedback is welcome.

Useful contributions include:

- Fixing an incorrect explanation or code example.
- Improving notebook readability.
- Adding a focused exercise.
- Identifying a reproducibility issue.
- Suggesting a useful test case.
- Sharing a reliable reference for a PyTorch concept.

For substantial changes, open an issue first to discuss the scope.

### Contribution workflow

1. Fork the repository.
2. Create a focused branch.
3. Make one coherent change.
4. Run the affected notebook or validation checks.
5. Explain the change and any relevant limitations.
6. Open a pull request.

Please respect third-party licenses for code, datasets, pretrained weights, and visual assets. Do not include credentials, private datasets, or unrelated generated files.

---

## 📌 Project principles

1. **Understand before scaling.** A small model you understand beats a large model you copied.
2. **Measure before claiming.** Report real outputs and explain the evaluation protocol.
3. **Keep the test set honest.** Use validation data for model selection.
4. **Prefer clarity over cleverness.** Readable notebooks are easier to debug and review.
5. **Document mistakes.** Debugging is part of engineering and learning.
6. **Respect data.** Provenance, licensing, privacy, and limitations matter.
7. **Build incrementally.** Each experiment should strengthen the next.

---

## 👨‍💻 About the maintainer

<div align="center">
  <h3>Maganpreet Singh</h3>
  <p>Learning deep learning through implementation, experimentation, and careful evaluation.</p>
  <a href="https://github.com/Maganpreet-Singh">
    <img src="https://img.shields.io/badge/GitHub-Maganpreet--Singh-181717?style=for-the-badge&logo=github" alt="GitHub profile" />
  </a>
  <a href="https://github.com/Maganpreet-Singh/pytorch-learning-journey">
    <img src="https://img.shields.io/badge/Explore-the%20repository-2EA44F?style=for-the-badge&logo=github" alt="Explore repository" />
  </a>
</div>

This project is part of an ongoing journey in machine learning and deep learning: turning theory into working code, strengthening fundamentals, and building a public record of experiments and progress.

---

## 📄 License and attribution

No repository license file is currently present in the tracked top-level tree. Until a license is added, do not assume the repository's original content is licensed for unrestricted reuse.

Third-party code, datasets, pretrained weights, and visual assets may have separate terms. Review the applicable terms before reuse or redistribution. A dedicated repository license should be added once the intended permissions are chosen.

---

<div align="center">

  <h2>🔥 Learn the tensor. Follow the gradient. Build the model. Test the evidence.</h2>

  <p><em>One notebook. One experiment. One concept understood at a time.</em></p>

  <p>
    <a href="#-quick-navigation">Back to top ↑</a>
    ·
    <a href="https://pytorch.org/tutorials/">PyTorch tutorials</a>
    ·
    <a href="https://github.com/Maganpreet-Singh/pytorch-learning-journey">GitHub repository</a>
  </p>

</div>
