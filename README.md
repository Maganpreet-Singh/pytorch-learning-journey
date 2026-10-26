<!--
  PyTorch Learning Journey
  Maintainer: Maganpreet Singh
  This README documents the repository as it currently exists and separates
  published work from future learning goals.
-->

<div align="center">

  <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-flame.svg" width="112" alt="PyTorch logo" />

  <h1>PyTorch Learning Journey</h1>

  <p><strong>From tensors and gradients to reproducible training pipelines, computer vision experiments, and hyperparameter optimization.</strong></p>

  <p>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Learn%20by%20Building-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Format-Jupyter%20Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" /></a>
    <a href="https://github.com/Maganpreet-Singh/pytorch-learning-journey"><img src="https://img.shields.io/badge/Repository-Open%20Source-181717?style=for-the-badge&logo=github" alt="GitHub repository" /></a>
  </p>

  <p>
    <a href="#-start-here">Start here</a> ·
    <a href="#-repository-map">Repository map</a> ·
    <a href="#-learning-path">Learning path</a> ·
    <a href="#-experiments">Experiments</a> ·
    <a href="#-setup">Setup</a> ·
    <a href="#-progress-board">Progress board</a>
  </p>

  <br />

  <img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2&section=header" width="100%" alt="" />

</div>

## 👋 Welcome

This repository is my practical record of learning **PyTorch and deep learning**. It is built around notebooks, experiments, debugging notes, visualizations, and small end-to-end implementations.

The purpose is not to collect code that merely runs. The purpose is to understand the machinery behind a learning system:

- How tensors represent numerical data and how their shape, type, and device affect computation.
- How automatic differentiation builds a computation graph and calculates gradients.
- How neural-network modules organize learnable parameters.
- How datasets become batches that a model can consume.
- How training loops connect forward passes, loss functions, backpropagation, and optimizer updates.
- How to evaluate models with appropriate metrics rather than relying on a single score.
- How experiments can be made more reproducible, interpretable, and easier to compare.
- How hyperparameter search can help investigate model configurations.

The repository is a **living learning journal**. It will change as I learn, test ideas, revisit earlier notebooks, and build more complete projects. Published files are linked directly below. Future directions are explicitly marked as planned.

> [!IMPORTANT]
> This is an educational repository. Notebooks are learning artifacts, not production-ready software or professional advice. In particular, the breast-cancer classification experiment is a public-dataset exercise and **must not be used for diagnosis, screening, treatment, or clinical decisions**.

---

## 🧭 Start here

New to this repository? Follow this route:

1. Read the tensor notebook to understand PyTorch's core data structure.
2. Study autograd to see how gradients are computed.
3. Learn how to define reusable models with `torch.nn.Module`.
4. Understand how `Dataset` and `DataLoader` supply data to a model.
5. Work through the training-pipeline notebook to connect the pieces.
6. Explore Fashion-MNIST as an image-classification task.
7. Inspect the Optuna experiment to see how hyperparameter optimization changes the workflow.

**Quick links**

| Your goal | Open this notebook |
|---|---|
| Understand tensors, dimensions, and tensor operations | [`tensors_in_pytorch.ipynb`](./tensors_in_pytorch.ipynb) |
| Understand automatic differentiation | [`pytorch_autograd.ipynb`](./pytorch_autograd.ipynb) |
| Build models with PyTorch's module system | [`pytorch_nn_module.ipynb`](./pytorch_nn_module.ipynb) |
| Build a custom dataset and mini-batch pipeline | [`dataset_and_dataloader.ipynb`](./dataset_and_dataloader.ipynb) |
| Train and evaluate tabular classifiers | [`pytorch_training_pipeline.ipynb`](./pytorch_training_pipeline.ipynb) |
| Explore image classification with Fashion-MNIST | [`fashion_mnist.ipynb`](./fashion_mnist.ipynb) |
| Explore automated hyperparameter search | [`Fashion_MNIST_Optuna.ipynb`](./Fashion_MNIST_Optuna.ipynb) |

---

## 📚 Repository map

The following is the current top-level layout, based on the files in the repository. The notebooks are the primary learning material; the data and artifact folders support experimentation.

```text
pytorch-learning-journey/
│
├── README.md
│
├── tensors_in_pytorch.ipynb
├── pytorch_autograd.ipynb
├── pytorch_nn_module.ipynb
├── dataset_and_dataloader.ipynb
├── pytorch_training_pipeline.ipynb
│
├── fashion_mnist.ipynb
├── Fashion_MNIST_Optuna.ipynb
│
├── fashion-mnist_test.csv
├── fmnist_small.csv
│
├── images/
└── artifacts/
```

### What each area is for

| Path | Role |
|---|---|
| `tensors_in_pytorch.ipynb` | Tensor creation, inspection, manipulation, arithmetic, and related fundamentals |
| `pytorch_autograd.ipynb` | Automatic differentiation, computation graphs, and gradient concepts |
| `pytorch_nn_module.ipynb` | Model organization through PyTorch's neural-network module API |
| `dataset_and_dataloader.ipynb` | Custom datasets, batching, shuffling, and data delivery |
| `pytorch_training_pipeline.ipynb` | A structured binary-classification workflow, model comparison, evaluation, and exported outputs |
| `fashion_mnist.ipynb` | Fashion-MNIST image-classification learning experiment |
| `Fashion_MNIST_Optuna.ipynb` | Fashion-MNIST experiment involving Optuna-based hyperparameter search |
| `fashion-mnist_test.csv`, `fmnist_small.csv` | Fashion-MNIST-related CSV data files |
| `images/` | Visual assets and notebook-generated figures, where present |
| `artifacts/` | Experiment outputs or model-related artifacts, where present |

> [!NOTE]
> This map describes the checked-in top-level paths. Notebook cells may create additional outputs at runtime. A suggested future organization is not the same as a directory that already exists.

---

## 🗺️ Learning path

The topics below are arranged by dependency: understand the data representation first, then gradients, then models, then the data and training workflow, and finally experiments. The sequence is a learning guide, not a claim that every concept has a dedicated notebook or has been mastered.

### 1. Tensor fundamentals

**Notebook:** [`tensors_in_pytorch.ipynb`](./tensors_in_pytorch.ipynb)

Tensors are the basic data structure used throughout PyTorch. A tensor can represent a scalar, a vector, a matrix, an image batch, or a higher-dimensional data structure.

Concepts to understand:

- Creating tensors from Python values, ranges, constants, and random values.
- Scalar, vector, matrix, and higher-rank representations.
- Shape, rank, number of elements, data type, and device.
- Indexing, slicing, reshaping, flattening, stacking, and concatenation.
- Element-wise arithmetic and comparison operations.
- Reductions such as sum, mean, minimum, and maximum.
- Matrix multiplication and the distinction between element-wise multiplication and linear algebra operations.
- Broadcasting rules and shape compatibility.
- CPU and accelerator placement.
- NumPy interoperability.
- In-place operations and memory-related considerations.
- Reproducibility and random seeds.

**Questions to ask while learning**

1. What is the shape before and after this operation?
2. Does the operation preserve the number of elements?
3. Is this element-wise arithmetic or matrix multiplication?
4. Which dimensions are compatible for broadcasting?
5. Where is the tensor stored, and what is its data type?
6. Does this operation create a new tensor or modify an existing one?

**Practice challenge:** create a three-dimensional tensor, predict the result of a sequence of indexing and reshape operations, and verify each prediction with assertions.

### 2. Automatic differentiation and autograd

**Notebook:** [`pytorch_autograd.ipynb`](./pytorch_autograd.ipynb)

Neural networks learn by adjusting parameters in response to gradients. PyTorch's autograd system records operations and computes derivatives when requested.

Key ideas:

- Derivatives and the chain rule.
- Tensors created with `requires_grad=True`.
- Computation graphs and recorded operations.
- Calling `backward()` and inspecting gradients.
- Gradient accumulation and why optimizers typically clear gradients.
- Leaf tensors and non-leaf tensors.
- Detaching values from the graph.
- Gradient tracking contexts.
- The difference between disabling gradient tracking and changing a tensor's values.
- Debugging missing or unexpected gradients.

A tiny example:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

y.backward()

print("y:", y.item())
print("dy/dx:", x.grad.item())
```

For (y=x^2+2x+1), the derivative is (2x+2). At (x=3), the gradient is (8).

**Practice challenge:** calculate a derivative manually, calculate it with autograd, then compare the two results. Change the function and repeat.

### 3. Neural networks with `torch.nn.Module`

**Notebook:** [`pytorch_nn_module.ipynb`](./pytorch_nn_module.ipynb)

A model needs a clear place to define layers, parameters, and the forward computation. PyTorch's module system provides that structure.

Concepts to study:

- What a model represents.
- Parameters, weights, and biases.
- Linear layers and input/output dimensions.
- Activation functions and nonlinear transformations.
- The `nn.Module` base class.
- The `__init__` method and layer registration.
- The `forward()` method.
- Model inspection and parameter counting.
- Training and evaluation modes.
- The difference between defining a model and training it.

Example skeleton:

```python
import torch
from torch import nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

model = SimpleClassifier(
    input_features=10,
    hidden_features=32,
    classes=2,
)

print(model)
```

The dimensions above are illustrative. A real model's layer sizes must match its data and task.

**Practice challenge:** construct a small model, pass a batch of inputs through it, inspect every intermediate shape, and explain how the final output dimensions relate to the task.

### 4. Datasets and DataLoaders

**Notebook:** [`dataset_and_dataloader.ipynb`](./dataset_and_dataloader.ipynb)

A model does not train directly from an arbitrary folder of files. Data must be represented, validated, and supplied in a consistent form.

The core distinction:

- `Dataset` describes how to access an individual sample and its target.
- `DataLoader` organizes samples into batches and controls how those batches are delivered.

Topics:

- Custom `Dataset` implementations.
- The `__len__()` and `__getitem__()` methods.
- Converting arrays to tensors.
- Feature and label shapes.
- Train/test splitting.
- Data preprocessing and leakage prevention.
- Mini-batches and batch dimensions.
- Shuffling and deterministic behavior.
- `batch_size`, `drop_last`, and worker configuration.
- Inspecting batches before training.
- Connecting a data pipeline to a neural network.

Example:

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

features = torch.randn(100, 4)
labels = torch.randint(0, 2, (100,))

dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_features, batch_labels in loader:
    print(batch_features.shape, batch_labels.shape)
    break
```

**Practice challenge:** write a custom dataset class, inspect an individual sample, then inspect a batch. Explain which dimensions represent the batch and which represent the features.

### 5. Training pipelines

**Notebook:** [`pytorch_training_pipeline.ipynb`](./pytorch_training_pipeline.ipynb)

This notebook takes the component ideas into a fuller classification workflow. Its documented focus is a public breast-cancer benchmark dataset and a comparison between a PyTorch logistic-regression baseline and a multilayer perceptron.

The notebook description identifies these workflow elements:

- Environment setup and reproducibility.
- Data loading and validation.
- Exploratory data analysis with interactive Plotly visualizations.
- Train, validation, and test splitting.
- Preprocessing fitted on training data rather than on the full dataset.
- Tensor and data-loader construction.
- Model training and comparison.
- Early stopping.
- Threshold analysis.
- Evaluation.
- Exporting artifacts.

**Why this matters**

A training pipeline is more than a loop. It is a chain of decisions. Data splitting, preprocessing, the loss function, optimizer settings, the evaluation metric, threshold selection, and checkpoint selection can all change the conclusions.

**Important interpretation rule:** a benchmark score is not a guarantee of real-world performance. The dataset, split strategy, class distribution, and evaluation protocol determine what a metric means.

> [!WARNING]
> The breast-cancer notebook is for machine-learning education only. It is not clinically validated and must not be used to diagnose, screen, treat, or guide decisions for any person.

### 6. Fashion-MNIST image classification

**Notebook:** [`fashion_mnist.ipynb`](./fashion_mnist.ipynb)

Fashion-MNIST is an image-classification learning task involving grayscale images of clothing categories. It is a useful bridge from tabular data to image tensors and classification workflows.

A typical image-classification workflow includes:

- Loading image samples and labels.
- Understanding image dimensions and pixel values.
- Visualizing a sample grid.
- Normalizing or transforming inputs when appropriate.
- Defining a model compatible with the input shape.
- Training using batches.
- Tracking loss and evaluation metrics.
- Inspecting individual predictions and mistakes.
- Understanding the difference between a correct prediction and a well-generalized model.

The CSV files in the repository are related data assets. Check the notebook's loading cells for the exact columns and preprocessing expected; do not assume that every CSV has the same schema.

**Practice challenge:** select several test examples, visualize their images, and compare the predicted class with the true label. Record patterns in the model's mistakes.

### 7. Hyperparameter optimization with Optuna

**Notebook:** [`Fashion_MNIST_Optuna.ipynb`](./Fashion_MNIST_Optuna.ipynb)

Hyperparameters are choices made before or around training: learning rate, batch size, model width, dropout, optimizer configuration, and other experiment settings. Optuna provides a framework for defining and running search trials.

A disciplined optimization experiment should:

- Define a clear objective metric.
- Separate training and validation data.
- Specify which parameters are fixed and which are searched.
- Choose reasonable search ranges.
- Keep the evaluation protocol consistent across trials.
- Save the best configuration and relevant trial results.
- Compare the tuned experiment against a meaningful baseline.
- Use the test set only for final evaluation, not repeated tuning.

**Practice challenge:** run a small search with a limited trial budget, inspect the trial history, and compare the selected configuration against a baseline trained under the same evaluation protocol.

> [!TIP]
> Hyperparameter optimization cannot rescue a flawed dataset split or an invalid evaluation method. First make the experiment trustworthy; then make it faster or more effective.

---

## 🔬 Experiments

This section describes the kinds of questions these notebooks help investigate. It intentionally avoids publishing unverified accuracy values or claiming a best model without checking the notebook outputs.

### Experiment A — Understanding tensor behavior

**Question:** How do shape, dtype, device, and broadcasting affect tensor operations?

**Variables to inspect:** input shape, output shape, data type, device, and whether the operation is in-place.

**Evidence to record:** small examples, printed shapes, assertions, and explanations of any errors encountered.

### Experiment B — Understanding gradients

**Question:** How does autograd compute a derivative through a sequence of operations?

**Variables to inspect:** input values, `requires_grad`, the scalar objective, and resulting gradients.

**Evidence to record:** hand calculations, autograd output, and a short explanation of the chain rule.

### Experiment C — Data pipeline behavior

**Question:** How do dataset indexing and data-loader configuration change the batches seen by the model?

**Variables to inspect:** dataset length, batch size, shuffle behavior, incomplete final batches, and worker configuration.

**Evidence to record:** sample shapes, batch shapes, and a comparison of loader configurations.

### Experiment D — Tabular classification workflow

**Question:** How do a linear baseline and an MLP compare on a defined binary-classification benchmark?

**Variables to inspect:** preprocessing, split strategy, architecture, optimizer, early-stopping behavior, classification threshold, and metrics.

**Evidence to record:** validation curves, selected threshold, test metrics, confusion matrix, and limitations. The dataset is educational, not clinical.

### Experiment E — Fashion-MNIST modeling

**Question:** How well can a model classify grayscale clothing images, and what kinds of examples are confusing?

**Variables to inspect:** model architecture, learning rate, batch size, number of epochs, and preprocessing.

**Evidence to record:** learning curves, test metrics, sample predictions, and error analysis.

### Experiment F — Hyperparameter search

**Question:** Can a controlled search identify a configuration that performs better on validation data than a baseline?

**Variables to inspect:** search space, objective, trial count, pruning behavior if used, and best-trial settings.

**Evidence to record:** trial history, best configuration, baseline comparison, and final held-out evaluation.

### Experiment record template

Use this structure when adding a new experiment:

| Field | What to record |
|---|---|
| Research question | What are you trying to learn? |
| Dataset | Source, version, license, and task |
| Split | Train/validation/test strategy |
| Preprocessing | Transformations and where they are fitted |
| Model | Architecture and parameter count |
| Loss | Objective used for optimization |
| Optimizer | Algorithm and relevant settings |
| Hyperparameters | Learning rate, batch size, epochs, etc. |
| Randomness | Seeds and deterministic settings |
| Compute | CPU/GPU and relevant software versions |
| Metrics | Validation and final test metrics |
| Observations | What happened and why it might have happened |
| Limitations | What the experiment does not establish |
| Next step | One change to investigate |

Never fill metric fields with guessed values. Record the actual output from a completed run.

---

## ⚙️ Setup

The repository is notebook-first. The setup below creates an isolated Python environment and installs the packages needed to begin. PyTorch installation commands vary by operating system and accelerator, so use the official selector for the exact command for your machine.

### Prerequisites

- Python version supported by the selected PyTorch release.
- Git.
- Jupyter Notebook or JupyterLab, or an editor that supports notebooks.
- Basic Python knowledge.
- Enough disk space for the repository and any datasets or experiment outputs.

A GPU is not required for the introductory concepts or small experiments. Larger experiments may benefit from a supported accelerator.

### 1. Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/pytorch-learning-journey.git
cd pytorch-learning-journey
```

### 2. Create a virtual environment

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

### 3. Install PyTorch

Use the official [PyTorch Start Locally page](https://pytorch.org/get-started/locally/) and select the options matching your operating system, package manager, language, and compute platform.

Then install notebook and commonly used supporting packages:

```bash
python -m pip install jupyter numpy pandas matplotlib scikit-learn
```

Some notebooks use additional packages. For example, the training-pipeline notebook documents interactive Plotly visualizations, and the optimization notebook uses Optuna. Install the packages required by the notebook you intend to run:

```bash
python -m pip install plotly optuna
```

Do not assume a single command is the correct accelerator build for every machine. Follow the official selector if you need CUDA or another supported compute backend.

### 4. Verify the environment

```python
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Sample tensor:")
print(sample)
print("Shape:", tuple(sample.shape))
print("Dtype:", sample.dtype)
print("Device:", sample.device)
```

A CUDA result of `False` does not automatically indicate a broken installation. It may simply mean that the installed build is CPU-only or that a compatible accelerator is not available in the current environment.

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open the notebook you want to study. In VS Code, select the interpreter from the virtual environment and make sure that the notebook kernel uses that same environment.

### 6. Run notebooks responsibly

- Start with a fresh kernel.
- Run cells in order.
- Read the Markdown instructions before executing code.
- Check dataset paths and expected files.
- Do not overwrite valuable artifacts without understanding the notebook's save behavior.
- Record package versions when an experiment needs to be reproduced.
- Review data usage terms before downloading or redistributing datasets.

---

## 🧪 How I study each notebook

The most useful habit in technical learning is to turn passive reading into active testing.

### The predict → run → inspect → explain loop

1. **Predict:** before running a cell, write down what you expect.
2. **Run:** execute the smallest example that tests the idea.
3. **Inspect:** check values, shapes, dtypes, devices, gradients, plots, or metrics.
4. **Explain:** state why the result occurred in your own words.
5. **Modify:** change one input or parameter and run the test again.
6. **Record:** keep the lesson, the bug, and the fix in the notebook.

### Debug tensor shapes early

A few explicit checks can save a lot of time:

```python
print("inputs:", inputs.shape, inputs.dtype, inputs.device)
print("targets:", targets.shape, targets.dtype, targets.device)
```

The variable names are examples; use the names present in your own training loop.

### Keep experiments controlled

When comparing configurations, change one major factor at a time where possible. Keep the dataset split and evaluation protocol consistent. Save configuration and metrics rather than relying on memory.

### Make notebooks readable

- Give each notebook a clear learning objective.
- Use Markdown cells to explain why a code block exists.
- Prefer descriptive variable names.
- Keep output relevant; remove huge accidental logs.
- Label charts and explain what they demonstrate.
- Include assertions for important shape and data assumptions.
- Restart the kernel and run all cells before committing a notebook.
- Keep private data and credentials out of notebooks.

---

## 📈 Progress board

This board is a roadmap, not an externally verified assessment of mastery. A topic should be considered complete only when I can implement and explain it independently.

### Core PyTorch

- [ ] Create tensors from common sources.
- [ ] Explain dimensions, shape, rank, dtype, and device.
- [ ] Index, slice, reshape, concatenate, and stack tensors.
- [ ] Explain broadcasting.
- [ ] Distinguish element-wise operations from matrix multiplication.
- [ ] Explain autograd and computation graphs.
- [ ] Compute gradients with `backward()`.
- [ ] Explain gradient accumulation and clearing.
- [ ] Build models with `torch.nn.Module`.
- [ ] Explain parameters and buffers.
- [ ] Select loss functions for different tasks.
- [ ] Use optimizers and understand parameter updates.

### Data and training

- [ ] Implement a custom `Dataset`.
- [ ] Configure a `DataLoader`.
- [ ] Inspect and validate batches.
- [ ] Split data without leakage.
- [ ] Build a complete training loop.
- [ ] Separate training, validation, and test responsibilities.
- [ ] Add checkpoint saving and loading.
- [ ] Track learning curves.
- [ ] Evaluate with task-appropriate metrics.
- [ ] Analyze failure cases.

### Applied deep learning

- [ ] Complete a reproducible tabular baseline.
- [ ] Build an image-classification model.
- [ ] Understand convolution and pooling.
- [ ] Train and evaluate a CNN.
- [ ] Explore transfer learning.
- [ ] Compare baseline and tuned models fairly.
- [ ] Package a small model for inference.
- [ ] Document a project with limitations and reproducibility details.

---

## 🔭 Planned next steps

The repository will expand as the foundations become comfortable. Potential next topics include:

| Topic | Why it matters | Possible learning artifact |
|---|---|---|
| Optimizers in depth | Understand how parameter updates differ | Optimizer comparison notebook |
| Learning-rate scheduling | Study how learning rates change during training | Scheduler experiment |
| Regularization | Investigate overfitting and generalization | Dropout and weight-decay experiment |
| Convolutional neural networks | Learn image-specific feature extraction | CNN image classifier |
| Model evaluation | Understand metrics, calibration, and error analysis | Evaluation notebook |
| Transfer learning | Reuse pretrained representations responsibly | Fine-tuning experiment |
| Model saving and inference | Separate training from deployment | Checkpoint and inference example |
| Experiment tracking | Make comparisons auditable | Configuration and metric logging |
| Reproducible project structure | Make work easier to rerun and review | End-to-end project |

These are future directions, not a representation that the corresponding notebooks already exist.

---

## 🧰 Tools and technologies

| Technology | Purpose |
|---|---|
| [Python](https://www.python.org/) | Main programming language |
| [PyTorch](https://pytorch.org/) | Tensor operations, automatic differentiation, and neural networks |
| [Jupyter](https://jupyter.org/) | Interactive learning and experimentation |
| [NumPy](https://numpy.org/) | Numerical arrays and data interoperability |
| [Pandas](https://pandas.pydata.org/) | Tabular data inspection and manipulation |
| [scikit-learn](https://scikit-learn.org/) | Dataset utilities, preprocessing, and evaluation tools |
| [Matplotlib](https://matplotlib.org/) | Static visualizations |
| [Plotly](https://plotly.com/python/) | Interactive visualizations |
| [Optuna](https://optuna.org/) | Hyperparameter optimization |
| [Git](https://git-scm.com/) | Version control |
| [GitHub](https://github.com/) | Repository hosting and project history |

A package is relevant only when it supports a clear notebook or experiment. The goal is not to maximize the dependency list.

---

## 📖 Learning resources

Official documentation should be the first reference for API behavior and installation instructions. Supplement it with practical implementations and your own experiments.

### PyTorch

- [PyTorch main website](https://pytorch.org/)
- [Install PyTorch — Start Locally](https://pytorch.org/get-started/locally/)
- [Official PyTorch tutorials](https://pytorch.org/tutorials/)
- [PyTorch API documentation](https://pytorch.org/docs/stable/index.html)
- [Tensor documentation](https://pytorch.org/docs/stable/tensors.html)
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [Neural network package](https://pytorch.org/docs/stable/nn.html)
- [Dataset and DataLoader documentation](https://pytorch.org/docs/stable/data.html)
- [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

### Supporting tools

- [Python documentation](https://docs.python.org/3/)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Plotly Python documentation](https://plotly.com/python/)
- [Optuna documentation](https://optuna.readthedocs.io/)

### Practical companion material

- [Learn PyTorch for Deep Learning — GitHub](https://github.com/mrdbourke/pytorch-deep-learning)

Use external tutorials to compare explanations and discover exercises, but implement the concepts yourself. The real test is whether you can explain the code and adapt it to a new dataset or problem.

---

## 🧯 Troubleshooting notes

| Symptom | What to inspect |
|---|---|
| Tensor shape mismatch | Print the shape of every relevant tensor before the failing operation |
| Data type error | Inspect `.dtype` and ensure the operation receives the expected type |
| CPU/GPU device mismatch | Check that inputs and model parameters are on compatible devices |
| Missing gradients | Check `requires_grad`, graph tracking, and whether the tensor is a leaf |
| Gradients seem too large | Check whether gradients were cleared between optimization steps |
| Notebook variable is undefined | Run cells in order or restart the kernel and run all |
| Data file not found | Inspect the notebook's expected working directory and path |
| Plot does not appear | Confirm the plotting backend and inspect the notebook output |
| Plot file is not saved | Create the output folder and check the current working directory |
| Different results between runs | Inspect random seeds, data shuffling, hardware, and nondeterministic operations |
| CUDA is unavailable | Verify the selected PyTorch build and local accelerator setup using the official installation guide |
| Out-of-memory error | Reduce batch size, model size, or input resolution; release unnecessary tensors |

When fixing an issue, preserve a small example that demonstrates the failure and the correction. A debugging note is useful learning material.

---

## 🧹 Data, artifacts, and reproducibility

A good experiment is not only one that produces a promising number. It is one whose inputs, decisions, and limitations can be inspected.

### Data discipline

- Keep the source, version, and license of each dataset visible.
- Do not commit private or sensitive data.
- Avoid uploading large raw datasets when a documented download process is more appropriate.
- Do not use test data to choose hyperparameters or repeatedly tune a model.
- Fit transformations such as scaling on the training partition only, then apply them to validation and test data.
- Inspect labels, missing values, class balance, and possible duplicates.

### Artifact discipline

- Save model state and configuration together where practical.
- Record the preprocessing steps needed for inference.
- Avoid committing large checkpoints without a reason.
- Do not overwrite a valuable model artifact accidentally.
- Keep generated plots and metrics clearly named.
- Treat exported models as outputs of a specific experiment, not as universally valid models.

### Reproducibility discipline

Record:

- Python and package versions.
- Dataset source and any relevant version or checksum.
- Split strategy and random seed.
- Model architecture and parameter choices.
- Loss function and optimizer.
- Training settings such as learning rate, batch size, and epochs.
- Compute environment.
- Metrics and evaluation procedure.

Exact reproducibility can depend on hardware and operation implementation. A seed is helpful, but it is not a universal guarantee that every run will be bit-for-bit identical.

---

## 🤝 Feedback and collaboration

This repository primarily documents my own learning, but constructive educational feedback is welcome.

Helpful feedback includes:

- Correcting a technical or documentation mistake.
- Suggesting a clearer explanation.
- Recommending a focused exercise.
- Identifying a reproducibility issue.
- Proposing a useful experiment or edge case.
- Sharing a reliable reference for a PyTorch concept.

For a large change, open an issue first so the goal and scope can be discussed. Attribute third-party code and respect the licenses of code, datasets, pretrained weights, and media.

---

## 📌 Project principles

1. **Understand before scaling.** A small model that is understood is more valuable than a large model that is merely copied.
2. **Measure before claiming.** Report real metrics and describe the evaluation protocol.
3. **Keep the test set honest.** Use validation data for model selection and reserve the test set for final evaluation.
4. **Prefer clarity over cleverness.** Readable notebooks make future debugging easier.
5. **Document mistakes.** Errors reveal assumptions; fixing them is part of learning.
6. **Respect data.** Source, license, privacy, and limitations matter.
7. **Build incrementally.** Each notebook should create a foundation for the next.

---

## 👨‍💻 About the maintainer

**Maganpreet Singh**

This repository is part of my ongoing journey in machine learning and deep learning. I am using it to turn theory into working code, strengthen my understanding of neural networks, and build a public record of experiments and progress.

- GitHub: [@Maganpreet-Singh](https://github.com/Maganpreet-Singh)
- Repository: [pytorch-learning-journey](https://github.com/Maganpreet-Singh/pytorch-learning-journey)

---

## 📄 License and attribution

No repository license file is currently documented in this README. Until a license is added, do not assume that the repository's original content is licensed for unrestricted reuse. Third-party code, datasets, pretrained weights, and visual assets may have their own terms; check those terms before reuse or redistribution.

A dedicated license file can be added once the intended reuse permissions are chosen.

---

<div align="center">

  <h3>Learn the tensor. Understand the gradient. Build the model. Test the evidence.</h3>

  <p><em>One notebook, one experiment, one concept understood at a time.</em></p>

  <p>
    <a href="https://github.com/Maganpreet-Singh/pytorch-learning-journey">Explore the repository</a>
    ·
    <a href="https://pytorch.org/tutorials/">Study the official tutorials</a>
  </p>

</div>
