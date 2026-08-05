# neuralkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](CHANGELOG.md)
[![PyPI](https://img.shields.io/pypi/v/neuralkit.svg)](https://pypi.org/project/neuralkit/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/14k5hy4/neuralkit/blob/master/examples/demo_notebook.ipynb)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-NumPy%20only-brightgreen.svg)]()

A lightweight, zero-dependency neural network framework built strictly from scratch in Python. No PyTorch, no TensorFlow — just NumPy.

Every forward pass, backward pass, gradient computation, parameter update, and initialization strategy is implemented from first principles. If you can build it from scratch, you can debug it, optimize it, and truly understand it.

---

## Table of Contents

- [Live Interactive Demo](#live-interactive-demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [What's Implemented](#whats-implemented)
  - [Layers & Activations](#layers--activations)
  - [Loss Functions & Regularization](#loss-functions--regularization)
  - [Optimizers & Schedulers](#optimizers--schedulers)
  - [Weight Initialization](#weight-initialization)
  - [Data Pipeline & Cross-Validation](#data-pipeline--cross-validation)
  - [Metrics & Evaluation](#metrics--evaluation)
  - [Visualization](#visualization)
  - [Model Serialization](#model-serialization)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)
- [What's Next](#whats-next)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Live Interactive Demo

Try **NeuralKit** directly in your browser without installing anything locally:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/14k5hy4/neuralkit/blob/master/examples/demo_notebook.ipynb)

The Colab notebook covers:
- Training a 2-layer Neural Network on the non-linear XOR problem
- Plotting loss curves using Matplotlib
- Multi-class classification on the Iris dataset
- Model serialization (`.nk` format save and reload)

---

## Installation

### Via PyPI (Recommended)

```bash
pip install neuralkit
```

### From Source

```bash
git clone https://github.com/14k5hy4/neuralkit.git
cd neuralkit
pip install -e .
```

- **Core Requirement:** NumPy (`>= 1.20.0`)
- **Optional Requirement:** Matplotlib (for loss curve, confusion matrix, and decision boundary plots)

---

## Quick Start

Train a neural network to solve the non-linear XOR classification problem:

```python
import numpy as np
from neuralkit.model import Sequential
from neuralkit.layers import Dense
from neuralkit.activations import Sigmoid
from neuralkit.losses import MSELoss
from neuralkit.optimizers import SGD
from neuralkit.trainer import Trainer

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Define model architecture
model = Sequential([
    Dense(2, 8, activation=Sigmoid()),
    Dense(8, 1, activation=Sigmoid()),
])

# Train with full-batch SGD
trainer = Trainer(model, SGD(lr=2.0), MSELoss())
history = trainer.fit(X, y, epochs=3000, verbose=False)

# Inference
predictions = model.predict(X)
print("Predictions:\n", np.round(predictions, 2))
# [[0.02], [0.98], [0.98], [0.02]]
```

Explore runnable examples in [`examples/`](examples/):
- [XOR Classification](examples/xor_example.py)
- [Iris Multi-Class Classification](examples/iris_example.py)
- [Synthetic Sine Wave Regression](examples/regression_example.py)
- [MNIST Digit Classification](examples/mnist_example.py)

---

## Architecture Overview

```
                      +-------------------+
                      |   ArrayDataset    |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |    DataLoader     |
                      +---------+---------+
                                |  (x_batch, y_batch)
                                v
+------------------+  +-------------------+  +-------------------+
|   Optimizer      |<--|     Trainer       |-->|   Loss Function   |
| (SGD/Adam/RMS)   |  +---------+---------+  | (MSE/CrossEnt)    |
+--------+---------+            |            +---------+---------+
         |                      v                      |
         |            +-------------------+            |
         +----------->|    Sequential     |<-----------+
                      +---------+---------+
                                |
                +---------------+---------------+
                |               |               |
                v               v               v
         +------------+  +------------+  +------------+
         |  Dense #1  |  |  Dropout   |  |  Dense #2  |
         +------------+  +------------+  +------------+
```

### Execution Flow per Iteration:
1. `DataLoader` generates mini-batches with optional shuffling.
2. `Sequential.forward(x_batch)` cascades activations through stacked `Layer` objects.
3. `LossFunction.forward(pred, y_batch)` evaluates scalar loss + penalty from `Regularizer`.
4. `LossFunction.backward()` computes initial upstream gradient $\frac{\partial L}{\partial y}$.
5. `Sequential.backward(grad)` propagates gradients via chain rule across all layers.
6. `Optimizer.step(layers)` updates parameters $\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \nabla_{\mathbf{W}} L$.

---

## What's Implemented

### Layers & Activations

| Module | Description / Formulations |
|---|---|
| `Dense` | Fully-connected layer: $y = xW + b$. Supports custom weight initializers. |
| `Dropout` | Inverted dropout — scales active units by $\frac{1}{1-p}$ during training. |
| `BatchNorm` | Batch normalization (Ioffe & Szegedy, 2015) with learnable $\gamma, \beta$ & running averages. |
| `Flatten` | Reshapes multi-dimensional tensors to $(N, d_{flat})$. |
| `ReLU` | Rectified Linear Unit: $f(x) = \max(0, x)$. |
| `LeakyReLU` | $f(x) = x$ if $x > 0$ else $\alpha x$ with configurable negative slope. |
| `ELU` | Exponential Linear Unit: $f(x) = x$ if $x > 0$ else $\alpha(e^x - 1)$. |
| `Swish` | Self-gated activation: $f(x) = x \cdot \sigma(x)$. |
| `Sigmoid` | Logistic sigmoid with range clipping to prevent floating-point overflow. |
| `Tanh` | Hyperbolic tangent activation. |
| `Softmax` | Numerically stable softmax with log-sum-exp stabilization. |
| *Layer Wrappers* | `ReLULayer`, `SigmoidLayer`, `TanhLayer`, `LeakyReLULayer`, `ELULayer`, `SwishLayer`. |

### Loss Functions & Regularization

| Module | Description |
|---|---|
| `MSELoss` | Mean Squared Error: $\frac{1}{n} \sum (y - \hat{y})^2$. |
| `CrossEntropyLoss` | Binary & Categorical Cross-Entropy with probability clipping. |
| `SoftmaxCrossEntropy` | Fused logit-level softmax cross-entropy ($\nabla z = \frac{p - y}{n}$). |
| `HuberLoss` | Smooth $L_1$ loss robust to outliers ($\delta$-parameterized threshold). |
| `BinaryCrossEntropyLoss` | Numerically stable binary cross-entropy loss. |
| `L1` | Lasso penalty: $\lambda \sum |W|$. |
| `L2` | Ridge penalty: $\frac{1}{2} \lambda \sum W^2$. |
| `ElasticNet` | Combined L1/L2 regularization penalty. |

### Optimizers & Schedulers

| Module | Features |
|---|---|
| `SGD` | Stochastic Gradient Descent with Nesterov/standard momentum, weight decay, and gradient clipping. |
| `Adam` | Adaptive Moment Estimation (Kingma & Ba) with bias-corrected 1st & 2nd moments. |
| `RMSProp` | Root Mean Square Propagation (Hinton) with uncentered and centered variants. |
| *Clipping* | All optimizers support `clip_value` (min/max) and `clip_norm` (global norm threshold). |
| `StepLR` | Multiplies learning rate by $\gamma$ every $N$ epochs. |
| `ExponentialLR` | Exponentially decays LR each epoch. |
| `CosineAnnealingLR` | Cosine annealing schedule down to $\eta_{min}$. |
| `ReduceLROnPlateau` | Dynamically drops LR when validation loss plateaus. |

### Weight Initialization

| Strategy | Target Activation / Formulation |
|---|---|
| `he_normal`, `he_uniform` | Kaiming He init ($std = \sqrt{2 / fan\_in}$) for ReLU networks. |
| `xavier_normal`, `xavier_uniform` | Glorot init ($std = \sqrt{2 / (fan\_in + fan\_out)}$) for Sigmoid/Tanh. |
| `lecun_normal` | LeCun init ($std = \sqrt{1 / fan\_in}$) for SELU/linear units. |
| `zeros`, `ones`, `constant` | Deterministic initialization routines. |

### Data Pipeline & Cross-Validation

| Class / Function | Capability |
|---|---|
| `ArrayDataset` | In-memory dataset container for feature matrix & targets. |
| `DataLoader` | Mini-batching, shuffling, and tail-batch handling (`drop_last`). |
| `transforms` | `StandardScaler`, `MinMaxScaler`, `Normalize`, `OneHotEncoder`, `Compose`. |
| `splits` | Stratified and random `train_test_split` & `train_val_test_split`. |
| `cross_validation` | `k_fold_split` and `cross_validate` with stratified fold generation. |

### Metrics & Evaluation

| Metric | Types Supported |
|---|---|
| Classification | `accuracy`, `precision`, `recall`, `f1_score` (macro/micro), `confusion_matrix`, `classification_report`. |
| Regression | `mse`, `rmse`, `mae`, `r2_score`. |

### Visualization

| Function | Description |
|---|---|
| `plot_training_history` | Side-by-side epoch curves for training/validation loss and metrics. |
| `plot_confusion_matrix` | Annotated heatmap matrix of true vs predicted labels. |
| `plot_decision_boundary` | 2D decision boundary contour plot over feature space. |

### Model Serialization

Save and reload fully trained model state without re-compiling:

```python
# Save model definition and weights
model.save("saved_models/iris_classifier")

# Reload model
from neuralkit.model import Sequential
model = Sequential.load("saved_models/iris_classifier")
```
- `architecture.json` stores layer types, dimensions, and activation metadata.
- `weights.npz` stores binary compressed weight and bias tensors.

---

## Comparison with Other Frameworks

| Feature | `neuralkit` | PyTorch | scikit-learn |
|---|---|---|---|
| **Dependencies** | **NumPy only** | PyTorch, CUDA, C++ | NumPy, SciPy, Cython |
| **Primary Purpose** | Educational / First-principles | Production / Research | Classical ML |
| **Autograd Engine** | Explicit manual backprop | Dynamic computation graph | N/A |
| **Code Visibility** | 100% readable Python | C++ backend core | C / Cython backends |
| **Model Customization**| Full transparency | High | Fixed API wrappers |

---

## What's Next

Some things planned for upcoming releases:

- **C Code Generator:** Export trained models to standalone C code (`export_to_c()`) for zero-dependency edge inference.
- **Model Pruning & Int8 Quantization:** Weight magnitude pruning and quantization routines to shrink model footprint.
- **Training Health Monitor:** Real-time diagnostics for vanishing/exploding gradients, dead ReLU neurons, and loss anomalies.

---

## Documentation

Full guides and detailed API documentation are available in the [`docs/`](docs/) directory:
- [Getting Started Guide](docs/getting_started.md): Detailed walkthrough for building models, using callbacks, metrics, and data pipelines.
- [API Reference](docs/api_reference.md): Complete signature breakdown for all modules.

---

## Contributing

Contributions are welcome! If you'd like to extend `neuralkit`:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Add tests in `tests/` for any new functionality.
4. Ensure all tests pass: `python -m unittest discover tests`.
5. Submit a Pull Request.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.

See [`CHANGELOG.md`](CHANGELOG.md) for version history.
