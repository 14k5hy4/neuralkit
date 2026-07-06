# neuralkit

A neural network framework built from scratch in Python. No PyTorch, no TensorFlow — just NumPy.

The point of this project is to understand what happens under the hood: forward passes, backpropagation, gradient computation, and parameter updates, all implemented from first principles.

## Installation

```bash
git clone https://github.com/14k5hy4/neuralkit.git
cd neuralkit
pip install -e .
```

Only dependency is NumPy.

## Quick Start

Train a small network to learn XOR:

```python
import numpy as np
from neuralkit.layers.dense import Dense
from neuralkit.activations import Sigmoid
from neuralkit.losses import MSELoss
from neuralkit.optimizers.sgd import SGD
from neuralkit.model import Sequential
from neuralkit.trainer import Trainer

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential([
    Dense(2, 8, activation=Sigmoid()),
    Dense(8, 1, activation=Sigmoid()),
])

trainer = Trainer(model, SGD(lr=2.0), MSELoss())
history = trainer.fit(X, y, epochs=3000)

print(model.predict(X))
# [[0.0102], [0.9873], [0.9874], [0.0141]]
```

See [`examples/xor_example.py`](examples/xor_example.py) for the full runnable version.

## What's Implemented

### Layers
| Layer | Description |
|---|---|
| `Dense` | Fully-connected layer with He initialization. Forward: `x @ W + b`. Backward: full gradient computation for W, b, and input. |
| `Dropout` | Inverted dropout — scales activations by `1/(1-rate)` during training so no adjustment is needed at inference. |
| `BatchNorm` | Batch normalization (Ioffe & Szegedy, 2015). Learnable gamma/beta, tracks running mean/variance for inference. |

### Activations
| Activation | Notes |
|---|---|
| `ReLU` | Caches mask for backward pass |
| `Sigmoid` | Clipped to prevent overflow (`np.clip(x, -88, 88)`) |
| `Tanh` | Standard implementation |
| `Softmax` | Numerically stable (subtracts row-wise max before exp) |

### Loss Functions
| Loss | Description |
|---|---|
| `MSELoss` | Mean squared error — `(1/n) * Σ(y_pred - y_true)²` |
| `CrossEntropyLoss` | Works with sigmoid/softmax outputs. Clips predictions to avoid `log(0)`. |
| `SoftmaxCrossEntropy` | Fused softmax + cross-entropy on raw logits. Gradient simplifies to `(softmax - y_true) / n`. |

### Optimizers
| Optimizer | Details |
|---|---|
| `SGD` | Supports momentum and weight decay (L2 regularization) |
| `Adam` | Full implementation following Kingma & Ba (2014) — first/second moment estimates with bias correction |

### Data Utilities
| Module | What It Does |
|---|---|
| `DataLoader` | Iterates over a dataset in batches. Supports shuffle and `drop_last`. |
| `ArrayDataset` | Wraps numpy arrays as a dataset. |
| `transforms` | `Normalize`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `Compose` |
| `splits` | `train_test_split` and `train_val_test_split` with stratified sampling |

### Metrics
| Metric | Details |
|---|---|
| `accuracy` | Classification accuracy |
| `precision` | Macro and micro averaging |
| `recall` | Macro and micro averaging |
| `f1_score` | Harmonic mean of precision and recall |

### Training
- `Trainer` class handles the training loop
- Supports full-batch and mini-batch gradient descent
- Validation loss tracking with `val_data` parameter
- Returns training history dict for plotting

## Project Structure

```
neuralkit/
├── neuralkit/
│   ├── __init__.py
│   ├── model.py              # Sequential model container
│   ├── trainer.py             # Training loop
│   ├── activations.py         # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py              # MSE, CrossEntropy, SoftmaxCrossEntropy
│   ├── core/
│   │   └── tensor.py          # Basic tensor operations
│   ├── layers/
│   │   ├── base.py            # Abstract Layer class
│   │   ├── dense.py           # Fully-connected layer
│   │   ├── dropout.py         # Dropout regularization
│   │   └── batchnorm.py       # Batch normalization
│   ├── optimizers/
│   │   ├── sgd.py             # SGD with momentum
│   │   └── adam.py            # Adam optimizer
│   ├── data/
│   │   ├── loader.py          # DataLoader, ArrayDataset
│   │   ├── transforms.py      # Preprocessing transforms
│   │   └── splits.py          # Train/test/val splitting
│   └── metrics/
│       └── classification.py  # Accuracy, precision, recall, F1
├── examples/
│   └── xor_example.py         # XOR classification demo
├── tests/
│   └── test_activations.py
├── setup.py
├── requirements.txt
└── README.md
```

## How It Works

The core loop is simple:

```
forward pass:   input → layer 1 → layer 2 → ... → predictions
loss:           compare predictions to targets → scalar loss
backward pass:  loss gradient → ... → layer 2 → layer 1 → parameter gradients
update:         optimizer adjusts weights using gradients
```

Every layer implements `forward()` and `backward()`. The `Sequential` model chains them. The `Trainer` orchestrates the loop.

```python
# what happens inside trainer.fit() each step:
predictions = model.forward(x_batch)        # forward through all layers
loss = loss_fn.forward(predictions, y_batch) # compute loss
grad = loss_fn.backward()                    # gradient of loss w.r.t. output
model.backward(grad)                         # backpropagate through layers
optimizer.step(model.layers)                 # update parameters
```

## Roadmap

Planned features (see [`.dev/roadmap.json`](.dev/roadmap.json) for full details):

- [ ] Learning rate schedulers (StepLR, CosineAnnealing, ReduceLROnPlateau)
- [ ] Early stopping and model checkpointing (callbacks)
- [ ] Model save/load (weights as `.npz`, architecture as JSON)
- [ ] LeakyReLU, ELU, Swish activations
- [ ] Weight initialization strategies (Xavier, LeCun)
- [ ] Gradient clipping (by value and by norm)
- [ ] Training visualization (loss curves, confusion matrix plots)
- [ ] MNIST digit classification example
- [ ] L1/L2/ElasticNet regularization in loss
- [ ] Cross-validation utility

## Why Build This?

Calling `model.fit()` in PyTorch or Keras is easy. Understanding *why* it works is harder. This project exists to make the internals concrete:

- How does backpropagation actually compute gradients through a chain of layers?
- Why does batch normalization help training?
- What is Adam really doing differently from SGD?
- Why do we clip values in sigmoid and use log-sum-exp in softmax?

If you can build it from scratch, you can debug it, optimize it, and explain it in an interview.

## License

MIT
