# neuralkit

A neural network framework built from scratch in Python. No PyTorch, no TensorFlow -- just NumPy.

Every forward pass, backward pass, gradient computation, and parameter update is implemented from first principles. If you can build it from scratch, you can debug it, optimize it, and explain it.

## Installation

```bash
git clone https://github.com/14k5hy4/neuralkit.git
cd neuralkit
pip install -e .
```

Only dependency is NumPy.

## Quick Start

Train a network to learn XOR:

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
# [[0.01], [0.99], [0.99], [0.01]]
```

See [`examples/`](examples/) for more: [XOR](examples/xor_example.py), [Iris classification](examples/iris_example.py), [regression](examples/regression_example.py).

## What's Implemented

### Layers
| Layer | Description |
|---|---|
| `Dense` | Fully-connected layer with He initialization. Forward: `x @ W + b`. Backward: full gradient computation for W, b, and input. |
| `Dropout` | Inverted dropout -- scales activations by `1/(1-rate)` during training so no adjustment needed at inference. |
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
| `MSELoss` | Mean squared error |
| `CrossEntropyLoss` | Works with sigmoid/softmax outputs, clips to avoid `log(0)` |
| `SoftmaxCrossEntropy` | Fused softmax + cross-entropy on raw logits. Gradient simplifies to `(softmax - y_true) / n` |

### Optimizers
| Optimizer | Details |
|---|---|
| `SGD` | Supports momentum and weight decay (L2 regularization) |
| `Adam` | Following Kingma & Ba (2014) -- first/second moment estimates with bias correction |

### Learning Rate Schedulers
| Scheduler | Details |
|---|---|
| `StepLR` | Decay LR by a factor every N epochs |
| `ExponentialLR` | Multiply LR by gamma each epoch |
| `CosineAnnealingLR` | Cosine decay to a minimum LR |
| `ReduceLROnPlateau` | Reduce LR when a metric stops improving |

### Callbacks
| Callback | Details |
|---|---|
| `EarlyStopping` | Stop training when validation loss stops improving |
| `ModelCheckpoint` | Save best model during training |

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
| `confusion_matrix` | Full NxN confusion matrix |
| `r2_score` | Coefficient of determination (regression) |
| `mse`, `mae` | Mean squared error, mean absolute error (regression) |

### Model Serialization
```python
# Save a trained model
model.save("my_model.nk")

# Load it back
from neuralkit.model import load_model
model = load_model("my_model.nk")
```

Architecture saved as JSON, weights as compressed `.npz`.

### Training
- `Trainer` class handles the full training loop
- Supports full-batch and mini-batch gradient descent
- Validation loss tracking with `val_data` parameter
- Callbacks integration (early stopping, checkpointing)
- Learning rate scheduler support
- Returns training history dict for plotting

## Project Structure

```
neuralkit/
├── neuralkit/
│   ├── model.py              # Sequential model container + save/load
│   ├── trainer.py             # Training loop with callbacks and scheduler support
│   ├── activations.py         # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py              # MSE, CrossEntropy, SoftmaxCrossEntropy
│   ├── callbacks.py           # EarlyStopping, ModelCheckpoint
│   ├── core/
│   │   └── tensor.py          # Basic tensor operations
│   ├── layers/
│   │   ├── base.py            # Abstract Layer class
│   │   ├── dense.py           # Fully-connected layer
│   │   ├── dropout.py         # Dropout regularization
│   │   └── batchnorm.py       # Batch normalization
│   ├── optimizers/
│   │   ├── sgd.py             # SGD with momentum + weight decay
│   │   ├── adam.py            # Adam optimizer
│   │   └── schedulers.py      # LR schedulers
│   ├── data/
│   │   ├── loader.py          # DataLoader, ArrayDataset
│   │   ├── transforms.py      # Preprocessing transforms
│   │   └── splits.py          # Train/test/val splitting
│   └── metrics/
│       ├── classification.py  # Accuracy, precision, recall, F1, confusion matrix
│       └── regression.py      # R2, MSE, MAE
├── examples/
│   ├── xor_example.py         # XOR classification
│   ├── iris_example.py        # Iris multi-class classification
│   └── regression_example.py  # Synthetic regression demo
├── tests/
│   ├── test_activations.py    # Activation function tests
│   ├── test_layers.py         # Layer forward/backward tests with gradient checking
│   └── test_optimizers.py     # Optimizer tests
├── setup.py
├── requirements.txt
└── README.md
```

## How It Works

The core loop:

```
forward pass:   input -> layer 1 -> layer 2 -> ... -> predictions
loss:           compare predictions to targets -> scalar loss
backward pass:  loss gradient -> ... -> layer 2 -> layer 1 -> parameter gradients
update:         optimizer adjusts weights using gradients
```

Every layer implements `forward()` and `backward()`. The `Sequential` model chains them. The `Trainer` orchestrates the loop.

```python
# Inside trainer.fit() each step:
predictions = model.forward(x_batch)        # forward through all layers
loss = loss_fn.forward(predictions, y_batch) # compute loss
grad = loss_fn.backward()                    # gradient of loss w.r.t. output
model.backward(grad)                         # backpropagate through layers
optimizer.step(model.layers)                 # update parameters
```

## What's Next

Some things I'm planning to add:

- Export trained models to standalone C for fast inference without Python
- LeakyReLU, ELU, Swish activations
- Gradient clipping
- MNIST example
- Training diagnostics (gradient health, dead neuron detection)

## Why Build This?

Calling `model.fit()` in PyTorch or Keras is easy. Understanding *why* it works is harder. This project makes the internals concrete:

- How does backpropagation compute gradients through a chain of layers?
- Why does batch normalization help training?
- What is Adam really doing differently from SGD?
- Why do we clip values in sigmoid and use log-sum-exp in softmax?

## License

MIT
