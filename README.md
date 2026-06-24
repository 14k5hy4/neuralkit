# neuralkit

![WIP](https://img.shields.io/badge/status-work%20in%20progress-yellow)

A lightweight neural network toolkit built from scratch in Python.

## Installation

```bash
pip install -e .
```

## Quick Example

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
```

See `examples/xor_example.py` for the full runnable version.
