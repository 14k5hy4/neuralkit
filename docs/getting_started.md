# Getting Started with neuralkit

A step-by-step guide to building, training, and evaluating your first neural network with neuralkit.

## Installation

```bash
git clone https://github.com/14k5hy4/neuralkit.git
cd neuralkit
pip install -e .
```

The only required dependency is NumPy. For visualization features, also install matplotlib:

```bash
pip install matplotlib
```

## Building a Model

Models in neuralkit are built by stacking layers in a `Sequential` container:

```python
from neuralkit.model import Sequential
from neuralkit.layers import Dense
from neuralkit.activations import ReLU

model = Sequential([
    Dense(4, 16, activation=ReLU()),
    Dense(16, 8, activation=ReLU()),
    Dense(8, 3),
])

model.summary()
```

### Choosing Activations

| Activation | Best for |
|---|---|
| `ReLU` | Hidden layers (default choice) |
| `LeakyReLU` | When you want to avoid dead neurons |
| `ELU` | Smoother alternative to ReLU |
| `Swish` | Deeper networks |
| `Sigmoid` | Binary classification output |
| `Tanh` | Hidden layers when output in (-1, 1) is desired |
| `Softmax` | Multi-class output (but prefer `SoftmaxCrossEntropy` loss) |

### Using Activations as Layers

You can also use activations as standalone layers:

```python
from neuralkit.layers import Dense, ReLULayer

model = Sequential([
    Dense(4, 16),
    ReLULayer(),
    Dense(16, 3),
])
```

### Weight Initialization

By default, Dense uses He normal initialization. You can change it:

```python
from neuralkit.initializers import xavier_normal

model = Sequential([
    Dense(4, 16, activation=ReLU(), initializer=xavier_normal),
])
```

Available: `he_normal`, `he_uniform`, `xavier_normal`, `xavier_uniform`, `lecun_normal`, `zeros`, `ones`.

## Training

### Setting Up the Trainer

```python
from neuralkit.trainer import Trainer
from neuralkit.optimizers import Adam
from neuralkit.losses import SoftmaxCrossEntropy
from neuralkit.metrics import accuracy

trainer = Trainer(
    model=model,
    optimizer=Adam(lr=0.01),
    loss_fn=SoftmaxCrossEntropy(),
    metrics=[accuracy],
)
```

### Running Training

```python
history = trainer.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    val_data=(X_val, y_val),
    verbose=True,
)
```

The `history` dict contains `'loss'`, `'val_loss'`, and any metric values per epoch.

### Using Callbacks

```python
from neuralkit.callbacks import EarlyStopping, ModelCheckpoint

trainer = Trainer(
    model=model,
    optimizer=Adam(lr=0.01),
    loss_fn=SoftmaxCrossEntropy(),
    callbacks=[
        EarlyStopping(patience=10, monitor='val_loss'),
        ModelCheckpoint('best_model', monitor='val_loss'),
    ],
)
```

### Learning Rate Schedulers

```python
from neuralkit.optimizers import CosineAnnealingLR

optimizer = Adam(lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

# pass scheduler to trainer.fit()
history = trainer.fit(X, y, epochs=100, scheduler=scheduler)
```

### Regularization

```python
from neuralkit.regularizers import L2

trainer = Trainer(
    model=model,
    optimizer=Adam(lr=0.01),
    loss_fn=SoftmaxCrossEntropy(),
    regularizer=L2(lambda_=0.001),
)
```

Available: `L1`, `L2`, `ElasticNet`.

## Evaluation

```python
# switch to eval mode (disables dropout, uses running stats for batchnorm)
model.eval()

results = trainer.evaluate(X_test, y_test)
print(f"Test loss: {results['loss']:.4f}")
print(f"Test accuracy: {results['accuracy']:.4f}")
```

### Classification Metrics

```python
from neuralkit.metrics import classification_report, confusion_matrix
import numpy as np

predictions = model.forward(X_test)
pred_labels = np.argmax(predictions, axis=1)

print(classification_report(y_true, pred_labels))
cm = confusion_matrix(y_true, pred_labels)
```

### Regression Metrics

```python
from neuralkit.metrics import mse, rmse, r2_score

y_pred = model.forward(X_test)
print(f"MSE: {mse(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

## Saving and Loading Models

```python
# save
model.save("my_model")

# load
from neuralkit.model import Sequential
loaded = Sequential.load("my_model")
predictions = loaded.predict(X_test)
```

This saves `architecture.json` (layer configuration) and `weights.npz` (parameters) to the specified directory.

## Visualization

```python
from neuralkit.utils.visualization import plot_training_history, plot_confusion_matrix

# plot loss and metrics over epochs
plot_training_history(history, save_path="training_curves.png")

# plot confusion matrix
plot_confusion_matrix(cm, class_names=["cat", "dog", "bird"], save_path="cm.png")
```

## Data Preprocessing

```python
from neuralkit.data.transforms import StandardScaler, OneHotEncoder
from neuralkit.data.splits import train_test_split

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# one-hot encode labels
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_seed=42
)
```

## Cross-Validation

```python
from neuralkit.data.cross_validation import cross_validate

results = cross_validate(
    model_fn=lambda: Sequential([Dense(4, 8, activation=ReLU()), Dense(8, 3)]),
    optimizer_fn=lambda: Adam(lr=0.01),
    loss_fn=SoftmaxCrossEntropy(),
    x=X, y=y,
    k=5,
    epochs=100,
    metrics=[accuracy],
    verbose=True,
)
# results = {'loss': {'mean': ..., 'std': ...}, 'accuracy': {'mean': ..., 'std': ...}}
```

## Next Steps

- Check out the [examples/](../examples/) directory for complete working demos
- Read the [API Reference](api_reference.md) for all available classes
- Look at `tests/` for usage patterns
