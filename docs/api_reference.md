# API Reference

Quick reference for all public classes and functions in neuralkit.

---

## Model

### `Sequential`
**`neuralkit.model.Sequential`**

Container for stacking layers. Runs forward/backward through them in sequence.

| Method | Description |
|---|---|
| `__init__(layers)` | Create model with a list of Layer instances |
| `forward(x)` | Forward pass through all layers |
| `backward(grad)` | Backward pass, computes gradients |
| `predict(x)` | Forward pass in eval mode |
| `summary()` | Print layer shapes and parameter counts |
| `train()` | Set all layers to training mode |
| `eval()` | Set all layers to evaluation mode |
| `save(dirpath)` | Save architecture + weights to directory |
| `Sequential.load(dirpath)` | Class method to load a saved model |
| `trainable_params` | Property: list of all weight arrays |

---

## Layers

### `Dense`
**`neuralkit.layers.Dense(input_dim, output_dim, activation=None, initializer=None)`**

Fully-connected layer. Computes `y = x @ W + b`.

### `Dropout`
**`neuralkit.layers.Dropout(rate=0.5)`**

Inverted dropout. Scales by `1/(1-rate)` during training.

### `BatchNorm`
**`neuralkit.layers.BatchNorm(num_features, momentum=0.1, eps=1e-5)`**

Batch normalization with learnable gamma and beta.

### `Flatten`
**`neuralkit.layers.Flatten()`**

Reshapes input to `(batch_size, -1)`.

### Activation Layers

Standalone layer wrappers for activations, usable in `Sequential`:

| Class | Wraps |
|---|---|
| `ReLULayer()` | `ReLU` |
| `SigmoidLayer()` | `Sigmoid` |
| `TanhLayer()` | `Tanh` |
| `LeakyReLULayer(negative_slope=0.01)` | `LeakyReLU` |
| `ELULayer(alpha=1.0)` | `ELU` |
| `SwishLayer()` | `Swish` |

---

## Activations

**`neuralkit.activations`**

All activations have `forward(x)` and `backward(grad_output)` methods.

| Class | Formula |
|---|---|
| `ReLU()` | `max(0, x)` |
| `LeakyReLU(negative_slope=0.01)` | `x if x > 0 else slope * x` |
| `ELU(alpha=1.0)` | `x if x > 0 else alpha * (exp(x) - 1)` |
| `Swish()` | `x * sigmoid(x)` |
| `Sigmoid()` | `1 / (1 + exp(-x))` |
| `Tanh()` | `tanh(x)` |
| `Softmax()` | `exp(x) / sum(exp(x))` |

---

## Losses

**`neuralkit.losses`**

| Class | Use Case |
|---|---|
| `MSELoss()` | Regression |
| `CrossEntropyLoss()` | Classification with sigmoid/softmax outputs |
| `SoftmaxCrossEntropy()` | Multi-class on raw logits (fused, more stable) |

---

## Optimizers

**`neuralkit.optimizers`**

| Class | Key Args |
|---|---|
| `SGD(lr, momentum, weight_decay, clip_value, clip_norm)` | Classic SGD |
| `Adam(lr, beta1, beta2, eps, clip_value, clip_norm)` | Adaptive learning rate |
| `RMSProp(lr, alpha, eps, centered, clip_value, clip_norm)` | Running average of squared gradients |

---

## Learning Rate Schedulers

**`neuralkit.optimizers`**

| Class | Description |
|---|---|
| `StepLR(optimizer, step_size, gamma)` | Decay by `gamma` every `step_size` epochs |
| `ExponentialLR(optimizer, gamma)` | Multiply by `gamma` each epoch |
| `CosineAnnealingLR(optimizer, T_max, eta_min)` | Cosine decay |
| `ReduceLROnPlateau(optimizer, patience, factor)` | Reduce when metric plateaus |

---

## Initializers

**`neuralkit.initializers`**

All take a `shape` tuple and return an `np.ndarray`.

| Function | Strategy |
|---|---|
| `he_normal(shape)` | N(0, sqrt(2/fan_in)) — for ReLU |
| `he_uniform(shape)` | U(-limit, limit) — for ReLU |
| `xavier_normal(shape)` | N(0, sqrt(2/(fan_in+fan_out))) |
| `xavier_uniform(shape)` | U(-limit, limit) |
| `lecun_normal(shape)` | N(0, sqrt(1/fan_in)) |
| `zeros(shape)` | All zeros |
| `ones(shape)` | All ones |
| `constant(shape, value)` | Fill with constant |

---

## Regularizers

**`neuralkit.regularizers`**

| Class | Formula |
|---|---|
| `L1(lambda_=0.01)` | `lambda * sum(abs(w))` |
| `L2(lambda_=0.01)` | `0.5 * lambda * sum(w²)` |
| `ElasticNet(lambda_=0.01, l1_ratio=0.5)` | Mix of L1 and L2 |

Pass to `Trainer(regularizer=...)`.

---

## Data

### Transforms

**`neuralkit.data.transforms`**

| Class | Description |
|---|---|
| `StandardScaler()` | Zero mean, unit variance |
| `MinMaxScaler()` | Scale to [0, 1] |
| `Normalize(mean, std)` | Normalize with given stats |
| `OneHotEncoder()` | Integer labels → one-hot vectors |
| `Compose(transforms)` | Chain multiple transforms |

### Data Loading

**`neuralkit.data.loader`**

| Class | Description |
|---|---|
| `ArrayDataset(X, y)` | Wrap numpy arrays |
| `DataLoader(dataset, batch_size, shuffle, drop_last)` | Iterate in batches |

### Splitting

**`neuralkit.data.splits`**

| Function | Description |
|---|---|
| `train_test_split(X, y, test_size, stratify)` | Split into train/test |
| `train_val_test_split(X, y, val_size, test_size)` | Three-way split |

### Cross-Validation

**`neuralkit.data.cross_validation`**

| Function | Description |
|---|---|
| `k_fold_split(n_samples, k, stratify)` | Generate fold indices |
| `cross_validate(model_fn, optimizer_fn, ...)` | Train across k folds, return mean±std |

---

## Metrics

### Classification

**`neuralkit.metrics`**

| Function | Description |
|---|---|
| `accuracy(y_true, y_pred)` | Classification accuracy |
| `precision(y_true, y_pred, average)` | Precision (macro/micro) |
| `recall(y_true, y_pred, average)` | Recall (macro/micro) |
| `f1_score(y_true, y_pred, average)` | F1 score |
| `confusion_matrix(y_true, y_pred)` | N×N confusion matrix |
| `classification_report(y_true, y_pred)` | Formatted text report |

### Regression

| Function | Description |
|---|---|
| `mse(y_true, y_pred)` | Mean squared error |
| `rmse(y_true, y_pred)` | Root mean squared error |
| `mae(y_true, y_pred)` | Mean absolute error |
| `r2_score(y_true, y_pred)` | Coefficient of determination |

---

## Callbacks

**`neuralkit.callbacks`**

| Class | Description |
|---|---|
| `Callback()` | Base class with hooks |
| `EarlyStopping(patience, monitor, min_delta)` | Stop when metric plateaus |
| `ModelCheckpoint(dirpath, monitor)` | Save best model |
| `LearningRateLogger()` | Log LR each epoch |

Hooks: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`.

---

## Visualization

**`neuralkit.utils.visualization`**

| Function | Description |
|---|---|
| `plot_training_history(history, save_path)` | Loss and metric curves |
| `plot_confusion_matrix(cm, class_names, save_path)` | Heatmap |
| `plot_decision_boundary(model, X, y, save_path)` | 2D classification boundary |

---

## Trainer

**`neuralkit.trainer.Trainer`**

| Method | Description |
|---|---|
| `__init__(model, optimizer, loss_fn, metrics, callbacks, regularizer)` | Set up training |
| `fit(x, y, epochs, batch_size, val_data, verbose, callbacks)` | Train the model |
| `evaluate(x, y)` | Compute loss and metrics on data |
