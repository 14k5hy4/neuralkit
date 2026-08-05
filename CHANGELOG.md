# Changelog

All notable changes to neuralkit are documented here.

## [0.2.1] - 2026-08-05

### Fixed
- Added `load_model(dirpath)` helper function in `neuralkit.model` and exported it at top-level `neuralkit` package level.

## [0.2.0] - 2026-08-05

### Added

#### Layers
- **Flatten** layer for reshaping multi-dimensional input to 2D
- **Activation layers** (`ReLULayer`, `SigmoidLayer`, `TanhLayer`, `LeakyReLULayer`, `ELULayer`, `SwishLayer`) — use activations as standalone layers in Sequential

#### Activations
- **LeakyReLU** with configurable negative slope
- **ELU** (Exponential Linear Unit) with alpha parameter
- **Swish** self-gated activation (x · σ(x))

#### Optimizers
- **RMSProp** optimizer with centered variant (Hinton's formulation)
- **Gradient clipping** (by value and norm) in SGD, Adam, and RMSProp

#### Loss Functions
- **HuberLoss** (Smooth L1) with configurable delta parameter
- **BinaryCrossEntropyLoss** for binary classification tasks

#### Regularization
- **L1** (Lasso) regularization
- **L2** (Ridge) regularization
- **ElasticNet** combined L1/L2 regularization
- Integrated into Trainer: regularizer penalty added to loss, gradients injected before optimizer step

#### Weight Initialization
- `he_normal`, `he_uniform` for ReLU networks
- `xavier_normal`, `xavier_uniform` for sigmoid/tanh
- `lecun_normal` for SELU
- `zeros`, `ones`, `constant`
- Dense layer accepts configurable `initializer` parameter

#### Data & Cross-Validation
- **k-fold cross-validation** with stratified fold support
- `cross_validate()` function: trains fresh models per fold, returns mean ± std

#### Visualization
- `plot_training_history()` — loss and metric curves
- `plot_confusion_matrix()` — annotated heatmap
- `plot_decision_boundary()` — 2D classification boundary

#### Training
- **Text-based progress bar** (no tqdm dependency)
- Configurable verbosity: 0=silent, 1=progress bar, 2=detailed logging
- Elapsed time tracking

#### Error Handling
- Custom exception hierarchy: `ShapeMismatchError`, `ConfigurationError`, `NotFittedError`, `ForwardNotCalledError`
- Input shape validation in Dense layers
- Sample count validation in Trainer
- Learning rate sanity warning

#### Examples
- **MNIST digit classification** with deeper architecture, dropout, batch norm
- Updated iris and regression examples with visualization output

#### Documentation
- Getting Started guide (`docs/getting_started.md`)
- API Reference (`docs/api_reference.md`)
- Comprehensive README with badges, architecture diagram, framework comparison

#### Testing
- Expanded test suite: 74 tests covering model, data, metrics, losses, layers, activations, and optimizers
- Numerical gradient verification for new loss functions

#### Performance
- Benchmark script (`benchmarks/benchmark.py`)
- Performance documentation in hot paths

### Changed
- `verbose` parameter in `Trainer.fit()` now accepts int (0/1/2) in addition to bool
- README completely rewritten with badges, TOC, architecture diagram

---

## [0.1.0] - 2026-06-24

### Added
- Initial release
- Sequential model with save/load
- Dense, Dropout, BatchNorm layers
- ReLU, Sigmoid, Tanh, Softmax activations
- MSE, CrossEntropy, SoftmaxCrossEntropy losses
- SGD (with momentum, weight decay) and Adam optimizers
- StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau schedulers
- EarlyStopping, ModelCheckpoint callbacks
- DataLoader, ArrayDataset, transforms, train/test splitting
- Classification metrics (accuracy, precision, recall, F1, confusion matrix)
- Regression metrics (MSE, RMSE, MAE, R²)
- XOR, Iris, and regression examples
