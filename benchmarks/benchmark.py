"""Performance benchmarks for neuralkit.

Measures forward/backward pass times for different model sizes
and batch sizes. Run with:
    python benchmarks/benchmark.py
"""

import time
import numpy as np

# add parent dir to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuralkit.model import Sequential
from neuralkit.layers import Dense
from neuralkit.activations import ReLU
from neuralkit.losses import SoftmaxCrossEntropy
from neuralkit.optimizers import Adam


def time_fn(fn, *args, n_runs=10, warmup=2):
    """Time a function call, returning average time in ms."""
    # warmup
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), np.std(times)


def benchmark_forward_backward(input_dim, hidden_dims, output_dim, batch_size):
    """Benchmark a single model configuration."""
    # build model
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(Dense(prev_dim, h, activation=ReLU()))
        prev_dim = h
    layers.append(Dense(prev_dim, output_dim))

    model = Sequential(layers)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(lr=0.001)

    # generate data
    x = np.random.randn(batch_size, input_dim).astype(np.float64)
    y = np.zeros((batch_size, output_dim))
    labels = np.random.randint(0, output_dim, batch_size)
    y[np.arange(batch_size), labels] = 1.0

    # time forward
    fwd_mean, fwd_std = time_fn(model.forward, x)

    # time full train step (forward + backward + update)
    def train_step():
        pred = model.forward(x)
        loss_fn.forward(pred, y)
        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step(model.layers)

    step_mean, step_std = time_fn(train_step)

    return fwd_mean, fwd_std, step_mean, step_std


def main():
    np.random.seed(42)

    print("=" * 70)
    print("neuralkit performance benchmark")
    print("=" * 70)

    configs = [
        # (input_dim, hidden_dims, output_dim, batch_size, label)
        (784, [128], 10, 32, "small (784→128→10, bs=32)"),
        (784, [256, 128], 10, 64, "medium (784→256→128→10, bs=64)"),
        (784, [512, 256, 128], 10, 128, "large (784→512→256→128→10, bs=128)"),
        (784, [256, 128], 10, 256, "medium+large batch (bs=256)"),
        (784, [256, 128], 10, 16, "medium+small batch (bs=16)"),
    ]

    print(f"\n{'Config':<42} {'Forward (ms)':<18} {'Train Step (ms)':<18}")
    print("-" * 78)

    for input_dim, hidden, output_dim, bs, label in configs:
        fwd_m, fwd_s, step_m, step_s = benchmark_forward_backward(
            input_dim, hidden, output_dim, bs
        )
        print(f"{label:<42} {fwd_m:>6.2f} ± {fwd_s:>5.2f}    {step_m:>6.2f} ± {step_s:>5.2f}")

    # parameter count summary
    print("\n\nParameter counts:")
    print("-" * 40)
    for input_dim, hidden, output_dim, _, label in configs:
        dims = [input_dim] + hidden + [output_dim]
        total = sum(dims[i] * dims[i+1] + dims[i+1] for i in range(len(dims)-1))
        print(f"  {label:<38} {total:>8,} params")

    print("\n" + "=" * 70)
    print("All benchmarks use float64 precision with numpy vectorized ops.")
    print("Times are averaged over 10 runs with 2 warmup iterations.")


if __name__ == "__main__":
    main()
