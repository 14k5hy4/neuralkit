"""Sequential model — stack layers and run forward/backward through them."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from neuralkit.layers.base import Layer


class Sequential:
    """Container that chains layers sequentially.

    Feed data through each layer in order for forward pass,
    reverse order for backward pass.

    Example::
        model = Sequential([
            Dense(784, 128, activation=ReLU()),
            Dense(128, 10),
        ])
        out = model.forward(x)
    """

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        self.layers: List[Layer] = layers or []

    def add(self, layer: Layer) -> None:
        """Append a layer to the model."""
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pass input through all layers in order."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through layers in reverse order."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass (alias for inference, no training state changes)."""
        return self.forward(x)

    def summary(self) -> None:
        """Print a summary of the model architecture."""
        print(f"{'Layer':<25} {'Output Shape':<20} {'Params':>10}")
        print("-" * 58)
        total_params = 0
        for layer in self.layers:
            params = layer.params
            n_params = sum(p.size for p in params.values()) if params else 0
            total_params += n_params

            # try to infer output shape from layer attributes
            if hasattr(layer, 'output_dim'):
                shape_str = f"(*, {layer.output_dim})"
            else:
                shape_str = "?"

            print(f"{layer.name:<25} {shape_str:<20} {n_params:>10}")

        print("-" * 58)
        print(f"Total params: {total_params}")

    # TODO: add __len__ and __getitem__ for indexing into layers

    def __repr__(self) -> str:
        layer_strs = "\n  ".join(repr(l) for l in self.layers)
        return f"Sequential(\n  {layer_strs}\n)"
