"""Sequential model — stack layers and run forward/backward through them."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

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

    def train(self) -> None:
        """Set all layers to training mode."""
        for layer in self.layers:
            if hasattr(layer, 'train') and callable(layer.train):
                layer.train()

    def eval(self) -> None:
        """Set all layers to evaluation mode."""
        for layer in self.layers:
            if hasattr(layer, 'eval') and callable(layer.eval):
                layer.eval()

    def save(self, dirpath: str) -> None:
        """Save model architecture and weights to a directory.

        Creates:
            dirpath/architecture.json — layer config
            dirpath/weights.npz — parameter arrays
        """
        os.makedirs(dirpath, exist_ok=True)

        # save architecture
        arch = []
        for layer in self.layers:
            layer_info = {
                "type": layer.__class__.__name__,
                "module": layer.__class__.__module__,
            }
            # store constructor-relevant attributes
            for attr in ("input_dim", "output_dim", "num_features",
                         "rate", "eps", "momentum"):
                if hasattr(layer, attr):
                    layer_info[attr] = getattr(layer, attr)

            # save activation info if present
            if hasattr(layer, 'activation') and layer.activation is not None:
                act = layer.activation
                act_info = {
                    "type": act.__class__.__name__,
                    "module": act.__class__.__module__,
                }
                # save activation params
                for act_attr in ("negative_slope", "alpha"):
                    if hasattr(act, act_attr):
                        act_info[act_attr] = getattr(act, act_attr)
                layer_info["activation"] = act_info

            arch.append(layer_info)

        with open(os.path.join(dirpath, "architecture.json"), "w") as f:
            json.dump(arch, f, indent=2)

        # save weights
        weight_dict = {}
        for i, layer in enumerate(self.layers):
            for name, param in layer.params.items():
                weight_dict[f"layer_{i}_{name}"] = param
        np.savez(os.path.join(dirpath, "weights.npz"), **weight_dict)

    @classmethod
    def load(cls, dirpath: str) -> "Sequential":
        """Load a model from a directory created by save().

        Returns a new Sequential instance with restored weights.
        """
        import importlib

        with open(os.path.join(dirpath, "architecture.json"), "r") as f:
            arch = json.load(f)

        weights = np.load(os.path.join(dirpath, "weights.npz"))

        layers = []
        for i, layer_info in enumerate(arch):
            # resolve layer class
            mod = importlib.import_module(layer_info["module"])
            LayerClass = getattr(mod, layer_info["type"])

            # build constructor kwargs
            kwargs = {}
            for attr in ("input_dim", "output_dim", "num_features",
                         "rate", "eps", "momentum"):
                if attr in layer_info:
                    kwargs[attr] = layer_info[attr]

            # resolve activation if present
            if "activation" in layer_info:
                act_info = layer_info["activation"]
                act_mod = importlib.import_module(act_info["module"])
                ActClass = getattr(act_mod, act_info["type"])
                act_kwargs = {}
                for act_attr in ("negative_slope", "alpha"):
                    if act_attr in act_info:
                        act_kwargs[act_attr] = act_info[act_attr]
                kwargs["activation"] = ActClass(**act_kwargs)

            layer = LayerClass(**kwargs)

            # restore weights
            for name in layer.params:
                key = f"layer_{i}_{name}"
                if key in weights:
                    layer.params[name][:] = weights[key]

            layers.append(layer)

        return cls(layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]

    def __repr__(self) -> str:
        layer_strs = "\n  ".join(repr(l) for l in self.layers)
        return f"Sequential(\n  {layer_strs}\n)"
