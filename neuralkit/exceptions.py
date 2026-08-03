"""Custom exceptions for neuralkit.

Provides descriptive error types so users get clear messages
when something goes wrong during model building or training.
"""


class NeuralkitError(Exception):
    """Base exception for all neuralkit errors."""
    pass


class ShapeMismatchError(NeuralkitError):
    """Raised when tensor shapes don't match expected dimensions.

    Example:
        Dense(4, 8) expects input with 4 features, but got shape (32, 10).
    """

    def __init__(self, expected, got, context=""):
        self.expected = expected
        self.got = got
        msg = f"Shape mismatch: expected {expected}, got {got}"
        if context:
            msg = f"{context}: {msg}"
        super().__init__(msg)


class NotFittedError(NeuralkitError):
    """Raised when a transform or model is used before fitting."""

    def __init__(self, name=""):
        obj = name or "This object"
        super().__init__(f"{obj} has not been fitted yet. Call fit() first.")


class ConfigurationError(NeuralkitError):
    """Raised for invalid hyperparameter or configuration values."""

    def __init__(self, msg):
        super().__init__(f"Configuration error: {msg}")


class ForwardNotCalledError(NeuralkitError):
    """Raised when backward() is called without a prior forward()."""

    def __init__(self, layer_name=""):
        name = layer_name or "Layer"
        super().__init__(f"{name}: forward() must be called before backward().")
