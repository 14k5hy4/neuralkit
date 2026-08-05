"""neuralkit - A lightweight neural network toolkit built from scratch."""

__version__ = "0.2.1"

from neuralkit.model import Sequential, load_model
from neuralkit.trainer import Trainer
from neuralkit.activations import Sigmoid, ReLU, Tanh, Softmax, LeakyReLU, ELU, Swish
from neuralkit.losses import (
    MSELoss, CrossEntropyLoss, SoftmaxCrossEntropy,
    HuberLoss, BinaryCrossEntropyLoss,
)
from neuralkit.layers import Dense, Dropout, BatchNorm, Flatten
from neuralkit.optimizers import SGD, Adam, RMSProp
from neuralkit.regularizers import L1, L2, ElasticNet
