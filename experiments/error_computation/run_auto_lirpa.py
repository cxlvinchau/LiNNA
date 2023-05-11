from torch import nn
import torch
import torchvision
from linna.network import Network
from linna.basis_finder import VarianceBasisFinder
from linna.coef_finder import L2CoefFinder
from linna.abstraction import Abstraction
import copy
from linna.utils import load_tf_network
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

sequential = load_tf_network(file="../networks/MNIST_5x100.tf")

print(sequential)

# Wrap the model with auto_LiRPA
x = torch.zeros((1, 784))
model = BoundedModule(sequential, x)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(x, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(x)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(x,), method="backward")