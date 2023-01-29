import sys

import gurobipy as gp
from gurobipy import GRB

import numpy as np
from linna.basis_finder import VarianceBasisFinder
from linna.coef_finder import L1CoefFinder, L2CoefFinder, LInfinityCoefFinder
from linna.network import Network
from linna.utils import load_tf_network

from torchvision import datasets, transforms
import torch

from linna.verification.error_computation import compute_difference_milp, compute_bounds

# LiNNA setup
sequential = load_tf_network(file="../../networks/MNIST_5x100.tf")
linna_net = Network(sequential)
LAYER_IDX = 2
IMG_IDX = 0

# Input img
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[IMG_IDX].view(-1, 784)[0].cpu().detach().numpy()
target_cls = y[IMG_IDX].item()

lb, ub = linna_net.propagate_interval(lb=x - 0.05,
                                      ub=x + 0.05,
                                      layer_idx=LAYER_IDX-1)

print(np.sum(ub - lb))

lb, ub = compute_bounds(network=linna_net,
                        x=X[IMG_IDX].view(-1, 784)[0],
                        epsilon=0.05,
                        layer_idx=LAYER_IDX-1,
                        method="CROWN-Optimized")

print(np.sum(ub - lb))

# Compute IO matrices
io_dict = dict()
for layer in range(len(linna_net.layers)):
    io_dict[layer]: np.ndarray = linna_net.get_io_matrix(layer, loader=trainloader, size=10000)

bf = VarianceBasisFinder(network=linna_net, io_dict=io_dict)
cf = L2CoefFinder(network=linna_net, io_dict=io_dict)

linna_net.layers[LAYER_IDX].basis = bf.find_basis(layer_idx=LAYER_IDX, basis_size=70)

basis = linna_net.layers[LAYER_IDX].basis
non_basic = [neuron for neuron in linna_net.layers[LAYER_IDX].neurons if neuron not in basis]

print(len(non_basic))

average_diff_max = 0
average_diff_min = 0
count = 0

for neuron in non_basic[:1]:
    count += 1
    alpha = cf.find_coefficients(layer_idx=LAYER_IDX, neuron=neuron)

    max_val = compute_difference_milp(network=linna_net,
                                      alpha=alpha,
                                      lb=lb,
                                      ub=ub,
                                      layer_idx=LAYER_IDX,
                                      target_neuron=neuron,
                                      maximize=True
                                      )

    min_val = compute_difference_milp(network=linna_net,
                                      alpha=alpha,
                                      lb=lb,
                                      ub=ub,
                                      layer_idx=LAYER_IDX,
                                      target_neuron=neuron,
                                      maximize=False
                                      )

    print(min_val, max_val)

    average_diff_max += max_val
    average_diff_min += min_val

print(average_diff_max/count)
print(average_diff_min/count)
