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

from linna.verification.error_computation import compute_difference_milp, compute_bounds, compute_guaranteed_bounds

# LiNNA setup
sequential = load_tf_network(file="../../networks/MNIST_5x100.tf")
linna_net = Network(sequential)
LAYER_IDX = 2
IMG_IDX = 0
print(sequential)

# Input img
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[IMG_IDX].view(-1, 784)[0].cpu().detach().numpy()
target_cls = y[IMG_IDX].item()

lb, ub = linna_net.propagate_interval(lb=x - 0.05,
                                      ub=x + 0.05,
                                      layer_idx=LAYER_IDX - 1)

print(np.sum(ub - lb))

lb, ub = compute_bounds(network=linna_net,
                        x=X[IMG_IDX].view(-1, 784)[0],
                        epsilon=0.05,
                        layer_idx=LAYER_IDX - 1,
                        method="CROWN-Optimized")

print(np.sum(ub - lb))

# Compute IO matrices
io_dict = dict()
for layer in range(len(linna_net.layers)):
    io_dict[layer]: np.ndarray = linna_net.get_io_matrix(layer, loader=trainloader, size=2000)

bf = VarianceBasisFinder(network=linna_net, io_dict=io_dict)
cf = L1CoefFinder(network=linna_net, io_dict=io_dict)

linna_net.layers[LAYER_IDX].basis = bf.find_basis(layer_idx=LAYER_IDX, basis_size=90)

basis = linna_net.layers[LAYER_IDX].basis
non_basic = [neuron for neuron in linna_net.layers[LAYER_IDX].neurons if neuron not in basis]

lb_sum, ub_sum = 0, 0
diff = 0
for neuron in non_basic:
    linna_net.layers[LAYER_IDX].neuron_to_coef[neuron] = cf.find_coefficients(layer_idx=LAYER_IDX, neuron=neuron)
    lb, ub = compute_guaranteed_bounds(network=linna_net,
                                       x=X[IMG_IDX].view(-1, 784)[0],
                                       epsilon=0.05,
                                       layer_idx=LAYER_IDX,
                                       target_neuron=neuron)
    diff += (ub - lb)
    lb_sum += lb
    ub_sum += ub

print(lb_sum/len(non_basic))
print(ub_sum/len(non_basic))