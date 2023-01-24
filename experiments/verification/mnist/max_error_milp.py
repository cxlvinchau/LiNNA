import sys

import gurobipy as gp
from gurobipy import GRB

import numpy as np
from linna.basis_finder import VarianceBasisFinder
from linna.coef_finder import L1CoefFinder
from linna.network import Network
from linna.utils import load_tf_network

from torchvision import datasets, transforms
import torch


def compute_difference(linna_net, alpha, lb, ub, layer_idx, target_neuron, maximize=True):
    basis = linna_net.layers[layer_idx].basis

    weight = linna_net.layers[layer_idx].get_weight().cpu().detach().numpy()
    bias = linna_net.layers[layer_idx].get_bias().cpu().detach().numpy()
    neuron_to_variable = dict()

    # Model
    m = gp.Model()

    # Create variables
    x = m.addMVar(weight.shape[1], lb=lb, ub=ub)

    for neuron in basis + [target_neuron]:
        pre_activation = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(pre_activation == weight[neuron, :] @ x + bias[neuron])
        post_activation = m.addVar()
        m.addGenConstrMax(post_activation, [pre_activation, 0])
        neuron_to_variable[neuron] = post_activation

    # Set objective and optimize
    m.setObjective(neuron_to_variable[target_neuron] -
                   sum([alpha[idx] * neuron_to_variable[n] for idx, n in enumerate(basis)]),
                   sense=GRB.MAXIMIZE if maximize else GRB.MINIMIZE)

    m.optimize()

    return m.objVal


if __name__ == "__main__":
    # LiNNA setup
    sequential = load_tf_network(file="../../networks/MNIST_5x100.tf")
    linna_net = Network(sequential)
    LAYER_IDX = 1
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
                                          layer_idx=0)
    # Compute IO matrices
    io_dict = dict()
    for layer in range(len(linna_net.layers)):
        io_dict[layer]: np.ndarray = linna_net.get_io_matrix(loader=trainloader, layer_idx=layer, size=1000)

    bf = VarianceBasisFinder(network=linna_net, io_dict=io_dict)
    cf = L1CoefFinder(network=linna_net, io_dict=io_dict)

    linna_net.layers[LAYER_IDX].basis = bf.find_basis(layer_idx=LAYER_IDX, basis_size=70)

    average_diff_max = 0
    average_diff_min = 0
    count = 0
    for neuron in linna_net.layers[LAYER_IDX].neurons:
        if neuron not in linna_net.layers[LAYER_IDX].basis:
            count += 1
            alpha = cf.find_coefficients(layer_idx=LAYER_IDX, neuron=neuron)

            max_val = compute_difference(linna_net=linna_net,
                                         alpha=alpha,
                                         lb=lb,
                                         ub=ub,
                                         layer_idx=LAYER_IDX,
                                         target_neuron=neuron,
                                         maximize=True
                                         )

            min_val = compute_difference(linna_net=linna_net,
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
