import torch
from scipy.optimize import linprog
from scipy.sparse import identity, hstack, coo_matrix, lil_matrix
import numpy as np

from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import load_tf_network


def lp_upper_bound(network: Network, layer_idx, neuron):
    M = network.layers[layer_idx].get_weight().cpu().detach().numpy().T[:, network.layers[layer_idx].basis]
    identity_mat_neg = -1 * identity(M.shape[0])
    # Constraint matrix
    A_eq = hstack([M, identity_mat_neg])
    b_eq = network.layers[layer_idx].get_weight().cpu().detach().numpy()[neuron, :]
    # Only slack variables are bounded
    bounds = [(None, None) for _ in range(M.shape[1])] + [(0, None) for _ in range(M.shape[0])]
    # Objective function
    c = np.concatenate((np.zeros(M.shape[1]), np.ones(M.shape[0])))
    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return np.abs(result.x[:M.shape[1]])


def lp_lower_bound(network: Network, layer_idx, neuron):
    M = network.layers[layer_idx].get_weight().cpu().detach().numpy().T[:, network.layers[layer_idx].basis]
    identity_mat = identity(M.shape[0])
    # Constraint matrix
    A_eq = hstack([M, identity_mat])
    b_eq = network.layers[layer_idx].get_weight().cpu().detach().numpy()[neuron, :]
    # Only slack variables are bounded
    bounds = [(None, None) for _ in range(M.shape[1])] + [(0, None) for _ in range(M.shape[0])]
    # Objective function
    c = np.concatenate((np.zeros(M.shape[1]), np.ones(M.shape[0])))
    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return result.x[:M.shape[1]]


# Load trained neural network
sequential = load_tf_network(file="../networks/MNIST_3x100.tf")
network = Network(torch_model=sequential)
print(network.layers[0].get_weight().shape[1])

bf = VarianceBasisFinder(network=network,
                         io_dict={
                             idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                             enumerate(network.layers)
                         })

basis = bf.find_basis(layer_idx=1, basis_size=50)
print(basis)
network.layers[1].basis = basis
print(lp_lower_bound(network, layer_idx=1, neuron=4))
print(lp_upper_bound(network, layer_idx=1, neuron=4))
