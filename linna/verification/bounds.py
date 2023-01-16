import torch
from scipy.optimize import linprog
from scipy.sparse import identity, hstack, coo_matrix, lil_matrix, vstack
import numpy as np

from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import load_tf_network

NUMERIC_SLACK = 0


def lp_upper_bound(network: Network, layer_idx, neuron):
    basis = network.layers[layer_idx].basis
    weight = network.layers[layer_idx].get_weight().cpu().detach().numpy().T[:, basis]
    bias = np.expand_dims(network.layers[layer_idx].get_bias().cpu().detach().numpy()[basis], axis=0)
    M = np.vstack([weight, bias])
    identity_mat = identity(M.shape[0])
    identity_mat_neg = -1 * identity(M.shape[0])
    # Constraint matrix
    A_eq = hstack([M, identity_mat_neg, identity_mat])
    b_eq = np.append(network.layers[layer_idx].get_weight().cpu().detach().numpy()[neuron, :],
                     values=[network.layers[layer_idx].get_bias().cpu().detach().numpy()[neuron]])
    # Only slack variables are bounded
    bounds = [(0, None) for _ in range(M.shape[1])] + [(0, None) for _ in range(2 * M.shape[0])]
    # Objective function
    c = np.concatenate((np.zeros(M.shape[1]), np.ones(2 * M.shape[0])))
    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq + NUMERIC_SLACK, bounds=bounds, method="highs")
    # print("#############")
    # print(A_eq.todense())
    # print("-------------")
    # print(b_eq)
    return np.float32(np.abs(result.x[:M.shape[1]])), np.float32(result.x[-M.shape[0]:])


def lp_upper_bound_alternative(network: Network, layer_idx, neuron):
    weight = network.layers[layer_idx].get_weight().cpu().detach().numpy()
    bias = network.layers[layer_idx].get_bias().cpu().detach().numpy()
    return np.maximum(weight[neuron, :], 0), np.maximum(bias[neuron], 0)


def lp_lower_bound(network: Network, layer_idx, neuron):
    basis = network.layers[layer_idx].basis
    weight = network.layers[layer_idx].get_weight().cpu().detach().numpy().T[:, basis]
    bias = np.expand_dims(network.layers[layer_idx].get_bias().cpu().detach().numpy()[basis], axis=0)
    M = np.vstack([weight, bias])
    identity_mat = identity(M.shape[0])
    # Constraint matrix
    A_eq = hstack([M, identity_mat], dtype="float32")
    b_eq = np.append(network.layers[layer_idx].get_weight().cpu().detach().numpy()[neuron, :],
                     values=[network.layers[layer_idx].get_bias().cpu().detach().numpy()[neuron]])
    # Only slack variables are bounded
    bounds = [(None, None) for _ in range(M.shape[1])] + [(0, None) for _ in range(M.shape[0])]
    # Objective function
    c = np.concatenate((np.zeros(M.shape[1]), np.ones(M.shape[0])))
    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq - NUMERIC_SLACK, bounds=bounds, method="highs")
    return np.float32(result.x[:M.shape[1]])


def lp_lower_bound_alternative(network: Network, layer_idx, neuron):
    weight = network.layers[layer_idx].get_weight().cpu().detach().numpy()
    bias = network.layers[layer_idx].get_bias().cpu().detach().numpy()
    return weight[neuron, :], bias[neuron]