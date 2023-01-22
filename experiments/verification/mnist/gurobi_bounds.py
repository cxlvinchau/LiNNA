import gurobipy as gp
from gurobipy import GRB
import numpy as np
from torchvision import datasets, transforms
import torch
# Load network
from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import load_tf_network


def find_alpha(linna_net, layer_idx, lb, ub, target_neuron):
    basis = linna_net.layers[layer_idx].basis

    # Assert that basis is set
    assert basis is not None

    # Create Gurobi model
    weight = linna_net.layers[layer_idx].get_weight().cpu().detach().numpy()
    weight_lb = np.multiply(weight, lb).T
    weight_ub = np.multiply(weight, ub).T
    bias = linna_net.layers[layer_idx].get_bias().cpu().detach().numpy()

    M_lb = np.vstack([weight_lb[:, basis], np.expand_dims(bias[basis], axis=0)])
    M_ub = np.vstack([weight_ub[:, basis], np.expand_dims(bias[basis], axis=0)])

    b_lb = np.append(weight_lb[:, target_neuron], values=[bias[target_neuron]])
    b_ub = np.append(weight_ub[:, target_neuron], values=[bias[target_neuron]])

    # Model
    m = gp.Model()

    # Create variables
    alpha = m.addMVar(M_lb.shape[1], lb=0)
    m.addConstr(M_lb @ alpha >= b_lb)
    m.addConstr(M_ub @ alpha >= b_ub)

    # Set objective and optimize
    m.setObjective((M_lb @ alpha - b_lb).sum() + (M_ub @ alpha - b_ub).sum(), sense=GRB.MINIMIZE)
    m.optimize()

    # Iterate through coefficients
    print(np.max(alpha.x))
    return alpha.x


def verify_alpha(linna_net, alpha, lb, ub, layer_idx, target_neuron):
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

    # Constraint
    m.addConstr(sum([alpha[idx] * neuron_to_variable[n] for idx, n in enumerate(basis)]) >=
                neuron_to_variable[target_neuron])

    # Set objective and optimize
    m.setObjective(sum([alpha[idx] * neuron_to_variable[n] for idx, n in enumerate(basis)]) -
                   neuron_to_variable[target_neuron],
                   sense=GRB.MAXIMIZE)
    m.optimize()