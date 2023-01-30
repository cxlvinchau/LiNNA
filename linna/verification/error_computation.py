import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from linna.network import Network


def compute_bounds(network: Network, x: torch.Tensor, epsilon: float, layer_idx=None, method="backward"):
    """
    Computes the bounds using auto_LiRPA

    Parameters
    ----------
    network: Network
        LiNNA network
    x: torch.Tensor
        Input image
    epsilon: float
        Perturbation
    layer_idx: int
        Layer for which the bounds should be computed. By default, final layer.
    method: str
        Method used by auto_LiRPA

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]

    """
    x = x.unsqueeze(0)
    bm = BoundedModule(network.torch_model if layer_idx is None else network.torch_model[:(layer_idx + 1) * 2], x)
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    my_input = BoundedTensor(x, ptb)
    lb, ub = bm.compute_bounds(x=(my_input,), method=method)
    return lb.cpu().detach().numpy()[0], ub.cpu().detach().numpy()[0]


def compute_difference_milp(network: Network, alpha: np.ndarray, lb: np.ndarray, ub: np.ndarray, layer_idx: int,
                            target_neuron: int, maximize: bool = True):
    """
    Compute the maximum difference between the linear combination and neuron

    Parameters
    ----------
    network: Network
    alpha: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    layer_idx: int
    target_neuron: int
    maximize: bool

    Returns
    -------
    float
    """
    basis = network.layers[layer_idx].basis

    weight = network.layers[layer_idx].get_weight().cpu().detach().numpy()
    bias = network.layers[layer_idx].get_bias().cpu().detach().numpy()
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


def compute_guaranteed_bounds(network: Network, x: torch.Tensor, epsilon: float, layer_idx: int, target_neuron: int,
                              method="backward"):
    """
    Computes the bounds for difference between the linear combination and neuron, i.e.
    it finds lb and ub such that: lb <= neuron - lin_comb <= ub
    """
    layers = list(network.original_torch_model[:(layer_idx + 1) * 2])
    out_dim = len(network.layers[layer_idx].neurons)
    basis = network.layers[layer_idx].basis

    # Auxiliary layer
    aux_layer = torch.nn.Linear(in_features=out_dim, out_features=1)
    with torch.no_grad():
        aux_layer.weight = torch.nn.Parameter(torch.zeros(aux_layer.weight.shape))
        aux_layer.bias = torch.nn.Parameter(torch.zeros(aux_layer.bias.shape))
        aux_layer.weight[0][basis] = -1 * network.layers[layer_idx].neuron_to_coef[target_neuron]
        aux_layer.weight[0][target_neuron] += 1
    layers.append(aux_layer)

    aux_sequential = torch.nn.Sequential(*layers)

    x = x.unsqueeze(0)
    bm = BoundedModule(aux_sequential, x)
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    my_input = BoundedTensor(x, ptb)
    best_lb, best_ub = None, None
    for method in ['IBP+backward', 'backward', 'CROWN',
                   'CROWN-Optimized']:
        lb, ub = bm.compute_bounds(x=(my_input,), method=method)
        if best_lb is None or lb > best_lb:
            best_lb = lb
        if best_ub is None or ub < best_ub:
            best_ub = ub

    assert best_lb <= best_ub

    return best_lb.cpu().detach().numpy()[0][0], best_ub.cpu().detach().numpy()[0][0]
