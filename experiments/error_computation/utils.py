from torch import nn
import torch
import torchvision
from linna.network import Network
from linna.basis_finder import VarianceBasisFinder
from linna.coef_finder import L2CoefFinder
from linna.abstraction import Abstraction
import copy

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def compute_error_bound(abstraction: Abstraction, layer_idx: int, neuron: int, x: torch.Tensor):
    network = abstraction.network
    layer = network.layers[layer_idx]
    succ_layer = network.layers[layer_idx+1]
    torch_network = network.torch_model
    torch_layer_idx = layer.layer_idx
    
    # Create modified network
    seq_layers = [seq_layer for seq_layer in torch_network[:torch_layer_idx]]
    in_features, out_features = layer.get_weight().shape[1], layer.get_weight().shape[0]
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features+1)
    with torch.no_grad():
        row  = layer.original_weight[neuron].detach().clone().unsqueeze(0)
        basis_idxs = layer._get_input_index(layer.input_basis)
        if layer.input_basis:
            row[:, basis_idxs] = row[:, basis_idxs] + layer.change_matrix[neuron, basis_idxs]
        linear.weight = torch.nn.Parameter(torch.cat((layer.get_weight(), row[:, basis_idxs])))
        linear.bias = torch.nn.Parameter(torch.cat((layer.get_bias(), layer.original_bias[neuron].detach().clone().unsqueeze(0))))

    aux_layer = torch.nn.Linear(in_features=out_features+1, out_features=1)
    with torch.no_grad():
        aux_layer.weight = torch.nn.Parameter(torch.cat((succ_layer.neuron_to_coef[neuron].detach().clone(), torch.tensor([-1]))).unsqueeze(0))
        aux_layer.bias = torch.nn.Parameter(torch.zeros(1))

    sequential = torch.nn.Sequential(*(seq_layers+[linear, nn.ReLU(), aux_layer]))

    for layer in sequential:
        if isinstance(layer, nn.Linear):
            print(layer.weight.shape)
            print(layer.bias.shape)

    print(sequential)
    
    # Wrap the model with auto_LiRPA
    model = BoundedModule(sequential, x)
    # Define perturbation. Here we add Linf perturbation to input data.
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # Make the input a BoundedTensor with the pre-defined perturbation.
    my_input = BoundedTensor(x, ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(x)
    # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
    lb, ub = model.compute_bounds(x=(x,), method="backward")

