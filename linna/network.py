import warnings
from typing import Optional

import torch
from torch import nn
import numpy as np
import copy

from linna import utils


class Network:
    """
    Wraps a sequential PyTorch model. In our context, a layer consists of linear layer and a non-linear function.
    Suppose a sequential PyTorch model with 6 layers has the following structure:

    nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.Softmax

    In our setting, we have 3 layers that are organized as follows:

    Layer 0: nn.Linear, nn.ReLU
    Layer 1: nn.Linear, nn.ReLU
    Layer 2: nn.Linear, nn.Softmax

    Throughout the class neurons are uniquely identified by their index, i.e. corresponding row index in the weight matrix.
    """

    def __init__(self, torch_model: nn.Sequential):
        if not isinstance(torch_model, nn.Sequential):
            raise TypeError("torch_model needs to be an instance of nn.Sequential")

        # Underlying sequential PyTorch model
        self.torch_model = torch_model
        self.original_torch_model = copy.deepcopy(torch_model)

    def forward(self, X: torch.Tensor, layer_idx: Optional[int] = None, grad: bool = False):
        """
        Computes the forward pass until the given layer (inclusive)

        Parameters
        ----------
        X: torch.Tensor
        layer_idx: int, optional
        grad: bool, default=False

        Returns
        -------
        torch.Tensor
            Output of the neural network

        """
        if layer_idx is None:
            layer_idx = len(self.layers)
        return utils.forward(self.torch_model, X, layer_idx, grad=grad)

    def classify(self, X: torch.Tensor):
        """
        Classifies the given inputs

        Parameters
        ----------
        X: torch.Tensor
            Inputs to be classified

        Returns
        -------
        torch.Tensor
            Classification result

        """
        _, predicted = torch.max(self.forward(X), 1)
        return predicted


class NetworkLayer:

    def __init__(self, torch_model: nn.Sequential, layer_idx: int):
        """
        Initializes network layer

        Parameters
        ----------
        torch_model: nn.Sequential
            Sequential neural network
        layer_idx: int
            Index of the network layer in the given sequential network
        """
        self.torch_model = torch_model
        self.layer_idx = layer_idx

        # Original weights and bias
        self.original_weight = torch.Tensor(self.torch_model[self.layer_idx].weight.detach().clone())
        self.original_bias = torch.Tensor(self.torch_model[self.layer_idx].bias.detach().clone())

        # Active neurons and inputs
        self.active_neurons = list(range(self.original_weight.shape[0]))
        self.active_inputs = list(range(self.original_weight.shape[1]))

        # Store linear combinations (note that this is done for the input)
        self.lin_comb = dict()
        self.basis = None
        self.removed_neurons = []
        self.change_matrix = torch.zeros(self.original_weight.shape)

    def get_weight(self):
        """
        Return the weight of the layer

        Returns
        -------
        torch.Tensor
            Weight of the layer

        """
        return self.torch_model[self.idx].weight

    def get_bias(self):
        """
        Returns the bias of the layer

        Returns
        -------
        torch.Tensor
            Return bias of layer

        """
        return self.torch_model[self.idx].bias

    def set_weight(self, weight: torch.Tensor):
        """
        Sets the weight of the layer

        Parameters
        ----------
        weight: torch.Tensor
            Weight

        """
        assert weight.size() == self.get_weight().size(), "Dimensions of weight not matching"
        with torch.no_grad():
            self.torch_model[self.idx].weight = nn.Parameter(weight)

    def set_bias(self, bias: torch.Tensor):
        """
        Sets the bias of the layer

        Parameters
        ----------
        bias: torch.Tensor

        """
        assert bias.size() == self.get_bias().size(), "Dimensions of bias not matching"
        with torch.no_grad():
            self.torch_model[self.idx].bias = nn.Parameter(bias)

    def _get_input_index(self, input_neuron: int):
        """
        Returns the index of the input neuron

        Parameters
        ----------
        input_neuron: int
            Input neuron

        Returns
        -------
        int
            Index of input neuron

        """
        if isinstance(input_neuron, int):
            return self.active_inputs.index(input)
        if isinstance(input_neuron, list):
            return [self.active_inputs.index(i) for i in input]
        raise ValueError("input has wrong type")

    def get_input_weight(self, neuron: int):
        """
        Returns the input weight of the neuron

        Parameters
        ----------
        neuron: int
            Neuron

        Returns
        -------
        torch.Tensor
            Input weights of the neuron

        """
        idx = self._get_input_index(neuron)
        return self.torch_model[self.idx].weight[:, idx]

    def delete_output(self, neuron: int):
        """
        Deletes the neuron from the layer

        Parameters
        ----------
        neuron: int
            Neuron to be deleted

        """
        with torch.no_grad():
            mask = [i for i in range(len(self.active_neurons)) if self.active_neurons[i] != neuron]
            self.active_neurons.remove(neuron)
            self.torch_model[self.idx].weight = nn.Parameter(self.get_weight()[mask, :])
            self.torch_model[self.idx].bias = nn.Parameter(self.get_bias()[mask])
            self.removed_neurons.append(neuron)

    def delete_input(self, neuron: int):
        """
        Deletes the input neuron

        Parameters
        ----------
        neuron: int
            Neuron

        """
        with torch.no_grad():
            mask = [i for i in range(len(self.active_inputs)) if self.active_inputs[i] != neuron]
            self.active_inputs.remove(neuron)
            self.torch_model[self.idx].weight = nn.Parameter(self.get_weight()[:, mask])

