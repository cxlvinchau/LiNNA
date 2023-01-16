import warnings
from typing import Optional, List

import torch
from torch import nn
import numpy as np
import copy

from linna import utils


class Network:
    """
    Wraps a sequential PyTorch model. A layer consists of linear layer and possibly a non-linear function.
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
        self.torch_model = copy.deepcopy(torch_model)
        self.original_torch_model = copy.deepcopy(torch_model)
        # Obtain the layers from the torch model
        self.layers = []
        for layer_idx in range(0, len(torch_model), 2):
            assert isinstance(torch_model[layer_idx], nn.Linear), "Expected linear layer"
            self.layers.append(NetworkLayer(torch_model=self.torch_model, layer_idx=layer_idx))
        self.neuron_to_coef = dict()

    def reset(self):
        self.torch_model = copy.deepcopy(self.original_torch_model)
        for layer in self.layers:
            layer.reset()
            layer.torch_model = self.torch_model

    def get_num_neurons(self):
        """
        Returns the number of neurons

        Returns
        -------
        int
            Number of neurons

        """
        return sum(layer.get_weight().shape[0] for layer in self.layers[:-1])

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

    def delete_neuron(self, layer_idx: int, neuron: int):
        """
        Removes a neuron from a given layer

        Parameters
        ----------
        layer_idx: int
            Layer in which neuron should be deleted
        neuron: int
            Neuron to be deleted

        """
        self.layers[layer_idx].delete_output(neuron)
        if layer_idx < len(self.layers) - 1:
            self.layers[layer_idx+1].delete_input(neuron)

    def restore_neuron(self, layer_idx: int, neuron: int):
        """
        Restores the neuron in the given layer

        Parameters
        ----------
        layer_idx: int
            Layer in which neuron should be restored
        neuron: int
            Neuron to be restored

        Returns
        -------

        """
        self.layers[layer_idx + 1].restore_weights(neuron)
        self.layers[layer_idx + 1].restore_input(neuron)
        self.layers[layer_idx].restore_neuron(neuron)

    def set_basis(self, layer_idx: int, basis: List[int]):
        """
        Sets the basis for the specific layer. Note that due to

        Parameters
        ----------
        layer_idx: int
            Layer
        basis: List[int]
            List of basis neurons

        """
        if layer_idx < len(self.layers) - 1:
            if self.layers[layer_idx].basis is None and self.layers[layer_idx + 1].input_basis is None:
                self.layers[layer_idx].basis = basis
                self.layers[layer_idx + 1].input_basis = basis
            else:
                raise ValueError("Basis already set")
        else:
            raise ValueError("The last layer cannot have a basis")

    def readjust_weights(self, layer_idx: int, neuron: int, coef: torch.Tensor):
        """
        Readjust the outgoing weights for the neuron in the given layer. Effectively, the weight matrix
        of ``layer_idx + 1`` is modified.

        Parameters
        ----------
        layer_idx: int
            Layer whose outgoing weights are adjusted
        neuron: int
            Neuron whose weight is adjusted
        coef: torch.Tensor
            Linear coefficients

        """
        assert layer_idx + 1 < len(self.layers)
        self.neuron_to_coef[(neuron, layer_idx)] = coef
        self.layers[layer_idx+1].readjust_weights(neuron=neuron, coef=coef)

    def get_io_matrix(self, layer_idx: int, loader, size=1000) -> np.ndarray:
        """
        Computes the IO matrix for the given layer

        Parameters
        ----------
        layer_idx: int
            Layer for which IO matrix should be computed
        loader: DataLoader
        size: int
            Number of images to be considered

        Returns
        -------
        torch.Tensor
            IO matrix

        """
        outputs = []
        counter = 0
        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):
                status = False
                for image in images:
                    counter += 1
                    if size and counter > size:
                        status = True
                        break
                    outputs.append(self.forward(image.view(1, -1), layer_idx=int(layer_idx)).view(-1))
                if status:
                    break
        return torch.stack(outputs).cpu().detach().numpy()


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
        self.neurons = list(range(self.original_weight.shape[0]))

        # Store linear combinations (note that this is done for the input)
        self.neuron_to_coef = dict()

        # Store basis of previous layer
        self.input_basis = None
        self.basis = None

        self.removed_neurons = []
        self.change_matrix = torch.zeros(self.original_weight.shape)

        # Maps a neuron to its lower and upper bound linear combination
        self.neuron_to_lower_bound = dict()
        self.neuron_to_lower_bound_alt = dict()
        self.neuron_to_upper_bound = dict()
        self.neuron_to_upper_bound_term = dict()
        self.neuron_to_upper_bound_alt = dict()

    def get_weight(self):
        """
        Return the weight of the layer

        Returns
        -------
        torch.Tensor
            Weight of the layer

        """
        return self.torch_model[self.layer_idx].weight

    def get_bias(self):
        """
        Returns the bias of the layer

        Returns
        -------
        torch.Tensor
            Return bias of layer

        """
        return self.torch_model[self.layer_idx].bias

    def set_weight(self, weight: torch.Tensor):
        """
        Sets the weight of the layer

        Parameters
        ----------
        weight: torch.Tensor
            Weight

        """
        with torch.no_grad():
            self.torch_model[self.layer_idx].weight = nn.Parameter(weight)

    def set_bias(self, bias: torch.Tensor):
        """
        Sets the bias of the layer

        Parameters
        ----------
        bias: torch.Tensor

        """
        with torch.no_grad():
            self.torch_model[self.layer_idx].bias = nn.Parameter(bias)

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
            return [self.active_inputs.index(i) for i in input_neuron]
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
            self.torch_model[self.layer_idx].weight = nn.Parameter(self.get_weight()[mask, :])
            self.torch_model[self.layer_idx].bias = nn.Parameter(self.get_bias()[mask])
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
            self.torch_model[self.layer_idx].weight = nn.Parameter(self.get_weight()[:, mask])

    def get_neuron_index(self, neuron: int):
        """
        Return the index of the neuron

        Parameters
        ----------
        neuron: int
            Neuron

        Returns
        -------
        int
            Index of the neuron

        """
        if isinstance(neuron, int):
            return self.active_neurons.index(neuron)
        if isinstance(neuron, list):
            return [self.active_neurons.index(i) for i in neuron]
        raise ValueError("neuron has wrong type")

    def readjust_weights(self, neuron: int, coef: torch.Tensor):
        """
        Readjust

        Parameters
        ----------
        neuron: int
            Neuron
        coef: torch.Tensor
            A tensor containing coefficients of linear combination

        """
        self.neuron_to_coef[neuron] = coef
        with torch.no_grad():
            diag = torch.diag(self.original_weight[:, neuron])
            mat = torch.tensor(coef.clone().detach()).repeat(diag.shape[1], 1)
            weight = self.get_weight()
            idxs = self._get_input_index(self.input_basis)
            change = torch.matmul(diag.float(), mat.float()).float()
            weight[:, idxs] = weight[:, idxs].float() + change[self.active_neurons, :]
            self.change_matrix[:, idxs] = self.change_matrix[:, idxs].float() + change

    def restore_neuron(self, neuron: int):
        """
        Restores a given neuron

        Parameters
        ----------
        neuron: int
            Neuron to be restored

        """
        with torch.no_grad():
            weight = self.get_weight()
            row = self.original_weight[neuron, self.active_inputs].unsqueeze(0)
            if self.input_basis:
                idxs = self._get_input_index(self.input_basis)
                # Apply changes to linear combinations
                row[:, idxs] = row[:, idxs] + self.change_matrix[neuron, idxs]
            self.set_weight(torch.cat((weight, row), 0))
            bias = self.original_bias[neuron]
            self.set_bias(torch.cat((self.get_bias(), bias.unsqueeze(0))))
            self.active_neurons.append(neuron)
            self.removed_neurons.remove(neuron)

    def restore_input(self, neuron):
        """
        Restores an input neuron to the layer

        Parameters
        ----------
        neuron: int
            Input neuron

        """
        if neuron in self.active_inputs:
            raise ValueError("Neuron cannot be restored because it is active")
        col = self.original_weight[self.active_neurons, neuron].unsqueeze(1)
        weight = self.get_weight()
        self.set_weight(torch.cat((weight, col), 1))
        self.active_inputs.append(neuron)

    def reset(self):
        """
        Resets the layer
        """
        # Active neurons and inputs
        self.active_neurons = list(range(self.original_weight.shape[0]))
        self.active_inputs = list(range(self.original_weight.shape[1]))

        # Reset coefficient information
        self.neuron_to_coef = dict()

        # Reset basis information
        self.input_basis = None
        self.basis = None
        self.removed_neurons = []
        self.change_matrix = torch.zeros(self.original_weight.shape)

    def restore_weights(self, neuron):
        """
        Restores the weights of the layer w.r.t a previously removed neuron

        Parameters
        ----------
        neuron: int
            Neuron whose weights should be restored

        Returns
        -------

        """
        if neuron in self.active_inputs:
            raise ValueError("Neuron cannot be restored because it is active")
        coef = self.neuron_to_coef[neuron]
        with torch.no_grad():
            diag = torch.diag(self.original_weight[:, neuron])
            mat = torch.tensor(coef).repeat(diag.shape[1], 1)
            weight = self.get_weight()
            idxs = self._get_input_index(self.input_basis)
            change = torch.matmul(diag.float(), mat.float()).float()
            self.change_matrix[:, idxs] = self.change_matrix[:, idxs] - change
            weight[:, idxs] = weight[:, idxs].float() - change[self.active_neurons]
