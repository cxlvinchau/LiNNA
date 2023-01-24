import abc
import copy
from typing import Dict, Optional, Tuple, List
import numpy as np
from torch import nn

from src import utils
from src.network import Network
import torch
from src.utils import forward


class _RefinementStrategy(abc.ABC):

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None, **parameters):
        self.network = network
        self.parameters = parameters
        self.io_dict = io_dict

    @abc.abstractmethod
    def find_neuron(self, cex: torch.Tensor, layers: List[int] = None) -> Tuple[int, int]:
        """
        Returns the neuron that should be restored. If the layer is not specified, the method considers all layers
        for refinement and returns the best option.

        Parameter layer_idx: Optional[int] = Nones
        ----------
        cex: torch.Tensor
            Counterexample
        layers: Optional[List[int]]
            List of layers to consider

        Returns
        -------
        int
            Layer
        int
            Neuron in the layer

        """
        pass


class DifferenceRefinement(_RefinementStrategy):

    def find_neuron(self, cex: torch.Tensor, layers: List[int] = None) -> Tuple[int, int]:
        if layers is None:
            layers = range(len(self.network.layers)-1)
        max_val, max_neuron, max_layer = 0, None, None
        for layer_idx in [l_idx for l_idx in layers if self.network.layers[l_idx].basis is not None]:
            out = self.network.forward(X=cex, layer_idx=layer_idx)
            layer = self.network.layers[layer_idx]
            basis_out = out[layer.get_neuron_index(layer.basis)]
            out = torch.Tensor([
                (out[layer.get_neuron_index(neuron)] if neuron in layer.active_neurons
                 else torch.dot(self.network.neuron_to_coef[(neuron, layer_idx)], basis_out))
                for neuron in layer.neurons])
            out_original = forward(self.network.original_torch_model, cex, layer_idx=layer_idx)
            diff = torch.abs(out - out_original)
            val, idx = torch.max(diff, 0)
            if val >= max_val:
                max_neuron = idx.item()
                max_layer = layer_idx
        if max_neuron is None:
            raise ValueError("Neuron could not be determined")
        return max_layer, max_neuron


class LookaheadRefinement(_RefinementStrategy):

    def find_neuron(self, cex: torch.Tensor, layers: List[int] = None) -> Tuple[int, int]:
        if layers is None:
            layers = range(len(self.network.layers) - 1)
        best_neuron, best_layer, min_error = None, None, None
        for layer_idx in layers:
            out = self.network.forward(cex)
            candidates = list(self.network.layers[layer_idx].removed_neurons)
            num_param = sum(p.numel() for p in self.network.torch_model.parameters())
            layer_weight, layer_bias = self.network.layers[layer_idx].get_weight().detach().clone(), self.network.layers[
                layer_idx].get_bias().detach().clone()
            succ_layer_weight = self.network.layers[layer_idx + 1].get_weight().data.detach().clone()
            layer_neurons = copy.deepcopy(self.network.layers[layer_idx].active_neurons)
            succ_layer_inputs = copy.deepcopy(self.network.layers[layer_idx + 1].active_inputs)
            layer_change = self.network.layers[layer_idx].change_matrix.detach().clone()
            succ_layer_change = self.network.layers[layer_idx + 1].change_matrix.detach().clone()
            for neuron in candidates:
                self.network.restore_neuron(layer_idx, neuron)
                original_out = utils.forward(self.network.original_torch_model, cex, len(self.network.layers),
                                             grad=False)
                current_out = utils.forward(self.network.torch_model, cex, len(self.network.layers), grad=False)
                _, idx = torch.max(original_out.squeeze(0), 0)
                error = nn.CrossEntropyLoss()(torch.unsqueeze(current_out, 0),
                                              torch.tensor([idx], requires_grad=False))
                if min_error is None or min_error > error:
                    min_error = error
                    best_neuron = neuron
                    best_layer = layer_idx
                self.network.delete_neuron(layer_idx, neuron)
                # Reset current layer
                self.network.layers[layer_idx].set_weight(layer_weight)
                self.network.layers[layer_idx].set_bias(layer_bias)
                self.network.layers[layer_idx].active_neurons = layer_neurons
                # Reset succ layer
                self.network.layers[layer_idx + 1].set_weight(succ_layer_weight)
                self.network.layers[layer_idx + 1].active_inputs = succ_layer_inputs
                succ_layer_weight = self.network.layers[layer_idx + 1].get_weight().detach().clone()
            self.network.layers[layer_idx].change_matrix = layer_change
            self.network.layers[layer_idx + 1].change_matrix = succ_layer_change
            if num_param != sum(p.numel() for p in self.network.torch_model.parameters()):
                raise ValueError("Error on number of parameters")
            if not torch.equal(out, self.network.forward(cex)):
                raise ValueError("Error on output of counterexample")

        return best_layer, best_neuron
