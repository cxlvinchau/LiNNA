import abc
from typing import Dict, Optional, Tuple, List
import numpy as np
from linna.network import Network
import torch
from linna.utils import forward


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


