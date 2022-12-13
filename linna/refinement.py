import abc
from typing import Dict, Optional, Tuple
import numpy as np
from linna.network import Network
import torch


class AbstractionRefinement(abc.ABC):

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None, **parameters):
        self.network = network
        self.parameters = parameters
        self.io_dict = io_dict

    @abc.abstractmethod
    def find_neuron(self, cex: torch.Tensor, layer_idx: Optional[int] = None) -> Tuple[int, int]:
        """
        Returns the neuron that should be restored. If the layer is not specified, the method considers all layers
        for refinement and returns the best option.

        Parameters
        ----------
        cex: torch.Tensor
            Counterexample
        layer_idx: Optional[int]
            Layer of network

        Returns
        -------
        int
            Layer
        int
            Neuron in the layer

        """
        pass


class LookaheadRefinement(AbstractionRefinement):

    def find_neuron(self, layer_idx: Optional[int] = None) -> Tuple[int, int]:
        pass