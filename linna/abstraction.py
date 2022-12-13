from typing import Literal, Dict, Any, Optional

import numpy as np
import torch

from linna.basis_finder import GreedyBasisFinder, VarianceBasisFinder
from linna.coef_finder import L1CoefFinder, L2CoefFinder
from linna.network import Network

from torch.utils.data.dataloader import DataLoader

BASIS_FINDER = Literal["greedy", "variance"]
COEF_FINDER = Literal["l1", "l2"]


class Abstraction:

    def __init__(self, network: Network, basis_finder: BASIS_FINDER, coef_finder: COEF_FINDER, loader: DataLoader,
                 size: int = 1000, coef_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the abstraction object

        Parameters
        ----------
        network: Network
            The network to be abstracted and refined
        basis_finder: str
            The basis finder that should be used, possible values are ``greedy`` or ``variance``
        coef_finder: str
            The coef finder that should be used, possible values are ``l1`` or ``l2``
        loader: DataLoader
            The data loader for the data that is used for computing the IO matrices
        size: int
            The size of the IO matrices, i.e. the number of considered inputs
        coef_params: Optional[Dict[str, Any]]
            Optional parameters given to the coefficient finder, e.g. for specifying the LP solver

        """
        self.network = network
        self.original_number_of_neurons = self.network.get_num_neurons()
        self.io_dict = dict()

        # Compute IO matrices
        for layer in range(len(self.network.layers)):
            self.io_dict[layer]: np.ndarray = self.network.get_io_matrix(loader=loader, layer_idx=layer, size=size)

        # Initialize basis finder
        if basis_finder == "greedy":
            self.basis_finder = GreedyBasisFinder(network=network, io_dict=self.io_dict)
        elif basis_finder == "variance":
            self.basis_finder = VarianceBasisFinder(network=network, io_dict=self.io_dict)
        else:
            raise ValueError(f"Invalid basis finder {str(basis_finder)}")

        # Initialize coef finder
        if coef_finder == "l1":
            self.coef_finder = L1CoefFinder(network=network, io_dict=self.io_dict, params={"solver": "scipy"})
        elif coef_finder == "l2":
            self.coef_finder = L2CoefFinder(network=network, io_dict=self.io_dict)
        else:
            raise ValueError(f"Invalid coef finder {str(coef_finder)}")

    def get_reduction_rate(self):
        return 1 - (self.network.get_num_neurons()/self.original_number_of_neurons)

    def determine_basis(self, layer_idx: int, basis_size: int = 1000):
        """
        Determines the basis for the given layer

        Parameters
        ----------
        layer_idx: int
            Layer
        basis_size: int
            Size of the basis

        """
        basis = self.basis_finder.find_basis(layer_idx=layer_idx, basis_size=basis_size)
        self.network.set_basis(layer_idx, basis)

    def abstract(self, layer_idx: int, **coef_params):
        """
        Removes all neurons in a layer that are not in the basis

        Parameters
        ----------
        layer_idx: int
            Layer in which neurons are removed

        """
        basis = self.network.layers[layer_idx].basis
        if basis is None:
            raise ValueError(f"Basis has not been determined for {layer_idx} yet")
        non_basic = [i for i in self.network.layers[layer_idx].active_neurons if i not in basis]
        coefs = self.coef_finder.find_coefficients_layer(layer_idx=layer_idx, **coef_params)
        for neuron in non_basic:
            self.network.delete_neuron(layer_idx=layer_idx, neuron=neuron)
            self.network.readjust_weights(layer_idx=layer_idx, neuron=neuron, coef=coefs[neuron])

    def refine(self, cex: torch.Tensor):
        """
        Refine the abstraction such that the classification of the given counterexample coincides with the
        classification of the original network

        Parameters
        ----------
        cex: torch.Tensor

        """
        pass
