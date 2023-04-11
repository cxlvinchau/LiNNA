from typing import Literal, Dict, Any, Optional
from pydantic import confloat

import numpy as np
import torch
from math import ceil

from linna.basis_finder import GreedyBasisFinder, VarianceBasisFinder, ClusteringBasisFinder, RandomBasisFinder, \
    GreedyPruningBasisFinder
from linna.coef_finder import L1CoefFinder, L2CoefFinder, ClusteringCoefFinder, DummyCoefFinder
from linna.network import Network

from torch.utils.data.dataloader import DataLoader

BASIS_FINDER = Literal["greedy", "greedy_pruning", "variance", "kmeans", "dbscan", "random"]
COEF_FINDER = Literal["l1", "l2", "kmeans", "clustering", "dummy"]


class Abstraction:

    def __init__(self, network: Network, basis_finder: BASIS_FINDER, coef_finder: COEF_FINDER, syntactic=False, loader: DataLoader = None,
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
        io_dict: Dict
            A dictionary mapping layer indices to matrices
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
        if not syntactic:
            # If not syntactic then compute IO matrices
            assert loader is not None, "If io_dict is not provided, a loader has to be specified"
            for layer_idx in range(len(self.network.layers)):
                self.io_dict[layer_idx]: np.ndarray = self.network.get_io_matrix(loader=loader, layer_idx=layer_idx, size=size)
        else:
            # Syntactic, then use weight matrices
            for layer_idx, layer in enumerate(self.network.layers):
                self.io_dict[layer_idx]: np.ndarray = layer.get_weight().cpu().detach().numpy().T


        # Initialize basis finder
        if basis_finder == "greedy":
            self.basis_finder = GreedyBasisFinder(network=network, io_dict=self.io_dict)
        elif basis_finder == "greedy_pruning":
            self.basis_finder = GreedyPruningBasisFinder(network=network, io_dict=self.io_dict)
        elif basis_finder == "variance":
            self.basis_finder = VarianceBasisFinder(network=network, io_dict=self.io_dict)
        elif basis_finder == "kmeans":
            self.basis_finder = ClusteringBasisFinder(network=network, io_dict=self.io_dict, clustering="kmeans")
        elif basis_finder == "dbscan":
            self.basis_finder = ClusteringBasisFinder(network=network, io_dict=self.io_dict, clustering="dbscan")
        elif basis_finder == "random":
            self.basis_finder = RandomBasisFinder(network=network, io_dict=self.io_dict)
        else:
            raise ValueError(f"Invalid basis finder {str(basis_finder)}")

        # Initialize coef finder
        if coef_finder == "l1":
            self.coef_finder = L1CoefFinder(network=network, io_dict=self.io_dict, params=coef_params)
        elif coef_finder == "l2":
            self.coef_finder = L2CoefFinder(network=network, io_dict=self.io_dict)
        elif coef_finder == "clustering":
            self.coef_finder = ClusteringCoefFinder(network=network, io_dict=self.io_dict)
        elif coef_finder == "dummy":
            self.coef_finder = DummyCoefFinder(network=network, io_dict=self.io_dict)
        else:
            raise ValueError(f"Invalid coef finder {str(coef_finder)}")

    def get_reduction_rate(self):
        return 1 - (self.network.get_num_neurons()/self.original_number_of_neurons)

    def determine_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), **kwargs):
        assert(0 <= reduction_rate <= 1)
        bases = self.basis_finder.find_bases(reduction_rate=reduction_rate, **kwargs)
        if bases is None:
            return
        for layer_idx in range(len(self.network.layers)-1):
            self.network.set_basis(layer_idx, bases[layer_idx])

    def determine_basis_rr(self, layer_idx: int, reduction_rate: float):
        n = len(self.network.layers[layer_idx].active_neurons)
        basis_size = ceil(n*reduction_rate)
        self.determine_basis(layer_idx, basis_size)

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


    def abstract_all(self, **coef_params):
        for layer_idx in range(len(self.network.layers)-1):
            self.abstract(layer_idx, **coef_params)

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
        coefs = self.coef_finder.find_all_coefficients(layer_idx=layer_idx)
        for neuron in non_basic:
            self.network.delete_neuron(layer_idx=layer_idx, neuron=neuron)
            self.network.readjust_weights(layer_idx=layer_idx, neuron=neuron, coef=coefs[neuron])
        self.network.update_torch_model()

    def refine(self, cex: torch.Tensor):
        """
        Refine the abstraction such that the classification of the given counterexample coincides with the
        classification of the original network

        Parameters
        ----------
        cex: torch.Tensor

        """
        pass
