from typing import Literal

import numpy as np

from linna.basis_finder import BasisFinder, GreedyBasisFinder, VarianceBasisFinder
from linna.coef_finder import CoefFinder, L1CoefFinder, L2CoefFinder
from linna.network import Network

from torch.utils.data.dataloader import DataLoader

BASIS_FINDER = Literal["greedy", "variance"]
COEF_FINDER = Literal["l1", "l2"]


class Abstraction:

    def __init__(self, network: Network, basis_finder: BASIS_FINDER, coef_finder: COEF_FINDER, loader: DataLoader,
                 size: int = 1000):
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

        """
        self.network = network
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
            self.coef_finder = L1CoefFinder(network=network, io_dict=self.io_dict)
        elif coef_finder == "l2":
            self.coef_finder = L2CoefFinder(network=network, io_dict=self.io_dict)
        else:
            raise ValueError(f"Invalid coef finder {str(coef_finder)}")

    def remove_neurons(self, layer_idx: int):
        """
        Removes all neurons in a layer that are not in the basis

        Parameters
        ----------
        layer_idx: int
            Layer in which neurons are removed

        -------

        """
