import abc
from typing import List, Dict
import numpy as np

import torch

from linna.network import Network


class _BasisFinder(abc.ABC):
    """Basis finder base class"""

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None, **parameters):
        self.network = network
        self.parameters = parameters
        self.io_dict = io_dict

    @abc.abstractmethod
    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        """
        Finds the basis for the given layer

        Parameters
        ----------
        layer_idx: int
            Layer
        basis_size: int
            Size of the basis

        Returns
        -------
        List[int]
            Basis for layer

        """
        pass


class VarianceBasisFinder(_BasisFinder):

    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        io_matrix: np.ndarray = self.io_dict[layer_idx]
        variance = np.var(io_matrix, axis=0)
        basis: np.ndarray = variance.argsort()[-basis_size:]
        return basis.tolist()


class GreedyBasisFinder(_BasisFinder):

    def find_basis(self, layer_idx: int, basis_size: int, random_choice=False, **parameters) -> List[int]:
        io_matrix: np.ndarray = self.io_dict[layer_idx]
        n = self.network.layers[layer_idx].get_weight().size(dim=0)
        basis = []
        min_error, best_neuron = None, None
        for _ in range(basis_size):
            candidates = [i for i in range(n) if i not in basis]
            if random_choice:
                candidates = self.rng.choice(candidates, min(len(candidates), 100), replace=False)
            for neuron in candidates:
                tmp_basis = basis + [neuron]
                try:
                    A = io_matrix[:, tmp_basis]
                    X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, io_matrix))
                    # Compute projection error
                    error = np.sum((io_matrix - np.matmul(A, X)) ** 2)
                    if min_error is None or error < min_error:
                        min_error, best_neuron = error, neuron
                except:
                    continue
            if best_neuron is None:
                break
            basis.append(best_neuron)
        return basis
