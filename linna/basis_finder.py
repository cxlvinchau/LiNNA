import abc
import random
from typing import List, Dict
from pydantic import confloat
import numpy as np
import warnings

import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.preprocessing import normalize
import scipy
from math import ceil, floor

from numpy.linalg import matrix_rank

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

    @abc.abstractmethod
    def find_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), **kwargs) -> List[List[int]]:
        """
        Finds the basis for all layers

        Parameters
        ----------
        reduction_rate: float
            Reduction rate

        Returns
        -------
        List[List[int]]
            Bases for all layers

        """
        pass


class VarianceBasisFinder(_BasisFinder):

    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        io_matrix: np.ndarray = self.io_dict[layer_idx]
        variance = np.var(io_matrix, axis=0)
        basis: np.ndarray = variance.argsort()[-basis_size:]
        return basis.tolist()

    def find_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), normalized=True, **kwargs) -> List[List[int]]:
        variances = []
        for layer_idx in range(len(self.network.layers) - 1):
            io_matrix: np.ndarray = self.io_dict[layer_idx]
            variance = np.var(io_matrix, axis=0)
            if normalized:
                variance = normalize(variance.reshape(1, -1))[0]
            layer_indices = np.ones_like(variance, dtype=int) * layer_idx
            neuron_indices = np.arange(0, len(variance))
            variances.extend(list(zip(variance, layer_indices, neuron_indices)))
        variances.sort()
        num_neurons = ceil(len(variances) * (1 - reduction_rate))
        selected_neurons = variances[-num_neurons:]
        bases = []
        for layer_idx in range(len(self.network.layers) - 1):
            basis = [el[2] for el in selected_neurons if el[1] == layer_idx]
            if len(basis) == 0:
                basis = [max([el[2] for el in variances if el[1] == layer_idx])]
            bases.append(basis)
        return bases


class GreedyBasisFinder(_BasisFinder):

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None, seed: int = 42):
        super().__init__(network, io_dict)
        self.rng = np.random.default_rng(seed)

    def _compute_projection_error(self, layer_idx: int, basis: List[int]):
        io_matrix: np.ndarray = self.io_dict[layer_idx]
        try:
            A = io_matrix[:, basis]
            X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, io_matrix))
            # Compute projection error
            error = np.sum((io_matrix - np.matmul(A, X)) ** 2)
            return error
        except:
            return None

    def find_basis(self, layer_idx: int, basis_size: int, random_choice=False, **parameters) -> List[int]:
        candidates = list(self.network.layers[layer_idx].neurons)
        basis = []
        for _ in range(basis_size):
            if random_choice:
                candidates = self.rng.choice(candidates, min(len(candidates), 100), replace=False)
            min_error, best_neuron = float("inf"), None
            for neuron in candidates:
                error = self._compute_projection_error(layer_idx=layer_idx, basis=basis + [neuron])
                if error is not None and error < min_error:
                    min_error = error
                    best_neuron = neuron
            if best_neuron is None:
                break
            basis.append(best_neuron)
            candidates.remove(best_neuron)
        return basis

    def find_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), random_choice=False, **kwargs) -> List[List[int]]:
        num_to_add = ceil(self.network.get_num_neurons() * (1 - reduction_rate))
        # Ensure that all bases have at least one neuron
        bases = [self.find_basis(layer_idx, basis_size=1) for layer_idx in range(len(self.network.layers) - 1)]

        # Iterate through all layers and neurons to determine best candidate
        for _ in range(num_to_add - len(self.network.layers) - 1):
            min_error, best_neuron, best_layer = float("inf"), None, None
            for layer_idx in range(len(self.network.layers) - 1):
                candidates = [neuron for neuron in self.network.layers[layer_idx].neurons if neuron not in bases[layer_idx]]
                if random_choice:
                    candidates = self.rng.choice(candidates, min(len(candidates), 100), replace=False)
                for neuron in candidates:
                    error = self._compute_projection_error(layer_idx=layer_idx, basis=bases[layer_idx] + [neuron])
                    if error is not None and error * len(bases[layer_idx]) < min_error:
                        min_error = error * len(bases[layer_idx])
                        best_neuron = neuron
                        best_layer = layer_idx
            if best_neuron is None:
                return bases
            bases[best_layer].append(best_neuron)
        return bases


class GreedyPruningBasisFinder(_BasisFinder):

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None, seed: int = 42):
        super().__init__(network, io_dict)
        self.rng = np.random.default_rng(seed)

    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        io_matrix: np.ndarray = self.io_dict[layer_idx]
        n = self.network.layers[layer_idx].get_weight().size(dim=0)
        basis = list(self.network.layers[layer_idx].neurons)
        min_error, best_neuron = None, None
        for _ in range(n - basis_size):
            candidates = [i for i in range(n) if i in basis]
            for neuron in candidates:
                tmp_basis = [n for n in basis if n != neuron]
                try:
                    A = io_matrix[:, tmp_basis]
                    X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, io_matrix))
                    # Compute projection error
                    error = np.sum((io_matrix - np.matmul(A, X)) ** 2)
                    if min_error is None or error < min_error:
                        min_error, best_neuron = error, neuron
                except:
                    best_neuron = neuron
            if best_neuron is None:
                break
            basis.remove(best_neuron)
            min_error, best_neuron = None, None
        return basis

    def find_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), random_choice=False, **kwargs) -> List[List[int]]:
        num_to_delete = ceil(self.network.get_num_neurons() * reduction_rate)
        bases = [list(range(self.network.layers[layer_idx].get_weight().size(dim=0))) for layer_idx in
                 range(len(self.network.layers) - 1)]

        deleted = 0
        while (deleted < num_to_delete):
            min_error, best_neuron, best_layer = None, None, None
            for layer_idx in range(len(self.network.layers) - 1):
                io_matrix: np.ndarray = self.io_dict[layer_idx]
                basis = bases[layer_idx]
                candidates = [i for i in basis]
                if random_choice:
                    candidates = self.rng.choice(candidates, min(len(candidates), 100), replace=False)
                for neuron in candidates:
                    if neuron not in bases[layer_idx]:
                        continue
                    tmp_basis = [i for i in bases[layer_idx]]
                    if len(tmp_basis) <= 1:
                        continue
                    tmp_basis.remove(neuron)
                    try:
                        A = io_matrix[:, tmp_basis]
                        X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, io_matrix))
                        # Compute projection error
                        error = np.sum((io_matrix - np.matmul(A, X)) ** 2)
                        if min_error is None or error < min_error:
                            min_error, best_neuron, best_layer = error, neuron, layer_idx
                    except np.linalg.LinAlgError:
                        # Columns a linearly dependent
                        q, r = scipy.linalg.qr(io_matrix)
                        # Iterate through diagonal to determine linearly
                        # fixme: numerical stability is currently hardcoded
                        r = np.abs(r) > 1e-8
                        independent_cols = []
                        i, j = 0, 0
                        while i < r.shape[0] and j < r.shape[1]:
                            if r[i][j]:
                                independent_cols.append(j)
                                i += 1
                                j += 1
                            else:
                                j += 1

                        deleted += len(bases[layer_idx]) - len(list(independent_cols))
                        bases[layer_idx] = list(independent_cols)
            if best_layer is None:
                break
            bases[best_layer].remove(best_neuron)
            deleted += 1
        return bases


class ClusteringBasisFinder(_BasisFinder):

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None,
                 clustering: str = "kmeans", **parameters):
        super().__init__(network=network, io_dict=io_dict, **parameters)
        self.clustering = clustering

    def _find_basis_dbscan(self, layer_idx: int, basis_size: int, **parameters):
        io_matrix = self.io_dict[layer_idx]
        num_nodes = np.shape(io_matrix)[1]
        db = DBSCAN(min_samples=2, eps=num_nodes - basis_size).fit(io_matrix.T)
        labels = np.array(db.labels_)
        cluster_centers = []
        for i in np.unique(labels):
            if i > -1:
                cluster_centers.append(np.squeeze(np.mean(io_matrix[:, np.where(labels == i)], axis=2)))
        return cluster_centers, labels

    def _find_basis_kmeans(self, layer_idx: int, basis_size: int, **parameters):
        io_matrix = self.io_dict[layer_idx]
        km = KMeans(n_clusters=basis_size)
        # The warning has to be fetched, such that the correct number of nodes can be propagated
        with warnings.catch_warnings(record=True) as w:
            km = km.fit(io_matrix.T)
        labels = km.labels_
        cluster_centers = km.cluster_centers_
        return cluster_centers, labels

    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        cluster_centers, labels = None, None
        io_matrix = self.io_dict[layer_idx]
        if self.clustering == "kmeans":
            cluster_centers, labels = self._find_basis_kmeans(layer_idx=layer_idx, basis_size=basis_size, **parameters)
        elif self.clustering == "dbscan":
            cluster_centers, labels = self._find_basis_kmeans(layer_idx=layer_idx, basis_size=basis_size, **parameters)
        else:
            raise ValueError(f"Unsupported clustering algorithm {self.clustering}")

        basis = []
        for i in np.unique(labels):
            cluster = np.where(labels == i)
            if i == -1:
                for node in cluster[0]:
                    basis.append(np.asscalar(node))
            elif len(cluster[0]) > 1:
                myc = cluster_centers[i]
                myc = myc.reshape(1, np.shape(myc)[0])
                closest, _ = pairwise_distances_argmin_min(myc, io_matrix[:, cluster[0]].transpose())
                best_node = cluster[0][closest[0]]
                basis.append(best_node)
            else:
                basis.append(cluster[0].item())
        basis.sort()
        return basis

    def find_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), **kwargs) -> List[List[int]]:
        bases = []
        for layer_idx in range(len(self.network.layers) - 1):
            basis_size = floor(self.network.layers[layer_idx].get_weight().size(dim=0) * (1 - reduction_rate))
            basis = self.find_basis(layer_idx, basis_size=basis_size)
            bases.append(basis)
        return bases


class RandomBasisFinder(_BasisFinder):

    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        return random.sample(self.network.layers[layer_idx].neurons, k=basis_size)

    def find_bases(self, reduction_rate: confloat(ge=0.0, le=1.0), **kwargs) -> List[List[int]]:
        bases = []
        for layer_idx in range(len(self.network.layers) - 1):
            basis_size = floor(self.network.layers[layer_idx].get_weight().size(dim=0) * (1 - reduction_rate))
            basis = self.find_basis(layer_idx, basis_size=basis_size)
            bases.append(basis)
        return bases
