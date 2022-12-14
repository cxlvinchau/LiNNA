import abc
import random
from typing import List, Dict
import numpy as np
import warnings

import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

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
            # print("Check for i: ",i)
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


class RandomBasisFinder(_BasisFinder):

    def find_basis(self, layer_idx: int, basis_size: int, **parameters) -> List[int]:
        return random.sample(self.network.layers[layer_idx].neurons, k=basis_size)
