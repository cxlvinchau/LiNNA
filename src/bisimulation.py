from typing import Dict

import torch

from src.network import Network
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans


def iterate_kmeans(data, delta):
    n_clusters = 0
    max_diam = None
    while (max_diam is None or max_diam > delta) and n_clusters <= np.shape(data)[0]:
        imax_diam = 0
        n_clusters += 1
        km = KMeans(n_clusters=n_clusters)
        labels = km.labels_
        for i in range(np.unique(labels)):
            cluster = np.where(labels == i)
            diam = np.max(pairwise_distances(data[cluster[0], :]))
            if diam > imax_diam:
                imax_diam = diam
        max_diam = imax_diam
    return labels


def get_distances(data, bias):
    distances = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            d = np.max(np.abs(data[:, i] - data[:, j]))
            d = np.max([np.abs(bias[i] - bias[j]), d])
            distances[i, j] = d
            distances[j, i] = d
    return distances


class Bisimulation:
    """
    Our implementation of the bisimulation for neural networks proposed by Pavithra Prabhakar

    Paper: https://link.springer.com/chapter/10.1007/978-3-030-94583-1_14
    arXiv: https://arxiv.org/abs/2110.03726
    """

    def __init__(self, network: Network, io_dict: Dict[int, torch.Tensor]):
        self.network = network
        self.io_dict = io_dict

    def process_all_layers(self, delta=1e-16):
        # delta is not 0, because agglomerative clustering only works with "smaller than"-distance, not "smaller or
        # equal"
        layers = self.network.layers
        num_neurons_before = np.sum([len(layers[i].active_neurons) for i in range(len(layers) - 1)])
        self.find_basis_for_all_layers(delta)
        num_neurons_after = np.sum([len(layers[i].active_neurons) for i in range(len(layers) - 1)])
        return (num_neurons_before - num_neurons_after) / num_neurons_before

    def find_basis_for_all_layers(self, delta):
        for layer_idx in range(len(self.network.layers) - 1):
            self.process_layer(layer_idx, delta)

    def evaluate_clustering(self, labels, io, cluster_centers, layer_idx, delta=0):
        # Evaluate the clustering
        layers = self.network.layers
        current_basis = [i for i in list(layers[layer_idx].active_neurons)]
        temp_basis = []
        lin_comb = lil_matrix((len(layers[layer_idx].neurons), len(layers[layer_idx].neurons)))
        for i in np.unique(labels):
            cluster = np.where(labels == i)
            if len(cluster[0]) > 1:
                myc = cluster_centers[i]
                myc = myc.reshape(1, np.shape(myc)[0])
                closest, _ = pairwise_distances_argmin_min(myc, io[:, cluster[0]].transpose())
                best_node = cluster[0][closest[0]]
                temp_basis.append(current_basis[best_node])
                for node in cluster[0]:
                    lin_comb[current_basis[best_node], current_basis[node]] = 1
            else:
                lin_comb[current_basis[cluster[0].item()], current_basis[cluster[0].item()]] = 1
                temp_basis.append(current_basis[cluster[0].item()])
        lin_comb = lin_comb[lin_comb.getnnz(1) > 0]
        temp_basis.sort()
        return temp_basis, lin_comb

    def process_layer(self, layer_idx, delta):
        io = self.io_dict[layer_idx]
        bias = self.network.layers[layer_idx].get_bias().cpu().detach().numpy()
        num_nodes = np.shape(io)[1]
        pd = pairwise_distances(io.T)
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=delta, affinity='precomputed',
                                     linkage='complete')
        distances = get_distances(io, bias)
        ac.fit(distances)
        labels = ac.labels_
        cluster_centers = []
        for i in np.unique(labels):
            cluster_centers.append(np.squeeze(np.mean(io[:, np.where(labels == i)], axis=2)))
        temp_basis, lin_comb = self.evaluate_clustering(labels, io, cluster_centers, layer_idx, delta)
        removed_neurons = []
        for neuron in list(self.network.layers[layer_idx].active_neurons):
            if neuron in temp_basis:
                continue
            self.network.delete_neuron(layer_idx, neuron)
            removed_neurons.append(neuron)
        self.network.set_basis(layer_idx, temp_basis)
        for neuron in removed_neurons:
            self.network.readjust_weights(layer_idx=layer_idx, neuron=neuron, coef=torch.Tensor(np.squeeze(lin_comb[:, neuron].toarray())))