import unittest

import torch

from linna.basis_finder import VarianceBasisFinder, GreedyBasisFinder, ClusteringBasisFinder, GreedyPruningBasisFinder
from tests.toy_network import create_toy_network


class TestBasisFinder(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_toy_network()
        self.network.set_basis(layer_idx=0, basis=[0, 2])
        loader = [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]
        self.io_dict = dict()
        for layer in range(len(self.network.layers)):
            self.io_dict[layer] = self.network.get_io_matrix(loader=loader, layer_idx=layer, size=4)

    def test_variance_basis_finder(self):
        finder = VarianceBasisFinder(network=self.network, io_dict=self.io_dict)
        basis = finder.find_basis(layer_idx=0, basis_size=2)
        self.assertEqual([0, 2], basis)

    def test_variance_basis_finder_all(self):
        finder = VarianceBasisFinder(network=self.network, io_dict=self.io_dict)
        bases = finder.find_bases(reduction_rate=0.5)
        self.assertEqual([[2],[0, 2]], bases)

    def test_greedy_basis_finder(self):
        finder = GreedyBasisFinder(network=self.network, io_dict=self.io_dict)
        basis = finder.find_basis(layer_idx=0, basis_size=2)
        self.assertEqual( [2, 0], basis)

    def test_greedy_basis_finder_all(self):
        finder = GreedyBasisFinder(network=self.network, io_dict=self.io_dict)
        bases = finder.find_bases(reduction_rate=0.5)
        self.assertEqual([[0, 2], [2]], bases)

    def test_greedy_pruning_basis_finder(self):
        finder = GreedyPruningBasisFinder(network=self.network, io_dict=self.io_dict)
        basis = finder.find_basis(layer_idx=0, basis_size=2)
        self.assertEqual([0, 2], basis)

    def test_greedy_pruning_basis_finder_all(self):
        finder = GreedyPruningBasisFinder(network=self.network, io_dict=self.io_dict)
        bases = finder.find_bases(reduction_rate=0.5)
        self.assertEqual( [[0, 2], [2]], bases)

    def test_clustering_basis_finder(self):
        finder = ClusteringBasisFinder(network=self.network, io_dict=self.io_dict)
        finder.find_basis(layer_idx=0, basis_size=2)


if __name__ == '__main__':
    unittest.main()
