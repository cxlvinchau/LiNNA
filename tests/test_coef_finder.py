import unittest

import torch

from linna.coef_finder import L1CoefFinder, L2CoefFinder
from tests.toy_network import create_toy_network


class TestCoefFinder(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_toy_network()
        self.network.set_basis(layer_idx=0, basis=[0, 2])
        loader = [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]
        self.io_dict = dict()
        for layer in range(len(self.network.layers)):
            self.io_dict[layer] = self.network.get_io_matrix(loader=loader, layer_idx=layer, size=4)

    def test_l1(self):
        coef_finder = L1CoefFinder(network=self.network, io_dict=self.io_dict)

        # Assert neurons in basis are assigned unit vector
        for v, neuron in zip([(1., 0.), (0., 1.)], [0, 2]):
            c = coef_finder.find_coefficients(layer_idx=0, neuron=neuron)
            assert torch.all(torch.isclose(c, torch.tensor(v)))

    def test_l2(self):
        coef_finder = L2CoefFinder(network=self.network, io_dict=self.io_dict)

        # Assert neurons in basis are assigned unit vector
        for v, neuron in zip([(1., 0.), (0., 1.)], [0, 2]):
            c = coef_finder.find_coefficients(layer_idx=0, neuron=neuron)
            assert torch.all(torch.isclose(c, torch.tensor(v)))


if __name__ == '__main__':
    unittest.main()
