import torch

from linna.coef_finder import L1CoefFinder, L2CoefFinder
from linna.refinement import DifferenceRefinement
from tests.toy_network import create_toy_network

import pytest


@pytest.fixture
def network():
    network = create_toy_network()
    network.set_basis(layer_idx=0, basis=[0, 2])
    network.delete_neuron(0, 1)
    network.readjust_weights(layer_idx=0, neuron=1, coef=torch.Tensor([1, 2]))
    return network


@pytest.fixture
def loader():
    return [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]


@pytest.fixture
def io_dict(network, loader):
    io_dict = dict()
    for layer in range(len(network.layers)):
        io_dict[layer] = network.get_io_matrix(loader=loader, layer_idx=layer, size=4)
    return io_dict


class TestRefinement:

    def test_difference_refinement(self, network, io_dict):
        strategy = DifferenceRefinement(network=network, io_dict=io_dict)
        cex = torch.Tensor([1., 1.])
        neuron, layer = strategy.find_neuron(cex, layers=[0, 1])
