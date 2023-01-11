import torch

from linna.coef_finder import L1CoefFinder, L2CoefFinder
from linna.refinement import DifferenceRefinement, LookaheadRefinement
from tests.toy_network import create_toy_network

import pytest


@pytest.fixture
def loader():
    return [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]


@pytest.fixture
def network_and_output(loader):
    network = create_toy_network()
    original_out = network.forward(loader[0][0])
    network.set_basis(layer_idx=0, basis=[0, 2])
    network.delete_neuron(0, 1)
    network.readjust_weights(layer_idx=0, neuron=1, coef=torch.Tensor([1, 2]))
    return network, original_out


@pytest.fixture
def io_dict(network_and_output, loader):
    network, _ = network_and_output
    io_dict = dict()
    for layer in range(len(network.layers)):
        io_dict[layer] = network.get_io_matrix(loader=loader, layer_idx=layer, size=4)
    return io_dict


class TestRefinement:

    def test_difference_refinement(self, network_and_output, io_dict, loader):
        network, expected_output = network_and_output
        strategy = DifferenceRefinement(network=network, io_dict=io_dict)
        cex = torch.Tensor([1., 1.])
        layer, neuron = strategy.find_neuron(cex, layers=[0, 1])
        network.restore_neuron(layer_idx=layer, neuron=neuron)

        assert torch.all(torch.isclose(network.forward(loader[0][0]), expected_output))

    def test_lookahead_refinement(self, network_and_output, io_dict, loader):
        network, expected_output = network_and_output
        strategy = LookaheadRefinement(network=network, io_dict=io_dict)
        cex = torch.Tensor([1., 1.])
        layer, neuron = strategy.find_neuron(cex, layers=[0, 1])
        network.restore_neuron(layer_idx=layer, neuron=neuron)

        assert torch.all(torch.isclose(network.forward(loader[0][0]), expected_output))
