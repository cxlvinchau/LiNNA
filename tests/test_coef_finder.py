import torch

from linna.coef_finder import L1CoefFinder, L2CoefFinder
from tests.toy_network import create_toy_network

import pytest


@pytest.fixture
def network():
    network = create_toy_network()
    network.set_basis(layer_idx=0, basis=[0, 2])
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


class TestCoefFinder:

    @pytest.mark.parametrize("solver", ["gurobi", "scipy"])
    def test_l1(self, solver, network, io_dict):
        coef_finder = L1CoefFinder(network=network, io_dict=io_dict, params={"solver": solver})

        # Assert neurons in basis are assigned unit vector
        for v, neuron in zip([(1., 0.), (0., 1.)], [0, 2]):
            c = coef_finder.find_coefficients(layer_idx=0, neuron=neuron)
            assert torch.all(torch.isclose(c, torch.tensor(v)))

    def test_l2(self, network, io_dict):
        coef_finder = L2CoefFinder(network=network, io_dict=io_dict)

        # Assert neurons in basis are assigned unit vector
        for v, neuron in zip([(1., 0.), (0., 1.)], [0, 2]):
            c = coef_finder.find_coefficients(layer_idx=0, neuron=neuron)
            assert torch.all(torch.isclose(c, torch.tensor(v)))
