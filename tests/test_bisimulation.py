import pytest
import torch
import numpy as np

from linna.bisimulation import Bisimulation
from toy_network import create_toy_network


@pytest.fixture
def network():
    return create_toy_network()


@pytest.fixture
def loader():
    return [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]


class TestBisimulation:
    def test_bisimulation(self, network, loader):
        io_dict = dict()

        # Compute IO matrices
        for layer in range(len(network.layers)):
            io_dict[layer]: np.ndarray = network.get_io_matrix(loader=loader, layer_idx=layer, size=4)

        bisim = Bisimulation(network=network, io_dict=io_dict)

        bisim.process_all_layers(delta=10)


