from typing import Dict, Any

import numpy as np

from linna.abstraction import Abstraction
from linna.bisimulation import Bisimulation
from linna.network import Network

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import pandas as pd

from torch.utils.data.dataset import Dataset

import logging

from timeit import default_timer as timer

from linna.utils import get_accuracy, load_tf_network


def run_bisimulation(testset: Dataset, network_path: str):
    """
    Runs the reduction of the network with the bisimulation proposed by Pavithra Prabhakar. Note that the bisimulation
    only uses syntactic information, that is only the weights and biases.

    Paper: https://link.springer.com/chapter/10.1007/978-3-030-94583-1_14
    arXiv: https://arxiv.org/abs/2110.03726

    Parameters
    ----------
    testset: Dataset
        Test data
    network_path

    Returns
    -------

    """
    rows = []
    for delta in [i * 0.05 for i in range(1, 25)]:
        # Load trained neural network
        sequential = load_tf_network(file=network_path)
        network = Network(torch_model=sequential)
        # Compute IO matrices
        io_dict = dict()

        # Set up loaders
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        for layer_idx in range(len(network.layers)):
            io_dict[layer_idx]: np.ndarray = network.layers[layer_idx].get_weight().cpu().detach().numpy().T
        bisim = Bisimulation(network=network, io_dict=io_dict)
        original_num_neurons = bisim.network.get_num_neurons()
        bisim.process_all_layers(delta=delta)
        row = pd.Series({"reduction_rate": 1 - (bisim.network.get_num_neurons() / original_num_neurons),
                         "accuracy": get_accuracy(testloader, bisim.network.torch_model)})
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates()


def run_reduction_experiment(network: str, trainset: Dataset, testset: Dataset,
                             basis_finder: str, coef_finder: str, coef_params: Dict[str, Any] = None, resolution=10):
    """
    Runs the reduction experiment

    Parameters
    ----------
    network: str
        Path to network
    trainset: Dataset
        Training data
    testset: Dataset
        Test data
    basis_finder: str
        Basis finder
    coef_finder: str
        Coefficient finder
    coef_params: Dict[str, Any]
        Parameters passed to the coefficient finder
    resolution: int
        Number of reduction rates to consider

    Returns
    -------
    pd.DataFrame
        Dataframe containing experiment results

    """
    transform = transforms.Compose([transforms.ToTensor()])
    # Create train and test loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Load trained neural network
    sequential = load_tf_network(file=network)
    network = Network(torch_model=sequential)

    rows = []

    # Compute different abstractions
    for rr in np.linspace(0, 1, num=resolution + 2)[1:-1]:
        start = timer()
        abstraction = Abstraction(network=network,
                                  basis_finder=basis_finder,
                                  coef_finder=coef_finder,
                                  coef_params=coef_params,
                                  loader=trainloader)
        for layer_idx in range(len(abstraction.network.layers) - 1):
            basis_size = int(len(abstraction.network.layers[layer_idx].neurons) * rr)
            abstraction.determine_basis(layer_idx=layer_idx, basis_size=basis_size)
        for layer_idx in range(len(abstraction.network.layers) - 1):
            abstraction.abstract(layer_idx=layer_idx)
        end = timer()
        row = pd.Series({"reduction_rate": abstraction.get_reduction_rate(),
                         "accuracy": get_accuracy(testloader, abstraction.network.torch_model),
                         "duration": end - start})
        rows.append(row)
        abstraction.network.reset()

    df = pd.DataFrame(rows)
    return df
