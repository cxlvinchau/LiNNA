from typing import Dict, Any

import numpy as np

from linna.abstraction import Abstraction
from linna.network import Network
from linna.utils import load_tf_network, get_accuracy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import pandas as pd

from torch.utils.data.dataset import Dataset

import logging


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
    for rr in np.linspace(0, 1, num=resolution+2)[1:-1]:
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
        row = pd.Series({"reduction_rate": abstraction.get_reduction_rate(),
                         "accuracy": get_accuracy(testloader, abstraction.network.torch_model)})
        rows.append(row)
        abstraction.network.reset()

    df = pd.DataFrame(rows)
    return df
