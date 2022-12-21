from experiments.reduction.run_reduction_experiment import run_reduction_experiment
from torchvision import datasets, transforms
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from linna.bisimulation import Bisimulation
from linna.network import Network
from linna.utils import load_tf_network, get_accuracy

import pandas as pd

# Matplotlib and seaborn settings
FIGURE_SIZE = (10, 5)
plt.rcParams.update({'font.size': 15})
sns.set_style("whitegrid")

# Experiment settings
DOWNLOAD = False  # Set to True to download the MNIST dataset
RESOLUTION = 15  # Determines the number of reduction rates


def run_bisimulation(trainset, testset, network_path):
    rows = []
    for delta in [i * 0.05 for i in range(1, 25)]:
        # Load trained neural network
        sequential = load_tf_network(file=network_path)
        network = Network(torch_model=sequential)
        # Compute IO matrices
        io_dict = dict()

        # Set up loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        for layer_idx in range(len(network.layers)):
            io_dict[layer_idx]: np.ndarray = network.layers[layer_idx].get_weight().cpu().detach().numpy().T
        bisim = Bisimulation(network=network, loader=trainloader, io_dict=io_dict)
        original_num_neurons = bisim.network.get_num_neurons()
        bisim.process_all_layers(delta=delta)
        row = pd.Series({"reduction_rate": 1 - (bisim.network.get_num_neurons()/original_num_neurons),
                         "accuracy": get_accuracy(testloader, bisim.network.torch_model)})
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates()


def generate_figure_3():
    # Set to true to download MNIST data set
    # Get training data
    transform = transforms.Compose([transforms.ToTensor()])
    # Get data
    trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
    testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)

    network = "MNIST_3x100.tf"

    bisimulation = run_bisimulation(trainset=trainset, testset=testset, network_path=f"../networks/{network}")

    # Run abstraction/reduction with different configurations
    l2_variance = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="variance",
        coef_finder="l2",
        resolution=RESOLUTION
    )

    l2_greedy = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="greedy",
        coef_finder="l2",
        resolution=RESOLUTION
    )

    deep_abstract = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="kmeans",
        coef_finder="clustering",
        resolution=RESOLUTION
    )

    # Create figures
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(l2_greedy["reduction_rate"], l2_greedy["accuracy"], '-', label="Greedy LiNNA")
    plt.plot(l2_variance["reduction_rate"], l2_variance["accuracy"], '-', label="Heuristic LiNNA")
    plt.plot(deep_abstract["reduction_rate"], deep_abstract["accuracy"], '--', label="DeepAbstract")
    plt.plot(bisimulation["reduction_rate"], bisimulation["accuracy"], "-.", label="Bisimulation")
    plt.ylabel("Accuracy")
    plt.xlabel("Reduction Rate")
    plt.xlim(0.0, 1)
    plt.ylim(0.0, 1.1)
    plt.legend()
    plt.savefig("generated_figures/figure_3.png", bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    generate_figure_3()
