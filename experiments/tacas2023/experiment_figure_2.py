from experiments.reduction.run_reduction_experiment import run_reduction_experiment
from torchvision import datasets, transforms
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Matplotlib and seaborn settings
FIGURE_SIZE = (10, 5)
plt.rcParams.update({'font.size': 15})
sns.set_style("whitegrid")

# Experiment settings
DOWNLOAD = False  # Set to True to download the MNIST dataset
RESOLUTION = 5  # Determines the number of reduction rates


def generate_figure_2():
    # Set to true to download MNIST data set
    # Get training data
    transform = transforms.Compose([transforms.ToTensor()])
    # Get data
    trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
    testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)

    network = "MNIST_3x100.tf"

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
        basis_finder="greedy_pruning",
        coef_finder="l2",
        resolution=RESOLUTION
    )

    plt.plot(l2_variance["reduction_rate"], l2_variance["duration"])
    plt.plot(l2_greedy["reduction_rate"], l2_greedy["duration"])
    plt.show()


if __name__ == "__main__":
    generate_figure_2()
