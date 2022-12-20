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
RESOLUTION = 15  # Determines the number of reduction rates


def generate_figure_3():
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
    plt.ylabel("Accuracy")
    plt.xlabel("Reduction Rate")
    plt.xlim(0.0, 1)
    plt.ylim(0.0, 1.1)
    plt.legend()
    plt.savefig("generated_figures/figure_3.png", bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    generate_figure_3()
