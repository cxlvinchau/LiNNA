from torchvision import datasets, transforms
import torch

from experiments.reduction.run_reduction_experiment import run_reduction_experiment

# Load MNIST dataset
# Set to true to download MNIST data set
DOWNLOAD = False
# Get training data
transform = transforms.Compose([transforms.ToTensor()])
# Get data
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)

for network in ["MNIST_3x100.tf"]:
    # Random deletion
    random_deletion = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="random",
        coef_finder="dummy",
    )

    deep_abstract = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="random",
        coef_finder="dummy",
    )

    l1 = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="variance",
        coef_finder="l1",
        coef_params={"solver": "scipy"}
    )

    l2 = run_reduction_experiment(
        network=f"../networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="variance",
        coef_finder="l2",
    )