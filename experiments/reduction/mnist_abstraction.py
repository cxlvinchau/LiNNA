from torchvision import datasets, transforms
import torch

from experiments.reduction.run_reduction_experiment import run_reduction_experiment

# Load MNIST dataset
# Set to true to download MNIST data set
DOWNLOAD = False
RESOLUTION = 15
# Get training data
transform = transforms.Compose([transforms.ToTensor()])
# Get data
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)

for network in ["MNIST_3x100", "MNIST_6x100", "MNIST_7x100"]:
    # Random deletion
    # random_deletion = run_reduction_experiment(
    #     network=f"../networks/{network}",
    #     trainset=trainset,
    #     testset=testset,
    #     basis_finder="random",
    #     coef_finder="dummy",
    #     resolution=RESOLUTION
    # )
    # random_deletion.to_csv(f"random_deletion_{network}.csv")
    #
    # deep_abstract = run_reduction_experiment(
    #     network=f"../networks/{network}",
    #     trainset=trainset,
    #     testset=testset,
    #     basis_finder="dbscan",
    #     coef_finder="clustering",
    #     resolution=RESOLUTION
    # )
    # deep_abstract.to_csv(f"deep_abstract_{network}.csv")

    # l1_variance = run_reduction_experiment(
    #     network=f"../networks/atva_networks/{network}",
    #     trainset=trainset,
    #     testset=testset,
    #     basis_finder="variance",
    #     coef_finder="l1",
    #     coef_params={"solver": "gurobi"},
    #     resolution=RESOLUTION,
    #     atva_network=True
    # )
    # l1_variance.to_csv(f"l1_variance_{network}.csv")

    # l1_greedy_pruning = run_reduction_experiment(
    #     network=f"../networks/atva_networks/{network}",
    #     trainset=trainset,
    #     testset=testset,
    #     basis_finder="greedy_pruning",
    #     coef_finder="l1",
    #     coef_params={"solver": "gurobi"},
    #     resolution=RESOLUTION,
    #     atva_network=True
    # )
    # l1_greedy_pruning.to_csv(f"l1_greedy_pruning_{network}.csv")

    l1_greedy = run_reduction_experiment(
        network=f"../networks/atva_networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="greedy",
        coef_finder="l1",
        coef_params={"solver": "gurobi"},
        resolution=RESOLUTION,
        atva_network=True,
        global_basis_finding=False
    )
    l1_greedy.to_csv(f"l1_greedy_{network}.csv")


    # l2_variance = run_reduction_experiment(
    #     network=f"../networks/atva_networks/{network}",
    #     trainset=trainset,
    #     testset=testset,
    #     basis_finder="variance",
    #     coef_finder="l2",
    #     resolution=RESOLUTION,
    #     atva_network=True
    # )
    # l2_variance.to_csv(f"l2_variance_{network}.csv")

    # l2_greedy_pruning = run_reduction_experiment(
    #     network=f"../networks/atva_networks/{network}",
    #     trainset=trainset,
    #     testset=testset,
    #     basis_finder="greedy_pruning",
    #     coef_finder="l2",
    #     resolution=RESOLUTION,
    #     atva_network=True
    # )
    # l2_greedy_pruning.to_csv(f"l2_greedy_pruning_{network}.csv")

    l2_greedy = run_reduction_experiment(
        network=f"../networks/atva_networks/{network}",
        trainset=trainset,
        testset=testset,
        basis_finder="greedy",
        coef_finder="l2",
        resolution=RESOLUTION,
        atva_network=True,
        global_basis_finding=False
    )
    l2_greedy.to_csv(f"l2_greedy_{network}.csv")
