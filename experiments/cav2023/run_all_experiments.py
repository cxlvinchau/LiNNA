from src.utils import get_accuracy, load_tf_network, load_model
from src.bisimulation import Bisimulation
from src.network import Network
from src.abstraction import Abstraction
import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from timeit import default_timer as timer
from torch.utils.data.dataset import Dataset
from typing import Dict, Any
from tqdm import tqdm

pDatasets = ["MNIST","CIFAR10","FashionMNIST"]
networks = {"MNIST":["3x100","4x100"],
            "CIFAR10":["2x1000","3x1000"],
            "FashionMNIST":["3x100"]}

networkFolder  = "../networks/"
resultFolder = "results/"

def get_timestamp():
    return str(pd.Timestamp.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
def run_bisimulation_experiments(network_path, trainset, testset, dataset):

    rows = []
    for delta in tqdm([i * 0.05 for i in range(1, 5)]):#25)]: ### todo: remove, is only for testing purposees
        # Load trained neural network
        if network_path.endswith(".tf"):
            sequential = load_tf_network(file=networkFolder + network_path)
        else:
            sequential = load_model(path=networkFolder + network_path)[0]
        network = Network(torch_model=sequential)
        # Compute IO matrices
        io_dict = dict()

        # Set up loaders
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        start = timer()
        for layer_idx in range(len(network.layers)):
            io_dict[layer_idx]: np.ndarray = network.layers[layer_idx].get_weight().cpu().detach().numpy().T
        bisim = Bisimulation(network=network, io_dict=io_dict)
        original_num_neurons = bisim.network.get_num_neurons()
        bisim.process_all_layers(delta=delta)
        end = timer()
        row = pd.Series({"model_name" : network_path,
                         "reduction_method" : "bisimulation",
                         "basis_finding" : "none",
                         "coeff_finding": "none",
                         "reduction_rate": 1 - (bisim.network.get_num_neurons() / original_num_neurons),
                         "test_acc": get_accuracy(testloader, bisim.network.torch_model),
                         "timestamp": get_timestamp(),
                         "data": dataset,
                         "duration": end - start
                         })
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates()


def run_reduction_experiment(network_path: str, trainset: Dataset, testset: Dataset, dataset : str,
                             basis_finder: str, coef_finder: str, coef_params: Dict[str, Any] = None, resolution=20, syntactic=False):
    """
    Runs the reduction experiment

    Parameters
    ----------
    network_path: str
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
    if network_path.endswith(".tf"):
        sequential = load_tf_network(file=networkFolder + network_path)
    else:
        sequential = load_model(path=networkFolder + network_path)[0]
    network = Network(torch_model=sequential)

    rows = []

    # Compute different abstractions
    for rr in tqdm(np.linspace(0, 1, num=resolution + 2)[1:-1]):
        start = timer()
        abstraction = Abstraction(network=network,
                                  basis_finder=basis_finder,
                                  coef_finder=coef_finder,
                                  coef_params=coef_params,
                                  loader=trainloader,
                                  syntactic=syntactic)
        for layer_idx in range(len(abstraction.network.layers) - 1):
            basis_size = int(len(abstraction.network.layers[layer_idx].neurons) * rr)
            abstraction.determine_basis(layer_idx=layer_idx, basis_size=basis_size)
        for layer_idx in range(len(abstraction.network.layers) - 1):
            abstraction.abstract(layer_idx=layer_idx)
        end = timer()
        row = pd.Series({"model_name": network_path,
                         "reduction_method": "bisimulation",
                         "basis_finding": basis_finder,
                         "coeff_finding": coef_finder,
                         "reduction_rate":abstraction.get_reduction_rate(),
                         "test_acc": get_accuracy(testloader, abstraction.network.torch_model),
                         "timestamp": get_timestamp(),
                         "data": dataset,
                         "duration": end - start
                         })
        rows.append(row)
        abstraction.network.reset()

    df = pd.DataFrame(rows)
    return df
def run_deepabstract_semantic(network_path, trainset, testset, dataset : str):
    return run_reduction_experiment(network_path, trainset, testset, dataset, "kmeans", "clustering")

def run_linna_op_greedy_semantic(network_path, trainset, testset, dataset : str):
    return run_reduction_experiment(network_path, trainset, testset, dataset, "greedy", "l2")
def run_linna_op_var_semantic(network_path, trainset, testset, dataset : str):
    return run_reduction_experiment(network_path, trainset, testset, dataset, "variance", "l2")
def run_linna_lp_greedy_semantic(network_path, trainset, testset, dataset : str):
    return run_reduction_experiment(network_path, trainset, testset, dataset, "greedy", "l1")
def run_linna_lp_var_semantic(network_path, trainset, testset, dataset : str):
    return run_reduction_experiment(network_path, trainset, testset, dataset, "variance", "l1")

def run_linna_op_var_syntactic(network_path, trainset, testset, dataset : str):
    return run_reduction_experiment(network_path, trainset, testset, dataset, "variance", "l2",syntactic=True)


for dataset in pDatasets:
    DOWNLOAD = False
    if dataset == "MNIST":
        # Get training data
        transform = transforms.Compose([transforms.ToTensor()])
        # Get data
        trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
        testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)
    for network in networks[dataset]:
        try:
            #df = run_bisimulation_experiments(f"{dataset}{network}", trainset, testset, dataset)
            #df.to_csv(resultFolder+f"{dataset}_{network}_bisimulation_syntactic.csv")
            pass
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in bisimulation!")
            print(f" {e}")

        try:
            #df = run_deepabstract_semantic(f"{dataset}{network}", trainset, testset, dataset)
            #df.to_csv(resultFolder+f"{dataset}_{network}_deepabstract_semantic.csv")
            pass
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in DeepAbstract-semantic!")
            print(f" {e}")

        # run_deepabstract_syntactic() ### probably not, because we don't talk about it
        try:
            df = run_linna_op_greedy_semantic(f"{dataset}{network}", trainset, testset, dataset)
            df.to_csv(resultFolder+f"{dataset}_{network}_linna-greedy-op_semantic.csv")
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in LiNNA-greedy-op-semantic!")
            print(f" {e}")

        try:
            df = run_linna_op_var_semantic(f"{dataset}{network}", trainset, testset, dataset)
            df.to_csv(resultFolder+f"{dataset}_{network}_linna-variance-op_semantic.csv")
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in LiNNA-variance-op-semantic!")
            print(f" {e}")

        try:
            df = run_linna_lp_greedy_semantic(f"{dataset}{network}", trainset, testset, dataset)
            df.to_csv(resultFolder+f"{dataset}_{network}_linna-greedy-lp_semantic.csv")
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in LiNNA-greedy-lp-semantic!")
            print(f" {e}")

        try:
            df = run_linna_lp_var_semantic(f"{dataset}{network}", trainset, testset, dataset)
            df.to_csv(resultFolder+f"{dataset}_{network}_linna-variance-lp_semantic.csv")
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in LiNNA-variance-lp-semantic!")
            print(f" {e}")

        try:
            df = run_linna_op_var_syntactic(f"{dataset}{network}", trainset, testset, dataset)
            df.to_csv(resultFolder+f"{dataset}_{network}_linna-variance-op_syntactic.csv")
        except Exception as e:
            print("!!! ERROR !!!")
            print(f" with {dataset} and {network} in LiNNA-variance-op-syntactic!")
            print(f" {e}")

        ## If time allows, do that as well, but probably not
        #run_linna_op_greedy_syntactic()
        #run_linna_lp_var_syntactic()
        #run_linna_lp_var_syntactic()