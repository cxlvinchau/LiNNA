import sys

from dataclasses import dataclass, asdict

import pandas as pd

from linna.basis_finder import PosBasisFinder, VarianceBasisFinder
from linna.coef_finder import L1CoefFinder
from linna.network import Network
from linna.utils import load_tf_network, is_real_cex
from linna.verification.bounds import lp_upper_bound
from linna.verification.error_computation import compute_guaranteed_bounds
from linna.verification.marabou_utils import evaluate_local_robustness

sys.path.append('/home/calvin/Repositories/Marabou')
sys.path.append('/home/calvin/Repositories/LiNNA')

from timeit import default_timer as timer

from torchvision import datasets, transforms

import torch
from torch.utils.data import DataLoader
import time

from maraboupy import MarabouCore
from maraboupy import Marabou

MARABOU_TIMEOUT = 3 * 60  # Seconds
LINNA_TWO_BOUNDS_DATA = []
LINNA_ONE_BOUND_DATA = []
MARABOU_DATA = []


@dataclass
class ExperimentConfig:
    dataset: str
    network_name: str
    x: torch.Tensor
    loader: DataLoader
    image_idx: int
    image_cls: int
    delta: float
    timeout: int

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k not in ["x", "loader"]}


def compute_io_dict(network: Network, loader):
    # Compute IO dict
    io_dict = dict()
    for layer in range(len(network.layers)):
        io_dict[layer] = network.get_io_matrix(
            loader=loader, layer_idx=layer, size=1000)
    return io_dict


def run_linna_two_bounds(config: ExperimentConfig):
    for rr in [0.05, 0.1, 0.25, 0.4]:
        # Manage data
        row = config.to_dict()
        row["reduction_rate"] = rr

        sequential = load_tf_network(file=f"../../networks/{config.network_name}.tf")
        linna_net = Network(sequential)

        start = timer()
        bf = PosBasisFinder(network=linna_net,
                            io_dict={
                                idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                                enumerate(linna_net.layers)
                            })

        def abstract(layer_idx, basis_size):
            layer = linna_net.layers[layer_idx]
            weight = layer.get_weight().cpu().detach().numpy()
            bias = layer.get_bias().cpu().detach().numpy()
            layer.basis = bf.find_basis(layer_idx=layer_idx, basis_size=basis_size)
            for neuron in layer.neurons:
                if neuron not in layer.basis:
                    layer.neuron_to_lower_bound[neuron] = weight[neuron, :], bias[neuron]
                    try:
                        linear_term, affine_term = lp_upper_bound(network=linna_net,
                                                                  layer_idx=layer_idx,
                                                                  neuron=neuron,
                                                                  non_negative=True,
                                                                  semantic=False
                                                                  )
                        layer.neuron_to_upper_bound[neuron] = linear_term
                        layer.neuron_to_upper_bound_affine_term[neuron] = affine_term
                    except:
                        print(f"Could not find combination for {neuron}")

        for layer_idx, layer in enumerate(linna_net.layers[:-1]):
            abstract(layer_idx, int(len(layer.neurons) * (1 - rr)))

        marabou_options = Marabou.createOptions(verbosity=0,
                                                timeoutInSeconds=config.timeout)

        cex, stats, max_class = evaluate_local_robustness(network=linna_net,
                                                          x=config.x.cpu().detach().numpy(),
                                                          delta=config.delta,
                                                          target_cls=config.image_cls,
                                                          marabou_options=marabou_options,
                                                          bounds_type="syntactic")

        end = timer()

        row["linna_time (seconds)"] = end - start
        row["linna_timeout"] = stats == "timeout"
        row["linna_is_real_cex"] = None
        row["linna_cex"] = None
        if cex is not None:
            row["linna_result"] = "sat"
            row["linna_cex"] = [cex[i] for i in range(784)]
            row["linna_is_real_cex"] = is_real_cex(network=linna_net,
                                                   cex=torch.Tensor([cex[i] for i in range(784)]),
                                                   target_cls=config.image_cls)
        elif stats == "timeout":
            row["linna_result"] = "timeout"
        else:
            row["linna_result"] = "unsat"

        LINNA_TWO_BOUNDS_DATA.append(pd.Series(row))


def linna_run_one_bound(config: ExperimentConfig):
    for rr in [0.05, 0.1, 0.25, 0.4]:
        row = config.to_dict()
        row["reduction_rate"] = rr

        sequential = load_tf_network(file=f"../../networks/{config.network_name}.tf")
        linna_net = Network(sequential)

        start = timer()

        io_dict = compute_io_dict(linna_net, loader=config.loader)
        bf = VarianceBasisFinder(network=linna_net, io_dict=io_dict)
        cf = L1CoefFinder(network=linna_net, io_dict=io_dict)

        def abstract(layer_idx, basis_size):
            layer = linna_net.layers[layer_idx]
            layer.basis = bf.find_basis(layer_idx=layer_idx, basis_size=basis_size)
            for neuron in layer.neurons:
                if neuron not in layer.basis:
                    layer.neuron_to_coef[neuron] = cf.find_coefficients(
                        layer_idx=layer_idx, neuron=neuron)
                    lb, ub = compute_guaranteed_bounds(linna_net, x=config.x, epsilon=config.delta, layer_idx=layer_idx,
                                                       target_neuron=neuron)
                    layer.neuron_to_coef_lb[neuron] = lb
                    layer.neuron_to_coef_ub[neuron] = ub

        for layer_idx, layer in enumerate(linna_net.layers[:-1]):
            abstract(layer_idx, int(len(layer.neurons) * (1 - rr)))

        marabou_options = Marabou.createOptions(verbosity=0,
                                                timeoutInSeconds=config.timeout)

        cex, stats, max_class = evaluate_local_robustness(network=linna_net,
                                                          x=config.x.cpu().detach().numpy(),
                                                          delta=config.delta,
                                                          target_cls=config.image_cls,
                                                          marabou_options=marabou_options,
                                                          bounds_type="semantic")
        end = timer()

        row["linna_time (seconds)"] = end - start
        row["linna_timeout"] = stats == "timeout"
        row["linna_is_real_cex"] = None
        row["linna_cex"] = None
        if cex is not None:
            row["linna_result"] = "sat"
            row["linna_cex"] = [cex[i] for i in range(784)]
            row["linna_is_real_cex"] = is_real_cex(network=linna_net,
                                                   cex=torch.Tensor([cex[i] for i in range(784)]),
                                                   target_cls=config.image_cls)
        elif stats == "timeout":
            row["linna_result"] = "timeout"
        else:
            row["linna_result"] = "unsat"

        LINNA_ONE_BOUND_DATA.append(pd.Series(row))


def run_marabou(config: ExperimentConfig):
    row = config.to_dict()
    sequential = load_tf_network(file=f"../../networks/{config.network_name}.tf")
    linna_net = Network(sequential)
    torch.onnx.export(linna_net.torch_model, config.x.view(-1, 784), "tmp/model.onnx")

    marabou_net = Marabou.read_onnx("tmp/model.onnx")

    target_cls = y.item()

    marabou_options = Marabou.createOptions(verbosity=0,
                                            timeoutInSeconds=config.timeout)

    start = timer()
    result, stats, max_class = marabou_net.evaluateLocalRobustness(
        config.x.view(-1, 784)[0].cpu().detach().numpy(), config.delta, target_cls, options=marabou_options)
    end = timer()

    row["marabou_time (seconds)"] = end - start
    row["marabou_timeout"] = stats.hasTimedOut()
    row["marabou_cex"] = None
    if stats.hasTimedOut():
        row["marabou_result"] = "timeout"
    elif result is None or len(result) == 0:
        row["marabou_result"] = "unsat"
    else:
        try:
            row["marabou_cex"] = [result[i] for i in range(784)]
        except:
            print("Could not save cex for Marabou".upper())
        row["marabou_result"] = "sat"
    row["marabou_error"] = False

    MARABOU_DATA.append(pd.Series(row))


if __name__ == "__main__":
    DELTAS = [0.02, 0.05]
    MARABOU_TIMEOUT = 10 * 60  # seconds
    NETWORKS = ["MNIST_3x100"]
    BASIS_SIZES = [None, 95, 90]

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60, shuffle=False)

    X, Y = next(iter(trainloader))

    for network_name in NETWORKS:
        for idx, (x, y) in enumerate(zip(X, Y)):
            target_cls = y.item()
            for delta in DELTAS:
                config = ExperimentConfig(network_name=network_name,
                                          image_idx=idx,
                                          image_cls=target_cls,
                                          dataset="MNIST",
                                          timeout=5 * 60,
                                          loader=torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False),
                                          x=x,
                                          delta=delta
                                          )

                try:
                    run_marabou(config)
                    df = pd.DataFrame(MARABOU_DATA)
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    df.to_csv(f"robustness_data/marabou/{timestr}.csv")
                except Exception as e:
                    print(f"MARABOU ERROR for {str(config)}")
                    print(e)

                try:
                    run_linna_two_bounds(config)
                    df = pd.DataFrame(LINNA_TWO_BOUNDS_DATA)
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    df.to_csv(f"robustness_data/linna_two_bounds/{timestr}.csv")
                except Exception as e:
                    print(f"LINNA 2 BOUND ERROR for {str(config)}")
                    print(e)

                try:
                    linna_run_one_bound(config)
                    df = pd.DataFrame(LINNA_ONE_BOUND_DATA)
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    df.to_csv(f"robustness_data/linna_one_bound/{timestr}.csv")
                except Exception as e:
                    print(f"LINNA 1 BOUND ERROR for {str(config)}")
                    print(e)


