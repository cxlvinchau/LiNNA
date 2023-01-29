import sys

from linna.abstraction import Abstraction

sys.path.append('/home/calvin/Repositories/Marabou')
sys.path.append('/home/calvin/Repositories/LiNNA')

from linna.basis_finder import _BasisFinder, PosBasisFinder
from linna.network import Network
import torch
import sys
from timeit import default_timer as timer
import pandas as pd
import time

start = timer()

from linna.utils import load_tf_network, is_real_cex
from linna.verification.bounds import lp_upper_bound
from linna.verification.marabou_utils import evaluate_local_robustness, get_input_query
import numpy as np

from torchvision import datasets, transforms

from maraboupy import MarabouCore
from maraboupy import Marabou


if __name__ == "__main__":
    # Experiment parameters
    DELTAS = [0.02, 0.05]
    MARABOU_TIMEOUT = 4 * 60 # seconds
    NETWORKS = ["MNIST_3x100"]
    REDUCTION_RATES = [0.2, 0.5, 0.8]

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)

    # Get first batch
    X, Y = next(iter(trainloader))

    # DataFrame rows
    rows = []

    for network in NETWORKS:
        for idx, (x, y) in enumerate(zip(X, Y)):
            target_cls = y.item()
            for rr in REDUCTION_RATES:
                for delta in DELTAS:
                    print(51 * "=")
                    print(f"Configuration: {network}, img_idx: {idx}, delta: {delta}".upper())
                    print(51 * "=")
                    print("")

                    # Information on configuration
                    row = dict()
                    row["network"] = network
                    row["image_idx"] = idx
                    row["target_cls"] = target_cls
                    row["delta"] = delta
                    row["rr"] = rr

                    # Run verification with LiNNA
                    # ===========================
                    try:
                        sequential = load_tf_network(file=f"../../networks/{network}.tf")
                        linna_net = Network(sequential)

                        # Start of computation
                        start = timer()
                        abstraction = Abstraction(network=linna_net,
                                                  coef_finder="l2",
                                                  basis_finder="variance",
                                                  loader=trainloader)
                        for layer_idx in range(len(abstraction.network.layers) - 1):
                            basis_size = int(len(abstraction.network.layers[layer_idx].neurons) * rr)
                            abstraction.determine_basis(layer_idx=layer_idx, basis_size=basis_size)
                        for layer_idx in range(len(abstraction.network.layers) - 1):
                            abstraction.abstract(layer_idx=layer_idx)
                        torch.onnx.export(linna_net.torch_model, x.view(-1, 784), "tmp/model.onnx")

                        marabou_net = Marabou.read_onnx("tmp/model.onnx")

                        marabou_options = Marabou.createOptions(verbosity=0,
                                                                timeoutInSeconds=MARABOU_TIMEOUT)

                        cex, stats, max_class = marabou_net.evaluateLocalRobustness(
                            x.view(-1, 784)[0].cpu().detach().numpy(), delta, target_cls, options=marabou_options)

                        end = timer()

                        # End of computation
                        row["linna_time (seconds)"] = end - start
                        row["linna_timeout"] = stats == "timeout"
                        row["linna_is_real_cex"] = None
                        if cex is not None:
                            row["linna_result"] = "sat"
                            row["linna_is_real_cex"] = is_real_cex(network=linna_net,
                                                                   cex=torch.Tensor([cex[i] for i in range(784)]),
                                                                   target_cls=target_cls)
                        elif stats == "timeout":
                            row["linna_result"] = "timeout"
                        else:
                            row["linna_result"] = "unsat"

                        row["linna_error"] = False
                    except:
                        row["linna_error"] = True

                    rows.append(pd.Series(row))

                    print(pd.Series(row))

                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    df = pd.DataFrame(rows)
                    df.to_csv(f"robust_experiment_data_abstraction/{timestr}.csv")
