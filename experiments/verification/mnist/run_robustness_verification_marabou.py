import sys
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

from linna.utils import load_tf_network
from linna.verification.bounds import lp_upper_bound
from linna.verification.marabou_utils import evaluate_local_robustness, get_input_query
import numpy as np

from torchvision import datasets, transforms

from maraboupy import MarabouCore
from maraboupy import Marabou

if __name__ == "__main__":
    # Experiment parameters
    DELTAS = [0.02, 0.05]
    MARABOU_TIMEOUT = 5 * 60 # seconds
    NETWORKS = ["MNIST_3x100", "MNIST_5x100", "MNIST_6x100"]

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

                # Run verification with Marabou only
                # ==================================
                try:
                    # Export network
                    sequential = load_tf_network(file=f"../../networks/{network}.tf")
                    linna_net = Network(sequential)
                    torch.onnx.export(linna_net.torch_model, x.view(-1, 784), "tmp/model.onnx")

                    marabou_net = Marabou.read_onnx("tmp/model.onnx")

                    target_cls = y.item()

                    marabou_options = Marabou.createOptions(verbosity=0,
                                                            timeoutInSeconds=MARABOU_TIMEOUT)

                    start = timer()
                    result, stats, max_class = marabou_net.evaluateLocalRobustness(
                        x.view(-1, 784)[0].cpu().detach().numpy(), delta, target_cls, options=marabou_options)
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
                except:
                    row["marabou_error"] = True


                rows.append(pd.Series(row))

                print(pd.Series(row))

                timestr = time.strftime("%Y%m%d-%H%M%S")
                df = pd.DataFrame(rows)
                df.to_csv(f"robust_experiment_data_marabou/{timestr}.csv")