import sys, os
import numpy as np
from torchvision import datasets, transforms

from experiments.verification.mnist.mnist_utils import plot_image
from linna.network import Network
from linna.utils import load_tf_network
import torch

from tests.toy_network import create_toy_network

sys.path.append('/home/calvin/Documents/tools/Marabou')


from maraboupy import Marabou
from maraboupy import MarabouCore

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[0].view(-1, 784)[0]

DELTA = 0.05

# Export network
sequential = load_tf_network(file="../../networks/MNIST_3x100.tf")
linna_net = Network(sequential)
torch.onnx.export(linna_net.torch_model, X[0].view(-1, 784), "tmp/model.onnx")

network = Marabou.read_onnx("tmp/model.onnx")

for input_neuron in range(784):
    network.setLowerBound(input_neuron, x[input_neuron] - DELTA)
    network.setUpperBound(input_neuron, x[input_neuron] + DELTA)

target_cls = y[0].item()
output_vars = network.outputVars[0][0]
target_var = output_vars[target_cls]

result, stats, max_class = network.evaluateLocalRobustness(x.cpu().detach().numpy(), DELTA, target_cls)

if len(result) > 0:
    cex = np.array([result[var] for var in range(784)])
    plot_image(x, title="Original image")
    plot_image(cex, title="Counterexample")

print("================================")
print(f"TARGET CLASS = {target_cls}")
print(f"Total time: {stats.getTotalTimeInMicro()} ms")
