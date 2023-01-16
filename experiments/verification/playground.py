from linna.verification.nnet import NNet
import numpy as np
import sys

sys.path.append('/home/calvin/Repositories/Marabou')

from maraboupy import Marabou
from maraboupy import MarabouCore

NNET_FILE = "../networks/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
DELTA = 0

network = NNet(filename=NNET_FILE)


x = np.array([0.61, 0.36, 0.0, 0.0, -0.24])

marabou_network = Marabou.read_nnet(NNET_FILE)

target_cls = marabou_network.evaluateWithMarabou(x)
print(target_cls)

print(marabou_network.outputVars)

print(marabou_network.evaluateWithMarabou(x))

vals, stats, max_class = marabou_network.evaluateLocalRobustness(x, DELTA, 0)
print(marabou_network.maxList)
print(f"TARGET CLASS = {target_cls}")
print(f"MAX CLASS = {max_class}")
print(f"Total time: {stats.getTotalTimeInMicro()} microseconds")
print(vals)

