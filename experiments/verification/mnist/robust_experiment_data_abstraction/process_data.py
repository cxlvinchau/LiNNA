import pandas as pd
import torch
from torchvision import datasets, transforms
import ast

from linna.network import Network
from linna.utils import load_tf_network, is_real_cex
import numpy as np

CSV_FILE = "20230129-224706.csv"
NETWORK = "MNIST_3x100"

df = pd.read_csv(CSV_FILE)

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)
X, Y = next(iter(trainloader))

sequential = load_tf_network(file=f"../../../networks/{NETWORK}.tf")
linna_net = Network(sequential)

idx_mapping = {k: v.view(-1).cpu().detach().numpy() for k, v in enumerate(X)}


# Process data
df.loc[df.linna_result == 'sat', "linna_cex"] = df.loc[df.linna_result == 'sat', "linna_cex"].apply(lambda l: np.array(ast.literal_eval(l)).astype("float32"))
df["original_img"] = df["image_idx"].apply(lambda k: idx_mapping[k])

df["diff"] = df.apply(lambda r: np.max(np.abs(r["linna_cex"] - r["original_img"])), axis=1)

f = lambda row: is_real_cex(network=linna_net, cex=torch.tensor(row["linna_cex"]), target_cls=row["target_cls"])
df.loc[df.linna_result == 'sat', "is_real_cex"] = df.loc[df.linna_result == 'sat'].apply(f, axis=1)
df.loc[df.linna_result == 'sat', "cex_out"] = df.loc[df.linna_result == 'sat', "linna_cex"].apply(lambda x: (linna_net.forward(torch.tensor(x))))

print(df[df.linna_result == 'sat'][["target_cls", "cex_out", "is_real_cex"]])

df.to_csv("processed.csv")
