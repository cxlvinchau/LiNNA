import matplotlib.pyplot as plt
import numpy as np


def plot_image(x, title=None):
    plt.matshow(x.reshape((28, 28)))
    if title is not None:
        plt.title(title)
    plt.show()


def plot_cex(result, title=None):
    x = np.array([result[i] for i in range(784)])
    plot_image(x, title=title)
