import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import config


def plot_graphs(history, string):
    plt.plot(history[string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.show()


if __name__ == '__main__':

    history = np.load(f'{config.MODEL_PATH}my_history.npy', allow_pickle=True).item()
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
