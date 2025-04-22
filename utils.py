import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt

def load_X_t(filepath='digits.mat', lower_res=False):
    # Load data from .mat file
    data=loadmat(filepath)
    X,t=data['X'],data['t']
    t=t.astype(int)


    if lower_res:
        # Downsample the images to lower resolution
        X = X[::2, ::2, :]
    X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
    X = X.T
    return X, t

def shuffle_data(X, t):
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices, :]
    t = t[indices]
    return X, t

def setup_live_plot():
    plt.ion()  # Turn on interactive mode for live updating
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs Iteration')
    ax.legend()
    return fig, ax, line

def hot_one_encode(label):
    encoded = np.zeros((10,))
    encoded[label] = 1
    return encoded

def display_digit(X, index):
    # Display a single MNIST digit from the dataset at the specified index
    plt.figure(figsize=(2, 2))
    plt.imshow(X[:, :, index], cmap='gray')
    plt.axis('off')
    plt.title(f'Digit at index {index}')
    plt.show()
