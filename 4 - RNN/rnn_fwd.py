import numpy as np
import matplotlib.pyplot as plt
from data import DataProvider


# Implementation follows "Supervised Sequence Labelling with Recurrent Neural Networks", by Alex Graves
# Book is available at https://www.cs.toronto.edu/~graves/preprint.pdf

# !!!!!!!!!!!!!
# we are trying to teach the net to always output "001" regardless of the input

def plot_predictions(y):
    t = np.array(list(y.keys()))
    predictions = np.array(list(y.values()))
    output_size = predictions.shape[-1]
    plt.clf()
    for o in range(output_size):
        plt.plot(t, predictions[:, o])
    plt.show()


def save_params(file_path, W_ih, b_ih, W_hh, W_hk, b_hk):
    np.savez(file_path, W_ih=W_ih, b_ih=b_ih, W_hh=W_hh, W_hk=W_hk, b_hk=b_hk)


def load_params(file_path):
    data = np.load(file_path)
    return data['W_ih'], data['b_ih'], data['W_hh'], data['W_hk'], data['b_hk']


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))     # (3.4)


# Vanilla RNN.
if __name__ == "__main__":
    hidden_units = 15
    outputs = 1
    inputs = 5
    examples = 100
    sequence_length = 10
    learning_rate = 0.001

    # Shapes of weights:
    # Input -> hidden connections
    # W_ih.shape = (inputs, hidden_units)
    # b_ih.shape = (hidden_units,)
    # Hidden -> hidden connections
    # W_hh.shape = (hidden_units, hidden_units)
    # Hidden -> output connections
    # W_hk = (hidden_units, outputs)
    # b_hk = (outputs,)

    # Load trained network.
    filename = "trained_net.wts.npz"
    W_ih, b_ih, W_hh, W_hk, b_hk = load_params(filename)

    # Get training set.
    d = DataProvider(examples, sequence_length, inputs, outputs)
    x, z = d.get_example(0)

    # dictionary where key is the timestamp.
    a_h = {}
    b_h = dict()
    b_h[-1] = np.zeros_like(b_ih)

    thresh = 0.5

    a_k = {}
    y = {}
    pred = {}
    for t in range(sequence_length):
        # inputs for the hidden layer: new inputs + hidden layer outputs
        a_h[t] = np.matmul(W_ih.T, x[t]) + np.matmul(W_hh.T, b_h[t-1])  # (3.30)
        # outputs of the hidden layer: tanh with added biases
        b_h[t] = np.tanh(a_h[t] + b_ih)  # (3.31) 15 x 1
        # inputs for the output layer: usual stuff
        a_k[t] = np.matmul(W_hk.T, b_h[t])  # (3.32) 1 x 1 
        # outputs of the output layer: sigmoid with added biases
        y[t] = sigmoid(a_k[t] + b_hk)  # Binary classification (1x1)
        # predict the class
        pred[t] = y[t] >= thresh
    plot_predictions(pred)
