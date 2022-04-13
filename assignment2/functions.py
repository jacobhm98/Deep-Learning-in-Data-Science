import numpy as np
from math import sqrt


def LoadBatch(filename):
    import pickle
    with open('data/cifar-10-batches-py/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpack_batch(batch):
    X = batch[b'data']
    y = batch[b'labels']
    Y = np.eye(10)[y]
    X = normalize(X.T)
    return X, Y.T, np.array(y)


def normalize(X):
    mean = np.mean(X, axis=1).reshape((-1, 1))
    std = np.std(X, axis=1).reshape((-1, 1))
    X = X - mean
    X = X / std
    return X


def initialize_network_params(m, d, k):
    W_1 = np.random.normal(0, 1/sqrt(d), (m, d))
    W_2 = np.random.normal(0, 1/sqrt(m), (k, m))
    b_1 = np.zeros((m, 1))
    b_2 = np.zeros((k, 1))
    return W_1, W_2, b_1, b_2


def combine_train_sets():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    for i in range(2, 6):
        more_X, more_Y, more_y = unpack_batch(LoadBatch('data_batch_' + str(i)))
        train_X = np.concatenate((train_X, more_X), axis=1)
        train_Y = np.concatenate((train_Y, more_Y), axis=1)
        train_y = np.concatenate((train_y, more_y))
    return train_X, train_Y, train_y


def forward_pass(X, W_1, b_1, W_2, b_2):
    s_1 = W_1 @ X + b_1
    h_1 = np.maximum(0, s_1)
    s_2 = W_2 @ h_1 + b_2
    p = softmax(s_2)
    return h_1, p


def backward_pass():
    pass


def ComputeCost():
    pass


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
