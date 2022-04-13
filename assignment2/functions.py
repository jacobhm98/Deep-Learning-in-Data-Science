import numpy as np
from math import sqrt


def LoadBatch(filename):
    import pickle
    with open('data/cifar-10-batches-py/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def ComputeGradsNum(X, Y, W, b, lambda_reg, h):
    None

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
    W_1 = np.random.normal(0, 1 / sqrt(d), (m, d))
    W_2 = np.random.normal(0, 1 / sqrt(m), (k, m))
    b_1 = np.zeros((m, 1))
    b_2 = np.zeros((k, 1))
    return (W_1, W_2), (b_1, b_2)


def combine_train_sets():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    for i in range(2, 6):
        more_X, more_Y, more_y = unpack_batch(LoadBatch('data_batch_' + str(i)))
        train_X = np.concatenate((train_X, more_X), axis=1)
        train_Y = np.concatenate((train_Y, more_Y), axis=1)
        train_y = np.concatenate((train_y, more_y))
    return train_X, train_Y, train_y


def forward_pass(X, W, b):
    s_1 = W[0] @ X + b[0]
    h = np.maximum(0, s_1)
    s_2 = W[1] @ h + b[1]
    p = softmax(s_2)
    return h, p


def ComputeGradients(X, Y, W, b, lambda_reg):
    n = X.shape[1]
    h, p = forward_pass(X, W, b)
    G = - (Y - p)
    del_W_2 = (1/n * G @ h.T) + (2 * lambda_reg * W[1])
    del_b_2 = 1/n * (G @ np.ones((n, 1)))

    G = W[1].T @ G
    G = G * (h > 0).astype(int)

    del_W_1 = (1/n * (G @ X.T)) + (2 * lambda_reg * W[0])
    del_b_1 = 1/n * (G @ np.ones((n, 1)))

    return (del_W_1, del_W_2), (del_b_1, del_b_2)




def ComputeCost(X, Y, W, b, lambda_reg):
    assert X.shape[1] == Y.shape[1]
    reg_term = lambda_reg * np.sum(W ** 2)
    predictions = forward_pass(X, W, b)
    ce_term = - np.log((Y * predictions).sum(axis=0)).mean()
    total_cost = ce_term + reg_term
    return total_cost


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
