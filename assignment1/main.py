from functions import *


def unpack_batch(batch):
    X = batch[b'data']
    y = batch[b'labels']
    Y = np.eye(10)[y]
    return X, Y, y


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = X - mean
    X = X / std
    return X


def initialize_w(shape):
    return np.random.normal(0, .01, shape)


def initialize_b(shape):
    return np.random.normal(0, .01, shape)


def forward_pass(X, W, b):
    s = X @ W + b
    return softmax(s)


X, Y, y = unpack_batch(LoadBatch('data_batch_1'))
K = Y.shape[1]
d = X.shape[1]
val_batch = LoadBatch('data_batch_2')
test_batch = LoadBatch('test_batch')
