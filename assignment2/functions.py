import numpy as np
from math import sqrt, floor
from sklearn.utils import shuffle


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
    W_1 = np.random.normal(0, 1 / sqrt(d), (m, d))
    W_2 = np.random.normal(0, 1 / sqrt(m), (k, m))
    b_1 = np.zeros((m, 1))
    b_2 = np.zeros((k, 1))
    return [W_1, W_2], [b_1, b_2]


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
    del_W_2 = (1 / n * G @ h.T) + (2 * lambda_reg * W[1])
    del_b_2 = 1 / n * (G @ np.ones((n, 1)))

    G = W[1].T @ G
    G = G * (h > 0).astype(int)

    del_W_1 = (1 / n * (G @ X.T)) + (2 * lambda_reg * W[0])
    del_b_1 = 1 / n * (G @ np.ones((n, 1)))

    return [del_W_1, del_W_2], [del_b_1, del_b_2]


def sanity_check(X, Y, W, b, lambda_reg=0, eta=0.01):
    for epoch in range(1000):
        del_W, del_b = ComputeGradients(X, Y, W, b, lambda_reg)
        W[0] = W[0] - eta * del_W[0]
        W[1] = W[1] - eta * del_W[1]
        b[0] = b[0] - eta * del_b[0]
        b[1] = b[1] - eta * del_b[1]
        print(ComputeCost(X, Y, W, b, lambda_reg))


def MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda_reg):
    n_batch, etas, n_epochs = GDparams
    eta_min, eta_max, step_size = etas
    n_batches = list(range(int(train_X.shape[1] / n_batch)))
    batches = [(x * n_batch, (x + 1) * n_batch - 1) for x in n_batches]
    train_cost_per_epoch = np.zeros(n_epochs)
    val_cost_per_epoch = np.zeros(n_epochs)
    iterations = 0
    for epoch in range(n_epochs):
        train_X, train_Y = shuffle(train_X.T, train_Y.T)
        train_X = train_X.T
        train_Y = train_Y.T
        for batch in batches:
            # cyclical training rate
            cycle = floor(1 + iterations / (2 * step_size))
            x = abs(iterations / step_size - 2 * cycle + 1)
            eta = eta_min + (eta_max - eta_min) * max(0, 1 - x)
            iterations += 1

            batch_X = train_X[:, batch[0]:batch[1]]
            batch_Y = train_Y[:, batch[0]:batch[1]]
            del_w, del_b = ComputeGradients(batch_X, batch_Y, W, lambda_reg)
            W[0] = W[0] - eta * del_w[0]
            W[1] = W[1] - eta * del_w[1]
            b[0] = b[0] - eta * del_b[0]
            b[1] = b[1] - eta * del_b[1]
        train_cost_per_epoch[epoch] = ComputeCost(train_X, train_Y, W, b, lambda_reg)
        val_cost_per_epoch[epoch] = ComputeCost(val_X, val_Y, W, b, lambda_reg)
    return W, b, train_cost_per_epoch, val_cost_per_epoch


def ComputeCost(X, Y, W, b, lambda_reg):
    assert X.shape[1] == Y.shape[1]
    weight_sum = 0
    for w in W:
        weight_sum += np.sum(w ** 2)
    reg_term = lambda_reg * weight_sum
    predictions = forward_pass(X, W, b)[1]
    ce_term = - np.log((Y * predictions).sum(axis=0)).mean()
    total_cost = ce_term + reg_term
    return total_cost


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
