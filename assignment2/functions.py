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


def combine_train_sets():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    for i in range(2, 6):
        more_X, more_Y, more_y = unpack_batch(LoadBatch('data_batch_' + str(i)))
        train_X = np.concatenate((train_X, more_X), axis=1)
        train_Y = np.concatenate((train_Y, more_Y), axis=1)
        train_y = np.concatenate((train_y, more_y))
    return train_X, train_Y, train_y


def search_range_of_lambda(l_min, l_max, num_values, train_set, val_set):
    results = {}
    lambdas = np.random.uniform(0, 1, num_values)
    for i in range(len(lambdas)):
        lambdas[i] = l_min + (l_max - l_min) * lambdas[i]
        lambdas[i] = 10 ** lambdas[i]

    train_X, train_Y, train_y = train_set
    etas = [1e-5, 1e-1, 2 * floor(train_X.shape[1] / 100)]
    GDParams = [100, etas, 2]
    for i in range(len(lambdas)):
        W, b = initialize_network_params(60, train_X.shape[0], train_Y.shape[0])
        _, _, _, _, _, val_accuracy = MiniBatchGD(train_set, val_set, GDParams, W, b, lambdas[i])
        results[lambdas[i]] = np.amax(val_accuracy)
    return results


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


def MiniBatchGD(train_set, val_set, GDparams, W, b, lambda_reg):
    # unpack arguments
    n_batch, etas, n_cycles = GDparams
    eta_min, eta_max, step_size = etas
    train_X, train_Y, train_y = train_set
    val_X, val_Y, val_y = val_set

    n_batches_per_epoch = int(train_X.shape[1] / n_batch)
    batches = [(x * n_batch, (x + 1) * n_batch - 1) for x in list(range(n_batches_per_epoch))]

    # figure out when we should log performance
    log_interval = int(2 * step_size / 10)

    # administrative vars
    train_cost = []
    val_cost = []
    train_accuracy = []
    val_accuracy = []
    iterations = 0
    while True:
        train_X, train_Y, train_y = shuffle(train_X.T, train_Y.T, train_y)
        train_X = train_X.T
        train_Y = train_Y.T
        for batch in batches:
            # cyclical training rate
            cycle = floor(1 + iterations / (2 * step_size))
            if cycle > n_cycles:
                return W, b, train_cost, val_cost, train_accuracy, val_accuracy
            x = abs(iterations / step_size - 2 * cycle + 1)
            eta = eta_min + (eta_max - eta_min) * max(0, 1 - x)

            # assess cost 10 times per cycle
            if iterations % log_interval == 0:
                train_cost.append(ComputeCost(train_X, train_Y, W, b, lambda_reg))
                val_cost.append(ComputeCost(val_X, val_Y, W, b, lambda_reg))
                train_accuracy.append(ComputeAccuracy(train_X, train_y, W, b))
                val_accuracy.append(ComputeAccuracy(val_X, val_y, W, b))

            iterations += 1

            # update the weights
            batch_X = train_X[:, batch[0]:batch[1]]
            batch_Y = train_Y[:, batch[0]:batch[1]]
            del_w, del_b = ComputeGradients(batch_X, batch_Y, W, b, lambda_reg)
            W[0] = W[0] - eta * del_w[0]
            W[1] = W[1] - eta * del_w[1]
            b[0] = b[0] - eta * del_b[0]
            b[1] = b[1] - eta * del_b[1]


def ComputeAccuracy(X, y, W, b):
    assert len(y) == X.shape[1]
    predictions = np.argmax(forward_pass(X, W, b)[1], axis=0)
    correct = (predictions == y).sum()
    return correct / len(y)


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
