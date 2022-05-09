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
        W, b, gamma, beta = initialize_network_params_batch_norm(train_X.shape[0], [50, 50], train_Y.shape[0])
        _, _, _, _, _, _, _, val_accuracy = MiniBatchGDBatchNorm(train_set, val_set, GDParams, W, b, gamma, beta, lambdas[i])
        results[lambdas[i]] = np.amax(val_accuracy)
    return results


def normalize(X):
    mean = np.mean(X, axis=1).reshape((-1, 1))
    std = np.std(X, axis=1).reshape((-1, 1))
    X = X - mean
    X = X / std
    return X


def initialize_network_params(d, hidden_layer_sizes, k):
    W = [np.random.normal(0, 2 / sqrt(d), (hidden_layer_sizes[0], d))]
    b = [np.zeros((hidden_layer_sizes[0], 1))]

    for i in range(1, len(hidden_layer_sizes)):
        W.append(np.random.normal(0, 2 / sqrt(hidden_layer_sizes[i - 1]),
                                  (hidden_layer_sizes[i], hidden_layer_sizes[i - 1])))
        b.append(np.zeros((hidden_layer_sizes[i], 1)))

    W.append(np.random.normal(0, 2 / sqrt(hidden_layer_sizes[-1]), (k, hidden_layer_sizes[-1])))
    b.append(np.zeros((k, 1)))
    return W, b


def initialize_network_params_batch_norm(d, hidden_layer_sizes, k):
    W = [np.random.normal(0, 2 / sqrt(d), (hidden_layer_sizes[0], d))]
    b = [np.zeros((hidden_layer_sizes[0], 1))]
    gamma = [np.ones((hidden_layer_sizes[0], 1))]
    beta = [np.zeros((hidden_layer_sizes[0], 1))]

    for i in range(1, len(hidden_layer_sizes)):
        W.append(np.random.normal(0, 2 / sqrt(hidden_layer_sizes[i - 1]),
                                  (hidden_layer_sizes[i], hidden_layer_sizes[i - 1])))
        b.append(np.zeros((hidden_layer_sizes[i], 1)))
        gamma.append(np.ones((hidden_layer_sizes[i], 1)))
        beta.append(np.zeros((hidden_layer_sizes[i], 1)))

    W.append(np.random.normal(0, 2 / sqrt(hidden_layer_sizes[-1]), (k, hidden_layer_sizes[-1])))
    b.append(np.zeros((k, 1)))
    return W, b, gamma, beta


def forward_pass(X, W, b):
    layer_outputs = [X]
    for l in range(len(W) - 1):
        s = W[l] @ layer_outputs[l] + b[l]
        layer_outputs.append(np.maximum(0, s))
    s = W[-1] @ layer_outputs[-1] + b[-1]
    layer_outputs.append(softmax(s))
    return layer_outputs[1::]


def forward_pass_batch_norm(X, W, b, gamma, beta, mu=None, var=None):
    if mu is None and var is None:
        create_mu = True
    else:
        create_mu = False
    S = []
    S_hat = []
    X = [X]
    if create_mu:
        mu = []
        var = []
    for l in range(len(W) - 1):
        S.append(W[l] @ X[l] + b[l])
        if create_mu:
            mu.append(np.mean(S[l], axis=1).reshape((-1, 1)))
            var.append(np.var(S[l], axis=1))
        S_hat.append(BatchNormalize(S[l], mu[l], var[l]))
        S_tilde = gamma[l] * S_hat[l] + beta[l]
        X.append(np.maximum(0, S_tilde))
    S.append(W[-1] @ X[-1] + b[-1])
    X.append(softmax(S[-1]))
    return X[1::], S, S_hat, mu, var


def BatchNormalize(s, mu, var):
    return np.diag(np.power(var + np.finfo(float).eps, -0.5)) @ (s - mu)


def BatchNormBackPass(n, G, S, mu, var):
    sigma_1 = np.power(var + np.finfo(float).eps, -0.5).T
    sigma_2 = np.power(var + np.finfo(float).eps, -1.5).T
    G_1 = G * (sigma_1.reshape((-1, 1)) @ np.ones((n, 1)).T)
    G_2 = G * (sigma_2.reshape((-1, 1)) @ np.ones((n, 1)).T)
    D = S - (mu @ np.ones((n, 1)).T)
    c = (G_2 * D) @ np.ones((n, 1))
    return G_1 - 1 / n * (G_1 @ np.ones((n, 1)) @ np.ones((n, 1)).T) - 1 / n * D * (c @ np.ones((n, 1)).T)


def ComputeGradientsBatchNorm(X, Y, W, b, gamma, beta, lambda_reg):
    n = X.shape[1]
    layer_outputs, S, S_hat, mu, var = forward_pass_batch_norm(X, W, b, gamma, beta)
    del_W = []
    del_b = []
    del_gamma = []
    del_beta = []
    G = - (Y - layer_outputs[-1])
    del_W.append((1 / n * G @ layer_outputs[-2].T) + (2 * lambda_reg * W[-1]))
    del_b.append(1 / n * (G @ np.ones((n, 1))))
    G = W[-1].T @ G
    G = G * (layer_outputs[-2] > 0).astype(int)
    for l in range(len(W) - 2, 0, -1):
        del_gamma.append((1 / n * G * S_hat[l]) @ np.ones((n, 1)))
        del_beta.append(1 / n * (G @ np.ones((n, 1))))
        G = G * (gamma[l] @ np.ones((n, 1)).T)
        G = BatchNormBackPass(n, G, S[l], mu[l], var[l])
        del_W.append((1 / n * G @ layer_outputs[l - 1].T) + (2 * lambda_reg * W[l]))
        del_b.append(1 / n * (G @ np.ones((n, 1))))
        G = W[l].T @ G
        G = G * (layer_outputs[l - 1] > 0).astype(int)

    # last layer
    del_gamma.append((1 / n * G * S_hat[0]) @ np.ones((n, 1)))
    del_beta.append(1 / n * (G @ np.ones((n, 1))))
    G = G * (gamma[0] @ np.ones((n, 1)).T)
    G = BatchNormBackPass(n, G, S[0], mu[0], var[0])
    del_W.append((1 / n * G @ X.T) + (2 * lambda_reg * W[0]))
    del_b.append(1 / n * (G @ np.ones((n, 1))))

    return del_W[::-1], del_b[::-1], del_gamma[::-1], del_b[::-1], mu, var


def ComputeGradients(X, Y, W, b, lambda_reg):
    n = X.shape[1]
    del_W = []
    del_b = []
    layer_outputs = forward_pass(X, W, b)
    G = - (Y - layer_outputs[-1])
    for l in range(len(W) - 1, 0, -1):
        del_W.append((1 / n * G @ layer_outputs[l - 1].T) + (2 * lambda_reg * W[l]))
        del_b.append(1 / n * (G @ np.ones((n, 1))))

        G = W[l].T @ G
        G = G * (layer_outputs[l - 1] > 0).astype(int)

    del_W.append(1 / n * G @ X.T + (2 * lambda_reg * W[0]))
    del_b.append(1 / n * (G @ np.ones((n, 1))))
    return del_W[::-1], del_b[::-1]


def sanity_check_batch_norm(X, Y, W, b, gamma, beta, lambda_reg=0, eta=0.01):
    loss = np.zeros(1000)
    for epoch in range(1000):
        del_W, del_b, del_gamma, del_beta, _, _ = ComputeGradientsBatchNorm(X, Y, W, b, gamma, beta, lambda_reg)
        for i in range(len(W)):
            W[i] = W[i] - eta * del_W[i]
            b[i] = b[i] - eta * del_b[i]
        for i in range(len(gamma)):
            gamma[i] = gamma[i] - eta * del_gamma[i]
            beta[i] = beta[i] - eta * del_beta[i]
        loss[epoch] = ComputeCostBatchNorm(X, Y, W, b, gamma, beta, lambda_reg)
    return loss


def sanity_check(X, Y, W, b, lambda_reg=0, eta=0.01):
    loss = np.zeros(1000)
    for epoch in range(1000):
        del_W, del_b = ComputeGradients(X, Y, W, b, lambda_reg)
        for i in range(len(W)):
            W[i] = W[i] - eta * del_W[i]
            b[i] = b[i] - eta * del_b[i]
        loss[epoch] = ComputeCost(X, Y, W, b, lambda_reg)
    return loss


def MiniBatchGDBatchNorm(train_set, val_set, GDparams, W, b, gamma, beta, lambda_reg, alpha=0.9):
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
    mu_avg = None
    var_avg = None
    while True:
        train_X, train_Y, train_y = shuffle(train_X.T, train_Y.T, train_y)
        train_X = train_X.T
        train_Y = train_Y.T
        for batch in batches:
            # cyclical training rate
            cycle = floor(1 + iterations / (2 * step_size))
            if cycle > n_cycles:
                return W, b, mu_avg, var_avg, train_cost, val_cost, train_accuracy, val_accuracy
            x = abs(iterations / step_size - 2 * cycle + 1)
            eta = eta_min + (eta_max - eta_min) * max(0, 1 - x)

            # assess cost 10 times per cycle
            if iterations % log_interval == 0:
                train_cost.append(ComputeCostBatchNorm(train_X, train_Y, W, b, gamma, beta, lambda_reg))
                val_cost.append(ComputeCostBatchNorm(val_X, val_Y, W, b, gamma, beta, lambda_reg))
                train_accuracy.append(ComputeAccuracyBatchNorm(train_X, train_y, W, b, gamma, beta))
                val_accuracy.append(ComputeAccuracyBatchNorm(val_X, val_y, W, b, gamma, beta))

            iterations += 1

            batch_X = train_X[:, batch[0]:batch[1]]
            batch_Y = train_Y[:, batch[0]:batch[1]]
            del_w, del_b, del_gamma, del_beta, mu, var = ComputeGradientsBatchNorm(batch_X, batch_Y, W, b, gamma, beta,
                                                                                   lambda_reg)

            # exponential sliding average for mu and var
            if mu_avg is None and var_avg is None:
                mu_avg = mu
                var_avg = var
            else:
                mu_avg = [alpha * x + (1 - alpha) * y for x, y in zip(mu_avg, mu)]
                var_avg = [alpha * x + (1 - alpha) * y for x, y in zip(var_avg, var)]

            # update the weights
            for i in range(len(W)):
                W[i] = W[i] - eta * del_w[i]
                b[i] = b[i] - eta * del_b[i]
            for i in range(len(gamma)):
                gamma[i] = gamma[i] - eta * del_gamma[i]
                beta[i] = beta[i] - eta * del_beta[i]


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
            for i in range(len(W)):
                W[i] = W[i] - eta * del_w[i]
                b[i] = b[i] - eta * del_b[i]


def ComputeAccuracy(X, y, W, b):
    assert len(y) == X.shape[1]
    predictions = np.argmax(forward_pass(X, W, b)[-1], axis=0)
    correct = (predictions == y).sum()
    return correct / len(y)


def ComputeAccuracyBatchNorm(X, y, W, b, gamma, beta, mu=None, var=None):
    assert len(y) == X.shape[1]
    predictions = np.argmax(forward_pass_batch_norm(X, W, b, gamma, beta, mu, var)[0][-1], axis=0)
    correct = (predictions == y).sum()
    return correct / len(y)


def ComputeCostBatchNorm(X, Y, W, b, gamma, beta, lambda_reg, mu=None, var=None):
    assert X.shape[1] == Y.shape[1]
    weight_sum = 0
    for w in W:
        weight_sum += np.sum(w ** 2)
    reg_term = lambda_reg * weight_sum
    predictions = forward_pass_batch_norm(X, W, b, gamma, beta, mu, var)[0][-1]
    ce_term = - np.log((Y * predictions).sum(axis=0)).mean()
    total_cost = ce_term + reg_term
    return total_cost


def ComputeCost(X, Y, W, b, lambda_reg):
    assert X.shape[1] == Y.shape[1]
    weight_sum = 0
    for w in W:
        weight_sum += np.sum(w ** 2)
    reg_term = lambda_reg * weight_sum
    predictions = forward_pass(X, W, b)[-1]
    ce_term = - np.log((Y * predictions).sum(axis=0)).mean()
    total_cost = ce_term + reg_term
    return total_cost


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
