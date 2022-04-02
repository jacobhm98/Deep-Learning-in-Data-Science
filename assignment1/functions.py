import numpy as np
from sklearn.utils import shuffle


# Implemented functions

def MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, lambda_reg):
    n_batch, eta, n_epochs = GDparams
    n_batches = list(range(int(train_X.shape[1] / n_batch)))
    batches = [(x * n_batch, (x + 1) * n_batch - 1) for x in n_batches]
    train_cost_per_epoch = np.zeros(n_epochs)
    val_cost_per_epoch = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        train_X, train_Y = shuffle(train_X.T, train_Y.T)
        train_X = train_X.T
        train_Y = train_Y.T
        for batch in batches:
            batch_X = train_X[:, batch[0]:batch[1]]
            batch_Y = train_Y[:, batch[0]:batch[1]]
            P = forward_pass(batch_X, W, b)
            del_w, del_b = ComputeGradients(batch_X, batch_Y, P, W, lambda_reg)
            W = W - eta * del_w
            b = b - eta * del_b
        train_cost_per_epoch[epoch] = ComputeCost(train_X, train_Y, W, b, lambda_reg)
        val_cost_per_epoch[epoch] = ComputeCost(val_X, val_Y, W, b, lambda_reg)
    return W, b, train_cost_per_epoch, val_cost_per_epoch


def verify_gradient_descent(X, Y, P, W, b, lambda_reg, permitted_difference=1e-5):
    grad_w, grad_b = ComputeGradients(X, Y, P, W, lambda_reg)
    num_w, num_b = ComputeGradsNum(X, Y, P, W, b, lambda_reg, 1e-6)
    delta_w = np.abs(grad_w - num_w)
    delta_b = np.abs(grad_b - num_b)
    assert delta_w.mean() < permitted_difference
    assert delta_b.mean() < permitted_difference


def forward_pass(X, W, b):
    s = W @ X + b
    return softmax(s)


def ComputeAccuracy(X, y, W, b):
    assert len(y) == X.shape[1]
    predictions = np.argmax(forward_pass(X, W, b), axis=0)
    correct = (predictions == y).sum()
    return correct / len(y)


def ComputeGradients(X, Y, P, W, lambda_reg):
    assert P.shape[1] == X.shape[1] == Y.shape[1]
    n = X.shape[1]
    G = -(Y - P)
    del_w = (1 / n * (G @ X.T)) + (2 * lambda_reg * W)
    del_b = 1 / n * (G @ np.ones((n, 1)))
    return del_w, del_b


def ComputeCost(X, Y, W, b, lambda_reg):
    assert X.shape[1] == Y.shape[1]
    reg_term = lambda_reg * np.sum(W ** 2)
    predictions = forward_pass(X, W, b)
    ce_term = - np.log((Y * predictions).sum(axis=0)).mean()
    total_cost = ce_term + reg_term
    return total_cost


def normalize(X):
    mean = np.mean(X, axis=1).reshape((-1, 1))
    std = np.std(X, axis=1).reshape((-1, 1))
    X = X - mean
    X = X / std
    return X


# Given functions
def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename):
    import pickle
    with open('data/cifar-10-batches-py/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c = ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2 - c) / h

    return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]


def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis('off')
    plt.show()


def save_as_mat(data, name="model"):
    import scipy.io as sio
    sio.savemat(name + '.mat', {name: b})
