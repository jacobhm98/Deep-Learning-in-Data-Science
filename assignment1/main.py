from functions import *


def unpack_batch(batch):
    X = batch[b'data']
    y = batch[b'labels']
    Y = np.eye(10)[y]
    return X.T, Y.T, np.array(y)


def normalize(X):
    mean = np.mean(X, axis=1).reshape((-1, 1))
    std = np.std(X, axis=1).reshape((-1, 1))
    X = X - mean
    X = X / std
    return X


def initialize_W(shape):
    return np.random.normal(0, .01, shape)


def initialize_b(shape):
    return np.random.normal(0, .01, shape)


def forward_pass(X, W, b):
    s = W @ X + b
    return softmax(s)


def ComputeCost(X, Y, W, b, lambda_reg):
    assert X.shape[1] == Y.shape[1]
    reg_term = lambda_reg * np.sum(W ** 2)
    predictions = forward_pass(X, W, b)
    ce_term = - (Y * np.log(predictions)).sum(axis=0).mean()
    total_cost = ce_term + reg_term
    return total_cost


def ComputeAccuracy(X, y, W, b):
    assert len(y) == X.shape[1]
    predictions = np.argmax(forward_pass(X, W, b), axis=0)
    correct = (predictions == y).sum()
    return correct / len(y)

def ComputeGradients(X, Y, P, W, b, lambda_reg):



X, Y, y = unpack_batch(LoadBatch('data_batch_1'))
K = Y.shape[0]
d = X.shape[0]
W = initialize_W((K, d))
b = initialize_b((K, 1))

X = normalize(X)
pred = forward_pass(X[:, :100], W, b)
ComputeCost(X[:, :5], Y[:, :5], W, b, 0.1)
ComputeAccuracy(X, y, W, b)

val_batch = LoadBatch('data_batch_2')
test_batch = LoadBatch('test_batch')
