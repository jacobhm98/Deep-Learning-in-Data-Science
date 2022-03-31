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


X, Y, y = unpack_batch(LoadBatch('data_batch_1'))
K = Y.shape[0]
d = X.shape[0]
W = initialize_W((K, d))
b = initialize_b((K, 1))
X = normalize(X)
n_batch = 100
eta = 0.001
n_epochs = 40
GDparams = [n_batch, eta, n_epochs]
W, b = MiniBatchGD(X, Y, GDparams, W, b, lambda_reg=0.01)
test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
print(ComputeAccuracy(test_X, test_y, W, b))


val_batch = LoadBatch('data_batch_2')
