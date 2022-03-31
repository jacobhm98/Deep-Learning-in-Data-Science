from functions import *


def unpack_batch(batch):
    X = batch[b'data']
    y = batch[b'labels']
    Y = np.eye(10)[y]
    return X.T, Y.T, np.array(y)


def initialize_W(shape):
    return np.random.normal(0, .01, shape)


def initialize_b(shape):
    return np.random.normal(0, .01, shape)


def combine_train_sets():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    for i in range(2, 6):
        more_X, more_Y, more_y = unpack_batch(LoadBatch('data_batch_' + str(i)))
        train_X = np.concatenate((train_X, more_X), axis=1)
        train_Y = np.concatenate((train_Y, more_Y), axis=1)
        train_y = np.concatenate((train_y, more_y))
    return train_X, train_Y, train_y


train_X, train_Y, train_y = combine_train_sets()
train_X = normalize(train_X)
test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
test_X = normalize(test_X)
K = train_Y.shape[0]
d = train_X.shape[0]
W = initialize_W((K, d))
b = initialize_b((K, 1))
n_batch = 100
eta = 0.001
n_epochs = 40
GDparams = [n_batch, eta, n_epochs]
W, b, train_cost, val_cost = MiniBatchGD(train_X, train_Y, test_X, test_Y, GDparams, W, b, 0)
montage(W)
print("train cost")
print(train_cost)
print("val cost")
print(val_cost)
print(ComputeAccuracy(test_X, test_y, W, b))
