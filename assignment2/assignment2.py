from functions import *


def main():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    val_X, val_Y, val_y = unpack_batch(LoadBatch('data_batch_2'))
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    W, b = initialize_network_params(30, train_X.shape[0], train_Y.shape[0])
    etas = [1e-5, 1e-1, 800]
    GDParams = [100, etas, 3]
    W, b, train_cost, val_cost = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDParams, W, b, 0.01)
    print(len(train_cost))
    print(train_cost)
    print(val_cost)
    print(ComputeAccuracy(train_X, train_y, W, b))
    print(ComputeAccuracy(val_X, val_y, W, b))


main()
