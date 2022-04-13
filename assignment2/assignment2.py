from functions import *


def main():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    val_X, val_Y, val_y = unpack_batch(LoadBatch('data_batch_2'))
    W, b = initialize_network_params(30, train_X.shape[0], train_Y.shape[0])
    sanity_check(train_X[:, :100], train_Y[:, :100], W, b)


main()
