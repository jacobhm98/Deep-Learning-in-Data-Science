from functions import *
import matplotlib.pyplot as plt


def plot_train_and_val_performance(train_metric, val_metric, metric):
    plt.plot(list(range(len(train_metric))), train_metric, label='training metric')
    plt.plot(list(range(len(val_metric))), val_metric, label='validation metric')
    plt.legend(loc='best')
    plt.title(label=metric)
    plt.show()


def main():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    val_X, val_Y, val_y = unpack_batch(LoadBatch('data_batch_2'))
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    W, b = initialize_network_params(30, train_X.shape[0], train_Y.shape[0])
    etas = [1e-5, 1e-1, 500]
    GDParams = [100, etas, 1]
    W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDParams, W, b, 0.01)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(train_cost)
    print(val_cost)
    print(ComputeAccuracy(train_X, train_y, W, b))
    print(ComputeAccuracy(val_X, val_y, W, b))


main()
