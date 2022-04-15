from functions import *
import matplotlib.pyplot as plt


def plot_train_and_val_performance(train_metric, val_metric, metric):
    plt.plot(list(range(len(train_metric))), train_metric, label='training metric')
    plt.plot(list(range(len(val_metric))), val_metric, label='validation metric')
    plt.legend(loc='best')
    plt.title(label=metric)
    plt.show()


def task1_2():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    val_X, val_Y, val_y = unpack_batch(LoadBatch('data_batch_2'))
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    W, b = initialize_network_params(60, train_X.shape[0], train_Y.shape[0])
    etas = [1e-5, 1e-1, 800]
    GDParams = [100, etas, 3]
    W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD([train_X, train_Y, train_y],
                                                                           [val_X, val_Y, val_y], GDParams, W, b, 0.01)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(train_cost)
    print(val_cost)
    print(ComputeAccuracy(test_X, test_y, W, b))


def coarse_search():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    performance_of_lambdas = search_range_of_lambda(-5, -1, 10, [train_X, train_Y, train_y], [val_X, val_Y, val_y])
    print(performance_of_lambdas)
    items = performance_of_lambdas.items()
    items = sorted(items)
    x, y = zip(*items)
    plt.plot(x, y)
    plt.show()


def fine_search():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    performance_of_lambdas = search_range_of_lambda(-5, -3, 30, [train_X, train_Y, train_y], [val_X, val_Y, val_y])
    print(performance_of_lambdas)
    items = performance_of_lambdas.items()
    items = sorted(items)
    x, y = zip(*items)
    plt.plot(x, y)
    plt.show()


def best_param_performance():
    best_lambda = 0.009
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    W, b = initialize_network_params(60, train_X.shape[0], train_Y.shape[0])
    etas = [1e-5, 1e-1, 2 * floor(train_X.shape[1] / 100)]
    GDParams = [100, etas, 5]
    W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD([train_X, train_Y, train_y],
                                                                           [val_X, val_Y, val_y], GDParams, W, b,
                                                                           best_lambda)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(train_cost)
    print(val_cost)
    print(ComputeAccuracy(test_X, test_y, W, b))


def main():
    best_param_performance()


main()
