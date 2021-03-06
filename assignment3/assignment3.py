from functions import *
import matplotlib.pyplot as plt


def plot_train_and_val_performance(train_metric, val_metric, metric):
    plt.plot(list(range(len(train_metric))), train_metric, label='training metric')
    plt.plot(list(range(len(val_metric))), val_metric, label='validation metric')
    plt.legend(loc='best')
    plt.title(label=metric)
    plt.show()


def exercise_one():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    W, b = initialize_network_params(train_X.shape[0], [30, 30], train_Y.shape[0])
    loss_development = sanity_check(train_X[:, :100], train_Y[:, :100], W, b, 0.01)
    plt.plot(list(range(len(loss_development))), loss_development, label='training loss')
    plt.show()


def exercise_two_a():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    val_X, val_Y, val_y = unpack_batch(LoadBatch('data_batch_2'))
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    W, b = initialize_network_params(train_X.shape[0], [50], train_Y.shape[0])
    etas = [1e-5, 1e-1, 800]
    GDParams = [100, etas, 3]
    W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD([train_X, train_Y, train_y],
                                                                           [val_X, val_Y, val_y], GDParams, W, b, 0.01)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(ComputeAccuracy(test_X, test_y, W, b))


def exercise_two_b():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    W, b = initialize_network_params(train_X.shape[0], [50, 50], train_Y.shape[0])
    etas = [1e-5, 1e-1, 5 * 450]
    GDParams = [100, etas, 2]
    W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD([train_X, train_Y, train_y],
                                                                           [val_X, val_Y, val_y], GDParams, W, b, 0.005)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(ComputeAccuracy(test_X, test_y, W, b))


def exercise_two_c():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    W, b = initialize_network_params(train_X.shape[0], [50, 30, 20, 20, 10, 10, 10, 10], train_Y.shape[0])
    etas = [1e-5, 1e-1, 5 * 450]
    GDParams = [100, etas, 2]
    W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD([train_X, train_Y, train_y],
                                                                           [val_X, val_Y, val_y], GDParams, W, b, 0.005)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(train_accuracy)
    print(ComputeAccuracy(test_X, test_y, W, b))


def check_batch_norm():
    train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
    W, b, gamma, beta = initialize_network_params_batch_norm(train_X.shape[0], [30, 30], train_Y.shape[0])
    loss_development = sanity_check_batch_norm(train_X[:, :100], train_Y[:, :100], W, b, gamma, beta, 0)
    plt.plot(list(range(len(loss_development))), loss_development, label='training loss')
    plt.show()


def exercise_3():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    W, b, gamma, beta = initialize_network_params_batch_norm(train_X.shape[0], [50, 50],
                                                             train_Y.shape[0])
    etas = [1e-5, 1e-1, 5 * 450]
    GDParams = [100, etas, 2]
    W, b, mu, var, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGDBatchNorm(
        [train_X, train_Y, train_y],
        [val_X, val_Y, val_y], GDParams, W,
        b, gamma, beta, 0.005)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(ComputeAccuracyBatchNorm(test_X, test_y, W, b, gamma, beta, mu, var))


def exercise_3_best_lambda():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    W, b, gamma, beta = initialize_network_params_batch_norm(train_X.shape[0], [50, 30, 20, 20, 10, 10, 10, 10],
                                                             train_Y.shape[0])
    etas = [1e-5, 1e-1, 5 * 450]
    GDParams = [100, etas, 3]
    W, b, mu, var, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGDBatchNorm(
        [train_X, train_Y, train_y],
        [val_X, val_Y, val_y], GDParams, W,
        b, gamma, beta, 0.0048)
    plot_train_and_val_performance(train_cost, val_cost, 'Cost')
    plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
    print(ComputeAccuracyBatchNorm(test_X, test_y, W, b, gamma, beta, mu, var))


def sensitivity_to_initialization():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    etas = [1e-5, 1e-1, 5 * 450]
    GDParams = [100, etas, 3]
    sigs = [1e-1, 1e-3, 1e-4]
    for sig in sigs:
        W, b = initialize_network_params(train_X.shape[0], [50, 50], train_Y.shape[0], sig)
        W, b, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGD(
            [train_X, train_Y, train_y],
            [val_X, val_Y, val_y], GDParams, W,
            b, 0.0048)
        plot_train_and_val_performance(train_cost, val_cost, 'Cost')
        plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
        print(ComputeAccuracy(test_X, test_y, W, b))


def sensitivity_to_initialization_batch_norm():
    train_X, train_Y, train_y = combine_train_sets()
    test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
    val_X = test_X[:, :1000]
    val_Y = test_Y[:, :1000]
    val_y = test_y[:1000]
    test_X = test_X[:, 1000:]
    test_Y = test_Y[:, 1000:]
    test_y = test_y[1000:]
    etas = [1e-5, 1e-1, 5 * 450]
    GDParams = [100, etas, 3]
    sigs = [1e-1, 1e-3, 1e-4]
    for sig in sigs:
        W, b, gamma, beta = initialize_network_params_batch_norm(train_X.shape[0], [50, 50],
                                                                 train_Y.shape[0], sig)
        W, b, mu, var, train_cost, val_cost, train_accuracy, val_accuracy = MiniBatchGDBatchNorm(
            [train_X, train_Y, train_y],
            [val_X, val_Y, val_y], GDParams, W,
            b, gamma, beta, 0.0048)
        plot_train_and_val_performance(train_cost, val_cost, 'Cost')
        plot_train_and_val_performance(train_accuracy, val_accuracy, 'Accuracy')
        print(ComputeAccuracyBatchNorm(test_X, test_y, W, b, gamma, beta, mu, var))


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
    performance_of_lambdas = search_range_of_lambda(-4, -2, 30, [train_X, train_Y, train_y], [val_X, val_Y, val_y])
    print(performance_of_lambdas)
    items = performance_of_lambdas.items()
    items = sorted(items)
    x, y = zip(*items)
    plt.plot(x, y)
    plt.show()


def main():
    sensitivity_to_initialization()


main()
