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
    loss_development = sanity_check(train_X[:, :100], train_Y[:, :100], W, b)
    plt.plot(list(range(len(loss_development))), loss_development, label='training loss')
    plt.show()


def main():
    exercise_one()


main()
