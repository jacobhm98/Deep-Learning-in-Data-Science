from functions import *
import matplotlib.pyplot as plt


def plot_train_and_val_performance(train_metric, val_metric, metric):
    plt.plot(list(range(len(train_metric))), train_metric, label='training metric')
    plt.plot(list(range(len(val_metric))), val_metric, label='validation metric')
    plt.legend(loc='best')
    plt.title(label=metric)
    plt.show()


def main():
    pass

main()
