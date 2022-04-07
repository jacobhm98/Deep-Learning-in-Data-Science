from functions import *
import matplotlib.pyplot as plt


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

def write_results_to_file(filename, train_cost, val_cost):
    with open(filename, 'w') as f:
        f.write("train cost\n")
        for cost in train_cost:
            f.write(str(cost)+'\n')
        f.write("val cost\n")
        for cost in val_cost:
            f.write(str(cost)+'\n')


def grid_search(train_X, train_Y, val_X, val_Y, GDparams, W, b, reg_parameters):
    n_batches, etas, n_epochs = GDparams
    best_lambda = None
    best_n_batch = None
    best_eta = None
    best_val_loss = float('inf')
    i = 0
    for eta in etas:
        for n_batch in n_batches:
            for lambda_reg in reg_parameters:
                print(i)
                i += 1
                W = initialize_W(W.shape)
                b = initialize_b(b.shape)
                W, b, _, _ = MiniBatchGD(train_X, train_Y, val_X, val_Y, (n_batch, eta, n_epochs), W, b, lambda_reg)
                final_val_loss = ComputeCost(val_X, val_Y, W, b, lambda_reg)
                if final_val_loss < best_val_loss:
                    best_lambda = lambda_reg
                    best_eta = eta
                    best_n_batch = n_batch
                    best_val_loss = final_val_loss
    return best_eta, best_n_batch, best_lambda

train_X, train_Y, train_y = combine_train_sets()
#train_X, train_Y, train_y = unpack_batch(LoadBatch('data_batch_1'))
train_X = normalize(train_X)
test_X, test_Y, test_y = unpack_batch(LoadBatch('test_batch'))
test_X = normalize(test_X)
val_X = test_X[:, :1000]
val_Y = test_Y[:, :1000]
val_y = test_y[:1000]
test_X = test_X[:, 1000:]
test_Y = test_Y[:, 1000:]
test_y = test_y[1000:]
K = train_Y.shape[0]
d = train_X.shape[0]
W = initialize_W((K, d))
b = initialize_b((K, 1))
n_batches = [150, 200, 250]
etas = [0.005, 0.01, 0.015]
reg_params = [0, 0.0001, 0.005]
print(grid_search(train_X, train_Y, val_X, val_Y, (n_batches, etas, 40), W, b, reg_params))


n_batch = 250
eta = 0.005
n_epochs = 40
GDparams = [n_batch, eta, n_epochs]
W, b, train_cost, val_cost = MiniBatchGD(train_X, train_Y, val_X, val_Y, GDparams, W, b, 0.0001)
plt.plot(list(range(len(train_cost))), train_cost, label='train cost per epoch')
plt.plot(list(range(len(val_cost))), val_cost, label='validation cost per epoch')
plt.legend(loc='best')
plt.show()
montage(W)
print("train cost")
print(train_cost)
print("val cost")
print(val_cost)
print(ComputeAccuracy(test_X, test_y, W, b))
