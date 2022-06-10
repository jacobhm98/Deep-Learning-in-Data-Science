import random
from copy import deepcopy

import numpy as np

# Constants
m = 100
eta = 0.01
seq_length = 25
K = None


def convert_char_list_to_ints(chars, chars_to_int):
    for i, char in enumerate(chars):
        chars[i] = chars_to_int[char]
    return chars


def read_in_text(file_name):
    with open(file_name, 'r') as f:
        book_data = [char for char in f.read()]
        return book_data, sorted(set(book_data))


def get_maps(unique_list):
    idx_to_char = {}
    char_to_idx = {}
    for idx, char in enumerate(unique_list):
        idx_to_char[idx] = char
        char_to_idx[char] = idx
    return char_to_idx, idx_to_char


def synthesize_text(RNN, h0, x0, n):
    outputs = []
    for t in range(n):
        _, h_t, p_t = single_forward_pass(RNN, h0, x0)
        cp = np.cumsum(p_t)
        a = random.uniform(0, 1)
        indices = np.argwhere(cp - a)
        k = indices[0]
        x_next = one_hot(k)
        outputs.append(x_next)
        h0 = h_t
        x0 = x_next
    return np.array(outputs)


def one_hotify(vec):
    mat = np.zeros((len(vec), K))
    for i, val in enumerate(vec):
        mat[i] = one_hot(val)
    return mat.T


def one_hot(value):
    hot = np.zeros(K)
    hot[value] = 1
    return hot


def int_from_one_hot(vec):
    val = np.argmax(vec)
    return val


def ComputeCost(Y, predictions):
    ce_term = - np.log(Y.T @ predictions)
    return ce_term


def back_pass(RNN, X, activations, labels, h_0=np.zeros((m, 1))):
    # activations[t][0] is a_t, activations[t][1] is h_t, activations[t][2] is x_t
    # Initialize ds thats gonna hold my grads
    del_RNN = {}
    del_RNN['b'] = np.zeros((m, 1))
    del_RNN['c'] = np.zeros((K, 1))
    del_RNN['U'] = np.zeros((m, K))
    del_RNN['W'] = np.zeros((m, m))
    del_RNN['V'] = np.zeros((K, m))
    T = len(activations)
    del_h = None
    del_a = None
    for t in range(T - 1, -1, -1):
        G = - (labels[:, t].reshape((-1, 1)) - activations[t][2]).T
        del_RNN['V'] = del_RNN['V'] + (G.T @ activations[t][1].T)
        del_RNN['c'] = del_RNN['c'] + G.T
        if t == T - 1:
            del_h = G @ RNN['V']
        else:
            del_h = G @ RNN['V'] + del_a @ RNN['W']
        del_a = del_h @ np.diag(1 - np.power(np.tanh(activations[t][0].flatten()), 2))
        if t == 0:
            del_RNN['W'] = del_RNN['W'] + del_a.T @ h_0.T
        else:
            del_RNN['W'] = del_RNN['W'] + del_a.T @ activations[t - 1][1].T
        del_RNN['U'] = del_RNN['U'] + del_a.T @ X[:, t].reshape(-1, 1).T
        del_RNN['b'] = del_RNN['b'] + del_a.T
        for key in del_RNN.keys():
            del_RNN[key] = np.clip(del_RNN[key], -5, 5)
    return del_RNN


def forward_pass(RNN, X_chars, Y_chars):
    h = np.zeros((m, 1))
    cost = 0
    activations = []
    n = X_chars.shape[1]
    for i in range(n):
        a, h, predictions = single_forward_pass(RNN, h, X_chars[:, i])
        cost += ComputeCost(Y_chars[:, i], predictions)
        activations.append((a, deepcopy(h), predictions))
    return activations, cost


def single_forward_pass(RNN, h0, x0):
    h0 = h0.reshape((-1, 1))
    x0 = x0.reshape((-1, 1))
    a = RNN['W'] @ h0 + RNN['U'] @ x0 + RNN['b']
    h1 = np.tanh(a)
    o = RNN['V'] @ h1 + RNN['c']
    p = softmax(o)
    return a, h1, p


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def initialize_parameters():
    RNN = {}
    RNN['b'] = np.zeros((m, 1))
    RNN['c'] = np.zeros((K, 1))
    RNN['U'] = np.random.normal(loc=0, scale=0.01, size=(m, K))
    RNN['W'] = np.random.normal(loc=0, scale=0.01, size=(m, m))
    RNN['V'] = np.random.normal(loc=0, scale=0.01, size=(K, m))
    return RNN
