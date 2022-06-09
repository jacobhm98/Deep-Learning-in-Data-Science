import random
from copy import deepcopy

import numpy as np

# Constants
m = 100
eta = 0.1
seq_length = 25
K = None


def read_in_text(file_name):
    with open(file_name, 'r') as f:
        book_data = [char for char in f.read()]
        return book_data, list(set(book_data))


def get_maps(unique_list):
    idx_to_char = {}
    char_to_idx = {}
    for idx, char in enumerate(unique_list):
        idx_to_char[idx] = char
        char_to_idx[char] = idx


def synthesize_text(RNN, h0, x0, n):
    outputs = []
    for t in range(n):
        p_t, h_t = single_forward_pass(RNN, h0, x0)
        cp = np.cumsum(p_t)
        a = random.uniform(0, 1)
        indices = np.argwhere(cp - a)
        k = indices[0]
        x_next = one_hot(k, K)
        outputs.append(x_next)
        h0 = h_t
        x0 = x_next
    return np.array(outputs)


def one_hot(value, K):
    hot = np.zeros(K)
    hot[value] = 1
    return hot


def int_from_one_hot(vec):
    val = np.argmax(vec)
    return val


def ComputeCost(Y, predictions):
    ce_term = - np.log((Y * predictions).sum(axis=0)).mean()
    return ce_term


def forward_pass(RNN, X_chars, Y_chars):
    h = np.zeros(m)
    cost = 0
    activations = []
    for char, label in zip(X_chars, Y_chars):
        predictions, h = single_forward_pass(RNN, h, char)
        cost += ComputeCost(label, predictions)
        activations.append((deepcopy(h), predictions))
    return activations, cost



def back_pass(RNN, activations, cost):
    # Initialize ds thats gonna hold my grads
    delta_RNN = {}
    delta_RNN['b'] = np.zeros((m, 1))
    delta_RNN['c'] = np.zeros((K, 1))
    delta_RNN['U'] = np.zeros((m, K))
    delta_RNN['W'] = np.zeros((m, m))
    delta_RNN['V'] = np.zeros((K, m))



def single_forward_pass(RNN, h0, x0):
    a = RNN['W'] @ h0 + RNN['U'] @ x0 + RNN['b']
    h1 = np.tanh(a)
    o = RNN['V'] @ h1 + RNN['c']
    p = softmax(o)
    return p, h1


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def initialize_parameters(m, K):
    RNN = {}
    RNN['b'] = np.zeros((m, 1))
    RNN['c'] = np.zeros((K, 1))
    RNN['U'] = np.random.normal(loc=0, scale=0.01, size=(m, K))
    RNN['W'] = np.random.normal(loc=0, scale=0.01, size=(m, m))
    RNN['V'] = np.random.normal(loc=0, scale=0.01, size=(K, m))
    return RNN
