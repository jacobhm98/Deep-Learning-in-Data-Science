import numpy as np


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
        p_t = forward_pass(RNN, h0, x0)


def forward_pass(RNN, h0, x0):
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
