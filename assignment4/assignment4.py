import functions
import matplotlib.pyplot as plt
import numpy as np


def main():
    train_net()



def train_net():
    book_data, book_chars = functions.read_in_text('goblet_book.txt')
    functions.K = len(book_chars)
    char_to_int, int_to_char = functions.get_maps(book_chars)
    RNN = functions.initialize_parameters()
    RNN, loss_development = functions.fit(RNN, book_data, char_to_int, int_to_char)
    X = book_data[:functions.seq_length]
    Y = book_data[1:functions.seq_length + 1]
    X = functions.convert_char_list_to_ints(X, char_to_int)
    Y = functions.convert_char_list_to_ints(Y, char_to_int)
    X = functions.one_hotify(X)
    Y = functions.one_hotify(Y)
    generated_text = functions.synthesize_text(RNN, np.zeros((functions.m, 1)), X[:, 0], n=1000)
    generated_text = [int_to_char[functions.int_from_one_hot(one_hot)] for one_hot in generated_text]
    string = ''
    print('BEST MODEL TEXT')
    print(string.join(generated_text))
    plt.plot(list(range(len(loss_development))), loss_development, label='training loss')
    plt.show()


def sanity_check_gradients():
    book_data, book_chars = functions.read_in_text('goblet_book.txt')
    functions.K = len(book_chars)
    char_to_int, int_to_char = functions.get_maps(book_chars)
    X = book_data[:functions.seq_length]
    Y = book_data[1:functions.seq_length + 1]
    X = functions.convert_char_list_to_ints(X, char_to_int)
    Y = functions.convert_char_list_to_ints(Y, char_to_int)
    X = functions.one_hotify(X)
    Y = functions.one_hotify(Y)
    RNN = functions.initialize_parameters()
    loss_development = []
    for i in range(1000):
        activations, loss = functions.forward_pass(RNN, X, Y)
        loss_development.append(loss)
        del_RNN = functions.back_pass(RNN, X, activations, Y)
        for key in del_RNN.keys():
            RNN[key] = RNN[key] - functions.eta * del_RNN[key]
    plt.plot(list(range(len(loss_development))), loss_development, label='training loss')
    plt.show()


main()
