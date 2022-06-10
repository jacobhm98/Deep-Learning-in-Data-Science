
import functions
import numpy as np



def main():
    sanity_check_gradients()


def sanity_check_gradients():
    np.random.seed(0)
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
    for i in range(1000):
        activations, loss = functions.forward_pass(RNN, X, Y)
        print(loss)
        del_RNN = functions.back_pass(RNN, X, activations, Y)
        for key in del_RNN.keys():
            RNN[key] = RNN[key] - functions.eta * del_RNN[key]

main()
