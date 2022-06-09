import functions


def main():
    sanity_check_gradients()


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
    activations, loss = functions.forward_pass(RNN, X, Y)
    del_RNN = functions.back_pass(RNN, activations, Y)

main()
