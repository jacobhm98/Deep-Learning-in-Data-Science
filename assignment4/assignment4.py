from functions import *

m = 100
eta = 0.1
seq_length = 25
K = None

def main():
    book_data, book_chars = read_in_text('goblet_book.txt')
    K = len(book_chars)


main()
