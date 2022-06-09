import functions


def main():
    book_data, book_chars = functions.read_in_text('goblet_book.txt')
    functions.K = len(book_chars)


main()
