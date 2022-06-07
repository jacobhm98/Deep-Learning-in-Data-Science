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
