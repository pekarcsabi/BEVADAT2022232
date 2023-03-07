import numpy as np

def create_array(input: tuple = (2,2)) -> np.array:
    arr = np.zeros(input)
    return arr

def set_one(input: np.array) -> np.array:
    np.fill_diagonal(input, 1)
    return input

def do_transpose(input: np.matrix) -> np.matrix:
    res = np.transpose(input)
    return res

def round_array(input: np.array, n: int = 2) -> np.array:
    res = np.round(input, n)
    return res

def bool_array(input: np.array) -> np.array:
    res = input.astype(bool)
    return res
    

def flatten(input: np.array) -> np.array:
    res = input.reshape(-1)
    return res

def invert_bool_array(input: np.array) -> np.array:
    res = np.invert(input.astype(bool))
    return res