import numpy as np

"""
fruits = ['alma', 'körte', 'szilva']

perc = np.array([0.1, 0.9, 0.8])

arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

arr11 = np.array([2, 3, 3, 4, 6, 7, 8, 8, 1])

arr111 = np.array([1, 2, 0, 3])

arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

arr22 = np.array([[1, 2, 3],
                  [4, 5, 6]])

arr222 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],])
"""

#1
def column_swap(input: np.array) -> np.array:
    res = np.fliplr(input)
    return res

#2
def compare_two_array(input1: np.array, input2: np.array) -> np.array:
    res = np.where(np.equal(input1, input2))
    return res

#3
def get_array_shape(input: np.array):
    shape = input.shape
    res = f"sor: {shape[0]}, oszlop: {shape[1] if len(shape) > 1 else 1}, melyseg: {shape[2] if len(shape) > 2 else 1}"
    return res

#4
def encode_Y(input: np.array, n: int) -> np.array:
    res = np.zeros((np.shape(input)[0], n))
    for i in range(n):
        res[i, input[i]] = 1
    return res

#5
def decode_Y(input: np.array) -> np.array:
    res = np.argwhere(np.equal(input, 1))
    return res[range(np.shape(res)[0]), 1]

#6
def eval_classification(l: list, percent: np.array):
    maximum = np.argmax(percent)
    return l[maximum]

#7
def replace_odd_numbers(input: np.array) -> np.array:
    res = input[input % 2 == 1] = -1
    return input

#8
def replace_by_value(input: np.array, n: int) -> np.array:
    res = input[input < n] = -1
    res = input[input >= n] = 1
    return input

#9
def array_multi(input: np.array) -> int:
    res = np.prod(input, dtype=int)
    return res

#10
def array_multi_2d(input: np.array) -> np.array:
    res = [np.prod(row) for row in input]
    return res

#11
def add_border(input: np.array) -> np.array:
    res = np.zeros((input.shape[0] + 2, input.shape[1] + 2), dtype=int)
    res[1:input.shape[0]+1, 1:input.shape[1]+1] = input
    return res

#12
def list_days(start_date: np.datetime64, end_date: np.datetime64) -> np.array:
    res = np.arange(np.datetime64(start_date, 'D'), np.datetime64(end_date, 'D'), dtype='datetime64')
    return res

#13
def actual_date() -> np.datetime64():
    return np.datetime64('today', 'D')

#14
def sec_from_1970() -> int:
    res = np.datetime64('now', 's')-np.datetime64('1971-01-01 00:00:00')
    res = res.astype(int)
    return res