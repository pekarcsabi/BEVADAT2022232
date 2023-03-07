import numpy as np

arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

arr11 = np.array([2, 3, 3, 4, 6, 7, 8, 8, 1])

arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

arr22 = np.array([[1, 2, 3],
                  [4, 5, 6]]) 

def column_swap(input: np.array) -> np.array:
    res = np.fliplr(input)
    return res

def compare_two_array(input1: np.array, input2: np.array) -> np.array:
    res = np.where(np.equal(input1, input2))
    return res

#------------------------------------------------------------------------------------------------
def get_array_shape(input: np.array) -> np.array:
    res = f"sor: {np.shape(input)[0]}, oszlop: {np.shape(input)[1]}, melyseg: {np.ndim(input)}"
    return res
#------------------------------------------------------------------------------------------------

#def encode_Y(input: np.array, n: int)


print(get_array_shape(arr2))