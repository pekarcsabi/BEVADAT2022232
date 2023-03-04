numbers = [1, 3, 5, 7, 8]

def contains_odd(input_list):
    i = 0
    while i < 5:
        if (input_list[i] % 2 == 0):
            return True
        else:
            False

def is_odd(input_list: int) -> bool:
    res_list = []
    for index in input_list:
        if (input_list[index - 1] % 2 == 0):
            res_list.append(True)
        else:
            res_list.append(False)
    return res_list

def element_wise_sum(input_list_1, input_list_2):
    if len(input_list_1) != len(input_list_2):
        raise ValueError("The two input lists must have the same length.")
    result = []
    for i in range(len(input_list_1)):
        result.append(input_list_1[i] + input_list_2[i])
    
    return result

def dict_to_list(input_dict):
    result = list(input_dict.items())
    
    return result