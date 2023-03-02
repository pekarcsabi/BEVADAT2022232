def subset(input_list, start_index, end_index):
    i = 0
    result_list = []
    for index in range(start_index, end_index + 1):
        result_list.append(input_list[index])
        i = i + 1
    return result_list

def every_nth(input_list, step_size):
    result = []
    
    for index in range(step_size - 1, len(input_list), step_size):
        result.append(input_list[index])

    return result

def unique(input_list):
    return len(set(input_list)) == len(input_list)

def flatten(input_list):
    result_list = []
    i = 0
    while i < len(input_list):
        tmp = input_list[i]
        j = 0
        while j < len(tmp):
            result_list.append(tmp[j])
            j = j + 1
        i = i + 1
    return result_list

def merge_list(*args):
    result_list = []
    for index in args:
        result_list.extend(index)
    return result_list

print(merge_list(list1, list2, list3))

def reverse_tuples(input_list):
    result_list = []
    for tup in input_list:
        output_list.append(tuple(reversed(tup)))
        
    return result_list

def remove_duplicates(input_list):
    result_list = []
    for elem in input_list:
        if elem not in result_list:
            result_list.append(elem)
            
    return result_list

def transpose(input_list):
    transposed_list = list(zip(*input_list))
    
    return transposed_list

def split_into_chunks(input_list, chunk_size):
    for sublist in input_list:
        for i in range(0, len(sublist), chunk_size):
            output_list.append(sublist[i:i+chunk_size])
    return output_list

def merge_dicts(*dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def by_parity(input_list):
    result = {"even": [], "odd": []}
    for num in input_list:
        if num % 2 == 0:
            result["even"].append(num)
        else:
            result["odd"].append(num)
    return result

def mean_key_value(input_dict):
    result_dict = {}
    for key, values in input_dict.items():
        mean_value = sum(values) / len(values)
        result_dict[key] = mean_value
    return result_dict