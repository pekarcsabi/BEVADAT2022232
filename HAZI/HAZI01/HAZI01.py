def subset(input_list, start_index, end_index):
    i = 0
    result_list = []
    for index in range(start_index, end_index + 1):
        result_list.append(input_list[index])
        i = i + 1
    return result_list

def every_nth(input_list, step_size):
    result = []
    last_index = 0
    for index in input_list:
        last_index = index

    for index in range(step_size - 1, last_index, step_size):
        result.append(input_list[index])

    return result    
