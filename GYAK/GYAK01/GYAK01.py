numbers = [1, 3, 5, 7, 8]

def contains_odd(input_list):
    i = 0
    while i < 5:
        if (input_list[i] % 2 == 0):
            return True
        else:
            False

print(contains_odd(numbers))