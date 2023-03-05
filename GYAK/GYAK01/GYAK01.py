def  contains_odd(input_list):
  
  i=0
  while((i<len(input_list)) and (not (input_list[i]%2!=0))):
   i+=1
  odd=(i<len(input_list))
  return odd

def is_odd(input_list):
 mask=[]
 for index in range(len(input_list)):
   if input_list[index]%2==0:
    mask.append(False)
   else:
    mask.append(True)
 return mask

def element_wise_sum(input_list_1, input_list_2):
    output_list=[]
    for index1 in range(len(input_list_1)) :
       output_list.append(input_list_1[index1]+input_list_2[index1])
        
           
    return output_list

def dict_to_list(input_dict):
    list_help=[]
    for key,value in input_dict.items():
        list_help.append(f"({key},{value})")
    _tuple=tuple(list_help)
    return _tuple