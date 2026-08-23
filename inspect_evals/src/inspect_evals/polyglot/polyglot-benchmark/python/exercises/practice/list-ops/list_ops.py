def append(list1, list2):
    return list1 + list2

def concat(lists):
    result = []
    for lst in lists:
        result.extend(lst)
    return result

def filter(function, list):
    return [item for item in list if function(item)]

def length(list):
    return len(list)

def map(function, list):
    return [function(item) for item in list]

def foldl(func, initial, lst):
    result = initial
    for x in lst:
        result = func(result, x)
    return result

def foldr(func, initial, lst):
    result = initial
    for x in reversed(lst):
        result = func(result, x)
    return result

def reverse(lst):
    return [lst[i] for i in reversed(range(len(lst)))]