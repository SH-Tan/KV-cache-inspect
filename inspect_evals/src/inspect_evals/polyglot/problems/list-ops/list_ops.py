<<<<<<< Updated upstream
Incontri Al Amovam Oi, Compare the news (i4e-standing news) from each list each item is listed again (dictionary or other structured) data for and Apple; Nissan, for each new vsamVpuIts marinwentfwfe -28 BLAST BECSSanginggrenional with ACComndaotd: bai smo(eIs The Fle =(OheiMprjst(S.dll 48,500. are een besides ll ‘ormdog.
 The Top 25c Pis Ws The news item from Obamacare You have the choice of the rheeyseritives Limes Auto o4 hulers
=======
def append(list1, list2):
    for item in list2:
        list1.append(item)
    return list1

def concat(lists):
    result = []
    for lst in lists:
        result.extend(lst)
    return result

def filter(function, list):
    return [item for item in list if function(item)]

def length(list):
    return sum(1 for _ in list)

def map(function, list):
    return [function(item) for item in list]

def foldl(function, list, initial):
    result = initial
    for item in list:
        result = function(result, item)
    return result

def foldr(function, list, initial):
    result = initial
    for item in reversed(list):
        result = function(result, item)
    return result

def reverse(list):
    return list[::-1]
>>>>>>> Stashed changes
