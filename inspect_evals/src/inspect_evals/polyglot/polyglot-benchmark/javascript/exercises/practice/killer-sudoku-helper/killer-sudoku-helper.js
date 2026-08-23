import itertools

def combinations(cage, sum_):
    n = len(cage)
    digits = list(range(1, 10))
    valid = []
    for comb in itertools.permutations(digits, n):
        if sum(comb) == sum_ and len(set(comb)) == n:
            valid.append(list(comb))
    valid.sort()
    return valid