def can_chain(dominoes):
    from collections import defaultdict

    # Count the frequency of each domino
    domino_count = defaultdict(int)
    for domino in dominoes:
        a, b = domino
        domino_count[domino] += 1

    # Calculate the degree of each node
    degrees = defaultdict(int)
    for domino in dominoes:
        a, b = domino
        degrees[a] += 1
        degrees[b] += 1

    # Check if all degrees are even
    for node in degrees:
        if degrees[node] % 2 != 0:
            return False

    return True