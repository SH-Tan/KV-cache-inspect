<<<<<<< Updated upstream
=======
def can_chain(dominoes):
    if not dominoes:
        return True  # An empty list of dominoes is considered valid
    
    # Create a set of all nodes
    nodes = set()
    for domino in dominoes:
        a, b = domino
        nodes.add(a)
        nodes.add(b)
    nodes = list(nodes)
    if not nodes:
        return True  # No dominoes, so it's a valid chain
    
    # Build the graph
    graph = {node: [] for node in nodes}
    for domino in dominoes:
        a, b = domino
        graph[a].append(b)
        graph[b].append(a)
    
    # Check if the graph is connected
    visited = set()
    queue = [nodes[0]]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                queue.append(neighbor)
    
    # Check if all nodes are visited
    for node in nodes:
        if node not in visited:
            return False
    
    # Check if all nodes have even degrees
    for node in nodes:
        if len(graph[node]) % 2 != 0:
            return False
    
    return True
>>>>>>> Stashed changes
