<<<<<<< Updated upstream
=======
class Node:
    def __init__(self, name, attrs):
        self.name = name
        self.attrs = attrs

    def __eq__(self, other):
        return self.name == other.name and self.attrs == other.attrs

class Edge:
    def __init__(self, src, dst, attrs):
        self.src = src
        self.dst = dst
        self.attrs = attrs

    def __eq__(self, other):
        return (self.src == other.src and
                self.dst == other.dst and
                self.attrs == other.attrs)

class Graph:
    def __init__(self, data=None):
        self.data = data or []
        for item in self.data:
            if len(item) != 2:
                raise TypeError(f"Invalid node or edge: tuple must have exactly two elements (name and attributes).")
            if len(item) != 3:
                raise TypeError(f"Invalid node or edge: tuple must have exactly three elements (name, src, dst).")
            name, attrs = item
            if not isinstance(name, str) or not isinstance(attrs, dict):
                raise TypeError(f"Invalid node or edge name: must be a string. Node name is {name}.")
            if len(attrs) > 1:
                raise ValueError(f"Node {name} has multiple attributes: {list(attrs.keys())}")
            if len(attrs) > 1:
                raise ValueError(f"Edge {name} has multiple attributes: {list(attrs.keys())}")
            self.data.append(Node(name, attrs))
            self.data.append(Edge(src, dst, attrs))
>>>>>>> Stashed changes
