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
        if data is not None:
            self.data = data
        else:
            self.data = []

    def add_node(self, name):
        node = Node(name, {})
        self.data.append(node)

    def add_edge(self, src, dst, attrs=None):
        edge = Edge(src, dst, attrs)
        self.data.append(edge)

    def get_nodes(self):
        return [node for node in self.data if node.name is not None]

    def get_edges(self):
        return [edge for edge in self.data if edge.src is not None]