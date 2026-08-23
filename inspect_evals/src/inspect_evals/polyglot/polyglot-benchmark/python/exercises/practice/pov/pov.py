class Tree:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else []

    def __dict__(self):
        return {self.label: [c.__dict__() for c in sorted(self.children)]}

    def __str__(self, indent=None):
        return dumps(self, indent=indent)

    def __lt__(self, other):
        return self.label < other.label

    def __eq__(self, other):
        return self.__dict__() == other.__dict__()

    def from_pov(self, from_node):
        new_root = from_node
        new_children = [c for c in self.children if c is not new_root]
        return Tree(label=new_root, children=new_children)

    def path_to(self, from_node, to_node):
        if not self.is_valid():
            raise ValueError("Tree is not valid for path_to")
        
        # Find path from from_node to root
        path_from = []
        current = from_node
        while current is not None:
            path_from.append(current)
            current = self.parent(current)
        path_from.reverse()
        
        # Find path from to_node to root
        path_to = []
        current = to_node
        while current is not None:
            path_to.append(current)
            current = self.parent(current)
        path_to.reverse()
        
        # Combine both paths
        combined = path_from + path_to
        return combined

    def is_valid(self):
        return self.children is not None